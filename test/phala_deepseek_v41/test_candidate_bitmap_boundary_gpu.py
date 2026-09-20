"""Small real-kernel DSPARK candidate boundary test; no weights or model server.

Run each arm in a new process. Old arm stops BEFORE unsafe bitmap sort or sparse
metadata when actual producer IDs violate its contract. No CPU skip behavior.
"""
import argparse
import json
import sys
from types import SimpleNamespace

import torch

CONTEXT = 1048576
INPUT = 1048542
VERIFY = 6
BLOCK = 8
TOPK = 2048
INDEX_PAGE = 64
TABLE_TOKENS = CONTEXT + 512
BITMAP_BLOCKS = 131072


def production_lengths(data, ratio, arm):
    """Call the actual backend layout integration, not a fixture clamp.

    Old arm must run in the old image. Fixed arm must run in the corrected
    image; if the backend still publishes oversized lengths, its real producer
    fails our pre-sort invariant and the dangerous downstream calls are withheld.
    """
    core = data['backend'].make_core_attn_metadata(
        req_to_token=data['req_to_token'],
        req_pool_indices_repeated=data['request_ids'].long(),
        seq_lens_casual=data['raw'], max_seq_len=TABLE_TOKENS,
        out_loc=(data['raw'].long() - 1).clamp_min(0), need_compress=True,
    )
    return core.c1_topk_lengths_clamp1 if ratio == 1 else core.c2_topk_lengths_clamp1


def inputs(ratio, mixed):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor, store_fp4_index_k_cache,
    )
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    torch.manual_seed(20260920 + ratio)
    rows = 18 if mixed else VERIFY
    # Keep the incident logical stride16392 even for ratio2; no tiny alias table.
    pages = TABLE_TOKENS // INDEX_PAGE
    cache2d = torch.empty_strided((pages, INDEX_PAGE * 68),
        ((INDEX_PAGE * 68 + 511) // 512 * 512, 1), dtype=torch.uint8, device='cuda')
    # Real FP4 store layout, repeated physical pages. Every logical page has
    # allocated backing; identical page contents merely bound initialization cost.
    one_page = torch.zeros_like(cache2d[:1])
    keys = torch.randn((INDEX_PAGE, 128), device='cuda', dtype=torch.bfloat16)
    store_fp4_index_k_cache(keys, one_page, torch.arange(INDEX_PAGE, device='cuda'), page_size=INDEX_PAGE, rne=True)
    cache2d.copy_(one_page.expand_as(cache2d))
    cache = cache2d.view(pages, INDEX_PAGE, 1, 68)
    table = torch.arange(pages, dtype=torch.int32, device='cuda').expand(rows, -1).contiguous()
    query = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device='cuda')
    q, sf = quantize_fp4_indexer_tensor(query.flatten(0, 1), rne=True)
    q, sf = q.view(rows, 1, 32, 64), sf.view(rows, 1, 32)
    weights = torch.rand((rows, 32), device='cuda', dtype=torch.float32) / 32
    request_ids = torch.arange(rows // VERIFY, device='cuda', dtype=torch.int32).repeat_interleave(VERIFY)
    req_to_token = torch.arange(TABLE_TOKENS, dtype=torch.int32, device='cuda').expand(rows // VERIFY, -1).contiguous()
    # Construct only metadata operands/state. No production method is patched,
    # and no model runner/weights are initialized. This is the real backend type
    # so any helper methods called by the correction resolve to production code.
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.page_size = 256
    backend.swa_page_size = 128
    backend.max_context_len = CONTEXT
    backend.MAX_SEQ_LEN_FOR_CAPTURE = TABLE_TOKENS
    backend.req_to_token = req_to_token
    backend.low_ratios = (1, 2)
    backend.present_ratios = (1, 2)
    backend.index_topk = 512
    backend.cuda_int32_kwargs = dict(device='cuda', dtype=torch.int32)
    backend.trtllm_attn = False
    backend.is_dsv41 = True
    backend.is_dspark_draft = False
    backend.encoder_replay = False
    backend.token_to_kv_pool = SimpleNamespace(request_window=None,
        full_to_swa_index_mapping=torch.arange(TABLE_TOKENS, dtype=torch.int32, device='cuda'))
    return dict(q=q, sf=sf, cache=cache, page_table=table, weights=weights,
        request_ids=request_ids, req_to_token=req_to_token, backend=backend,
        raw=torch.zeros(rows, device='cuda', dtype=torch.int32))


def raw_lengths(generated, mixed):
    # Inclusive causal KV lengths for six verify queries after accepted prefix.
    values = [INPUT + generated + offset + 1 for offset in range(VERIFY)]
    if mixed:
        values += list(range(123, 129)) + [0] * VERIFY
    return values


def producer(data, ratio, arm):
    import deep_gemm
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_row_lens
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import amax_topk_blocks
    from sglang.srt.layers.attention.dsv4.indexer import deep_gemm_fp4_paged_mqa_logits
    lengths = production_lengths(data, ratio, arm)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths.unsqueeze(-1), INDEX_PAGE, deep_gemm.get_num_sms())
    dense = deep_gemm_fp4_paged_mqa_logits(
        (data['q'], data['sf']), data['cache'], data['weights'], lengths,
        data['page_table'], schedule, TABLE_TOKENS,
    )
    nblocks, valid = candidate_row_lens(lengths, TOPK)
    blocks = amax_topk_blocks(dense, lengths, nblocks, TOPK)
    return lengths, dense, blocks, valid


def producer_invariant(lengths, blocks, capacity=BITMAP_BLOCKS):
    count = ((lengths + BLOCK - 1) // BLOCK).clamp_max(TOPK)
    live = torch.arange(TOPK, device='cuda')[None, :] < count[:, None]
    values = blocks[live]
    return bool(((values >= 0) & (values < capacity)).all().item())


def chain(data, ratio):
    from sglang.kernels.ops.attention.dsv4.candidate_table import sort_candidate_blocks
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        SparseBlockTable, build_sparse_indexer_schedule, sparse_logits,
    )
    lengths, dense, blocks, valid = producer(data, ratio, 'fixed')
    # This chain is single-stream. SparseBlockTable.ready is intentionally unused
    # by sparse_logits; no imitation of production's alt-stream/event lifetime.
    physical = sort_candidate_blocks(blocks, lengths, data['page_table'], INDEX_PAGE)
    schedule = build_sparse_indexer_schedule(blocks, lengths, data['page_table'],
        INDEX_PAGE, data['q'].dtype, data['request_ids'])
    table = SparseBlockTable(blocks=blocks, schedule=schedule,
        phys_blocks=physical, valid_lens=valid, ready=None)
    sparse = sparse_logits(data['q'], data['sf'], data['cache'], data['weights'].to(torch.bfloat16), table)
    return lengths, dense, blocks, valid, physical, sparse


def validate(data, outputs, ratio, mixed):
    lengths, dense, blocks, valid, physical, sparse = outputs
    # Oracle only: production_lengths itself must use the actual patched layout.
    # Internal speculative lookahead is valid beyond the semantic context.
    # A repair must preserve it rather than silently change attention inputs.
    expected_lens = (data['raw'] // ratio).clamp_min(1)
    torch.testing.assert_close(lengths, expected_lens, rtol=0, atol=0)
    assert producer_invariant(lengths, blocks, TABLE_TOKENS // BLOCK)
    counts = ((lengths + 7) // 8).clamp_max(TOPK)
    live_blocks = torch.arange(TOPK, device='cuda')[None, :] < counts[:, None]
    assert (blocks[live_blocks] < (lengths[:, None].expand_as(blocks)[live_blocks] + 7) // 8).all()
    assert (physical[live_blocks] >= 0).all()
    # Identity logical→physical table means the transform must preserve block ID.
    torch.testing.assert_close(physical[live_blocks], blocks[live_blocks])
    assert (blocks[~live_blocks] == torch.iinfo(torch.int32).max).all()
    columns = blocks.long()[:, :, None] * 8 + torch.arange(8, device='cuda')[None, None, :]
    columns = columns.reshape(lengths.numel(), TOPK * 8)
    active = torch.arange(TOPK * 8, device='cuda')[None, :] < valid[:, None]
    assert (columns[active] < lengths[:, None].expand_as(columns)[active]).all()
    # BF16 sparse logits and FP32 dense logits are two native implementations
    # on the same packed KV. Compare only causally valid sparse output columns.
    expected = dense.gather(1, columns.clamp(0, dense.shape[1] - 1))
    assert torch.isfinite(sparse[active]).all()
    torch.testing.assert_close(sparse[active].float(), expected[active], rtol=0.03, atol=0.03)
    if mixed:
        assert lengths[-VERIFY:].tolist() == [1] * VERIFY
        assert valid[-VERIFY:].tolist() == [1] * VERIFY


def sort_controls():
    """Compare both bitmap specializations against an independent torch oracle."""
    from sglang.kernels.ops.attention.dsv4.candidate_table import sort_candidate_blocks
    for extent in (CONTEXT, TABLE_TOKENS, 2 * CONTEXT):
        lengths = torch.tensor([1, 16383, 16384, extent - 1, extent],
                               dtype=torch.int32, device='cuda')
        table = torch.arange(extent // INDEX_PAGE, dtype=torch.int32,
                             device='cuda').expand(5, -1).contiguous()
        selected = torch.full((5, TOPK), -1, device='cuda', dtype=torch.int32)
        expected = torch.full_like(selected, torch.iinfo(torch.int32).max)
        for row, length in enumerate(lengths.tolist()):
            n = (length + BLOCK - 1) // BLOCK
            if n <= TOPK:
                # Identity branch must ignore uninitialized/padded input.
                expected[row, :n] = torch.arange(n, device='cuda')
            else:
                ids = torch.cat((torch.arange(TOPK - 1, device='cuda'),
                                 torch.tensor([n - 1], device='cuda'))).int()
                selected[row] = ids[torch.randperm(TOPK, device='cuda')]
                expected[row] = ids.sort().values
        physical = sort_candidate_blocks(selected, lengths, table, INDEX_PAGE)
        torch.cuda.synchronize()
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        torch.testing.assert_close(physical, expected, rtol=0, atol=0)
    print('SORT_CONTROLS_PASS old_extent scratch_extent two_million_extent', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('legacy-producer', 'fixed'), required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU test NOT RUN: CUDA required'
    assert torch.cuda.get_device_capability()[0] == 10, 'SM10x required'
    # Publish the single-process configuration through the runtime's test API;
    # production metadata reads parallel configuration even without weights.
    from sglang.srt.runtime_context import get_context
    runtime_config = get_context().override_server_args(tp_size=1)
    runtime_config.install()
    import deep_gemm
    assert hasattr(deep_gemm, 'get_paged_sparse_mqa_logits_metadata')
    assert hasattr(deep_gemm, 'fp8_fp4_paged_sparse_mqa_logits')
    if args.arm == 'legacy-producer':
        data = inputs(1, False)
        data['raw'].copy_(torch.tensor(raw_lengths(31, False), device='cuda', dtype=torch.int32))
        lengths, dense, blocks, valid = producer(data, 1, args.arm)
        torch.cuda.synchronize()
        assert not producer_invariant(lengths, blocks), 'legacy producer did not reproduce invalid block: inspect fixture/source'
        print(json.dumps({'arm': args.arm, 'status': 'EXPECTED_PRODUCER_INVARIANT_FAILURE',
            'lengths': lengths.tolist(), 'max_block': int(blocks.max()),
            'bitmap_blocks': BITMAP_BLOCKS, 'unsafe_sort_and_sparse_metadata_executed': False}), flush=True)
        return 23
    checks = []
    sort_controls()
    for ratio in (1, 2):
        for mixed in (False, True):
            data = inputs(ratio, mixed)
            for generated in (0, 27, 28, 29, 30, 31):
                data['raw'].copy_(torch.tensor(raw_lengths(generated, mixed), device='cuda', dtype=torch.int32))
                # Validate actual producer before any bitmap kernel or schedule.
                staged = producer(data, ratio, 'fixed')
                torch.cuda.synchronize()
                assert producer_invariant(staged[0], staged[2], TABLE_TOKENS // BLOCK)
                del staged
                output = chain(data, ratio)
                torch.cuda.synchronize()
                validate(data, output, ratio, mixed)
                del output
                checks.append({'mode': 'eager', 'ratio': ratio, 'mixed': mixed, 'generated': generated})
            # Precompile/initialize every kernel on a side stream before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                warm = chain(data, ratio)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            del warm
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = chain(data, ratio)
            for generated in (0, 31, 27, 30, 31):
                data['raw'].copy_(torch.tensor(raw_lengths(generated, mixed), device='cuda', dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                validate(data, captured, ratio, mixed)
                checks.append({'mode': 'graph', 'ratio': ratio, 'mixed': mixed, 'generated': generated})
            del graph, captured, data
    print(json.dumps({'arm': 'fixed', 'status': 'PASS', 'checks': checks,
        'logical_context': CONTEXT, 'table_tokens': TABLE_TOKENS,
        'weights_loaded': False, 'single_stream_only': True}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
