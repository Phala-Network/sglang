"""The inputs and outputs of the V4.1 indexer backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, TypeVar

import msgspec
import torch

from sglang.srt.utils.common import ceil_align, ceil_div

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata


def get_tail_row_indices(
    full_rows_per_request: List[int],
    tail_rows_per_request: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Row indices of each request's tail, copied without a host sync."""
    rows, start = [], 0
    for n, t in zip(full_rows_per_request, tail_rows_per_request):
        rows.extend(range(start + n - t, start + n))
        start += n
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(rows, dtype=torch.int64, device=device)
    staged = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
    return staged.to(device, non_blocking=True)


class CandidateMetadata:
    def tail(self, rows_per_request: List[int]) -> CandidateMetadata:
        raise NotImplementedError(f"{type(self).__name__} is not a prefill publish")


class Selection(msgspec.Struct, frozen=True):
    page_indices: torch.Tensor  # index-K pool slots, ascending
    raw_indices: Optional[torch.Tensor]  # the same picks as compressed positions

    def reset(self) -> None:
        self.page_indices.fill_(-1)
        if self.raw_indices is not None:
            self.raw_indices.fill_(-1)


class RowShard(msgspec.Struct, frozen=True, kw_only=True):
    """This rank's share of a prefill chunk's query rows across a group whose
    ranks all hold the same rows.

    The indexer is replicated, so every rank of the attention-TP group would
    score every row identically. Each rank scores a contiguous share instead and
    the selections are all-gathered. A dense score depends only on its own query
    row, so the gathered selection is bitwise the replicated one."""

    group: GroupCoordinator
    rank: int
    world: int
    # The score kernel groups 128 // heads consecutive rows into one MMA tile;
    # aligned shares keep every tile's row set the same as unsharded.
    row_align: int

    def rows_per_rank(self, num_rows: int) -> int:
        return ceil_align(ceil_div(num_rows, self.world), self.row_align)

    def local_rows(self, num_rows: int) -> slice:
        per = self.rows_per_rank(num_rows)
        start = min(self.rank * per, num_rows)
        return slice(start, min(start + per, num_rows))

    def gather(self, local: torch.Tensor, num_rows: int) -> torch.Tensor:
        """``[num_rows, ...]`` from every rank's ``[rows_per_rank, ...]`` share;
        the last share's padding rows fall past ``num_rows`` and are dropped."""
        gathered = local.new_empty((self.world * local.shape[0], *local.shape[1:]))
        self.group.all_gather_into_tensor(gathered, local.contiguous())
        return gathered[:num_rows]


class PrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An eager extend: the backend projects the queries and gathers the index K."""

    indexer: DeepseekV41Indexer
    layer_id: int
    compress_ratio: int
    freqs_cis: torch.Tensor
    x: torch.Tensor  # [rows, hidden] bf16
    q_lora: torch.Tensor  # [rows, q_lora_rank]
    positions: torch.Tensor  # [rows] absolute position of each row

    # [rows] req_to_token row of each query row; None under prefill CP, where a
    # rank's rows are an interleaved subset of the batch's.
    req_rows: Optional[torch.Tensor]
    req_pool_indices: torch.Tensor  # [requests] req_to_token row of each request
    # [rows, pages] int32 at the KV page size; the index-K pool pages it with
    # expand_index_page_table.
    kv_page_table: torch.Tensor
    # Visible full-length context of each request at its newest row, and the
    # query rows of each request in row order.
    seq_lens_cpu: Optional[List[int]]
    rows_per_request: Optional[List[int]]
    rows_per_request_device: Optional[torch.Tensor]
    row_shard: Optional[RowShard] = None


class DecodeInputs(msgspec.Struct, frozen=True, kw_only=True):
    """Decode (one row per request) or verify (one row per draft token)."""

    indexer: DeepseekV41Indexer
    layer_id: int
    compress_ratio: int
    freqs_cis: torch.Tensor
    x: torch.Tensor  # [rows, hidden] bf16
    q_lora: torch.Tensor  # [rows, q_lora_rank]
    positions: torch.Tensor  # [rows] absolute position of each row
    req_rows: torch.Tensor  # [rows] req_to_token row of each query row
    paged_metadata: PagedIndexerMetadata
    is_verify: bool


class CapturedPrefillInputs(msgspec.Struct, frozen=True, kw_only=True):
    """An extend under the prefill graph: projected queries, paged index K."""

    indexer: DeepseekV41Indexer
    layer_id: int
    q: torch.Tensor  # [rows, heads, 128] bf16, roped
    weights: torch.Tensor  # [rows, heads] bf16 head weights
    paged_metadata: PagedIndexerMetadata


Metadata = TypeVar("Metadata", bound=CandidateMetadata)


class PrefillCandidates(Protocol[Metadata]):
    """The attention backend keeps a publish alive and hands it back unread."""

    def publish_prefill(
        self,
        inputs: PrefillInputs,
        out: Selection,
    ) -> Optional[Metadata]: ...

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[Metadata],
        out: Selection,
    ) -> None: ...


class DecodeCandidates(Protocol[Metadata]):
    def publish_decode(
        self,
        inputs: DecodeInputs,
        out: Selection,
    ) -> Optional[Metadata]: ...

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[Metadata],
        out: Selection,
    ) -> None: ...
