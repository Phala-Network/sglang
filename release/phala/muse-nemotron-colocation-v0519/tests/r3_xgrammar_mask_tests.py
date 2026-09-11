"""CPU token-mask/rollback tests against the actual cached Nemotron tokenizer."""
import itertools
import json
import time

import xgrammar as xg
from transformers import AutoTokenizer

from property_fixtures import cases


def main():
    tokenizer = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/cc84af2fe71647d87f4486c064f320e1e7535243", local_files_only=True)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=131072, stop_token_ids=[tokenizer.eos_token_id])
    compiler = xg.GrammarCompiler(info)
    rows = []
    for name, schema, expected in cases():
        started = time.monotonic()
        grammar = compiler.compile_json_schema(schema, max_whitespace_cnt=64)
        compile_seconds = time.monotonic() - started
        for index, items in enumerate(itertools.permutations(expected.items())):
            text = json.dumps(dict(items), ensure_ascii=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            matcher = xg.GrammarMatcher(grammar, max_rollback_tokens=200)
            mask = xg.allocate_token_bitmask(1, 131072)
            for token in ids:
                matcher.fill_next_token_bitmask(mask)
                value = int(mask[0, token // 32])
                assert (value & (1 << (token % 32))) != 0, (name, token, tokenizer.decode([token]))
                assert matcher.accept_token(token), (name, token)
                matcher.rollback(1)
                assert matcher.accept_token(token), (name, "rollback", token)
            assert matcher.accept_token(tokenizer.eos_token_id), (name, "eos")
            assert matcher.is_terminated()
            rows.append({"fixture": name, "permutation": index, "tokens": len(ids), "compile_s": round(compile_seconds, 4)})
    print(json.dumps({"tokenizer": tokenizer.name_or_path, "passed": len(rows), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
