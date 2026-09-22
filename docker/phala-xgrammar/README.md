# Single native XGrammar dependency

This engine requires `xgrammar==0.2.6+phala.union1`, not stock 0.2.1/0.2.6.
There is no claimed public wheel or publicly fetchable Phala native Git commit.
Use the `external-dependencies.json` and deterministic dependency delta from
the same pinned `sglang-serving-patches` checkout as the engine export.
The native input is frozen in serving-patches commit
`345f57ee3ff447679a9f4d426b6d61ada95c0e12`; later engine exports retain that
exact dependency manifest/delta. Check out that immutable commit before using
the recipe, or the final combined export whose manifest matches it.

In an existing Linux CPU build environment with the native build dependencies:

```bash
python docker/phala-xgrammar/build.py \
  --patch-repository /path/to/pinned-serving-patches \
  --source /path/to/new-xgrammar-source \
  --wheel-dir /path/to/wheels
python -m pip install --no-index --no-deps \
  /path/to/wheels/xgrammar-0.2.6+phala.union1-*.whl
```

The recipe fetches the immutable public upstream, applies the generated delta,
checks the complete source tree and license hash, and checks the DLPack submodule
identity before building. `--verify-only` stops before submodules/wheel build.
It verifies each declared submodule URL and gitlink against the patched tree.
It does not assume that installing the local-version pin from public PyPI works.
No SGLang image is built or published by this recipe.

The dependency record binds source commit `8830951a`/tree `b8dcb87b`, and the
tested wheel built from `8b5d3b86`/tree `09c66b3c`. The successor only adds tests,
not runtime/build bytes. Native CPU evidence is 655 passing tests and a stock
same-base semantic ablation; it is not engine/GPU/final-image qualification.

`--constrained-json-max-whitespace-cnt` restores the historical direct-JSON
setting. Unset preserves existing behavior. It does not change structural-tag
per-format limits or limit whitespace inside JSON strings. Invalid bounds,
non-XGrammar backends, compact conflicts and unsupported tokenizers fail closed.
The shared schema-position validator rejects constraints the native backend
would ignore in direct JSON, legacy tags and nested JSON/XML formats.

The recipe's `--verify-only` path was executed from a fresh directory against
public upstream on 2026-09-22 and produced tree `b8dcb87b50b99cc61dbab46e7dcfbd34f78a2f45`.
The installed union1 adapter also passed all 14 CPU tests, including nine native
cases, without skips. No new wheel or runtime image was built for these checks.
