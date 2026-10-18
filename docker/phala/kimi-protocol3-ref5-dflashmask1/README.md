# Kimi ref5 DFlash grammar-mask maintenance release

This directory archives the exact recipe used for the published image in
[release.json](release.json). Runtime source remains the immutable commit
`e77dff2b12ef22ac67b2ea2905e066cedb3f69c3`; this later documentation commit is
not its runtime revision. The official 0.5.21 compatibility-maintenance exception
is retained; shared main is not the deployed Kimi runtime.

The image derives from pinned ref5 and replaces only `dflash_utils.py`. It
clears the stale extend/decode grammar mask before DFlash verification; verify
callers still apply their current tree-position masks. Native XGrammar p6,
other dependencies and explicit customer strict behavior are inherited.

To recreate the build input from a checkout with the release source object:

```bash
SOURCE=e77dff2b12ef22ac67b2ea2905e066cedb3f69c3
CONTEXT=$(mktemp -d)
cp docker/phala/kimi-protocol3-ref5-dflashmask1/Dockerfile "$CONTEXT/Dockerfile"
git show "$SOURCE:python/sglang/srt/speculative/dflash_utils.py" > "$CONTEXT/dflash_utils.py"
docker build --pull=false --network=none --build-arg ENGINE_REVISION="$SOURCE" \
  -t kimi-ref5-dflashmask-rebuild:local "$CONTEXT"
```

This is a local rebuild recipe, not a promise of a byte-identical OCI digest.
Consume the already published digest from release.json; never overwrite its tag.
The original final image passed five focused installed mask tests and direct
`hf --help` / `hf download --help` checks. Source replay validates all 32
ordered patch trees. Source checks, installed-image checks and production
acceptance are distinct.

Natural TP8/B300 traffic completed after rollout, with no sustained zero
acceptance in the observed window. HTTP400 and aborts still occur; short,
light-load observation does not establish general error-rate improvement,
full-concurrency stability or semantic/SSE correctness. The optional image
decoder may fall back from nvImageCodec to PIL. No strict2 override or
fastsafetensors loading is introduced by this recipe.
