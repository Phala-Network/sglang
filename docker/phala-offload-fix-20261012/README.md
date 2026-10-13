# SGLang prefetch worker repair image

Frozen build inputs for [source 53c8f4e6](https://github.com/Phala-Network/sglang/commit/53c8f4e63f11887a398b727dd6bc30dbe4aad73c) and [Release v0.5.21-phala-offload-fix-20261012](https://github.com/Phala-Network/sglang/releases/tag/v0.5.21-phala-offload-fix-20261012). SGLang remains authoritative source; this directory records the exact three controller overlays used in the published image. The recipe and manifest were moved here from the deployment repository; that metadata relocation does not rebuild or change the deployed artifact.

Published image: `ghcr.io/phala-network/sglang:0.5.21-phala-offload-fix-20261012@sha256:9619a8004cbeda15ce999ab89d96b1e89b5476bd0507936253876621cc616f9e`.

`image-inputs.json` records the immutable DSPARK+usage base, source/patch revisions and before/after installed hashes. The Dockerfile rejects mismatched base bytes, installs only those three Python files, compiles them and verifies installed hashes. Native dependencies are inherited unchanged. `prepare_build.py` stages the three overlay files directly from the pinned source commit and verifies the canonical regression test by hash; the test remains in SGLang's source tree and is not copied into this recipe.

On a Linux amd64 Docker builder, use a clean SGLang checkout that contains the pinned source commit and stage a fresh build context:

```sh
staging_dir="$(mktemp -d)"
python prepare_build.py \
  --source-repo /path/to/sglang \
  --output "$staging_dir"
docker build --platform linux/amd64 --progress plain \
  -t sglang-offload-repair:local "$staging_dir"
```

A rebuild is a new artifact; its digest is not assumed equal to the immutable published image. Before publication or use, verify installed hashes and run `python -m unittest test.registered.unit.mem_cache.test_prefetch_io_failures` from the pinned source checkout, then run HF/download, module/serve CLI and native import checks. The published image passed 35 tests with two CUDA skips in the 37-test scope. See the [production acceptance record](https://github.com/Phala-Network/phala-models-compose/blob/main/production/deepseek-v4-1-flash-usw5-quad-tp2-sglang/acceptance.md) for image checks versus runtime rollout.
