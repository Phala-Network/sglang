# Official v0.5.23 maintenance image

This recipe builds one complete version-maintenance engine on the fixed official
CUDA13 amd64 base. Shared serving repairs occur once and Qwen behavior is conditional.
Current v0.5.20 main runtime consumers and older recipes remain intact. Embedded
Governor is not selected; external PIG owns admission. No image is published by
these tools, and this candidate has not passed final-image acceptance yet.

The engine version-line commit, serving-patches main commit and recipe main commit
are separate identities. Record all three; this main recipe/documentation commit
must never be presented as the complete v0.5.23 engine source.

Use Python 3.12+, Git and the authorized fixed Linux builder. First verify the
complete ordered source export using the existing patch repository tool:

```sh
python /path/to/sglang-serving-patches/scripts/stack.py verify \
  --stack-root /path/to/sglang-serving-patches/versions/v0.5.23 \
  --source /path/to/complete-sglang
python prepare.py --engine-repo /path/to/complete-sglang \
  --engine-commit FULL_VERSION_LINE_COMMIT \
  --patches-repo /path/to/sglang-serving-patches --patches-commit FULL_PATCH_MAIN_COMMIT \
  --recipe-commit FULL_RECIPE_MAIN_COMMIT --version PUBLIC_RELEASE_TAG \
  --output /new/build-context
```

Build from that context once, passing the exact `build_args` in `context.json` as
Docker build arguments. Use `--pull=false --network=none --iidfile IMAGE_ID_FILE`
and the `ghcr.io/phala-network/sglang:PUBLIC_RELEASE_TAG` destination. The context
contains a complete Git archive of srt; startup does not install or mount source.
Preparation rejects package changes outside srt and source deletions, which need
an explicit packaging update. The official native dependencies and CLI are retained.

On the actual final image, without source or executable mounts, run:

```sh
python verify_cli.py --image sha256:EXACT_BUILDER_IMAGE_ID \
  --context /new/build-context/context.json --output /new/cli-evidence \
  --nvidia-runtime
```

The optional NVIDIA runtime injects host driver libraries while hiding GPUs. The
checks directly invoke `hf --help`, `hf download --help`, SGLang serve help, and
the installed parser. They do not install packages, download models, or replace
the required model/GPU/PIG-chain/final configuration acceptance.

Before pushing an image, read back an immutable remote source tag resolving to
the exact complete version-line commit and its published, non-draft Release.
Do not move existing tags. Publish the same built image, read back its immutable
digest and add that mapping and the focused final-image evidence to the Release
receipt. Downloader and backend consume the same accepted tag@digest. Compose
and deployment evidence belong in phala-models-compose.
