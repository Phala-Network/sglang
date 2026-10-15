# GLM-5.3 protocol maintenance image

This source-bound incremental recipe installs the GLM-5.3 reasoning repair over
the immutable Phala7 image. It preserves official native dependencies and the
pinned model's original chat template. The only installed runtime change is
`serving_chat.py`; the matching focused tests are retained in the image.

Prepare the context on builder201 from the final reviewed complete engine commit:

```bash
python docker/phala-glm53-protocol/prepare_context.py \
  --engine-ref "$ENGINE_COMMIT" --patches-commit "$PATCHES_COMMIT" \
  --output /path/to/new/context
docker build -f /path/to/new/context/docker/phala-glm53-protocol/Dockerfile \
  --build-arg ENGINE_COMMIT="$ENGINE_COMMIT" \
  --build-arg PATCHES_COMMIT="$PATCHES_COMMIT" \
  -t "$CANDIDATE_IMAGE" /path/to/new/context
```

The installer checks the base module preimage, all context bytes and both source
identities, then updates the installed wheel's `RECORD`. Build-time tests import
the installed package. After building, repeat focused installed tests and direct
`hf --help`, `hf download --help`, and `sglang serve --help` against the final
image ID without source mounts. Image publication requires the exact remote
source tag and published Release first.

For GLM-5.3's always-thinking template, `none` wins over conflicting template
settings and closes only the final generation token prefix, including raw
`input_ids`, without mutating message history or caller IDs. An unclosed thinking
continuation conflicts with reasoning-off and returns a client error; plain and
closed answer prefixes remain supported. Omitted effort uses
a 1024-token reasoning cap with `--enable-strict-thinking`; explicit
`reasoning.max_tokens` supplies its own cap. Explicit official `low`, `high` and
`max` effort values retain the original template's semantics. Budgeted requests
reuse the reasoner backend around GLM47's answer/tool grammar so the cap applies
to ordinary text and tool requests. Final image and
GPU/direct-endpoint acceptance are separate from source CPU tests.
