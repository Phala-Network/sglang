#!/usr/bin/env bash
set -eu

IMAGE=${1:?IMAGE is required}
FIXTURES=${2:?absolute fixture directory containing test/ is required}
OUT=${3:?output directory is required}
mkdir -p "$OUT"
MOUNTS=(--mount "type=bind,src=$FIXTURES/test,dst=/qualification/test,readonly")
if [ "${4:-}" = --development-overlay ]; then
  while IFS= read -r path; do
    relative=${path#"$FIXTURES/"}
    MOUNTS+=(--mount "type=bind,src=$path,dst=/sgl-workspace/sglang/$relative,readonly")
  done < <(find "$FIXTURES/python" -type f -name '*.py')
fi
RUN=(docker run --rm --init --pull=never --network=none --cpus=4 --memory=12g
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void
  -e SGLANG_RUST_BUILD_MODE=never -e OMP_NUM_THREADS=1
  -e PYTHONPATH=/qualification -w /qualification "${MOUNTS[@]}"
  --entrypoint python3 "$IMAGE")

"${RUN[@]}" -m unittest discover -s /qualification/test/registered/unit/utils \
  -p 'test_engine_failure*.py' -v > "$OUT/engine-failure.log" 2>&1
"${RUN[@]}" /qualification/test/registered/unit/spec/test_dspark_accept_result_ownership.py \
  -v > "$OUT/dspark-ownership.log" 2>&1
docker image inspect "$IMAGE" --format '{{.Id}}' > "$OUT/tested-image-id.txt"
echo PASS_ENGINE_FAILURE_AND_DSPARK_CPU
