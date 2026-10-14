#!/usr/bin/env bash
# Execute inside the builder container, with only fixed tokenizer data mounted.
set -euo pipefail
mode="${1:?Use source or installed}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
export SGLANG_USE_CPU_ENGINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/libcuda.so.1
export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
case "$mode" in
  source)
    export PYTHONPATH="$root/python"
    ;;
  installed)
    unset PYTHONPATH
    installed="$(python -c 'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parent)')"
    test "$installed" != "$root/python/sglang"
    # Policy tests load a file directly. Point their source-relative address at
    # the actual installed package; this never masks that package on sys.path.
    mv "$root/python/sglang" "$root/python/sglang-build-input"
    ln -s "$installed" "$root/python/sglang"
    python - <<'PY'
import hashlib
import json
from pathlib import Path
import sglang
root = Path('/opt/phala-xing4')
package = Path(sglang.__file__).resolve().parent
manifest = json.loads((root / 'installed-source.json').read_text())
assert str(package) == manifest['installed_package']
for name, expected in manifest['runtime_sha256'].items():
    path = package / Path(name).relative_to('python/sglang')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print(json.dumps({'engine_commit': manifest['engine_commit'],
                  'installed_package': str(package),
                  'checked_runtime_modules': len(manifest['runtime_sha256']),
                  'executable_source_mount': False}))
PY
    ;;
  *) exit 2 ;;
esac
cd "$root"
python -m pytest -q --junitxml="${JUNIT_XML:-/tmp/xing4-cpu.xml}" \
  test/registered/unit/configs/test_xing_request_static_yarn_policy.py \
  test/registered/unit/model_executor/test_xing_request_static_yarn_torch.py \
  test/registered/unit/managers/test_xing_request_static_yarn_lifecycle.py \
  test/registered/unit/configs/test_request_static_yarn_policy.py \
  test/registered/unit/model_executor/test_request_static_yarn_torch.py \
  test/registered/unit/managers/test_request_static_yarn_admission_reject.py \
  test/registered/unit/models/test_xing4_native.py \
  test/registered/unit/models/test_xing4_tokenizer.py \
  test/registered/unit/models/test_xing4_runtime_auth.py \
  test/registered/unit/utils/test_startup_failure_diagnostics.py \
  test/registered/unit/test_profiler_export_failure.py
