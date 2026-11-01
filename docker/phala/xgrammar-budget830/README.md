# Pinned budget830 native input

This is a future release input, not a change to any default dependency, Dockerfile,
serving environment or benchmark image. Keep both O15 comparison arms and the
accepted f40/p6 runtime fixed. Do not merge the unrelated Kimi runtime candidate
or infer final-image acceptance from these native CPU gates.

`release-inputs.json` binds the exact reviewed native source6113e5fc, reconstructed
p6 source tree, upstream PR830, wheel and library hashes, build tools, and separate
fixture revisions. `p6-to-budget830.patch` is the complete signed candidate's
source delta, including its distinct package version. Its zero-context encoding
avoids whitespace-hook changes to patch context; exact intermediate tree checks
remain mandatory. The original p5 and p6
patches stay in the existing immutable release path at SGLang9d263bb062, rather
than being duplicated here. Supply a Git checkout containing that public release
commit as `--release-repository`; it need not be the active checkout.

## Source and existing-artifact verification

Run with ordinary Python (no native imports or wheel build):

```sh
python docker/phala/xgrammar-budget830/build.py \
  --release-repository /path/to/sglang-with-release-history \
  --source /path/to/new-native-source \
  --wheel /path/to/xgrammar-0.2.6rc2+phala.feedback.20261017.p6.budget830-cp312-cp312-linux_x86_64.whl
```

The destination must not exist. The script fetches the pinned upstream commit,
checks every patch hash and intermediate Git tree, and verifies the existing
wheel and contained native library. `--upstream-repository` can use a local
upstream object repository. No wheel is bundled or automatically downloaded.
Missing release objects, patches or artifact hashes fail closed. Source replay
has been demonstrated; byte-identical wheel rebuilding has not.

## Optional future CPU rebuild

Only explicit `--wheel-dir /path/to/output` enables building; omit `--wheel`.
Provision the exact Python packages and compiler/CMake/Ninja versions recorded
in the manifest in an isolated build environment first. The script performs no
package installation. It verifies the DLPack URL/commit, pins SOURCE_DATE_EPOCH,
uses Release/O3, four build/LTO jobs and source-prefix mapping, and builds CPU
bindings with C++ tests/cpptrace off and CUDA architectures empty. The original
package's GPU JIT sources remain unchanged but are not compiled or executed.
A rebuilt wheel is a new candidate until its hashes and actual-image behavior
are qualified; this is not a manylinux or reproducible-binary guarantee.

## Future final-image composition

The existing historical ref5 Dockerfile installs an already qualified wheel with
`--no-deps --no-index` after SHA256 verification. Reuse that pattern only in a
separately approved final-image composition: pin the engine/base-image digest,
copy this exact wheel, verify its wheel hash, install without dependency
resolution, and verify installed version/native hash. Record the source commit,
tree and native hash in release metadata. Do not change the frozen historical
ref5 Dockerfile, infer a new approved base image, or rebuild this wheel merely to
repackage it. No future image digest exists yet.

Mainline runtime/Python Kimi composition, fixture commits and dependency pins
must be reviewed independently. In particular, the native artifact does not
supply the unmerged engine changes. The fixture revisions are evidence references,
not cherry-pick instructions. Native6113 and fixtureea320 were approved; final
test-only4d8725 review was still pending when these pins were prepared.

## Qualification boundary

The pinned wheel passed an isolated actual-f40 CPU gate:32 API rows,189 functional
cases, then3 corrected-contract and6 ASGI cases. This used a device-free runc
container with networking disabled,2 CPU quota and4GiB memory, preserving the
resident service. It is native/component evidence, not final-image, GPU, live
service or O15 replacement acceptance. Retain the prior p6 negative controls.
Local original-source checks passed26 upstream budget cases,32 API rows,2 Inkling
cases, and272 synthetic Kimi cases;13 hardcoded `/tokenizer` fixture cases were
not locally qualified. No wheel build or runtime retest is required merely to
review this source/pin change.
