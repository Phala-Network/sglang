# Native HF CLI packaging

The GLM-5.3 r3 runtime retained `huggingface_hub` 1.30.0 and its console-script
metadata but omitted `/opt/sglang/bin/hf` when copying the final multi-stage
image. A native `entrypoint: ["hf", "download"]` therefore failed before the
model service could start. The main Dockerfile now copies the official console
script alongside the other selected runtime binaries.

This narrowly scoped overlay repairs the already-published immutable r3 image.
At build time only, it uses the existing distlib and the installed HF entrypoint
metadata to regenerate the normal console script. It installs no packages,
changes no dependency versions, preserves the NVIDIA entrypoint and all r3
runtime code, and fails on any unexpected base/version/entrypoint. Both copied
build-test files require SHA-256 guards. No script is run at production startup.

`test_native_cli.py` qualifies the actual image's `hf` and `hf download` help,
including the Compose-required `--revision` and `--max-workers` options. Image
qualification must also run `sglang serve --help` with writable compilation
caches, compare installed package/repair identities to r3, and exercise the
native downloader with fixed model metadata. Source tests do not prove a built
or published image, and CPU CLI tests do not prove GPU inference or attestation.

Freeze build inputs before building. Keep final image digest and release-result
digests out of those inputs. Follow the production release contract for clean
rebuild comparison, SBOM, provenance and registry read-back before rollout.
