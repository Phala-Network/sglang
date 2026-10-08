# SGLang 0.5.21 Phala control

This compatibility release restores the exact control source and original image recipe tested on 2026-10-06. It is rooted at official v0.5.21 commit `e00930c5489053f26d86b179cee0d087f846acbb`. Main is preserved.

`source-manifest.json` binds all 15 changed/new files before and after installation. The complete source tree includes their exact final bytes. `Dockerfile`, `install.py`, `patched/`, `tests/`, the canonical patch and privacy binding are copied from the original build context without changes. Build context is this directory, not the repository root.

Official linux/amd64 base: `docker.io/lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75`.

The already-tested local image ID is `sha256:82d8c91448b07878cde81cd0152242dbe3cf06422470ae1793efeb48895b5714`; its prior OCI export manifest is `sha256:aab2840f0aecbb0b5b6bb5574d6206348042b539c89af2567cdc299de4a4f386`. Neither identifier is asserted to be a registry pull reference. Registry publication is recorded after push/readback.

Changes retain protocol compatibility and privacy logging on current upstream. No experimental asynchronous HiCache, admin-key, packed attention, FastSafeTensors, or Governor patch is introduced. Original image metadata identifies the upstream base; this release reconstructs the complete patched Git identity after the original build rather than claiming that Git identity was embedded originally.

Evidence: all 15 installed final-image hashes and native/model imports passed; direct `hf --help` and `hf download --help` passed. Existing same-image TP4 startup and 29 functional actions passed; these do not constitute TP2 or four-replica acceptance. CPU protocol/privacy tests are retained in `tests/`.
