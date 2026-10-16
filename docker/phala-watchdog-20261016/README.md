# SGLang 0.5.21 watchdog and DSPARK repair

This maintenance release extends accepted source `53c8f4e63f11887a398b727dd6bc30dbe4aad73c`
and image `sha256:9619a8004cbeda15ce999ab89d96b1e89b5476bd0507936253876621cc616f9e`.
It retains all other runtime bytes, native libraries, dependencies and serving parameters.
The existing frozen 0.5.21 compatibility line remains the release input; current
unified main is a separate composition.

The common fix preserves SGLang's SIGQUIT ownership under Uvicorn 0.54's
multi-worker supervisor. Every spawned HTTP worker reads the same terminal
failure latch. A hard scheduler timeout or child crash publishes failure before
signaling cleanup; `/ready` and `/health` fail closed across workers. Idle time
does not count toward the active scheduler deadline. Hard timeouts prioritize
bounded termination over potentially blocking in-process diagnostics; soft
watchdogs retain their diagnostic behavior. If the parent ignores SIGQUIT or
exits without cleanup, the watchdog kills the saved process identities and exits
itself. A parent already gone before detection cannot be walked retrospectively;
the failed scheduler still exits. Host/container supervision owns any earlier
orphaned processes.

The DSPARK fix copies six accept results out of reusable graph buffers before
the scheduler's asynchronous consumer can race the next graph replay. It follows
upstream [PR #42812](https://github.com/sgl-project/sglang/pull/42812), head
`7a8f74963c9db8e21c98ac72cb65ba7b189e6f9c` (open at review). Result values,
shape, dtype and device remain unchanged. This is a demonstrated buffer-ownership
defect; the original USW5 stall's unique initiating operation is not established.

Stage from a clean frozen checkout with `prepare_build.py --source-repo REPO
--patches-commit SHA --output NEW_DIRECTORY`, then build that context once on
CPU201. The recipe verifies accepted base bytes and installs only ten reviewed
Python modules. Source tag and a published GitHub Release must exist before
registry publication. The release receipt binds the final image digest to the
source and patch commits.

Relevant tests are `utils/test_engine_failure.py`,
`utils/test_engine_failure_multiprocess.py` with its real four-worker fixture,
and `spec/test_dspark_accept_result_ownership.py` under `test/registered/unit`.
Run the final image without runtime-source overlays, including direct HF and
SGLang CLI checks. CPU/PyTorch storage tests are distinct from the optional CUDA
stream test and target GPU acceptance.

The repair makes a fatal engine unready and exits its failed processes. It does
not add automatic container restart; production's existing finite startup owner
and `restart: no` remain authoritative. PIG/ingress propagation and real model
acceptance must be checked independently on the authorized target.
