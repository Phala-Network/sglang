# K3.1 video and runtime repair image candidate

This Python delta packages video support and reviewed serving/cache repairs over the accepted `phala-dev-kimi-k31-3` digest. It retains all native and Python dependency layers, including the existing video decoder packages. The recipe performs no installation or model download.

Freeze the complete reviewed source after the video CPU regressions and ordered serving-patches replay pass. The approved K3.1 compatibility maintenance line remains the release source; `main` retains its fixed customer entrypoint. The release tag is immutable.

On the authorized CPU builder, stage the frozen Git checkout and data-only HF fixture inside the task directory, then run this recipe once:

```sh
python3 -B source/docker/phala/k31/video-20261107/build_candidate.py \
  --repo /opt/testdata/kimi-k31-video-20261107/source \
  --revision <complete-reviewed-source-commit> \
  --fixture /opt/testdata/kimi-k31-video-20261107/hf-fixture \
  --run-dir /opt/testdata/kimi-k31-video-20261107/attempt-1
```

`prepare_context.py` rejects unrelated runtime, build, native and dependency edits. Its explicit file-to-selector map runs the committed video regressions and CPU classes `TestPrefetchCommitOrdering` / `TestMambaTemporalCopy`; the original GPU HiCache/Mamba suites remain outside CPU qualification. Final-image checks also retain inherited scheduler/COW regressions, existing native/HF/parser/spec checks, `sglang serve --help`, and direct `hf --help` / `hf download --help`. Tests use bounded CPU-only containers with no network, GPU, source mounts or runtime installation. Source marker, OCI revision, package versions, pip-check output and inherited layer prefix must match the frozen source and base contract.

There is no publication or production-rollout function. Before pushing the qualified bytes to `ghcr.io/phala-network/sglang:phala-dev-kimi-k31-4`, read back the exact remote source tag and published GitHub Release. Recheck that the Git/image tag is unused before creating it; never move an existing tag. Add the actual registry digest to the release receipt after publication.

Only exact task containers are removed by this worker. It leaves build cache untouched and retains the base plus qualified candidate for explicit owner handoff and cleanup. CPU qualification does not establish GPU inference, video semantics, production readiness or performance; the serial deployment acceptance owns those results.
