# Phala SGLang source

This repository owns the complete downstream engine source. Develop shared serving fixes here, with model-specific behavior guarded by model or topology. Governor owns its controller/adapter/minimal hooks; TAIL owns transport and attestation. The separate serving-patches repository is an optional generated export, not another manually maintained source or release gate.

## Branches and integration

`main` currently retains the upstream-derived source baseline plus Phala repository administration. It is **not** a claim that all downstream model fixes or Governor integration have been accepted. The common v0.5.20 serving input ends at `codex/serving-v0520-0020-allowed-output-r1`; the frozen Governor composition input is `codex/serving-v0520-qwen38-governor-derived-r2`. Remaining model/input branches retain distinct work pending shared-source integration and affected tests. They are not separate mandatory PR pipelines.

The administrator consolidates shared changes and publishes complete source identities. Contributors hand off exact commits, tests, failures and compatibility constraints. Do not create one permanent branch/PR/profile workflow for every patch. A source commit, passing source tests, published image and accepted target deployment are separate states.

Obsolete intermediate branches are removed only when their commits remain reachable from a retained branch or tag. Minimal historical tips are preserved under `archive/20260920/`; older release tags are unchanged. The branch-to-commit-to-retained-ref map is [the cleanup manifest](repository-archive-20260920.json). Restore a removed branch at its recorded SHA if historical work needs continuation; archive tags do not imply a successful release.

## CI and releases

This fork does not run upstream hardware matrices, PR-label/state bots, upstream package publishing or scheduled fleet maintenance. Their inherited workflow files were removed from `main`; original versions remain in Git history. The temporary GHCR token broker is retired. No credentials or temporary token transfer are part of the new repository workflow.

Full source lint remains explicitly available through `workflow_dispatch`, with read-only repository permissions and a bounded execution time. It is not a per-patch PR gate. Its historical failures remain failures until fixed and revalidated. No old Actions results/logs are deleted to manufacture a green history. Run affected unit/regression checks for changed code; use authorized hardware for actual runtime checks. GPU/model/attestation qualification is not replaced by lint.

Published engine images belong to `ghcr.io/phala-network/sglang` and must identify the complete source commit. Release only after applicable image checks; no automatic release or deployment is triggered by repository cleanup. Before deployment, consult the [component version register](https://github.com/Phala-Network/phala-models-compose/blob/main/production/COMPONENT_VERSIONS.md) and recheck current version/digest and target compatibility.
