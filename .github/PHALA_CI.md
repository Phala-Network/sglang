# Candidate source checks

This maintenance branch automatically runs one `Phala source checks` workflow
on pull requests. It checks Python formatting/lint, configuration safety, the
upstream registered-test contract, and selected executable CPU source-method
regressions. Push does not duplicate the PR run.

The upstream GPU/vendor/nightly/release/robot workflows retain their bodies
and manual or reusable entry points, but no automatic triggers in this branch.
The full `Lint` workflow remains manually available, including Rust and docs
integration checks. This is deliberate scope selection, not a claim that those
checks or any GPU/model acceptance passed.

Historical fork tests without a qualified upstream runner registration now
live under `test/manual/unit/`, preserving relative path depth. Their paths do
not imply upstream discovery. The lightweight workflow names the source-only
subset explicitly; runtime-import tests and real torch GGUF regressions use
their corresponding prepared development environment.

The Muse template and Nemotron tokenizer are external pinned test artifacts.
Absent artifacts produce explicit skips here. Separate artifact-enabled local
runs, not this lightweight workflow, establish their CPU integration evidence.

Edits in a PR cannot deactivate default-branch `pull_request_target`,
`workflow_run` or scheduled workflows. Existing default-branch automation or
repository settings require separate owner action; this branch does not edit
main, disable workflows repository-wide, or authorize a deployment.
