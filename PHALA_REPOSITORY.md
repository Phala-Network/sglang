# Phala repository workflow

## Ownership

This repository owns the complete downstream engine source on official v0.5.20
baseline `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`. General serving repairs work
independently of Governor; model-specific behavior stays scoped to its model or
topology. Governor owns its controller/adapter/hooks; TAIL owns attestation and
transport.

Develop real source and synchronize the ordered
[serving patch export](https://github.com/Phala-Network/sglang-serving-patches)
through the existing export/replay process. Avoid a second hand-edited
implementation or a full patch system per model. Deployment configuration
consumes fixed complete engine and component identities.

## Main and release tags

`main` integrates downstream serving/model changes and Governor hooks; it is not
an upstream-only baseline. Integrate verified work from focused branches. Keep
long-term maintenance branches only for real compatibility needs.

Create immutable version tags from verified commits in `main` history. Record
engine commit, component versions, build inputs, image digest, validation scope
and limitations. Do not move published tags or overwrite image versions. Source
completion, image publication and deployment acceptance are distinct states.

Images belong to `ghcr.io/phala-network/sglang` and identify the complete engine
revision. Build and validate selected artifacts through the authorized release
process; repository cleanup does not initiate model deployment.

## Checks and contributions

[Phala source checks](.github/workflows/phala-source-check.yml) run source lint
and selected CPU regressions on pull requests or manual dispatch.
[Full lint](.github/workflows/lint.yml) is manual. Existing failures stay visible
until corrected. Use declared dependencies and tests relevant to changed code;
source checks do not replace GPU, native ABI or model acceptance.

Preserve upstream layout, licensing, tests and attribution for reviewable
upgrades. Keep root docs focused on entrypoints and Phala details under
`docs/phala`. Do not introduce a separate approval chain per patch or inherit
unrelated upstream hardware publishing/maintenance workflows.

## Historical work

Old branch names and completed migrations are historical inputs, not current
integration state. See [historical records](docs/phala/HISTORY.md). Remove obsolete
branches only after checking active consumers and retaining commits through
appropriate refs. Archive tags preserve evidence, not acceptance.
