# Kimi recovery image

Build-time composition of the frozen phala-3 image, the Mooncake recovery
runtime closure and the three-file Mamba admission repair. Runtime source
mounts and startup installation are not used. Entrypoint, environment,
command, Python ABI, Mooncake EP/PG and external PIG remain inherited.

Preparation must obtain exact Git blobs for the before/after engine commits,
validate the exported native manifest, and copy the frozen Mooncake verifier.
`source-lock.json` records engine, patches and Mooncake commits plus every
changed source hash. Only a context that passed these checks may be built.

Source regressions, final-image import/linkage/CLI checks, isolated HA tests,
GPU resource tests and production acceptance are separate gates. Exact source
tags and Releases must exist before image publication. RPC timeout settings
belong in Compose for every SDK consumer and owner; they are not silently
installed as global image defaults.
