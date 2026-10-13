# Build provenance and scope

Published image: `ghcr.io/phala-network/sglang:v0.5.21-phala-3-mcfix3@sha256:f077dab5c1d77930e4c117b609bd8bab8bee5216eb585e7f682ed0f6e3a23e13`. Config ID: `sha256:4176f3446ec02c59dd16963d95a49939bd2c74ab590161907c069d7efbb585d1`. The pinned base is phala-3 `sha256:d5abbec01e3e6d6f571cf8250871d72373a178dca1d05b398965c759798139cf`.

Complete SGLang source `e9a40ece22f8539d434ccc138336019ddbf2e06c` and serving-patches `a09c9f546e839cd3cedded5dea7c29145249ceff` use Releases `phala-v0.5.21-4`; Mooncake source `2214a3ca04e165ced3cab3cf604918d2ec22ed0f` uses Release `phala-v0.3.13-kimi-recovery-3`. See [source lock](source-lock.json) for the three Python preimages and postimages. The installer validates every preimage, payload hash and Python syntax before writing any target.

Dockerfile, installer and README were copied byte-for-byte from the reviewed SGLang image recipe formerly held under `image-builds/sglang/kimi-recovery-3`; source-lock was copied from its final mcfix3 release evidence. The recipe is now maintained in this repository. Their SHA256 identities are:

```json
{
  "Dockerfile": "08762b633d4e904fcc5b25ead09650a51ce334e51dde1d6a94c64466e76a751d",
  "install-overlay.py": "34dce309246b62c0adc3cbd193f7bc9ef3595a8839e396cf52153f6076b64d9d",
  "README.md": "03a418d79db180ed346f2e9ed1a49c788fab68f0ef04dc274eece17cb5e44072",
  "source-lock.json": "f277837bddca935324b528da4ecd78381553b29a319ca57f4df43e31e173bcd6"
}
```

The build context additionally requires `runtime/` containing four native targets and their private libraries, `verify-mooncake-recovery.py` from the exact Mooncake commit's `phala/recovery/verify-runtime.py`, the three exact engine blobs under `overlay/`, and `test_mamba_load_back_admission_cpu.py` from the engine commit. These binary/context inputs are not embedded in this source directory. Native closure manifest SHA256 is `e57c6b825fcc9bb102664c1c695dc3220aff3a009114933b5312e17f189e1736` (49 files). It was built once on 201 from a fixed HA cache plus verified main changes and reviewed overlay, then linked against unchanged dependencies.

This recipe records the completed image assembly and its input identities. A separate clean independent reproducibility rebuild was not performed. Runtime source mounts/startup installation are not deployment mechanisms. Final-image CPU/import/linkage/CLI checks passed; the separate native-closure TCP/etcd exercise does not establish final-image GPU/RDMA/TP or production acceptance. Current deployment and acceptance state is maintained in the [cluster acceptance record](https://github.com/Phala-Network/phala-models-compose/blob/main/production/kimi-k3-cluster/ACCEPTANCE.md).
