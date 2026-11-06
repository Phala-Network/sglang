# K3.1 scheduler repair candidate preparation

This recipe packages the frozen scheduler/COW repair and focused tests. It includes no production rollout; root owns the source handoff and deployment.

The recipe extends accepted image2 by digest. Its sparse layer contains only root-frozen Python runtime changes, focused regression tests, the source marker and these qualification helpers. It never invokes pip, a native compiler or a model downloader. `prepare_context.py` rejects Python changes outside five reviewed scheduler/COW files and unexpected build/native/dependency edits.

After root freezes a clean complete-source commit containing the repair and this recipe, stage that Git checkout and the existing data-only image2 HF fixture under `/opt/testdata/kimi-k31-usw3-repair-20261106` on CPU builder201. Then, once explicitly handed off, run:

```sh
python3 -B source/docker/phala/k31/repair-20261106/build_candidate.py \
  --repo /opt/testdata/kimi-k31-usw3-repair-20261106/source \
  --revision <root-frozen-40-character-commit> \
  --fixture /opt/testdata/kimi-k31-usw3-repair-20261106/hf-fixture \
  --run-dir /opt/testdata/kimi-k31-usw3-repair-20261106/attempt-1
```

The worker builds once using builder201's default Docker driver and fixed image2 layers, then checks the actual installed candidate without source/executable mounts or runtime installs. It adds the repair's committed CPU regressions to image2's existing native/HF/parser/spec/CLI qualification, compares package versions and pip-check results exactly with image2, and executes direct `hf --help` / `hf download --help` with network disabled. The source marker, OCI revision and changed source bytes must match the full frozen commit; the complete inherited layer prefix and runtime config must match image2.

Results and logs stay in the attempt directory. Test containers are removed by exact owned identities. The worker snapshots existing build-cache IDs, then prunes only new IDs; it never runs a broad prune or removes the shared builder. A base pulled by this attempt is untagged afterward. The candidate image is retained by exact image ID for the owner to inspect, publish or remove. Final ownership transfer must clear that pending cleanup receipt.

There is deliberately no registry publication function. The release owner must first verify an unused immutable tag, push the exact complete-source commit/tag, publish its GitHub Release, read both back, then publish the same qualified candidate bytes and read back the digest. Preserve the shared patches/Governor/full-engine provenance; Governor and Mooncake/native versions remain unchanged. CPU checks do not prove incident causality, GPU scheduling, production recovery or performance.
