# Xing4 SGLang 0.5.22 candidate

This source-owned recipe uses the frozen official 0.5.22 CUDA13 amd64 image and the complete candidate Python source. The official editable package path and official Rust/kernel binaries remain from the same 0.5.22 base. FlashInfer 0.7.1 and torch 2.14.1+cu130 are unchanged. No runtime source mounts or post-start edits are required.

`candidate-source.json` is generated from the exact engine commit; the build checks every tracked Python source hash. The only external build input is the already qualified XGrammar 0.2.7 cp312 wheel (SHA256 `05cbf9fdbd294310190bc6509e655a43d8a6b374c97fc882fc681cb10facc611`), with the existing control-character fix and unchanged 41 baseline bounded-string escape limitations. Its patch is retained here; no new XGrammar behavior is claimed.

Build on CPU201 only. Run the source-owned `qualify.py` against the final local image, with no source mount. Transfer the same Docker archive directly to 105 for GPU0 numerical/protocol/context qualification. This is an offline development candidate; registry and GitHub Release publication are outside this task.
