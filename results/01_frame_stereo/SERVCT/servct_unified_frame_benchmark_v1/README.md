# SERV-CT Unified Frame Benchmark v1

This package aggregates traceable local SERV-CT evidence for raw frame-based stereo methods.

Main protocol:
- Dataset: `dataset/servct_argos/honest_test`
- Samples: 8 holdout frames, `Experiment_2_009` through `Experiment_2_016`
- Evaluation resolution: original SERV-CT GT resolution
- Main mask: SERV-CT valid mask as used by the existing evaluators. Some legacy runs also exclude invalid/non-positive model predictions; see `valid_px` and notes.
- No retraining and no temporal refiners.

Important limitation:
Prediction arrays were not saved for these legacy runs, so this package does not recompute a stricter common mask or new pixel-level p50/p95. Runtime/VRAM are also not present in the SERV-CT metric files and remain empty.

Files:
- `servct_benchmark_full.csv`: complete benchmark table requested for the presentation.
- `servct_benchmark_slide_ready.csv` and `.md`: compact slide table.
- `servct_per_frame_metrics.csv`: holdout per-frame evidence.
- `servct_model_audit.csv`: every discovered/requested method and status.
- `servct_existing_results_audit.csv`: compatibility audit for reused metric files.
- `plots/`: presentation plots, with runtime plots explicitly marking missing runtime evidence.
- `qualitative/`: copied legacy montages for top methods where available.

## Runtime/VRAM Addendum

`servct_benchmark_full_with_runtime.csv` and
`servct_benchmark_slide_ready_with_runtime.csv` add measured runtime and peak
observed GPU memory for the same SERV-CT frame-based methods.

Runtime protocol caveat: the current external adapters are script-oriented and
do not all expose pure `load/preprocess/predict/postprocess` functions. The
runtime values are therefore labelled `adapter_end_to_end_per_frame`: model
loading, disk I/O, metric computation, and montage writing are included. They
are useful as a consistent local execution-cost comparison of the adapters, but
they should not be presented as pure inference latency.

Environment: RTX 3090 24 GB, driver 595.71.05. PyTorch CUDA measurements used
the local `ai` conda environment for S2M2 and the Fast-FoundationStereo local
conda environment for most external adapters. ONNX Runtime used the CUDA
execution provider. See `hardware_software_environment.json`.

One runtime row is missing: RAFT-Stereo RVC could not load under the existing
adapter architecture, although its previous accuracy CSV is retained in the
accuracy table.
