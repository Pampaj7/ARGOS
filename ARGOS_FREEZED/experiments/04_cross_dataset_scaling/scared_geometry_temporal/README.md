# SCARED-C D2 geometry/temporal sidecar

Use only validation dataset ID 2. `run_d2_temporal_audit.py` reuses the validated D2 flow, alignment, H4, cached raw stereo, and immutable checkpoint paths, writes compact CSV/JSON only here, and rejects D7 before checkpoint loading. H4's frozen torchvision ResNet-18 dependency is pinned at `../dependencies/resnet18-f37072fd.pth`; override it only with the identical SHA-256. Run a smoke with `--smoke --max-frames 12`; delete a successful smoke directory before full evaluation.
