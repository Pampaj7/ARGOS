# H4 frozen ResNet-18 dependency

`resnet18-f37072fd.pth` is the official torchvision ResNet-18 ImageNet checkpoint downloaded from `https://download.pytorch.org/models/resnet18-f37072fd.pth` on 2026-08-06. SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

It is a frozen external dependency required by the validated H4 feature extractor. The D2 sidecar verifies this exact SHA-256 before use and accepts an explicit `--resnet-checkpoint` only when it has the same hash.
