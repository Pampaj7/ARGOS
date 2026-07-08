# BiDAVideo License Audit

- Repository: `https://github.com/TomTomTommi/bidavideo.git`
- Local path: `external/bidavideo/`
- Commit: `dae817df1ceaafcb865ebd9c7aa44b16c535e856`
- License file: `external/bidavideo/LICENSE`
- License: MIT License
- Copyright: Copyright (c) 2024 Junpeng Jing

## Obligation

The MIT notice and permission text must be included in copies or substantial portions of the adapted software.

## ARGOS v2 Usage

ARGOS v2 copied/adapted only the small BiDAStabilizer architectural blocks required for causal experiments:

- `ResidualBlockNoBN`
- `ResidualBlocksWithInputConv`
- `flow_warp` convention
- layer/channel structure of the forward propagation and residual head

Each adapted source file includes upstream repository, source file, commit, license, and modification comments.

The external repository itself is kept under `external/bidavideo/` as a read-only reference and added to local `.git/info/exclude` to avoid accidentally committing it.
