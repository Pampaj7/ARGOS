# Architecture diagram specification

```text
Frozen stereo (L/R RGB -> raw disparity + validity)
             | raw only                         \
             v                                  \
Immutable bank: CS1 / CS2 / CS4 / CS8           |
             | original RGB + raw disparity      |
             v                                   |
Frozen SEA-RAFT: four direct current->anchor flows
             | no composition
             v
BiDA pull warp: grid + flow, support + FB consistency
             |
             v
Universal 17-channel candidate evidence
             |
             v
Shared encoder -> learned single-anchor retrieval
             | invalid score = -infinity
             v
Pairwise soft fusion with current raw
             |
       accepted? -- no --> exact raw fallback
             | yes
             v
        refined output

Independent current raw -----------------------> raw-only bank append
Refined output --------------------------------X no memory write
```

The paper figure should visually isolate the output path from the raw-only memory-update path, show four independent SEA-RAFT arrows, and place H4 and the spatial critic outside the main-method boundary.
