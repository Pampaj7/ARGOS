# Bounded H4 baseline

This isolated ARGOS v2 baseline combines CODD-style learned fusion with BiDA-style alignment and a corrected recurrent disparity state. It is evaluated in four-frame causal windows, resetting state before the next window. Indefinite recurrence drifts; bounded H4 remains the comparison showing that corrected state is useful only over a short horizon.

H4 transferred positively on seen and evaluated unseen SCARED-C backbones, but the immutable raw multi-anchor method beat it on strict common support in the frozen transfer audit. The canonical checkpoint and provenance are recorded in `MANIFEST.json`. No main-package module imports this baseline.
