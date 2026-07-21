# ARGOS v2 — multi-domain A2 proposal audit

## Hypothesis

The detector-only M1 experiment failed because it operated on a frozen A2
proposal trained solely on SCARED-C.  A detector can abstain from a bad proposal
but cannot turn it into a useful one.  This experiment therefore trains the
**existing A2 model and loss unchanged** on a mixture of:

* SCARED-C corrected temporal GT, with S2M2-S, RAFT-Stereo and StereoAnywhere;
* D4D curated Zivid anchor GT, with **S2M2-S only**.

All inference still uses frozen stereo, SEA-RAFT and the canonical causal
BiDA t-1 warp.  No parameter of the backbones or flow model is trainable.

## Why D4D is S2M2-S-only

The cache audit on specimen-2 shows RAFT-Stereo has a global disparity-scale
failure (12.452 px EPE) and StereoAnywhere has 7.936 px EPE under the frozen
wrappers.  This is not an honest temporal-refinement target for a bounded
3-pixel residual.  Training it as one would conflate proposal generation with
an uncorrectable global stereo error and cannot establish backbone-agnostic
temporal transfer.

S2M2-S has 0.874 px anchor EPE and high correlation with the same curated
Zivid geometry, making it the only admissible D4D supervised source here.

## Frozen architecture and loss

* `LearnedT1Refiner("A2", tau_px=3.0)`, unchanged 39,299-parameter model.
* Existing A2 loss: geometry, correction, raw-error gate, memory-usefulness
  gate and update regularization; no D4D-specific loss or domain identity.
* A single domain-balanced sampler selects domain, then `(backbone, sequence)`
  group, then a causal pair.  It never mixes temporal pairs across a sequence
  or backbone.

## Split and stop rules

| role | SCARED-C | D4D |
|---|---|---|
| train | existing 13 non-dataset-7 sequences / all 3 seen backbones | specimen-1 curated anchors / S2M2-S |
| validation/selection | dataset-7 keyframes 1–2 / all 3 seen backbones | specimen-2 curated anchors / S2M2-S |
| final only | dataset-7 keyframes 3–4; Fast-FoundationStereo/CREStereo afterward | specimen-3 anchors |

SERV-CT, StereoMIS, Fast-FoundationStereo, CREStereo and D4D specimen-3
labels are not loaded before a candidate passes both seen validation domains.

Proposal promotion requires positive geometry gain and nontrivial oracle
recovery on both seen domains.  Unconditional update safety is reported but is
not a proposal gate: the validated A2 baseline is deliberately paired with a
separate authorization module.  The next stage must retrain and evaluate that
authorization on the frozen selected proposal; *that* composition carries the
clean-input safety gate.  A proposal that fails geometry is not evaluated on
unseen domains.
