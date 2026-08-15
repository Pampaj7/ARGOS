#!/bin/bash
# Autonomous recompute with the support-masked canonical H4 variant.
# Waits for the in-flight verification, refuses to continue unless it shows zero invalid
# pixels, then recomputes DRENDS (5 recordings x 3 backbones) and SCARED-C D2/D7.
#   setsid nohup scripts/run_masked_recompute.sh > logs/masked_recompute.log 2>&1 < /dev/null &
set -u
A=/dtu/p1/leopam/ARGOS
PY=$A/.miniconda/envs/argos/bin/python
M=model_design.comparison.canonical_h4_masked:factory
DIAG=$A/ARGOS_hand/results/invalid_refined_diagnosis
R="Vid14_Pancreas_High Vid10_Liver_Med Vid11_Liver_High Vid12_Pancreas_Ext Vid13_Pancreas_Med"
cd "$A" || exit 1

echo "[$(date -Is)] waiting for verification jobs"
while bjobs 2>/dev/null | grep -q argos_diag; do sleep 30; done

echo "[$(date -Is)] checking verification"
"$PY" - <<'PY' || { echo "VERIFICATION FAILED - not recomputing"; exit 1; }
import json,sys,glob
files=sorted(glob.glob('/dtu/p1/leopam/ARGOS/ARGOS_hand/results/invalid_refined_diagnosis/*_masked_summary.json'))
if len(files)<2: sys.exit(f"expected 2 masked summaries, found {len(files)}")
for f in files:
    d=json.load(open(f))
    print(f"  {d['recording']}: frames={d['frames']} invalid={d['invalid_refined_pixels']}")
    if d['invalid_refined_pixels']!=0: sys.exit(f"still invalid in {f}")
    if d['frames']<1400: sys.exit(f"truncated run in {f}")
print("  verification PASS: zero invalid pixels")
PY

echo "[$(date -Is)] launching DRENDS recompute"
for BB in RAFT-Stereo CREStereo Fast-FoundationStereo; do
    setsid nohup "$A/scripts/run_drends_backbone.sh" --backbone "$BB" --module "$M" --recordings $R \
        --output "$A/ARGOS_hand/results/drends_masked/$BB" > "$A/logs/masked_drends_$BB.log" 2>&1 < /dev/null &
    sleep 10
done
while bjobs 2>/dev/null | grep -q drends_bb; do sleep 60; done
echo "[$(date -Is)] DRENDS recompute done"

echo "[$(date -Is)] launching SCARED-C D2/D7 recompute"
setsid nohup "$A/scripts/run_scared_masked.sh" > "$A/logs/masked_scared.log" 2>&1 < /dev/null &
while bjobs 2>/dev/null | grep -q scared_msk; do sleep 60; done
echo "[$(date -Is)] ALL DONE"
