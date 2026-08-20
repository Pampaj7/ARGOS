#!/bin/bash
# Kill the A5/A6/A7 queue once A6 has its twelve epochs, so A7 never starts.
#
# A7 (half width) was dropped on the PI's call: a capacity point is arbitrary in a way
# A5 and A6 are not -- A5 buys back 45% of the module's runtime if the cue is idle, and
# A6 removes a whole family of evidence rather than tuning a number. The job's for-loop
# was parsed at launch, so editing the script cannot stop A7; this can.
set -u
RUN=/dtu/p1/leopam/ARGOS/ARGOS_hand/original_h4/model_design/training_runs/ablation_A6_geometry_only/training_history.csv
JOB=$1
while true; do
    DONE=$(( $(wc -l < "$RUN" 2>/dev/null || echo 1) - 1 ))
    if [ "$DONE" -ge 12 ]; then
        echo "A6 finished at $DONE epochs; stopping $JOB before A7 starts"
        bkill "$JOB"
        exit 0
    fi
    bjobs "$JOB" >/dev/null 2>&1 || { echo "job $JOB already gone"; exit 0; }
    sleep 120
done
