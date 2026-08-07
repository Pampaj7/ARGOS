#!/usr/bin/env bash
set -u
ps -ef | grep -E 'run_hybrid_temporal_memory_oracle_audit|launch_hybrid_memory' | grep -v grep || true
ls -la /dtu/p1/leopam/ARGOS/ARGOS-V2/results/hybrid_temporal_memory_oracle_audit || true
for log in /dtu/p1/leopam/ARGOS/ARGOS-V2/results/hybrid_temporal_memory_oracle_audit/validation_gpu*.log; do
  test -f "$log" && tail -n 40 "$log"
done
