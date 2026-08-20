#!/bin/bash
# Submits the paired runtime measurement only when no argos_* job is left on the node.
while bjobs -w 2>/dev/null | grep -qE "argos_(a3b|twineval|a6eval|nofb|dclosure|a3bres)"; do
    sleep 600
done
exec bash /dtu/p1/leopam/ARGOS/scripts/run_runtime_quiet_pair.sh
