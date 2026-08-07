# ARGOS v2 controlled audit scripts

`self_check.py` preserves the D7/training lock. `run_d2_temporal_audit.py` is a D2-only GPU sidecar; use `--smoke --max-frames 12` first and remove successful smoke output before a full run. `preflight_external.py` never runs inference. `audit_servct_static.py` is CPU-only raw stereo geometry; temporal refiners are N/A.
