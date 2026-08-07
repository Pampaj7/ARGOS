# StereoMIS zero-shot no-reference preflight

No dense geometric reference is available. The only permitted future output is labelled **zero-shot no-reference temporal transfer**. `scripts/preflight_external.py StereoMIS` must pass before a separate frozen GPU evaluation; it never opens D7 and does not tune thresholds. Current result: [`PREFLIGHT.json`](PREFLIGHT.json), `BLOCKED_CACHE_INCOMPLETE`.
