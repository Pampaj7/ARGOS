#!/usr/bin/env python3
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preflight_external import audit

class ExternalPreflightTest(unittest.TestCase):
    def test_d4d_fails_closed_on_readiness_blocker(self):
        result = audit("D4D")
        self.assertEqual(result["status"], "BLOCKED_VERSION_PIN_REQUIRED")
        self.assertEqual(result["cache_readiness_status"], "BLOCKED_VERSION_PIN_REQUIRED")

if __name__ == "__main__": unittest.main()
