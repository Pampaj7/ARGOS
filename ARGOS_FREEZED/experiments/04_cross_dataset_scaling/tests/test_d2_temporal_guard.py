#!/usr/bin/env python3
"""Static guard for immutable temporal histories; no GPU or dataset access."""
import json
import subprocess
import sys
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_d2_temporal_audit.py"


class D2TemporalGuardTest(unittest.TestCase):
    def test_full_history_starts_at_sixteen(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--static-check"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["min_immutable_temporal_index"], 16)

    def test_d7_is_rejected(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--static-check", "--dataset-id", "7"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__": unittest.main()
