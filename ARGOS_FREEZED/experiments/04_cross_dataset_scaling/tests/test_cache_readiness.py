#!/usr/bin/env python3
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cache_manifests import classify

class CacheReadinessTest(unittest.TestCase):
    def test_discrepancy_or_missing_flow_fails_closed(self):
        self.assertEqual(classify(cache_rows=832, expected_causal_rows=156, flow_present=False), "BLOCKED_VERSION_PIN_REQUIRED")
        self.assertEqual(classify(cache_rows=156, expected_causal_rows=156, flow_present=False), "BLOCKED_FLOW_CACHE_MISSING")

if __name__ == "__main__": unittest.main()
