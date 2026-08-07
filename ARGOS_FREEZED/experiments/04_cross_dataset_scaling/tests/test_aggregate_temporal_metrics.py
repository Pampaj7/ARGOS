#!/usr/bin/env python3
import sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "scripts"))
from aggregate_temporal_metrics import aggregate

class AggregateTest(unittest.TestCase):
    def test_count_weighting_and_no_fabricated_diagnostics(self):
        rows = []
        for method, epe in (("raw", 2.0), ("h4", 1.0), ("immutable", 1.5)):
            rows.append({"backbone":"B","sequence":"S","frame_id":"1","method":method,"age_frames":"0","epe":str(epe),"epe_count":"10","gt_tce":"nan","gt_tce_count":"0","rgt_tce":"nan","nr_tce":"nan","nr_tce_count":"0","stereo_photo":"nan","stereo_photo_count":"0","temporal_photo":"nan","temporal_photo_count":"0"})
            rows.append({"backbone":"B","sequence":"S","frame_id":"1","method":method,"age_frames":"1","epe":str(epe),"epe_count":"10","gt_tce":str(epe),"gt_tce_count":"10","rgt_tce":str(epe/2),"nr_tce":str(epe/3),"nr_tce_count":"10","stereo_photo":"0.1","stereo_photo_count":"10","temporal_photo":"0.2","temporal_photo_count":"10"})
        summary, pareto = aggregate(rows); h4 = next(r for r in summary if r["scope"]=="B" and r["method"]=="h4" and r["age_frames"]==1)
        self.assertEqual(h4["gain_vs_raw_epe"], 1.0); self.assertEqual(h4["track_jitter"], "N/A"); self.assertTrue(pareto)

if __name__ == "__main__": unittest.main()
