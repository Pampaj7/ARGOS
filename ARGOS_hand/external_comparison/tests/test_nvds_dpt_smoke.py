import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nvds_dpt_smoke", ROOT / "run_nvds_dpt_d2_smoke.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
sys.path.insert(0, str(ROOT))
FULL = importlib.import_module("run_nvds_dpt_d2_full")


class NVDSSmokeTest(unittest.TestCase):
    def test_official_command_is_strict_and_left_only(self):
        source = (ROOT / "run_nvds_dpt_d2_smoke.py").read_text()
        self.assertIn('"--strict_resume"', source)
        self.assertNotIn("load_frame_lr", source)
        self.assertIn('info.seq_dir / "left"', source)
        self.assertIn("stdout=log_handle", source)
        self.assertNotIn("stdout=subprocess.PIPE", source)

    def test_full_d2_frame_contract(self):
        source = (ROOT / "run_nvds_dpt_d2_full.py").read_text()
        self.assertIn('"dataset_2_keyframe_2": 1033', source)
        self.assertIn('"dataset_2_keyframe_3": 1102', source)
        self.assertIn('"dataset_2_keyframe_4": 2114', source)
        self.assertEqual(1033 + 1102 + 2114, 4249)

    def test_official_stdout_opw_parser_accepts_space_separated_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "official_stdout.txt"
            report.write_text("**********initial**********\nall: 10.0 mean: 1.0 frames: 10\n**********forward**********\nall: 2.0 mean: 0.2 frames: 10\n**********backward**********\nall: 3.0 mean: 0.3 frames: 10\n**********Mixing**********\nall: 4.0 mean: 0.4 frames: 10\n")
            self.assertEqual(RUNNER._opw(report)["mix"], {"total": 4.0, "mean": 0.4, "frames": 10})

    def test_full_aggregate_is_total_weighted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = {}
            for sequence, frames, total in (("dataset_2_keyframe_2", 2, 2.0), ("dataset_2_keyframe_3", 8, 4.0)):
                path = root / sequence; path.mkdir()
                manifest = {"input": {"frames": frames}, "evidence": {"sha256": sequence},
                            "diagnostic": {"opw_raw_forward_backward_mix": {kind: {"total": total, "mean": total / frames, "frames": frames} for kind in ("initial", "forward", "backward", "mix")}}}
                (path / "run_manifest.json").write_text(json.dumps(manifest))
                manifests[sequence] = manifest
            with patch.object(FULL, "RESULT", root):
                aggregate = FULL._aggregate(manifests)
            self.assertEqual(aggregate["opw"]["mix"], {"total": 6.0, "micro_mean": 0.6, "macro_mean": 0.75, "frames": 10})

    def test_dirty_upstream_rejected_before_copy(self):
        locked = json.loads((ROOT / "upstreams.lock.json").read_text())["upstreams"]["nvds"]
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / "LICENSE").write_bytes((ROOT / "upstreams/nvds/LICENSE").read_bytes())
            replies = iter((locked["origin"], locked["commit"], " M infer_NVDS_dpt_bi.py"))
            with patch.object(RUNNER, "_git", side_effect=lambda *_: next(replies)):
                with self.assertRaisesRegex(RuntimeError, "dirty"):
                    RUNNER._verify_upstream(repo)

    def test_manifest_evidence_output_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mix_uint16").mkdir()
            output = root / "mix_uint16/frame_000000.png"
            output.write_bytes(b"uint16-placeholder")
            stdout = root / "official_stdout.txt"
            stdout.write_text("**********initial**********\nall:1.0,mean:0.1,frames:10\n**********forward**********\nall:2.0,mean:0.2,frames:10\n**********backward**********\nall:3.0,mean:0.3,frames:10\n**********Mixing**********\nall:4.0,mean:0.4,frames:10\n")
            source_hash = "source-records"
            evidence = {"source_input_records_sha256": source_hash,
                        "output": {"mix_uint16_tree_sha256": RUNNER._tree_sha256(root / "mix_uint16")}, "opw": RUNNER._opw(stdout)}
            evidence_path = root / "official_execution_evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            manifest = {"evidence": {"path": evidence_path.name, "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()},
                        "input": {"source_input_records_sha256": source_hash},
                        "output": {"mix_uint16_tree_sha256": evidence["output"]["mix_uint16_tree_sha256"],
                                   "official_stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest()},
                        "diagnostic": {"opw_raw_forward_backward_mix": evidence["opw"]}}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            RUNNER._validate_published(root)
            evidence["opw"]["mix"]["mean"] = 99.0
            evidence_path.write_text(json.dumps(evidence))
            manifest["evidence"]["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "OPW"):
                RUNNER._validate_published(root)

    def test_manifest_opw_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "mix_uint16").mkdir(); (root / "mix_uint16/frame.png").write_bytes(b"x")
            stdout = root / "official_stdout.txt"
            stdout.write_text("**********initial**********\nall:1,mean:1,frames:1\n**********forward**********\nall:1,mean:1,frames:1\n**********backward**********\nall:1,mean:1,frames:1\n**********Mixing**********\nall:1,mean:1,frames:1\n")
            opw = RUNNER._opw(stdout); evidence = {"source_input_records_sha256": "source", "output": {"mix_uint16_tree_sha256": RUNNER._tree_sha256(root / "mix_uint16")}, "opw": opw}
            evidence_path = root / "official_execution_evidence.json"; evidence_path.write_text(json.dumps(evidence))
            manifest = {"evidence": {"path": evidence_path.name, "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()}, "input": {"source_input_records_sha256": "source"}, "output": evidence["output"] | {"official_stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest()}, "diagnostic": {"opw_raw_forward_backward_mix": dict(opw)}}
            manifest["diagnostic"]["opw_raw_forward_backward_mix"]["mix"] = {"total": 2.0, "mean": 2.0, "frames": 1}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "OPW"):
                RUNNER._validate_published(root)


if __name__ == "__main__":
    unittest.main()
