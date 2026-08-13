import json
from pathlib import Path
import hashlib
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UpstreamLockTest(unittest.TestCase):
    def test_locked_upstreams_match_head_origin_license_and_status(self):
        locked = json.loads((ROOT / "upstreams.lock.json").read_text())["upstreams"]
        declarations = subprocess.check_output(["git", "config", "--file", str(ROOT.parents[1] / ".gitmodules"), "--get-regexp", r"^submodule\..*\.(path|url)$"], text=True)
        submodules = {}
        for line in declarations.splitlines():
            key, value = line.split(None, 1)
            _, name, field = key.split(".", 2)
            submodules.setdefault(name, {})[field] = value
        for name, item in locked.items():
            repo = ROOT / "upstreams" / name
            relative = repo.relative_to(ROOT.parents[1]).as_posix()
            self.assertIn({"path": relative, "url": item["origin"]}, submodules.values())
            self.assertTrue((repo / ".git").exists() and len(item["commit"]) == 40 and item["license"] == "MIT")
            git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
            self.assertEqual(git("rev-parse", "HEAD"), item["commit"])
            self.assertEqual(git("remote", "get-url", "origin"), item["origin"])
            self.assertEqual(hashlib.sha256((repo / "LICENSE").read_bytes()).hexdigest(), item["license_sha256"])
            self.assertEqual(git("status", "--porcelain"), "")
            index = subprocess.check_output(["git", "-C", str(ROOT.parents[1]), "ls-files", "-s", "--", relative], text=True).split()
            self.assertEqual(index[:2], ["160000", item["commit"]])
            status = subprocess.check_output(["git", "-C", str(ROOT.parents[1]), "submodule", "status", "--", relative], text=True)
            self.assertTrue(status.startswith(f" {item['commit']} {relative} "))
            self.assertEqual(git("-c", "submodule.recurse=false", "status", "--porcelain"), "")
        protocols = [json.loads(path.read_text()) for path in (ROOT / "protocols").glob("*.json")]
        self.assertTrue(all(not item["online_or_h4"] for item in protocols))
        self.assertEqual({item["method"] for item in protocols if item["future_frames_required"]}, {"nvds_plus_bidirectional_offline", "bidastabilizer_bidirectional_offline"})

    def test_static_upstream_source_evidence(self):
        nvds = (ROOT / "upstreams/nvds/infer_NVDS_dpt_bi.py").read_text()
        bida = (ROOT / "upstreams/bidavideo/models/core/bidastabilizer.py").read_text()
        self.assertTrue("seq_len = 4" in nvds and "mixing" in nvds.lower())
        self.assertTrue("backward-time" in bida and "forward-time" in bida)
