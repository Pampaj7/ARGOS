import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_critic_or_h4_imports_in_core():
    imports = []
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
    joined = " ".join(imports).lower()
    assert "critic" not in joined and "hard_negative" not in joined and "codd" not in joined


def test_core_imports_without_argos_v2_on_pythonpath():
    environment = os.environ.copy(); environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", f"import sys; sys.path.insert(0,{str(ROOT / 'src')!r}); import argos_freezed"], cwd="/tmp", env=environment)
    assert completed.returncode == 0
