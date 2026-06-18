import argparse
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.argos_paths import ROOT_DIR, ARGOS_ENV_PYTHON


ROOT = ROOT_DIR
PYTHON = ARGOS_ENV_PYTHON


COMMANDS = [
    {
        "name": "scoreboard",
        "cwd": ROOT,
        "cmd": [str(PYTHON), "scripts/reports/make_servct_scoreboard.py"],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for item in COMMANDS:
        print(f"[ARGOS] {item['name']}")
        print(" ".join(item["cmd"]))
        if not args.dry_run:
            subprocess.run(item["cmd"], cwd=item["cwd"], check=True)


if __name__ == "__main__":
    main()
