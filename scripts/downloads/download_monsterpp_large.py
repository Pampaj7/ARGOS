from pathlib import Path
import os
import time

from huggingface_hub import hf_hub_download


DEFAULT_STEREO_ROOT = Path(__file__).resolve().parents[2] / "stereo"
TARGET_DIR = Path(
    os.environ.get(
        "ARGOS_MONSTERPP_CHECKPOINT_DIR",
        DEFAULT_STEREO_ROOT / "MonSter-plusplus/MonSter++/checkpoints",
    )
)
FORCE_DOWNLOAD = os.environ.get("ARGOS_FORCE_DOWNLOAD", "").lower() in {"1", "true", "yes"}


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    log("downloading MonSter++ Mix_all_large.pth")
    path = hf_hub_download(
        repo_id="cjd24/MonSter-plusplus",
        repo_type="model",
        filename="Mix_all_large.pth",
        local_dir=TARGET_DIR,
        force_download=FORCE_DOWNLOAD,
    )
    size_gb = Path(path).stat().st_size / (1024 ** 3)
    log(f"done Mix_all_large.pth -> {path} ({size_gb:.2f} GiB)")


if __name__ == "__main__":
    main()
