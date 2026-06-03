from pathlib import Path
import os
import subprocess
import time

from huggingface_hub import hf_hub_download, snapshot_download


S2M2_DIR = Path("/home/pampaj/Desktop/stereo/s2m2/weights/pretrain_weights")
ENDOSLAM_DIR = Path("/home/pampaj/Desktop/stereo/datasets/EndoSLAM")


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def process_running(pattern):
    result = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def wait_for_scared():
    log("waiting for SCARED download job to finish before starting extra downloads")
    while process_running("download_scared_full.py"):
        time.sleep(60)
    log("SCARED download job is no longer running; starting extra queue")


def download_s2m2_weights():
    S2M2_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["CH256NTR3.pth", "CH384NTR3.pth"]:
        log(f"downloading S2M2 weight {filename}")
        path = hf_hub_download(
            repo_id="minimok/s2m2",
            repo_type="model",
            filename=filename,
            local_dir=S2M2_DIR,
        )
        size_gb = Path(path).stat().st_size / (1024 ** 3)
        log(f"done {filename} -> {path} ({size_gb:.2f} GiB)")


def download_endoslam():
    ENDOSLAM_DIR.mkdir(parents=True, exist_ok=True)
    log(f"downloading EndoSLAM mirror into {ENDOSLAM_DIR}")
    path = snapshot_download(
        repo_id="introvoyz041/EndoSLAM",
        repo_type="dataset",
        local_dir=ENDOSLAM_DIR,
        resume_download=True,
    )
    log(f"done EndoSLAM snapshot -> {path}")


def main():
    wait_for_scared()
    download_s2m2_weights()
    download_endoslam()
    log("extra training download queue complete")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
