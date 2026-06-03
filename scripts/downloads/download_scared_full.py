from pathlib import Path
import time

from huggingface_hub import hf_hub_download


REPO_ID = "kairuo/scared"
TARGET_DIR = Path("/home/pampaj/Desktop/stereo/Fast-FoundationStereo/data/surgical_stereo/scared")

FILES = [
    "README.md",
    "code.zip",
    "dataset_1.zip",
    "dataset_2.zip",
    "dataset_3.zip",
    "dataset_4.zip",
    "dataset_5.zip",
    "dataset_6.zip",
    "dataset_7.zip",
    "dataset_8.zip",
    "dataset_9.zip",
    "test_dataset_8.zip",
    "test_dataset_9.zip",
]


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%F %T')}] Starting SCARED full download into {TARGET_DIR}", flush=True)
    for filename in FILES:
        print(f"[{time.strftime('%F %T')}] downloading {filename}", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=TARGET_DIR,
            resume_download=True,
        )
        size_gb = Path(path).stat().st_size / (1024 ** 3)
        print(f"[{time.strftime('%F %T')}] done {filename} -> {path} ({size_gb:.2f} GiB)", flush=True)
    print(f"[{time.strftime('%F %T')}] SCARED full download complete", flush=True)


if __name__ == "__main__":
    main()
