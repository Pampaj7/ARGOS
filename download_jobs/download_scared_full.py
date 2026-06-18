from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from huggingface_hub import hf_hub_download


REPO_ID = "kairuo/scared"
TARGET_DIR = Path("/home/pampaj/Desktop/stereo/Fast-FoundationStereo/data/surgical_stereo/scared")
MAX_WORKERS = 8

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


def download_one(filename):
    out_path = TARGET_DIR / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        size_gb = out_path.stat().st_size / (1024 ** 3)
        return filename, out_path, size_gb, "already_present"

    path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=TARGET_DIR,
            resume_download=True,
        )
    )
    size_gb = path.stat().st_size / (1024 ** 3)
    return filename, path, size_gb, "downloaded"


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[{time.strftime('%F %T')}] Starting SCARED full download into {TARGET_DIR} "
        f"with {MAX_WORKERS} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, filename): filename for filename in FILES}
        for future in as_completed(futures):
            filename = futures[future]
            try:
                name, path, size_gb, status = future.result()
            except Exception as exc:
                print(f"[{time.strftime('%F %T')}] failed {filename}: {exc!r}", flush=True)
                raise
            print(
                f"[{time.strftime('%F %T')}] {status} {name} -> {path} ({size_gb:.2f} GiB)",
                flush=True,
            )
    print(f"[{time.strftime('%F %T')}] SCARED full download complete", flush=True)


if __name__ == "__main__":
    main()
