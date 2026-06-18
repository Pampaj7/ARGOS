#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

ARCHIVE_EXTS = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz')


def archive_stem(path: Path) -> str:
    name = path.name
    for suffix in ['.tar.gz', '.tar.xz', '.tgz', '.txz', '.zip', '.tar']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def dataset_root_for_archive(path: Path) -> Path:
    parts = path.parts
    if 'raw' in parts:
        raw_idx = parts.index('raw')
        return Path(*parts[:raw_idx])
    return path.parent


def output_dir_for_archive(path: Path) -> Path:
    return dataset_root_for_archive(path) / 'raw' / 'extracted' / archive_stem(path)


def has_existing_payload(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    for child in out_dir.iterdir():
        if child.name in {'_EXTRACT_DONE.json', '_EXTRACT_FAILED.json'}:
            continue
        return True
    return False


def disk_usage(path: Path) -> str:
    try:
        return subprocess.check_output(['du', '-sh', str(path)], text=True).split()[0]
    except Exception:
        return ''


def extract_zip(path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out_dir)


def extract_tar(path: Path, out_dir: Path) -> None:
    with tarfile.open(path) as tf:
        tf.extractall(out_dir, filter='data')


def extract(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.name.lower().endswith('.zip'):
        extract_zip(path, out_dir)
    else:
        extract_tar(path, out_dir)


def find_archives(dataset_dir: Path) -> list[Path]:
    archives = []
    for path in dataset_dir.rglob('*'):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(lower.endswith(ext) for ext in ARCHIVE_EXTS):
            archives.append(path)
    return sorted(archives, key=lambda p: (str(dataset_root_for_archive(p)), p.name))


def write_manifest(dataset_dir: Path, manifest: list[dict]) -> None:
    (dataset_dir / 'extraction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, default=Path('dataset'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    archives = find_archives(args.dataset_dir)
    manifest = []
    print(f'Found {len(archives)} archives under {args.dataset_dir}', flush=True)
    for idx, archive in enumerate(archives, 1):
        out_dir = output_dir_for_archive(archive)
        done_path = out_dir / '_EXTRACT_DONE.json'
        failed_path = out_dir / '_EXTRACT_FAILED.json'
        row = {
            'archive': str(archive),
            'archive_size_bytes': archive.stat().st_size,
            'output_dir': str(out_dir),
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        print(f'[{idx}/{len(archives)}] {archive} -> {out_dir}', flush=True)
        if has_existing_payload(out_dir) and done_path.exists() and not args.force:
            row['status'] = 'skipped_existing_done'
            row['output_size'] = disk_usage(out_dir)
            print(f'  skip: already extracted ({row["output_size"]})', flush=True)
            manifest.append(row)
            write_manifest(args.dataset_dir, manifest)
            continue
        if has_existing_payload(out_dir) and not args.force:
            row['status'] = 'skipped_existing_payload_no_done_marker'
            row['output_size'] = disk_usage(out_dir)
            print(f'  skip: payload exists without marker ({row["output_size"]})', flush=True)
            manifest.append(row)
            write_manifest(args.dataset_dir, manifest)
            continue
        if args.force and out_dir.exists():
            shutil.rmtree(out_dir)
        try:
            t0 = time.time()
            extract(archive, out_dir)
            elapsed = time.time() - t0
            row['status'] = 'extracted'
            row['elapsed_sec'] = elapsed
            row['output_size'] = disk_usage(out_dir)
            row['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            failed_path.unlink(missing_ok=True)
            done_path.write_text(json.dumps(row, indent=2) + '\n')
            print(f'  done: {row["output_size"]} in {elapsed/60:.1f} min', flush=True)
        except Exception as exc:
            row['status'] = 'failed'
            row['error'] = repr(exc)
            row['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            out_dir.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(json.dumps(row, indent=2) + '\n')
            print(f'  FAILED: {exc!r}', flush=True)
        manifest.append(row)
        write_manifest(args.dataset_dir, manifest)
    write_manifest(args.dataset_dir, manifest)
    print(f'Wrote {args.dataset_dir / "extraction_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
