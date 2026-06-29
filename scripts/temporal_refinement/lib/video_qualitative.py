from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def colorize_scalar(value: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    arr = value.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgb = np.stack([norm, 1.0 - np.abs(norm - 0.5) * 2.0, 1.0 - norm], axis=-1)
    rgb[~finite] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR))


def make_board(tiles: Sequence[tuple[str, np.ndarray]], panel_size: tuple[int, int] = (320, 256), cols: int = 4) -> np.ndarray:
    panel_w, panel_h = panel_size
    label_h = 24
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * (panel_h + label_h), cols * panel_w, 3), 255, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for idx, (label, tile) in enumerate(tiles):
        row, col = divmod(idx, cols)
        x = col * panel_w
        y = row * (panel_h + label_h)
        if tile.ndim == 2:
            tile_rgb = np.repeat(tile[..., None], 3, axis=2)
        else:
            tile_rgb = tile[..., :3]
        resized = resize_rgb(tile_rgb, (panel_w, panel_h))
        image.paste(Image.fromarray(resized), (x, y + label_h))
        draw.text((x + 5, y + 5), label[:38], fill=(0, 0, 0))
    return np.array(image)


def write_mp4(path: Path, frames: Iterable[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise RuntimeError(f"No frames supplied for video {path}")
    first = first.astype(np.uint8, copy=False)
    h, w = first.shape[:2]
    try:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open OpenCV VideoWriter for {path}")
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame in iterator:
            writer.write(cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except ModuleNotFoundError:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(first.tobytes())
        for frame in iterator:
            proc.stdin.write(frame.astype(np.uint8, copy=False).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
    except Exception:
        proc.kill()
        raise
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr[-1000:]}")
