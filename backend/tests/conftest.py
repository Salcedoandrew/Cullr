from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


def make_scene(seed: int = 0, width: int = 1200, height: int = 800) -> np.ndarray:
    """A synthetic 'photo': gradients, shapes and fine texture.

    It has enough high-frequency detail that blur destroys a measurable amount of
    Laplacian energy, which is what the sharpness tests need.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = 60 + 90 * (xx / width) + 25 * np.sin(xx / 37.0)
    base[..., 1] = 70 + 80 * (yy / height) + 20 * np.cos(yy / 29.0)
    base[..., 2] = 90 + 70 * ((xx + yy) / (width + height))

    for _ in range(24):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(12, 70))
        color = tuple(int(v) for v in rng.integers(20, 235, size=3))
        cv2.rectangle(base, (cx, cy), (cx + r, cy + r), color, -1)
    for _ in range(12):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(10, 45))
        color = tuple(int(v) for v in rng.integers(20, 235, size=3))
        cv2.circle(base, (cx, cy), r, color, -1)

    # Fine texture: thin lines + grain, the part blur removes first.
    for _ in range(220):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        x1 = min(width - 1, x0 + int(rng.integers(3, 22)))
        y1 = min(height - 1, y0 + int(rng.integers(1, 5)))
        base[y0:y1, x0:x1] += float(rng.integers(25, 90))
    base += rng.normal(0, 3.0, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def blur(bgr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return bgr.copy()
    return cv2.GaussianBlur(bgr, (0, 0), sigma)


def save_jpeg(path: Path, bgr: np.ndarray, captured_at: str | None = None,
              quality: int = 92) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if captured_at:
        exif = Image.Exif()
        exif[36867] = captured_at          # DateTimeOriginal
        exif[272] = "SyntheticCam"         # Model
        img.save(path, format="JPEG", quality=quality, exif=exif)
    else:
        img.save(path, format="JPEG", quality=quality)
    return path


def shift_image(bgr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(bgr, matrix, (bgr.shape[1], bgr.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


@pytest.fixture
def scene() -> np.ndarray:
    return make_scene(1)


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    """A small shoot: two singletons, a 3-frame burst, and one exact duplicate."""
    root = tmp_path / "shoot"
    root.mkdir()

    first = make_scene(11)
    save_jpeg(root / "IMG_0001.jpg", first, "2024:05:01 10:00:00")
    save_jpeg(root / "IMG_0001_copy.jpg", first, "2024:05:01 10:00:01")

    burst = make_scene(12)
    times = ["2024:05:01 10:10:00", "2024:05:01 10:10:00", "2024:05:01 10:10:01"]
    for index, stamp in enumerate(times):
        frame = shift_image(burst, index * 6, index * 2)
        if index == 2:
            frame = blur(frame, 5.0)
        save_jpeg(root / f"IMG_010{index}.jpg", frame, stamp)

    save_jpeg(root / "IMG_0200.jpg", make_scene(13), "2024:05:01 11:00:00")
    return root
