"""Image decoding, EXIF extraction, preview rendering and perceptual hashes."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import ANALYSIS_LONG_EDGE, IMAGE_EXTENSIONS, RAW_EXTENSIONS, THUMB_LONG_EDGE

try:  # HEIC/HEIF is standard on modern bodies; the plugin is a soft dependency.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    HEIF_AVAILABLE = False

try:  # RAW decoding is an optional extra (`pip install .[raw]`).
    import rawpy

    RAWPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    rawpy = None
    RAWPY_AVAILABLE = False

_EXIF_IFD = 0x8769
# Stable EXIF tag numbers (they do not change between Pillow versions).
_TAG_MAKE = 271
_TAG_MODEL = 272
_TAG_DATETIME = 306
_TAG_DATETIME_ORIGINAL = 36867
_TAG_SUBSEC_ORIGINAL = 37521
_TAG_ISO = 34855
_TAG_EXPOSURE_TIME = 33434
_TAG_FNUMBER = 33437
_TAG_FOCAL_LENGTH = 37386
_TAG_LENS = 42036


@dataclass
class ImageMeta:
    width: int
    height: int
    captured_at: float | None = None
    camera: str | None = None
    lens: str | None = None
    iso: int | None = None
    shutter: float | None = None
    aperture: float | None = None
    focal_len: float | None = None


def content_key(path: Path) -> str:
    """Cheap change detector: size + mtime. Avoids re-hashing 40 MB RAF files."""
    st = path.stat()
    return hashlib.sha1(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:20]


def iter_images(root: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    out: list[Path] = []
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if not any(part.startswith(".") for part in p.relative_to(root).parts):
                out.append(p)
    return out


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _as_int(value) -> int | None:
    f = _as_float(value)
    return int(f) if f is not None else None


def _parse_exif_datetime(value) -> float | None:
    if not value:
        return None
    text = str(value).strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _read_exif(img: Image.Image) -> dict:
    meta: dict = {}
    try:
        exif = img.getexif()
    except Exception:
        return meta
    if not exif:
        return meta
    try:
        exif_ifd = exif.get_ifd(_EXIF_IFD) or {}
    except Exception:
        exif_ifd = {}

    def pick(tag: int):
        if tag in exif_ifd:
            return exif_ifd[tag]
        return exif.get(tag)

    meta["captured_at"] = _parse_exif_datetime(
        pick(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
    )
    subsec = pick(_TAG_SUBSEC_ORIGINAL)
    if meta["captured_at"] is not None and subsec:
        try:
            meta["captured_at"] += float("0." + str(subsec).strip())
        except ValueError:
            pass
    make = str(exif.get(_TAG_MAKE) or "").strip()
    model = str(exif.get(_TAG_MODEL) or "").strip()
    camera = " ".join(x for x in (make, model) if x) or None
    if camera and camera.count(make) > 1:
        camera = model or make
    meta["camera"] = camera
    lens = pick(_TAG_LENS)
    meta["lens"] = str(lens).strip() if lens else None
    meta["iso"] = _as_int(pick(_TAG_ISO))
    meta["shutter"] = _as_float(pick(_TAG_EXPOSURE_TIME))
    meta["aperture"] = _as_float(pick(_TAG_FNUMBER))
    meta["focal_len"] = _as_float(pick(_TAG_FOCAL_LENGTH))
    return meta


def _open_pil(path: Path) -> Image.Image:
    """Open any supported image, using rawpy for RAW files when available."""
    if path.suffix.lower() in RAW_EXTENSIONS and RAWPY_AVAILABLE:
        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return Image.open(io.BytesIO(thumb.data))
                return Image.fromarray(thumb.data)
        except Exception:
            pass  # fall through to Pillow (DNG sometimes opens directly)
    return Image.open(path)


def _fit(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize so the long edge is exactly ``long_edge`` (up or down).

    A constant analysis size is what makes Laplacian-variance thresholds
    comparable across cameras. Upscaling a small file is intentional: it cannot
    carry more detail than it has, and the resulting softness is real.
    """
    w, h = img.size
    if not w or not h:
        return img
    scale = long_edge / float(max(w, h))
    if abs(scale - 1.0) < 0.01:
        return img
    resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def load_image(path: Path, long_edge: int = ANALYSIS_LONG_EDGE) -> tuple[np.ndarray | None, ImageMeta, str | None]:
    """Decode ``path`` into a BGR analysis frame plus metadata.

    Returns ``(bgr_or_none, meta, error)``. Unreadable files never raise: the
    pipeline records the error and keeps going.
    """
    try:
        with _open_pil(path) as raw_img:
            meta_dict = _read_exif(raw_img)
            img = ImageOps.exif_transpose(raw_img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            meta = ImageMeta(width=img.width, height=img.height, **meta_dict)
            img = _fit(img, long_edge)
            arr = np.asarray(img.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - per-file failure must not kill a scan
        return None, ImageMeta(width=0, height=0), f"{type(exc).__name__}: {exc}"
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, meta, None


def render_jpeg(path: Path, long_edge: int = THUMB_LONG_EDGE, quality: int = 82) -> bytes | None:
    """Render a cached preview/thumbnail. Called on demand, never in the hot loop."""
    try:
        with _open_pil(path) as raw_img:
            img = ImageOps.exif_transpose(raw_img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img = _fit(img, long_edge)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def gray_of(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- #
# Perceptual hashes (own implementation: no imagehash/scipy dependency)
# --------------------------------------------------------------------------- #

_PHASH_SIZE = 32
_PHASH_BITS = 8


def phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash. Resistant to resize/JPEG, not to pose changes."""
    small = cv2.resize(gray, (_PHASH_SIZE, _PHASH_SIZE), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small) / 255.0)
    low = dct[:_PHASH_BITS, :_PHASH_BITS]
    flat = low.flatten()
    median = np.median(flat[1:])  # drop DC: it only encodes average brightness
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: horizontal gradient sign, robust to exposure shifts."""
    small = cv2.resize(gray, (_PHASH_BITS + 1, _PHASH_BITS), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:].astype(np.int16) > small[:, :-1].astype(np.int16)
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hashes_to_hex(value: int) -> str:
    return f"{value:016x}"


def hex_to_hash(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def dct_signature(gray: np.ndarray, cells: int = 8) -> np.ndarray:
    """Small normalised DCT signature used for embedding-free structural similarity."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(small)[:cells, :cells].flatten()
    norm = float(np.linalg.norm(dct))
    return dct / norm if norm > 0 else dct


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
