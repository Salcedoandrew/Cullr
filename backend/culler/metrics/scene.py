"""Scene embeddings for duplicate confirmation (DINOv2-small over ONNX).

Perceptual hashes cannot tell a re-encoded copy from the same shot with a hand
moved or an expression changed. A scene embedding can: the two are close in
embedding space even when the hashes differ by more than the gate allows.
``similarity.find_groups`` already accepts such vectors (its two-stage gate
confirms loose hash matches with a cosine), so this module is the missing piece
that produces them.

Model: DINOv2 ViT-S/14, 384-d CLS token, L2 normalised. The weights are Meta's
(Apache-2.0, code and weights), converted to ONNX by the `onnx-community`
conversion on Hugging Face and pinned by SHA-256 in ``cli.py``. The fp16
conversion is used because it is half the size and measured within 0.0001
cosine of the fp32 graph on the same frames.

Everything is optional: without ``onnxruntime`` installed or the model file
present, ``available`` is False, ``embed`` returns None, and grouping is exactly
what it was before scene embeddings existed.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

# Pinned by cli.fetch-model. Keep the three in step with the shipped bundle.
MODEL_FILENAME = "dinov2_vits14_fp16.onnx"
MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_fp16.onnx"
)
# Pinned: the exact graph the duplicate threshold was calibrated against. Both
# the CLI and the app verify it, so a changed upstream export cannot silently
# alter grouping.
MODEL_SHA256 = "16845e153bbaf3fd1ef8a2154c454940f901f6fe8a80dd4c6c319eaecdb4d2ee"
EMBEDDING_DIM = 384
IMAGE_SIZE = 224

# Calibrated on this Mac, not inherited: a genuine near-duplicate pair (the same
# frame exported twice) scored 0.908, while the nearest unrelated pair in the
# same 12-frame sample scored 0.63. photo-pipeline's 0.92 would have missed the
# real pair, so the default sits at 0.88 with that gap either side.
SIMILARITY_THRESHOLD = 0.88

# ImageNet statistics, as DINOv2's own preprocessing uses.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def default_model_path() -> Path:
    """Where the downloaded model lives.

    Deliberately the writable data directory, not the app bundle: the weights
    are a user-initiated download, and a frozen app must never write inside its
    own (ad-hoc signed) bundle. Same folder rule as the catalog.
    """
    from ..db import default_data_dir

    return default_data_dir() / "models" / MODEL_FILENAME


@lru_cache(maxsize=1)
def runtime_available() -> bool:
    """True when the bundled scene runtime is installed and can load.

    A spec check is not enough for a frozen app: a bundle that collected the
    Python package but not ``libonnxruntime`` would pass it and then fail at the
    first embedding. Importing the module is the real test, and it is cached
    because the import costs a fraction of a second once per process.
    """
    try:
        import onnxruntime  # noqa: F401 - the import is the check
    except Exception:  # noqa: BLE001 - ImportError, or OSError from the dylib
        return False
    return True


def model_present(model_path: Path | str | None = None) -> bool:
    return (Path(model_path) if model_path else default_model_path()).is_file()


def model_verified(model_path: Path | str | None = None) -> bool:
    """Hash-check the model. Only for explicit download actions: 42 MB per call."""
    path = Path(model_path) if model_path else default_model_path()
    return path.is_file() and _digest(path) == MODEL_SHA256


def semantic_available(model_path: Path | str | None = None) -> bool:
    """True when the runtime and a model file are both present.

    Cheap enough for `/api/capabilities`: a filesystem stat and an import-system
    check, never a 42 MB hash and never an ONNX session. A corrupt file is
    caught when the session is built (`SceneEmbedder.available`).
    """
    return runtime_available() and model_present(model_path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    target: Path | str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch the pinned scene model, verifying it before it is installed.

    The download lands beside the target as `.download` and is renamed only
    after the SHA-256 matches, so a truncated or tampered transfer can never be
    used. ``progress(done, total)`` is called per megabyte; ``total`` is 0 when
    the server sends no Content-Length. Raises on any failure.
    """
    path = Path(target) if target else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _digest(path) == MODEL_SHA256:
            return path
        path.unlink()                 # present but not the pinned graph

    tmp = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, \
                tmp.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        digest = _digest(tmp)
        if digest != MODEL_SHA256:
            raise ValueError(f"scene model checksum mismatch: {digest}")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def preprocess(bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray | None:
    """BGR frame -> the 1x3xSxS ImageNet-normalised tensor DINOv2 expects.

    Shortest side scaled to ``size``, then centre crop: the standard ViT eval
    transform, and the same region the subject-sharpness axis measures.
    """
    import cv2

    if bgr is None or bgr.ndim != 3:
        return None
    height, width = bgr.shape[:2]
    if height < 8 or width < 8:
        return None
    scale = size / float(min(height, width))
    resized = cv2.resize(
        bgr,
        (max(size, round(width * scale)), max(size, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    top = (resized.shape[0] - size) // 2
    left = (resized.shape[1] - size) // 2
    crop = resized[top:top + size, left:left + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb.transpose(2, 0, 1) - _MEAN) / _STD)[None]


class SceneEmbedder:
    """Lazily-loaded ONNX session producing one unit vector per photo.

    One session is shared across analysis threads; ``onnxruntime`` sessions are
    thread-safe for ``run``. The session is built with a single intra-op thread
    on purpose: ``pipeline`` already analyses eight photos in flight, so a
    pool per call would oversubscribe the CPU. Measured at 224 px on this Mac:
    41 ms per image on one thread, 31 ms when a single call was allowed eight
    (a number to distrust once eight calls run at once); end to end the scan
    costs about 40 ms more per photo.
    """

    def __init__(self, model_path: Path | str | None = None, threads: int = 1):
        self.model_path = Path(model_path) if model_path else default_model_path()
        self.threads = max(1, int(threads))
        self._session = None
        self._broken = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._session is not None or self._broken:
            return self._session
        with self._lock:
            if self._session is not None or self._broken:
                return self._session
            if not self.model_path.is_file():
                self._broken = True
                return None
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = self.threads
                options.inter_op_num_threads = 1
                options.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self.model_path), options, providers=["CPUExecutionProvider"]
                )
            except Exception:                 # missing runtime, bad file, ...
                self._session = None
                self._broken = True
            return self._session

    def embed(self, bgr: np.ndarray | None) -> tuple[float, ...] | None:
        """Unit-length scene vector, or None when the embedder cannot run."""
        session = self._load()
        if session is None:
            return None
        tensor = preprocess(bgr)
        if tensor is None:
            return None
        try:
            raw = session.run(None, {"pixel_values": tensor})[0][0, 0]
        except Exception:
            return None
        vector = np.asarray(raw, dtype=np.float32)
        if vector.size != EMBEDDING_DIM:
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-9:
            return None
        return tuple(float(value) for value in vector / norm)


# --------------------------------------------------------------------------- #
# Storage: fp16 BLOBs, 768 bytes per photo (5,745 photos ~ 4.4 MB)
# --------------------------------------------------------------------------- #

def embedding_bytes(vector) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float16).tobytes()


def embedding_from_bytes(blob: bytes | None) -> tuple[float, ...] | None:
    if not blob:
        return None
    try:
        vector = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    except (TypeError, ValueError):
        return None
    if vector.size != EMBEDDING_DIM or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return tuple(float(value) for value in vector / norm)
