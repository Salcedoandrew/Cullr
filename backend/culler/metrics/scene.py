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

import importlib.util
import threading
from pathlib import Path

import numpy as np

# Pinned by cli.fetch-model. Keep the three in step with the shipped bundle.
MODEL_FILENAME = "dinov2_vits14_fp16.onnx"
MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_fp16.onnx"
)
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


def semantic_available(model_path: Path | str | None = None) -> bool:
    """True when the optional runtime and the model file are both present.

    Cheap enough for `/api/capabilities`: it checks the filesystem and the
    import system, but never builds an ONNX session.
    """
    path = Path(model_path) if model_path else default_model_path()
    if not path.is_file():
        return False
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except (ImportError, ValueError):
        return False


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
