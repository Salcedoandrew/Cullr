"""Face, eye-state and expression analysis via MediaPipe Face Landmarker.

Why MediaPipe and not InsightFace: the InsightFace `buffalo_l` weights are
non-commercial, and both reference projects that use them ship a licensing gate
for exactly that reason. MediaPipe's Face Landmarker bundle is Apache-2.0 and
provides the 478-point mesh *and* the ARKit-style blendshapes that facet found
more reliable than landmark geometry for closed eyes and subtle smiles
(`analyzers/face_blendshapes.py` there).

The module degrades silently: if ``mediapipe`` or the model bundle is missing,
``FaceAnalyzer.available`` is False and the pipeline simply scores without face
axes.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import resource_path
from .faceid import MIN_ID_PIXELS, describe_face

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Blendshape names (ARKit order used by the MediaPipe bundle).
BLINK_L = "eyeBlinkLeft"
BLINK_R = "eyeBlinkRight"
SMILE_L = "mouthSmileLeft"
SMILE_R = "mouthSmileRight"
FROWN_L = "mouthFrownLeft"
FROWN_R = "mouthFrownRight"

MIN_FACE_PIXELS = 24        # ignore detections smaller than this (no usable detail)
FACE_CROP_SIZE = 192        # normalised crop size for per-face sharpness
SMALL_FACE_AREA = 0.02      # fraction of frame below which a face is "background"

# Eyes-open scores live on the same 0..10 scale as facet's
# `FaceAnalyzer.compute_eyes_open_score`, so blink gates are interchangeable.
SCORE_MAX = 10.0
NEUTRAL = 5.0


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    eyes_open: float | None = None      # 0..10
    smile: float | None = None          # 0..10, 5 = neutral
    yaw: float | None = None
    pitch: float | None = None
    sharpness: float = 0.0              # Laplacian variance of the normalised crop
    area_ratio: float = 0.0
    # Exposure of the subject, measured on the face crop. Frame-wide clipping
    # cannot tell a blown face from a blown white studio backdrop, so the
    # subject region is what the exposure gate uses when a face is present.
    clip_hi: float | None = None
    clip_lo: float | None = None
    mean_luma: float | None = None
    # Identity descriptor (see metrics/faceid.py). Deliberately excluded from
    # ``to_dict``: it is written to the catalog, never sent to the browser.
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def is_small(self) -> bool:
        return self.area_ratio < SMALL_FACE_AREA

    def to_dict(self) -> dict:
        return {
            "x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h),
            "eyes_open": None if self.eyes_open is None else round(self.eyes_open, 2),
            "smile": None if self.smile is None else round(self.smile, 2),
            "yaw": None if self.yaw is None else round(self.yaw, 1),
            "pitch": None if self.pitch is None else round(self.pitch, 1),
            "sharpness": round(self.sharpness, 2),
            "area_ratio": round(self.area_ratio, 4),
            "clip_hi": None if self.clip_hi is None else round(self.clip_hi, 4),
            "clip_lo": None if self.clip_lo is None else round(self.clip_lo, 4),
            "mean_luma": None if self.mean_luma is None else round(self.mean_luma, 4),
        }


@dataclass
class FaceMetrics:
    faces: list[Face] = field(default_factory=list)
    available: bool = True

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def usable(self) -> list[Face]:
        return [f for f in self.faces if not f.is_small]

    @property
    def primary(self) -> Face | None:
        """Largest usable face — the subject for blink purposes."""
        usable = self.usable or self.faces
        return max(usable, key=lambda f: f.area) if usable else None

    @property
    def eyes_open(self) -> float | None:
        primary = self.primary
        return primary.eyes_open if primary else None

    @property
    def worst_eyes_open(self) -> float | None:
        values = [f.eyes_open for f in self.usable if f.eyes_open is not None]
        return min(values) if values else None

    @property
    def smile(self) -> float | None:
        primary = self.primary
        return primary.smile if primary else None

    @property
    def face_sharpness(self) -> float | None:
        values = [f.sharpness for f in self.usable] or [f.sharpness for f in self.faces]
        return max(values) if values else None

    @property
    def frontal(self) -> float | None:
        """1.0 for a frontal primary face, falling off with |yaw| and |pitch|."""
        primary = self.primary
        if primary is None or primary.yaw is None:
            return None
        angle = max(abs(primary.yaw), abs(primary.pitch or 0.0))
        return float(max(0.0, 1.0 - (angle / 75.0)))

    @property
    def area_ratio(self) -> float:
        return float(min(1.0, sum(f.area_ratio for f in self.usable)))

    @property
    def subject_exposure(self) -> tuple[float, float, float] | None:
        """(clip_hi, clip_lo, mean_luma) of the primary subject face."""
        primary = self.primary
        if primary is None or primary.mean_luma is None:
            return None
        if primary.area_ratio < SMALL_FACE_AREA:
            return None
        return (primary.clip_hi or 0.0, primary.clip_lo or 0.0, primary.mean_luma)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "count": self.count,
            "faces": [f.to_dict() for f in self.faces],
        }


def blendshapes_to_scores(name_to_score: dict[str, float]) -> tuple[float, float]:
    """Map blendshape activations to ``(eyes_open, smile)`` on the 0..10 scale.

    eyes_open uses the *stronger* blink because a wink or one closed eye still
    ruins a portrait: ``(1 - max(blinkL, blinkR)) * 10``. A blink activation of
    0.6 lands at 4.0, which is the default closed-eye cutoff.
    smile is neutral-5 anchored: smile blendshapes have no negative half, so the
    frown blendshapes supply it (the same construction as facet).
    """
    blink = max(name_to_score.get(BLINK_L, 0.0), name_to_score.get(BLINK_R, 0.0))
    eyes_open = (1.0 - blink) * SCORE_MAX
    smile = (name_to_score.get(SMILE_L, 0.0) + name_to_score.get(SMILE_R, 0.0)) / 2.0
    frown = (name_to_score.get(FROWN_L, 0.0) + name_to_score.get(FROWN_R, 0.0)) / 2.0
    smile_score = NEUTRAL + (smile - frown) * NEUTRAL
    return (
        float(max(0.0, min(SCORE_MAX, eyes_open))),
        float(max(0.0, min(SCORE_MAX, smile_score))),
    )


def pose_from_matrix(matrix) -> tuple[float, float, float] | None:
    """(yaw, pitch, roll) in degrees from MediaPipe's 4x4 transform."""
    try:
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 3:
            return None
        rot = m[:3, :3]
        norms = np.linalg.norm(rot, axis=0)
        if not np.all(np.isfinite(norms)) or np.any(norms < 1e-9):
            return None
        rot = rot / norms
    except (TypeError, ValueError):
        return None
    yaw = math.degrees(math.atan2(rot[0, 2], rot[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, rot[1, 2]))))
    roll = math.degrees(math.atan2(rot[1, 0], rot[1, 1]))
    return yaw, pitch, roll


class FaceAnalyzer:
    """Thread-safe (one landmarker per thread) face analyser."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else self.default_model_path()
        self.available = False
        self.load_error: str | None = None
        self._local = threading.local()
        self._probe()

    @staticmethod
    def default_model_path() -> Path:
        # models/ ships inside the app bundle, or sits at the repository root
        # next to backend/ when running from source.
        return resource_path("models", MODEL_FILENAME)

    def _probe(self) -> None:
        try:
            import mediapipe  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            # Name the module: inside a packaged app this is the difference
            # between "broken build" and knowing what to add to the bundle.
            detail = getattr(exc, "name", None) or exc.__class__.__name__
            self.load_error = f"mediapipe unavailable ({detail}: {exc})"
            return
        if not self.model_path.exists():
            self.load_error = (
                f"model bundle missing at {self.model_path}; download it with "
                "`python -m culler.cli fetch-model`"
            )
            return
        landmarker = self._build()
        if landmarker is None:
            return
        self._local.landmarker = landmarker
        self.available = True

    def _build(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=10,
                min_face_detection_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            return vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"landmarker failed to load: {exc}"
            return None

    def _landmarker(self):
        landmarker = getattr(self._local, "landmarker", None)
        if landmarker is None:
            landmarker = self._build()
            self._local.landmarker = landmarker
        return landmarker

    def analyze(self, bgr: np.ndarray) -> FaceMetrics:
        if not self.available:
            return FaceMetrics(faces=[], available=False)
        landmarker = self._landmarker()
        if landmarker is None:
            return FaceMetrics(faces=[], available=False)
        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect(image)
        except Exception:  # noqa: BLE001 - never fail a whole scan on one frame
            return FaceMetrics(faces=[], available=True)

        h, w = bgr.shape[:2]
        frame_area = float(h * w) or 1.0
        faces: list[Face] = []
        meshes = result.face_landmarks or []
        shapes = result.face_blendshapes or []
        matrices = result.facial_transformation_matrixes or []

        for idx, mesh in enumerate(meshes):
            xs = [p.x for p in mesh]
            ys = [p.y for p in mesh]
            x0 = max(0, int(min(xs) * w))
            y0 = max(0, int(min(ys) * h))
            x1 = min(w, int(max(xs) * w))
            y1 = min(h, int(max(ys) * h))
            bw, bh = max(0, x1 - x0), max(0, y1 - y0)
            if bw < MIN_FACE_PIXELS or bh < MIN_FACE_PIXELS:
                continue

            face = Face(x=x0, y=y0, w=bw, h=bh)

            if idx < len(shapes) and shapes[idx]:
                scores = {c.category_name: float(c.score) for c in shapes[idx]}
                face.eyes_open, face.smile = blendshapes_to_scores(scores)
            if idx < len(matrices):
                pose = pose_from_matrix(matrices[idx])
                if pose:
                    face.yaw, face.pitch, _roll = pose

            # Expand the crop by 40 % so the chin and eye line are included.
            pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
            cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
            cx1, cy1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
            crop = bgr[cy0:cy1, cx0:cx1]
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (FACE_CROP_SIZE, FACE_CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)
                face.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Exposure is measured on the inner 60 % of the *unpadded* box: skin,
            # eyes and mouth. The padded crop would let a white backdrop or the
            # highlights in hair decide whether the subject is blown.
            ix = int(bw * 0.2)
            iy = int(bh * 0.2)
            inner = bgr[y0 + iy:y1 - iy, x0 + ix:x1 - ix]
            if inner.size:
                inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
                inner_area = float(inner_gray.size) or 1.0
                face.clip_hi = float((inner_gray >= 250).sum()) / inner_area
                face.clip_lo = float((inner_gray <= 5).sum()) / inner_area
                face.mean_luma = float(inner_gray.mean()) / 255.0
            face.area_ratio = (bw * bh) / frame_area

            # Identity descriptor, for clustering people across the shoot. Tiny
            # detections are skipped: a 30 px face upscaled to the 112 px
            # template carries no identity detail, only noise.
            if min(bw, bh) >= MIN_ID_PIXELS:
                try:
                    face.embedding = describe_face(bgr, mesh, w, h)
                except Exception:  # noqa: BLE001 - a descriptor must never fail a scan
                    face.embedding = None
            faces.append(face)

        faces.sort(key=lambda f: f.area, reverse=True)
        return FaceMetrics(faces=faces, available=True)
