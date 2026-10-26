"""Face identity descriptors: landmark alignment plus a compact embedding.

Why not a learned face-recognition network: the permissive option (OpenCV Zoo's
SFace) is another model download and a heavier dependency, and the accurate ones
(InsightFace ``buffalo_l``, which both reference projects ship) are non-commercial
weights that this project deliberately avoids. The embedding here is built from
what MediaPipe already returns, needs no download and no training, and needs to
answer one modest question well: *which of these faces are the same person?*
within a single shoot.

Two classical ingredients do that:

* **Alignment.** A similarity transform fitted to the five-point ArcFace
  template normalises scale, rotation and eye position. Without it two photos of
  one person differ mostly by framing and the descriptor is useless.
* **Structure + texture.** An 8x8 DCT block of the aligned crop captures coarse
  structure (eye line, nose, mouth, hairline); a 3x3 grid of gradient-orientation
  histograms captures edge texture (the DCT/eigenface and LBPH families
  respectively). Cosine similarity on the concatenation is the identity signal.

The descriptor is deterministic, small (117 floats) and cheap.
"""

from __future__ import annotations

import cv2
import numpy as np

ALIGNED_SIZE = 112          # ArcFace's canonical crop size
MIN_ID_PIXELS = 40          # smaller detections cannot be described reliably
DCT_CELLS = 8               # 8x8 low-frequency block, DC removed -> 63 values
HOG_GRID = 3                # 3x3 spatial cells
HOG_BINS = 6                # orientation bins over 0..pi
W_DCT = 1.0
W_HOG = 1.4

DESCRIPTOR_DIM = (DCT_CELLS * DCT_CELLS - 1) + HOG_GRID * HOG_GRID * HOG_BINS

# Five-point template (left eye, right eye, nose, left mouth, right mouth) on a
# 112x112 crop, as used by ArcFace and every aligner built on it.
_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# MediaPipe canonical face-mesh indices. Eye centres are the mean of the two
# corners; the rest are single vertices.
_LEFT_EYE = (33, 133)
_RIGHT_EYE = (362, 263)
_NOSE_TIP = 1
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


def face_landmarks_to_points(mesh, width: int, height: int) -> np.ndarray | None:
    """Five alignment points in pixel coordinates from a MediaPipe mesh."""
    try:
        if mesh is None or len(mesh) < 468:
            return None

        def px(index: int) -> tuple[float, float]:
            point = mesh[index]
            return float(point.x) * width, float(point.y) * height

        left = px(_LEFT_EYE[0]), px(_LEFT_EYE[1])
        right = px(_RIGHT_EYE[0]), px(_RIGHT_EYE[1])
        points = np.array(
            [
                ((left[0][0] + left[1][0]) / 2.0, (left[0][1] + left[1][1]) / 2.0),
                ((right[0][0] + right[1][0]) / 2.0, (right[0][1] + right[1][1]) / 2.0),
                px(_NOSE_TIP),
                px(_MOUTH_LEFT),
                px(_MOUTH_RIGHT),
            ],
            dtype=np.float32,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(points)):
        return None
    # Guard against degenerate geometry (mesh collapsed by an occlusion).
    if float(np.ptp(points[:, 0])) < 4.0 or float(np.ptp(points[:, 1])) < 4.0:
        return None
    return points


def align_face(bgr: np.ndarray, points: np.ndarray,
               size: int = ALIGNED_SIZE) -> np.ndarray | None:
    """Warp a face onto the canonical template. Returns ``None`` on degeneracy."""
    if bgr is None or bgr.size == 0 or points is None or len(points) != 5:
        return None
    target = _TEMPLATE * (size / float(ALIGNED_SIZE))
    matrix, _inliers = cv2.estimateAffinePartial2D(
        points, target, method=cv2.LMEDS,
    )
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None
    # Reject absurd transforms instead of describing a smeared crop.
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not 0.2 <= scale <= 20.0:
        return None
    # warpAffine maps destination -> source through the inverse of the matrix,
    # so the points->template matrix places the face on the template directly.
    return cv2.warpAffine(bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def describe_aligned(aligned: np.ndarray) -> np.ndarray:
    """L2-normalised identity descriptor of an aligned BGR face crop.

    Cosine similarity between two descriptors is their dot product.
    """
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    # CLAHE first: it equalises local contrast, so a shaded side of the face in
    # one frame and a flat-lit one in the next still produce a comparable
    # structure block. The z-score afterwards removes the residual DC.
    local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(small)
    field = local.astype(np.float32) / 255.0
    field -= float(field.mean())
    std = float(field.std())
    if std > 1e-6:
        field /= std

    dct = cv2.dct(field)[:DCT_CELLS, :DCT_CELLS].flatten()[1:]
    dct /= max(float(np.linalg.norm(dct)), 1e-9)
    dct *= W_DCT

    gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bin_index = np.minimum((angle / np.pi * HOG_BINS).astype(np.int32), HOG_BINS - 1)

    side = field.shape[0] // HOG_GRID
    histograms: list[np.ndarray] = []
    for row in range(HOG_GRID):
        for col in range(HOG_GRID):
            sl = (slice(row * side, (row + 1) * side),
                  slice(col * side, (col + 1) * side))
            hist = np.bincount(bin_index[sl].ravel(), weights=magnitude[sl].ravel(),
                               minlength=HOG_BINS).astype(np.float32)
            hist /= max(float(np.linalg.norm(hist)), 1e-9)
            histograms.append(hist)
    hog = np.concatenate(histograms) * W_HOG

    vector = np.concatenate([dct, hog]).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    return vector / norm


def describe_face(bgr: np.ndarray, mesh, width: int, height: int) -> np.ndarray | None:
    """Alignment + descriptor in one call; ``None`` when the face is unusable."""
    points = face_landmarks_to_points(mesh, width, height)
    if points is None:
        return None
    aligned = align_face(bgr, points)
    if aligned is None:
        return None
    vector = describe_aligned(aligned)
    return vector if float(np.linalg.norm(vector)) > 0 else None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two (already normalised) descriptors, clamped to [-1, 1]."""
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Storage helpers: float32 BLOBs, ~470 bytes per face
# --------------------------------------------------------------------------- #

def embedding_bytes(vector: np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def embedding_from_bytes(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    try:
        vector = np.frombuffer(blob, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.size != DESCRIPTOR_DIM or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None
