"""Configuration: analysis sizes, shoot presets, scoring weights and gates.

The numbers here are calibrated for a fixed-size analysis frame (long edge
1600 px). Sharpness metrics are resolution dependent, so analysing a normalised
preview is what makes one threshold valid for a 12 MP and a 61 MP body.

Threshold provenance (see docs/RESEARCH.md):
  * Laplacian-variance blur screens: photo-pipeline (default 80, 30..250 range)
    and facet (min(10, var/50)).
  * Eyes-closed cutoff 4.0/10 and pose gate: facet.
  * Bounded per-group lead tie-breaks: facet `utils/selection.py`.
  * Per-genre profiles: facet `cull_profiles`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def bundle_root() -> Path:
    """Where read-only shipped files live.

    Inside a frozen app (PyInstaller) the resources are unpacked next to the
    executable, so ``__file__`` no longer points into the repository. Everything
    that ships with the app — the face model, the built frontend — resolves
    through here.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)

ANALYSIS_LONG_EDGE = 1600
THUMB_LONG_EDGE = 512
# 2: per-face identity descriptors (people clustering) are part of analysis.
ANALYSIS_VERSION = 2

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
    ".heic", ".heif", ".avif",
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef", ".srw",
}
RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef", ".srw",
}

# Axes that can contribute to the composite score. Missing axes are dropped
# from numerator AND denominator, so a landscape is never punished for having
# no eyes (the pattern used by photo-pipeline's axis_composite_score).
AXES = (
    "sharpness", "exposure", "noise", "contrast",
    "eyes_open", "face_sharpness", "subject",
)


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    weights: dict[str, float]
    strictness: int
    keep_min_per_group: int = 1
    eyes_closed_max: float = 4.0
    # Absolute focus gate applied to the *face crop* (192 px normalised).
    # Calibrated on studio portraits: a sharp face crop measures 450-550, a
    # 1 px Gaussian blur ~135, a 2 px blur ~35. Frame-wide Laplacian variance
    # is not used as an absolute gate because it swings 6x with content.
    face_blur_gate: float = 55.0


PRESETS: dict[str, Preset] = {
    "portrait": Preset(
        key="portrait",
        label="Portrait",
        description="Single subject, shallow depth of field. Eyes and face sharpness weighted heavily; soft backgrounds are not penalised.",
        weights={"sharpness": 0.26, "exposure": 0.18, "noise": 0.06,
                 "contrast": 0.06, "eyes_open": 0.22, "face_sharpness": 0.14,
                 "subject": 0.08},
        strictness=50,
        keep_min_per_group=1,
        eyes_closed_max=4.0,
        face_blur_gate=55.0,
    ),
    "wedding": Preset(
        key="wedding",
        label="Wedding / event",
        description="Many faces per frame, emotion matters. Lenient on softness because shallow depth of field is normal, strict on closed eyes.",
        weights={"sharpness": 0.24, "exposure": 0.16, "noise": 0.05,
                 "contrast": 0.05, "eyes_open": 0.26, "face_sharpness": 0.14,
                 "subject": 0.10},
        strictness=35,
        keep_min_per_group=1,
        eyes_closed_max=5.0,
        face_blur_gate=45.0,
    ),
    "event": Preset(
        key="event",
        label="Event / documentary",
        description="Mixed subjects and venues. Balanced gates; tolerates some motion as long as a frame in the group is sharp.",
        weights={"sharpness": 0.30, "exposure": 0.20, "noise": 0.08,
                 "contrast": 0.08, "eyes_open": 0.16, "face_sharpness": 0.10,
                 "subject": 0.08},
        strictness=50,
        keep_min_per_group=1,
        eyes_closed_max=4.0,
        face_blur_gate=55.0,
    ),
    "sports": Preset(
        key="sports",
        label="Sports / action",
        description="Fast bursts, motion blur is often intentional. Non-face blur allowed; the sharpest frame in each burst wins.",
        weights={"sharpness": 0.34, "exposure": 0.18, "noise": 0.05,
                 "contrast": 0.05, "eyes_open": 0.10, "face_sharpness": 0.12,
                 "subject": 0.16},
        strictness=85,
        keep_min_per_group=1,
        eyes_closed_max=2.0,
        face_blur_gate=85.0,
    ),
    "landscape": Preset(
        key="landscape",
        label="Landscape / travel",
        description="No faces: eyes axes are absent and therefore ignored. Sharpness and dynamic range dominate.",
        weights={"sharpness": 0.40, "exposure": 0.24, "noise": 0.12,
                 "contrast": 0.16, "subject": 0.08},
        strictness=60,
        keep_min_per_group=1,
        eyes_closed_max=0.0,
        face_blur_gate=0.0,
    ),
}

DEFAULT_PRESET = "wedding"


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class CullingSettings:
    """User-tunable settings for one library scan."""

    preset: str = DEFAULT_PRESET
    strictness: int | None = None          # 0..100, None = preset default
    keep_min_per_group: int | None = None
    detect_blinks: bool = True
    group_duplicates: bool = True
    # Similarity gates (pHash 64-bit Hamming distance).
    duplicate_distance: int = 6            # facet: 90% similarity => 6
    similar_distance: int = 14             # looser candidate gate within a burst
    burst_window_seconds: float = 2.0      # EXIF capture-time window
    similar_window_seconds: float = 8.0    # same-scene window
    min_group_size: int = 2
    # Embedding confirmation (only when an embedder is installed).
    embedding_threshold: float = 0.90
    # Relative blur gate: reject when a frame is this fraction of its group median.
    relative_blur_ratio: float | None = None
    # People: cluster faces across the shoot and guarantee every subject appears
    # in at least one keeper (Aftershoot-style "key face" coverage).
    person_coverage: bool = True
    # Cosine similarity for joining an existing person, and for merging two
    # clusters. Deliberately high: measured separation on real portraits is
    # intra-person 0.94-1.00 vs inter-person up to 0.88, and a false merge can
    # silently drop the coverage guarantee (one merged cluster counted as
    # covered by a frame that only contains one of the two people). Splitting a
    # person into two clusters only costs an extra keeper.
    person_match: float = 0.90
    person_merge: float = 0.95
    person_min_photos: int = 2          # frames before a person must be covered
    person_min_face_ratio: float = 0.015  # face size that counts as a subject
    # "Cull to target" (beta): keep roughly this fraction of the shoot, ranked by
    # measured quality. None means the gates decide (the "auto target" mode).
    # Coverage still wins, so the achieved count can sit slightly above it.
    target_ratio: float | None = None
    # Export: rewrite an existing <file>.xmp instead of leaving it alone.
    overwrite_xmp: bool = True
    # Similarity floor for *suggesting* a merge between two people. Proposals are
    # only ever confirmed by the photographer, so this can sit below the automatic
    # match threshold (0.90) and still be safe.
    person_propose: float = 0.84

    def resolved_preset(self) -> Preset:
        return PRESETS.get(self.preset, PRESETS[DEFAULT_PRESET])

    def effective_strictness(self) -> int:
        preset = self.resolved_preset()
        return preset.strictness if self.strictness is None else int(self.strictness)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CullingSettings":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def gates(self) -> "Gates":
        preset = self.resolved_preset()
        t = max(0, min(100, self.effective_strictness())) / 100.0
        eyes = (preset.eyes_closed_max if t == 0.5
                else lerp(preset.eyes_closed_max + 1.5, preset.eyes_closed_max - 1.5, t))
        return Gates(
            blur_abs=ABS_BLUR_FLOOR,
            blur_rel=lerp(0.30, 0.65, t),
            face_blur=lerp(preset.face_blur_gate * 0.65,
                           preset.face_blur_gate * 1.60, t),
            eyes_closed_max=max(0.0, min(10.0, eyes)),
            clip_hi_max=lerp(0.12, 0.03, t),
            clip_lo_max=lerp(0.35, 0.08, t),
            mean_luma_min=lerp(0.05, 0.10, t),
            mean_luma_max=lerp(0.95, 0.88, t),
            keep_min_per_group=(preset.keep_min_per_group
                                if self.keep_min_per_group is None
                                else max(0, self.keep_min_per_group)),
        )

    def weights(self) -> dict[str, float]:
        return dict(self.resolved_preset().weights)


# Frame-wide Laplacian variance below this is mush regardless of content; it is
# only a safety net. Real blur decisions are relative to the group, or absolute
# on the face crop (see Preset.face_blur_gate).
ABS_BLUR_FLOOR = 5.0


@dataclass(frozen=True)
class Gates:
    blur_abs: float
    blur_rel: float
    face_blur: float
    eyes_closed_max: float
    clip_hi_max: float
    clip_lo_max: float
    mean_luma_min: float
    mean_luma_max: float
    keep_min_per_group: int
