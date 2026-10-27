"""Per-photo scoring and explainable accept/reject decisions.

The composite is a weighted mean over *populated* axes: an axis that does not
apply (no face -> no eyes) is dropped from numerator and denominator, so a
landscape is never punished for having no eyes. This is the pattern from
photo-pipeline's `axis_composite_score`.

Every rejection carries a machine reason that the UI can show. Unexplained
rejections are what make photographers distrust auto-culling, so the reason is
part of the data model, not a log line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import AXES, CullingSettings, Gates
from .metrics.faces import FaceMetrics
from .metrics.technical import TechnicalMetrics

# Normalisation windows, calibrated on 1600 px analysis frames. Log mapping
# because Laplacian variance is heavy-tailed (a busy cityscape can hit 10x a
# clean portrait at the same acuity).
SHARPNESS_LO = 8.0
SHARPNESS_HI = 400.0
FACE_SHARPNESS_LO = 25.0
FACE_SHARPNESS_HI = 450.0
# Centre-crop variance needs its own window: the crop sits on the subject, so it
# measures higher than the frame-wide value at the same acuity. Measured on 40
# frames of a real 72-frame wedding set at 1600 px: 3.7 / 34.3 / 155.5
# (min / median / max), and 1.5-7.3 after a 2 px Gaussian blur.
SUBJECT_SHARPNESS_LO = 6.0
SUBJECT_SHARPNESS_HI = 200.0

# A face smaller than this is background crowd, not a subject: its crop has to
# be upscaled to be measured, so its sharpness is not trustworthy.
MIN_FACE_AREA_FOR_FOCUS = 0.02

REASON_PRIORITY = ("unreadable", "blurry", "overexposed", "underexposed", "closed_eyes")


@dataclass
class PhotoScore:
    axes: dict[str, float | None] = field(default_factory=dict)
    quality: float = 0.0
    verdict: str = "keep"
    reason: str | None = None
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "axes": {k: (None if v is None else round(v, 4)) for k, v in self.axes.items()},
            "quality": round(self.quality, 4),
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _log_norm(value: float, lo: float, hi: float) -> float:
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return float(math.log1p(value) - math.log1p(lo)) / (math.log1p(hi) - math.log1p(lo))


def sharpness_axis(lap: float) -> float:
    return _log_norm(lap, SHARPNESS_LO, SHARPNESS_HI)


def face_sharpness_axis(lap: float | None) -> float | None:
    if lap is None:
        return None
    return _log_norm(lap, FACE_SHARPNESS_LO, FACE_SHARPNESS_HI)


def exposure_inputs(tech: TechnicalMetrics,
                    faces: FaceMetrics | None = None) -> tuple[float, float, float]:
    """(clip_hi, clip_lo, mean_luma) used to judge exposure.

    When a subject face is present, clipping is taken from the subject region
    only: whole-frame clipping cannot distinguish a blown face from a
    deliberately white studio backdrop or a bright sky behind the subject.
    Mean luminance is blended 70/30 so the overall mood still counts a little.
    """
    subject = faces.subject_exposure if faces is not None else None
    if subject is None:
        return tech.clip_hi, tech.clip_lo, tech.mean_luma
    clip_hi, clip_lo, mean = subject
    return (
        clip_hi,
        clip_lo,
        0.3 * tech.mean_luma + 0.7 * mean,
    )


def exposure_axis(tech: TechnicalMetrics, faces: FaceMetrics | None = None) -> float:
    clip_hi, clip_lo, mean_luma = exposure_inputs(tech, faces)
    penalty = 0.0
    penalty += max(0.0, clip_hi - 0.01) * 6.0
    penalty += max(0.0, clip_lo - 0.05) * 2.0
    deviation = abs(mean_luma - 0.5)
    penalty += max(0.0, deviation - 0.22) * 1.4
    # A harsh two-peak histogram (blown sky over crushed shadows) can clip
    # nothing by the fractions above and still be unusable. facet charges 0.6
    # per unit of bimodality above 1.0 on its 0..10 axis; this axis is 0..1, so
    # the same term is 0.06. A smooth frame sits at or below 1.0 and pays
    # nothing. Ranker only: no rejection reason is added for it.
    penalty += max(0.0, tech.bimodality - 1.0) * 0.06
    return float(max(0.0, min(1.0, 1.0 - penalty)))


def noise_axis(tech: TechnicalMetrics) -> float:
    # Immerkaer's estimator reads ~3 sigma on clean JPEGs (compression artefacts
    # count as noise), climbs past 15 on genuinely noisy high-ISO frames.
    return float(max(0.0, min(1.0, 1.0 - (tech.noise - 3.0) / 15.0)))


def contrast_axis(tech: TechnicalMetrics) -> float:
    return float(max(0.0, min(1.0, tech.contrast / 62.0)))


def subject_sharpness_axis(tech: TechnicalMetrics, faces: FaceMetrics) -> float | None:
    """Subject focus for frames where no face can carry it.

    facet measures sharpness inside a segmentation mask; cullr uses a centre
    crop (``metrics/technical.py``, ported from facet's saliency metrics). When
    a usable face is present the face crop already measures the subject, so the
    axis is dropped from the composite instead of counting sharpness twice.
    Ranker only: a soft subject costs rank, never a rejection (no new reason).
    """
    if faces.usable or tech.subject_sharpness is None:
        return None
    return _log_norm(tech.subject_sharpness, SUBJECT_SHARPNESS_LO, SUBJECT_SHARPNESS_HI)


def color_axis(tech: TechnicalMetrics) -> float:
    """Palette variety, facet's 0..10 harmony mapped onto the 0..1 axes.

    Deliberately the smallest weight in the presets: a clean minimal portrait
    has little hue variety and must not be punished for it, so this only ever
    separates best-of-group near-ties. Never a rejection reason.
    """
    return float(max(0.0, min(1.0, tech.color_harmony / 10.0)))


def eyes_axis(faces: FaceMetrics) -> float | None:
    value = faces.worst_eyes_open
    if value is None:
        return None
    return float(max(0.0, min(1.0, value / 10.0)))


def subject_axis(faces: FaceMetrics) -> float | None:
    if not faces.usable:
        return None
    frontal = faces.frontal
    area = min(1.0, faces.area_ratio / 0.15)
    frontal = 0.6 if frontal is None else frontal
    return float(max(0.0, min(1.0, 0.6 * frontal + 0.4 * area)))


def build_axes(tech: TechnicalMetrics, faces: FaceMetrics) -> dict[str, float | None]:
    return {
        "sharpness": sharpness_axis(tech.sharpness),
        "subject_sharpness": subject_sharpness_axis(tech, faces),
        "exposure": exposure_axis(tech, faces),
        "noise": noise_axis(tech),
        "contrast": contrast_axis(tech),
        "color": color_axis(tech),
        "eyes_open": eyes_axis(faces),
        "face_sharpness": face_sharpness_axis(faces.face_sharpness),
        "subject": subject_axis(faces),
    }


def composite(axes: dict[str, float | None], weights: dict[str, float]) -> float:
    total = 0.0
    weighted = 0.0
    for name in AXES:
        value = axes.get(name)
        weight = float(weights.get(name, 0.0))
        if value is None or weight <= 0:
            continue
        total += weight
        weighted += weight * float(value)
    return float(weighted / total) if total > 0 else 0.0


def face_blink_is_reliable(faces: FaceMetrics, max_angle: float = 55.0) -> bool:
    """Appearance-based blink scores degrade on extreme profiles."""
    primary = faces.primary
    if primary is None or primary.yaw is None:
        return True
    return max(abs(primary.yaw), abs(primary.pitch or 0.0)) <= max_angle


def technical_reason(
    tech: TechnicalMetrics,
    faces: FaceMetrics,
    gates: Gates,
    settings: CullingSettings,
    group_median_sharpness: float | None = None,
) -> str | None:
    """Return the first rejection reason, or None when the frame is acceptable.

    Blur is judged three ways, because each alone fails somewhere:

    1. frame-wide floor -- catches mush, ignores content;
    2. face-crop gate -- an absolute, content-independent "are the eyes sharp"
       check, valid whenever a subject face is large enough;
    3. group-relative -- "soft for this burst", which is what actually decides
       between two otherwise-good frames, and is immune to the 6x swing that
       frame-wide Laplacian variance shows across scene types.

    Blur and highlight rejections are switchable (detect_blur /
    detect_highlights, the Customize toggles); underexposure has no switch,
    so a dark frame is always rejected. The toggles only gate rejection --
    the quality score still counts softness and exposure, so they still
    rank frames inside a group and under cull-to-target.
    """
    if settings.detect_blur:
        if tech.sharpness < gates.blur_abs:
            return "blurry"

        primary = faces.primary
        if (gates.face_blur > 0 and primary is not None
                and primary.area_ratio >= MIN_FACE_AREA_FOR_FOCUS
                and primary.sharpness > 0
                and primary.sharpness < gates.face_blur):
            return "blurry"

        if group_median_sharpness is not None and group_median_sharpness > 15.0:
            ratio = settings.relative_blur_ratio
            ratio = gates.blur_rel if ratio is None else ratio
            if tech.sharpness < group_median_sharpness * ratio:
                return "blurry"

    clip_hi, clip_lo, mean_luma = exposure_inputs(tech, faces)
    if settings.detect_highlights:
        if clip_hi > gates.clip_hi_max and mean_luma > 0.55:
            return "overexposed"
        if mean_luma > gates.mean_luma_max:
            return "overexposed"
    if mean_luma < gates.mean_luma_min:
        return "underexposed"
    if clip_lo > gates.clip_lo_max and mean_luma < 0.25:
        return "underexposed"

    if settings.detect_blinks and faces.usable and face_blink_is_reliable(faces):
        worst = faces.worst_eyes_open
        if worst is not None and worst <= gates.eyes_closed_max:
            return "closed_eyes"
    return None


def score_photo(
    tech: TechnicalMetrics,
    faces: FaceMetrics,
    settings: CullingSettings,
    group_median_sharpness: float | None = None,
    relative_to_group: bool = True,
) -> PhotoScore:
    axes = build_axes(tech, faces)
    quality = composite(axes, settings.weights())
    gates = settings.gates()
    reason = technical_reason(
        tech, faces, gates,
        settings,
        group_median_sharpness if relative_to_group else None,
    )
    return PhotoScore(
        axes=axes,
        quality=quality,
        verdict="reject" if reason else "keep",
        reason=reason,
        details={
            "sharpness_axis": axes["sharpness"] or 0.0,
            "exposure_axis": axes["exposure"] or 0.0,
        },
    )


def group_median_sharpness(values: list[float]) -> float | None:
    usable = sorted(v for v in values if v > 0)
    if len(usable) < 3:
        return None
    mid = len(usable) // 2
    if len(usable) % 2:
        return usable[mid]
    return (usable[mid - 1] + usable[mid]) / 2.0
