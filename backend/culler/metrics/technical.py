"""Objective image quality metrics.

All metrics are computed on the normalised analysis frame (long edge 1600 px),
which is what makes the thresholds comparable across camera resolutions.

Design notes
------------
* Sharpness uses Laplacian variance *after a mild Gaussian pre-blur*. Raw
  Laplacian variance is inflated by sensor noise, which makes a high-ISO frame
  look sharper than it is; the blur removes pixel-level noise while leaving
  edges (the signal) intact. Tenengrad is computed alongside as a second
  opinion, because Laplacian variance drops on very low-contrast (fog, snow)
  frames that are actually in focus.
* Exposure is measured from the luma histogram: clipping fractions plus mean
  luminance. Silhouette / low-key frames are handled by the gate in
  ``scoring.py`` rather than here, since that decision needs the face boxes.
* A two-peak histogram (blown sky above crushed shadows) can clip nothing by
  those fractions and still be unusable, so the histogram's shape is measured
  too: ``bimodality`` is facet's ``-excess_kurtosis``, and ``shadow_clipped`` /
  ``highlight_clipped`` flag masses that the strict clip fractions miss.
  Ported from facet `analyzers/technical.py`; see NOTICE.md.
* Noise uses Immerkaer's fast estimator on a centre crop, so vignetting and
  border artefacts do not dominate.
* Subject sharpness stands in for facet's saliency mask without a 2.4 GB
  segmentation model: the middle 50 % of the frame is the subject region
  (`centre_crop`), and its Laplacian variance is reported raw (for the axis
  window in ``scoring.py``) and in facet's 0..10 normalisation. Ported from
  facet `models/saliency_scorer.py` (MIT): `subject_sharpness`,
  `bg_separation`. See NOTICE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Immerkaer's 3x3 noise kernel.
_NOISE_KERNEL = np.array(
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32
)


@dataclass
class TechnicalMetrics:
    sharpness: float          # Laplacian variance of the denoised luma frame
    tenengrad: float          # mean squared Sobel gradient magnitude
    clip_hi: float            # fraction of pixels at/above 250
    clip_lo: float            # fraction of pixels at/below 5
    mean_luma: float          # 0..1
    p95_luma: float
    noise: float              # sigma estimate on the 0..255 scale
    contrast: float           # luma standard deviation on the 0..255 scale
    # Centre-crop Laplacian variance; None when the frame is too small for a
    # crop. Rows read back from the catalog predate these columns, so both are
    # optional.
    subject_sharpness: float | None = None
    bg_separation: float | None = None   # min(10, centre/frame variance * 2)
    # facet's histogram shape: ~0 unimodal, ~2 for two equal peaks.
    bimodality: float = 0.0
    # Masses the strict clip fractions miss (see SHADOW_MASS / HIGHLIGHT_MASS).
    shadow_clipped: bool = False
    highlight_clipped: bool = False

    def to_dict(self) -> dict:
        return {
            "sharpness": round(self.sharpness, 3),
            "tenengrad": round(self.tenengrad, 3),
            "clip_hi": round(self.clip_hi, 5),
            "clip_lo": round(self.clip_lo, 5),
            "mean_luma": round(self.mean_luma, 4),
            "p95_luma": round(self.p95_luma, 4),
            "noise": round(self.noise, 3),
            "contrast": round(self.contrast, 3),
            "subject_sharpness": (None if self.subject_sharpness is None
                                 else round(self.subject_sharpness, 3)),
            "subject_sharpness_10": (None if self.subject_sharpness is None
                                     else round(subject_sharpness_score(self.subject_sharpness), 3)),
            "bg_separation": (None if self.bg_separation is None
                              else round(self.bg_separation, 3)),
            "bimodality": round(self.bimodality, 4),
            "shadow_clipped": self.shadow_clipped,
            "highlight_clipped": self.highlight_clipped,
        }


def laplacian_variance(gray: np.ndarray, denoise_sigma: float = 0.8) -> float:
    """Edge-energy focus measure. Higher = sharper."""
    if denoise_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), denoise_sigma)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# The subject region, as a fraction of each side. Wedding subjects are usually
# centred, and a blurred background cannot drag this measure down the way it
# drags the frame-wide one (docs/RESEARCH.md 4.4).
SUBJECT_CROP_RATIO = 0.5
SUBJECT_CROP_MIN_SIDE = 32


def centre_crop(gray: np.ndarray, ratio: float = SUBJECT_CROP_RATIO) -> np.ndarray | None:
    """The middle ``ratio`` of the frame, or None when it would be a sliver."""
    h, w = gray.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    if ch < SUBJECT_CROP_MIN_SIDE or cw < SUBJECT_CROP_MIN_SIDE:
        return None
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return gray[y0:y0 + ch, x0:x0 + cw]


def subject_sharpness_score(variance: float) -> float:
    """facet's 0..10 normalisation: ``min(10, sqrt(variance) / 7)``.

    Ported from facet `models/saliency_scorer.py` (MIT), which measures the
    variance inside a BiRefNet subject mask; here the region is a centre crop.
    """
    return float(min(10.0, np.sqrt(max(float(variance), 0.0)) / 7.0))


def subject_region_sharpness(gray: np.ndarray) -> tuple[float | None, float | None]:
    """(centre-crop variance, bg_separation) for the subject region.

    ``bg_separation`` is facet's ``min(10, subject/bg ratio * 2)`` collapsed to
    one ratio: a shallow-DoF portrait separates its subject 3-8x, a landscape
    sits near 1x. Computed against the whole frame rather than a true
    background mask, so treat it as a hint, not a measurement.
    """
    crop = centre_crop(gray)
    if crop is None:
        return None, None
    subject = laplacian_variance(crop)
    frame = laplacian_variance(gray)
    separation = 0.0 if frame <= 0 else min(10.0, (subject / frame) * 2.0)
    return subject, separation


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def immerkaer_noise(gray: np.ndarray, min_side: int = 128) -> float:
    """Estimate Gaussian noise sigma (0..255 scale)."""
    h, w = gray.shape[:2]
    if h < min_side or w < min_side:
        return 0.0
    # Centre crop: borders carry lens vignetting and JPEG ringing.
    ch, cw = min(h, max(min_side, h // 2)), min(w, max(min_side, w // 2))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = gray[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    conv = cv2.filter2D(crop, cv2.CV_32F, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(conv)) / 6.0)


# Masses facet treats as visibly crushed / blown: below 30 and above 225 of
# the 0..255 luma range, at 15 % / 10 % of the frame. Stricter than cullr's
# clip fractions (5 / 250) on purpose -- these are flags for the photographer,
# not rejection gates.
SHADOW_MASS = 0.15
HIGHLIGHT_MASS = 0.10
SHADOW_LEVEL = 30
HIGHLIGHT_LEVEL = 225


def histogram_bimodality(hist: np.ndarray) -> float:
    """Two-peak-ness of a luma histogram, as facet measures it.

    Ported from facet `analyzers/technical.py` (`get_histogram_data`):
    ``-excess_kurtosis(histogram)``. A smooth unimodal frame reads ~0 (negative
    for a heavy-tailed one) and two separated peaks read ~2, so 1.0 separates
    them. Uses the normalised histogram already in hand, so it costs no extra
    pass over the frame.
    """
    mass = float(hist.sum())
    if mass <= 0:
        return 0.0
    values = np.arange(hist.size, dtype=np.float64)
    p = hist / mass
    mean = float(np.dot(values, p))
    var = float(np.dot((values - mean) ** 2, p))
    if var <= 1e-9:
        return 0.0
    fourth = float(np.dot((values - mean) ** 4, p))
    return float(3.0 - fourth / (var * var))


def analyse(bgr: np.ndarray) -> TechnicalMetrics:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    total = float(gray.size) or 1.0
    subject, separation = subject_region_sharpness(gray)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64) / total
    clip_hi = float(hist[250:].sum())
    clip_lo = float(hist[:6].sum())
    shadow_mass = float(hist[:SHADOW_LEVEL].sum())
    highlight_mass = float(hist[HIGHLIGHT_LEVEL + 1:].sum())
    mean_luma = float(np.dot(np.arange(256), hist) / 255.0)
    cdf = np.cumsum(hist)
    p95 = float(np.searchsorted(cdf, 0.95)) / 255.0
    contrast = float(gray.std())
    return TechnicalMetrics(
        sharpness=laplacian_variance(gray),
        tenengrad=tenengrad(gray),
        clip_hi=clip_hi,
        clip_lo=clip_lo,
        mean_luma=mean_luma,
        p95_luma=p95,
        noise=immerkaer_noise(gray),
        contrast=contrast,
        subject_sharpness=subject,
        bg_separation=separation,
        bimodality=histogram_bimodality(hist),
        shadow_clipped=shadow_mass > SHADOW_MASS,
        highlight_clipped=highlight_mass > HIGHLIGHT_MASS,
    )
