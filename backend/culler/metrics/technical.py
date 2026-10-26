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
* Noise uses Immerkaer's fast estimator on a centre crop, so vignetting and
  border artefacts do not dominate.
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
        }


def laplacian_variance(gray: np.ndarray, denoise_sigma: float = 0.8) -> float:
    """Edge-energy focus measure. Higher = sharper."""
    if denoise_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), denoise_sigma)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


def analyse(bgr: np.ndarray) -> TechnicalMetrics:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    total = float(gray.size) or 1.0
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64) / total
    clip_hi = float(hist[250:].sum())
    clip_lo = float(hist[:6].sum())
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
    )
