from __future__ import annotations

import cv2
import numpy as np

from culler.metrics.technical import (
    analyse, immerkaer_noise, laplacian_variance, subject_sharpness_score,
)
from conftest import blur, make_scene


def test_sharpness_orders_sharp_above_blurred(scene):
    sharp = analyse(scene)
    soft = analyse(blur(scene, 4.0))
    assert sharp.sharpness > soft.sharpness * 5
    assert sharp.tenengrad > soft.tenengrad


def test_laplacian_variance_is_content_dependent(scene):
    """Documents *why* the blur gate is group-relative: same acuity, new content."""
    values = [laplacian_variance(np.dstack([make_scene(seed)] * 3)) for seed in (1, 2, 3)]
    assert max(values) / min(values) < 6  # but also not identical
    assert len(set(round(v, 3) for v in values)) == 3


def test_exposure_detects_clipping():
    white = np.full((600, 800, 3), 252, dtype=np.uint8)
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    mid = np.full((600, 800, 3), 128, dtype=np.uint8)

    assert analyse(white).clip_hi > 0.9
    assert analyse(white).mean_luma > 0.9
    assert analyse(black).clip_lo > 0.9
    assert analyse(black).mean_luma < 0.05
    assert analyse(mid).clip_hi < 0.01
    assert analyse(mid).clip_lo < 0.01


def test_noise_estimator_grows_with_noise(scene):
    rng = np.random.default_rng(7)
    clean = immerkaer_noise(scene)
    noisy = np.clip(scene.astype(np.float32)
                    + rng.normal(0, 18, scene.shape).astype(np.float32), 0, 255).astype(np.uint8)
    assert immerkaer_noise(noisy) > clean * 2


def test_subject_sharpness_prefers_a_sharp_centre_over_a_soft_frame():
    """Centre-crop variance, the classical stand-in for facet's subject region.

    Calibration, on 40 frames of a real 72-frame wedding set at the 1600 px
    analysis size: centre-crop Laplacian variance 3.7 / 34.3 / 155.5
    (min / median / max), and 1.5-7.3 once the crop is blurred by 2 px. That
    spread is why the axis window is 6..200 rather than the frame-wide 8..400.
    """
    scene = make_scene(2)
    soft = blur(scene, 3.5)
    sharp_centre = soft.copy()
    h, w = scene.shape[:2]
    quarter_h, quarter_w = h // 4, w // 4
    sharp_centre[quarter_h:3 * quarter_h, quarter_w:3 * quarter_w] = \
        scene[quarter_h:3 * quarter_h, quarter_w:3 * quarter_w]

    subject = analyse(sharp_centre)
    uniform = analyse(soft)
    assert subject.subject_sharpness > uniform.subject_sharpness * 3


def test_bg_separation_is_the_centre_to_frame_ratio():
    metrics = analyse(make_scene(3))
    ratio = metrics.subject_sharpness / metrics.sharpness
    assert abs(metrics.bg_separation - min(10.0, ratio * 2.0)) < 1e-6


def test_subject_metrics_are_zero_on_a_flat_frame():
    flat = np.full((400, 600, 3), 120, dtype=np.uint8)
    metrics = analyse(flat)
    assert metrics.subject_sharpness == 0.0
    assert metrics.bg_separation == 0.0


def test_facet_subject_score_is_the_sqrt_over_seven_normalisation():
    # facet models/saliency_scorer.py: min(10, sqrt(subject variance) / 7).
    assert subject_sharpness_score(0.0) == 0.0
    assert subject_sharpness_score(49.0) == 1.0
    assert subject_sharpness_score(1225.0) == 5.0
    assert subject_sharpness_score(4900.0) == 10.0
    assert subject_sharpness_score(1e6) == 10.0


def test_bimodality_separates_two_peaks_from_one():
    """facet's ``-kurtosis(hist)``: ~0 unimodal, 2.0 for two equal peaks.

    Measured on the synthetic frames below: two equal masses at 30 and 220 read
    2.0, a broad gradient 30..220 reads 1.2, and a normal-ish frame ~0. facet's
    threshold of 1.0 sits between the first two.
    """
    two_peak = np.zeros((400, 600, 3), dtype=np.uint8)
    two_peak[:, :300] = 30
    two_peak[:, 300:] = 220
    ramp = np.tile(np.linspace(30, 220, 600, dtype=np.uint8), (400, 1))
    gradient = np.dstack([ramp] * 3)

    harsh = analyse(two_peak)
    smooth = analyse(gradient)
    assert harsh.bimodality > 1.5
    assert smooth.bimodality < 1.5
    assert harsh.bimodality > smooth.bimodality + 0.5
    # The point of the metric: neither frame clips by the existing definition.
    assert harsh.clip_hi == 0.0 and harsh.clip_lo == 0.0


def test_shadow_and_highlight_flags_are_not_the_clipping_fractions():
    shadowed = np.full((400, 600, 3), 128, dtype=np.uint8)
    shadowed[:, :240] = 10       # 40 % deep shadow, none of it at/below 5
    blown = np.full((400, 600, 3), 128, dtype=np.uint8)
    blown[:, :120] = 240         # 20 % bright, none of it at/above 250

    assert analyse(shadowed).shadow_clipped is True
    assert analyse(shadowed).highlight_clipped is False
    assert analyse(blown).highlight_clipped is True
    assert analyse(blown).shadow_clipped is False

    neutral = analyse(np.full((400, 600, 3), 128, dtype=np.uint8))
    assert neutral.shadow_clipped is False
    assert neutral.highlight_clipped is False


def test_color_harmony_orders_a_colourful_frame_above_a_grey_one():
    """facet's HSV-histogram entropy, scaled so the 15.49-bit maximum is 10.

    Measured: 40 real frames read 0.80-7.39 (median 6.75), a flat grey frame
    0.00, and a full hue ramp 4.83.
    """
    grey = np.full((400, 600, 3), 128, dtype=np.uint8)
    hsv = np.zeros((400, 600, 3), dtype=np.uint8)
    hsv[..., 0] = np.tile(np.linspace(0, 179, 600, dtype=np.uint8), (400, 1))
    hsv[..., 1] = 255
    hsv[..., 2] = 200
    colourful = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    assert analyse(grey).color_harmony == 0.0
    assert analyse(colourful).color_harmony > 4.0


def test_color_harmony_stays_on_the_facet_scale(scene):
    assert 0.0 <= analyse(scene).color_harmony <= 10.0


def test_metrics_are_finite_on_flat_image():
    flat = np.full((400, 600, 3), 120, dtype=np.uint8)
    metrics = analyse(flat)
    assert metrics.sharpness == 0.0
    assert metrics.contrast == 0.0
    assert np.isfinite(metrics.noise)
