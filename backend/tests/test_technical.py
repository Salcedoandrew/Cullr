from __future__ import annotations

import numpy as np

from culler.metrics.technical import analyse, immerkaer_noise, laplacian_variance
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


def test_metrics_are_finite_on_flat_image():
    flat = np.full((400, 600, 3), 120, dtype=np.uint8)
    metrics = analyse(flat)
    assert metrics.sharpness == 0.0
    assert metrics.contrast == 0.0
    assert np.isfinite(metrics.noise)
