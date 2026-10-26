from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from culler.metrics.faces import (
    Face, FaceAnalyzer, FaceMetrics, blendshapes_to_scores, pose_from_matrix,
)

FIXTURES = Path(__file__).resolve().parents[2] / ".research" / "fixtures"


def test_eyes_open_maps_blink_activation():
    open_eyes = blendshapes_to_scores({"eyeBlinkLeft": 0.02, "eyeBlinkRight": 0.05})
    blinking = blendshapes_to_scores({"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95})
    one_eye = blendshapes_to_scores({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.05})

    assert open_eyes[0] > 9.0
    assert blinking[0] < 1.0
    # A wink is still a spoiled portrait: the stronger blink decides.
    assert one_eye[0] == pytest.approx(1.0, abs=0.01)


def test_smile_is_neutral_at_zero_and_positive_when_smiling():
    neutral = blendshapes_to_scores({})[1]
    smiling = blendshapes_to_scores({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8})[1]
    frowning = blendshapes_to_scores({"mouthFrownLeft": 0.8, "mouthFrownRight": 0.8})[1]
    assert neutral == pytest.approx(5.0)
    assert smiling > 8.0
    assert frowning < 3.0


def test_pose_from_identity_is_frontal():
    identity = np.eye(4)
    yaw, pitch, roll = pose_from_matrix(identity)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) < 1e-6


def test_pose_rejects_degenerate_matrix():
    assert pose_from_matrix(np.zeros((4, 4))) is None
    assert pose_from_matrix(None) is None


def test_worst_eyes_open_ignores_tiny_background_faces():
    metrics = FaceMetrics(faces=[
        Face(0, 0, 400, 400, eyes_open=9.0, area_ratio=0.06, sharpness=400),
        Face(0, 0, 40, 40, eyes_open=1.0, area_ratio=0.001, sharpness=50),
    ])
    assert metrics.eyes_open == pytest.approx(9.0)
    assert metrics.worst_eyes_open == pytest.approx(9.0)
    assert metrics.primary.area_ratio == 0.06


def test_worst_eyes_open_counts_all_subject_faces():
    metrics = FaceMetrics(faces=[
        Face(0, 0, 400, 400, eyes_open=9.0, area_ratio=0.06, sharpness=400),
        Face(0, 0, 300, 300, eyes_open=2.0, area_ratio=0.04, sharpness=300),
    ])
    assert metrics.worst_eyes_open == pytest.approx(2.0)
    assert metrics.eyes_open == pytest.approx(9.0)  # primary = largest


@pytest.mark.skipif(not (FIXTURES / "portrait_open_eyes.jpg").exists(),
                    reason="Wikimedia fixtures not downloaded")
def test_real_portrait_eyes_are_open():
    from culler.imaging import load_image

    analyzer = FaceAnalyzer()
    if not analyzer.available:
        pytest.skip(f"face model unavailable: {analyzer.load_error}")
    bgr, _meta, _err = load_image(FIXTURES / "portrait_open_eyes.jpg")
    metrics = analyzer.analyze(bgr)
    assert metrics.count >= 1
    assert metrics.eyes_open > 6.0
    assert metrics.face_sharpness > 100


@pytest.mark.skipif(not (FIXTURES / "portrait_closed_eyes.jpg").exists(),
                    reason="Wikimedia fixtures not downloaded")
def test_real_closed_eyes_are_detected():
    from culler.imaging import load_image

    analyzer = FaceAnalyzer()
    if not analyzer.available:
        pytest.skip(f"face model unavailable: {analyzer.load_error}")
    bgr, _meta, _err = load_image(FIXTURES / "portrait_closed_eyes.jpg")
    metrics = analyzer.analyze(bgr)
    assert metrics.count >= 1
    assert metrics.eyes_open < 4.0, "closed eyes must fall under the 4.0/10 gate"
