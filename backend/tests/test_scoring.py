from __future__ import annotations

from culler.config import CullingSettings
from culler.metrics.faces import Face, FaceMetrics
from culler.metrics.technical import TechnicalMetrics
from culler.scoring import composite, exposure_axis, score_photo, technical_reason


def tech(sharp=120.0, clip_hi=0.0, clip_lo=0.02, mean=0.5, noise=4.0,
         contrast=60.0, subject=None, bimodality=0.0,
         color=0.0) -> TechnicalMetrics:
    return TechnicalMetrics(
        sharpness=sharp, tenengrad=sharp * 60, clip_hi=clip_hi, clip_lo=clip_lo,
        mean_luma=mean, p95_luma=0.9, noise=noise, contrast=contrast,
        subject_sharpness=subject, bimodality=bimodality, color_harmony=color,
    )


def face_metrics(eyes=None, sharpness=None, area=0.05, yaw=0.0,
                 clip_hi=None, clip_lo=None, mean_luma=None) -> FaceMetrics:
    if eyes is None and sharpness is None and mean_luma is None:
        return FaceMetrics(faces=[])
    return FaceMetrics(faces=[Face(
        x=100, y=100, w=400, h=400, eyes_open=eyes, smile=5.0, yaw=yaw, pitch=0.0,
        sharpness=sharpness or 400.0, area_ratio=area,
        clip_hi=clip_hi, clip_lo=clip_lo, mean_luma=mean_luma,
    )])


def gate(settings: CullingSettings):
    return settings.gates()


def test_missing_axes_are_dropped_not_zeroed():
    weights = {"sharpness": 0.5, "eyes_open": 0.5}
    assert composite({"sharpness": 0.6, "eyes_open": None}, weights) == 0.6
    assert composite({"sharpness": 0.6, "eyes_open": 0.2}, weights) == 0.4


def test_landscape_photo_is_not_penalised_for_having_no_faces():
    settings = CullingSettings(preset="landscape")
    result = score_photo(tech(sharp=300), face_metrics(), settings)
    assert result.axes["eyes_open"] is None
    assert result.quality > 0.5
    assert result.verdict == "keep"


def test_absolute_floor_rejects_mush():
    settings = CullingSettings()
    reason = technical_reason(tech(sharp=2.0), face_metrics(), gate(settings), settings)
    assert reason == "blurry"


def test_face_gate_rejects_soft_eyes():
    settings = CullingSettings(preset="portrait")
    soft_face = face_metrics(eyes=9.0, sharpness=25.0, area=0.05)
    reason = technical_reason(tech(sharp=120), soft_face, gate(settings), settings)
    assert reason == "blurry"


def test_face_gate_ignores_tiny_background_faces():
    settings = CullingSettings(preset="portrait")
    tiny = face_metrics(eyes=9.0, sharpness=10.0, area=0.005)
    reason = technical_reason(tech(sharp=120), tiny, gate(settings), settings)
    assert reason is None


def test_group_relative_gate_catches_the_soft_frame_of_a_burst():
    settings = CullingSettings(preset="wedding")
    gates = gate(settings)
    reason = technical_reason(tech(sharp=18), face_metrics(), gates, settings,
                              group_median_sharpness=120.0)
    assert reason == "blurry"
    # Same frame alone is fine: it is only "soft for this group".
    assert technical_reason(tech(sharp=18), face_metrics(), gates, settings) is None


def test_relative_gate_needs_a_meaningful_median():
    settings = CullingSettings()
    gates = gate(settings)
    assert technical_reason(tech(sharp=6.5), face_metrics(), gates, settings,
                            group_median_sharpness=8.0) is None


def test_overexposure_rejected():
    settings = CullingSettings()
    reason = technical_reason(tech(clip_hi=0.25, mean=0.85), face_metrics(),
                              gate(settings), settings)
    assert reason == "overexposed"


def test_bright_backdrop_does_not_reject_a_well_exposed_subject():
    """Studio white seamless blows out globally; the face is what matters."""
    settings = CullingSettings(preset="portrait")
    subject = face_metrics(eyes=9.0, sharpness=400, clip_hi=0.01, clip_lo=0.0,
                           mean_luma=0.62)
    # 38 % of the frame is pure white backdrop, mean well above the mid-point.
    reason = technical_reason(tech(clip_hi=0.38, mean=0.74), subject,
                              gate(settings), settings)
    assert reason is None


def test_blown_face_is_rejected_even_if_frame_is_mid_tone():
    settings = CullingSettings(preset="portrait")
    subject = face_metrics(eyes=9.0, sharpness=400, clip_hi=0.35, mean_luma=0.93)
    reason = technical_reason(tech(clip_hi=0.02, mean=0.5), subject,
                              gate(settings), settings)
    assert reason == "overexposed"


def test_underexposure_rejected():
    settings = CullingSettings()
    reason = technical_reason(tech(mean=0.02, clip_lo=0.6), face_metrics(),
                              gate(settings), settings)
    assert reason == "underexposed"


def test_closed_eyes_rejected():
    settings = CullingSettings(preset="wedding")
    faces = face_metrics(eyes=2.0, sharpness=400)
    assert technical_reason(tech(), faces, gate(settings), settings) == "closed_eyes"


def test_open_eyes_pass():
    settings = CullingSettings(preset="portrait")
    faces = face_metrics(eyes=8.5, sharpness=400)
    assert technical_reason(tech(), faces, gate(settings), settings) is None


def test_blink_detection_can_be_disabled():
    settings = CullingSettings(detect_blinks=False)
    faces = face_metrics(eyes=1.0, sharpness=400)
    assert technical_reason(tech(), faces, gate(settings), settings) is None


def test_blur_detection_can_be_disabled():
    # Each of the three blur paths is skipped when the toggle is off.
    mush = CullingSettings(detect_blur=False)
    assert technical_reason(tech(sharp=2.0), face_metrics(), gate(mush), mush) is None
    soft = CullingSettings(preset="portrait", detect_blur=False)
    soft_face = face_metrics(eyes=9.0, sharpness=25.0, area=0.05)
    assert technical_reason(tech(sharp=120), soft_face, gate(soft), soft) is None
    burst = CullingSettings(preset="wedding", detect_blur=False)
    assert technical_reason(tech(sharp=18), face_metrics(), gate(burst), burst,
                            group_median_sharpness=120.0) is None


def test_highlights_detection_can_be_disabled():
    settings = CullingSettings(detect_highlights=False)
    assert technical_reason(tech(clip_hi=0.25, mean=0.85), face_metrics(),
                            gate(settings), settings) is None


def test_highlights_off_still_rejects_underexposure():
    # There is no toggle for dark frames: they are always rejected.
    settings = CullingSettings(detect_highlights=False)
    assert technical_reason(tech(mean=0.02, clip_lo=0.6), face_metrics(),
                            gate(settings), settings) == "underexposed"


def test_detection_toggles_default_on():
    settings = CullingSettings()
    assert settings.detect_blur is True
    assert settings.detect_highlights is True
    # Libraries stored before the toggles existed carry no keys for them.
    stored = CullingSettings.from_dict({})
    assert stored.detect_blur is True
    assert stored.detect_highlights is True
    assert CullingSettings.from_dict({"detect_blur": False}).detect_blur is False


def test_profile_face_blink_is_not_trusted():
    settings = CullingSettings(preset="wedding")
    faces = face_metrics(eyes=1.0, sharpness=400, yaw=80.0)
    assert technical_reason(tech(), faces, gate(settings), settings) is None


def test_strictness_raises_the_face_gate():
    lenient = CullingSettings(preset="portrait", strictness=0)
    strict = CullingSettings(preset="portrait", strictness=100)
    assert strict.gates().face_blur > lenient.gates().face_blur
    assert strict.gates().eyes_closed_max < lenient.gates().eyes_closed_max


def test_subject_sharpness_ranks_a_sharp_subject_over_a_soft_one():
    settings = CullingSettings(preset="landscape")
    sharp = score_photo(tech(sharp=120, subject=140.0), face_metrics(), settings)
    soft = score_photo(tech(sharp=120, subject=4.0), face_metrics(), settings)
    assert sharp.axes["subject_sharpness"] is not None
    assert sharp.axes["subject_sharpness"] > soft.axes["subject_sharpness"]
    assert sharp.quality > soft.quality


def test_subject_sharpness_is_absent_when_a_face_measures_the_subject():
    # With a usable face the face crop is the subject: counting a centre crop as
    # well would weigh sharpness twice.
    faces = face_metrics(eyes=9.0, sharpness=400)
    result = score_photo(tech(subject=140.0), faces, CullingSettings(preset="portrait"))
    assert result.axes["subject_sharpness"] is None


def test_subject_sharpness_never_rejects_on_its_own():
    settings = CullingSettings()
    soft_subject = tech(sharp=60.0, subject=1.0)
    assert technical_reason(soft_subject, face_metrics(), gate(settings), settings) is None


def test_exposure_axis_penalises_a_two_peak_histogram():
    smooth = tech(mean=0.49, bimodality=0.0)
    harsh = tech(mean=0.49, bimodality=2.0)
    assert exposure_axis(smooth) == 1.0
    assert exposure_axis(harsh) < exposure_axis(smooth) - 0.05


def test_bimodality_never_becomes_a_rejection_reason():
    settings = CullingSettings()
    harsh = tech(mean=0.49, bimodality=3.0)
    assert technical_reason(harsh, face_metrics(), gate(settings), settings) is None


def test_colour_axis_ranks_but_never_rejects():
    settings = CullingSettings(preset="portrait")
    faces = face_metrics(eyes=9.0, sharpness=400)
    plain = score_photo(tech(color=0.2), faces, settings)
    vivid = score_photo(tech(color=9.5), faces, settings)
    assert vivid.axes["color"] > plain.axes["color"]
    assert technical_reason(tech(color=0.2), faces, gate(settings), settings) is None


def test_colour_weight_is_small_enough_not_to_punish_simplicity():
    # A minimal near-grey portrait is *supposed* to have low hue variety: the
    # axis is a tie-break, so its whole range is worth less than 0.05 quality.
    settings = CullingSettings(preset="portrait")
    faces = face_metrics(eyes=9.0, sharpness=400)
    plain = score_photo(tech(color=0.2), faces, settings)
    vivid = score_photo(tech(color=9.5), faces, settings)
    assert vivid.quality > plain.quality
    assert vivid.quality - plain.quality < 0.05
    assert plain.verdict == "keep"


def test_quality_is_bounded():
    settings = CullingSettings()
    for sharp in (0.0, 5.0, 50.0, 400.0, 5000.0):
        result = score_photo(tech(sharp=sharp), face_metrics(), settings)
        assert 0.0 <= result.quality <= 1.0


def test_unreadable_photo_never_becomes_a_keeper():
    settings = CullingSettings()
    result = score_photo(tech(sharp=0.0), face_metrics(), settings)
    assert result.verdict == "reject"
