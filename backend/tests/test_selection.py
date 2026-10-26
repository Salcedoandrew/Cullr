from __future__ import annotations

from culler.selection import SelectionSignals, lead_score, rank, select


def sig(photo_id: int, quality: float, eyes=None, face_sharpness=None, smile=None,
        reason=None, captured_at=None) -> SelectionSignals:
    return SelectionSignals(
        photo_id=photo_id, quality=quality, eyes_open=eyes,
        face_sharpness=face_sharpness, smile=smile, reason=reason,
        captured_at=captured_at, order=photo_id,
    )


def test_higher_quality_wins():
    a, b = sig(1, 0.50), sig(2, 0.70)
    assert rank([a, b])[0].photo_id == 2


def test_open_eyes_break_a_near_tie():
    blinking = sig(1, 0.600, eyes=1.0)
    open_eyes = sig(2, 0.595, eyes=9.5)
    assert lead_score(open_eyes) > lead_score(blinking)
    assert rank([blinking, open_eyes])[0].photo_id == 2


def test_open_eyes_never_rescue_a_bad_frame():
    soft = sig(1, 0.40, eyes=10.0, reason="blurry")
    sharp_but_blinking = sig(2, 0.75, eyes=1.0, reason="closed_eyes")
    # Even ignoring reasons, a 0.35 quality gap is far beyond the bounded bonus.
    assert lead_score(sharp_but_blinking) > lead_score(soft)


def test_tie_falls_back_to_earlier_capture():
    a = sig(1, 0.6, captured_at=100.0)
    b = sig(2, 0.6, captured_at=100.4)
    assert rank([b, a])[0].photo_id == 1


def test_keep_min_per_group_keeps_two_and_rejects_the_rest():
    signals = [sig(1, 0.8), sig(2, 0.7), sig(3, 0.6), sig(4, 0.5)]
    result = select([1, 2, 3, 4], signals, keep_min=2)
    assert result.keeper_ids == [1, 2]
    assert sorted(result.duplicate_ids) == [3, 4]
    assert result.lead_id == 1


def test_rejected_frames_cannot_be_the_lead():
    # The best-scoring frame is technically unusable, so the lead must be the
    # best *acceptable* frame -- you cannot deliver a blurry keeper.
    signals = [sig(1, 0.7), sig(2, 0.9, reason="blurry"), sig(3, 0.8)]
    result = select([1, 2, 3], signals, keep_min=1)
    assert result.lead_id == 3
    assert result.keeper_ids == [3]
    # Frame 2 keeps its own reason ("blurry"); only frame 1 is a duplicate.
    assert result.duplicate_ids == [1]


def test_group_with_no_acceptable_frame_yields_no_keepers():
    signals = [sig(1, 0.7, reason="blurry"), sig(2, 0.6, reason="closed_eyes")]
    result = select([1, 2], signals, keep_min=1)
    assert result.keeper_ids == []
    assert result.duplicate_ids == []
    assert result.lead_id == 1


def test_selection_requires_members():
    try:
        select([], [], keep_min=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty group")
