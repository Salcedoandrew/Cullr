"""Choosing the keeper(s) inside a similar/burst group.

The lead is chosen by a *bounded* tie-break on top of quality, following facet's
`utils/selection.py` rationale: a meaningful quality gap must never be
overturned by a blink bonus, but two near-identical frames with equal quality
should be separated by open eyes, a sharper face and a real smile.

Bounds (on the 0..10 signal scale, centred on neutral 5):
    eyes         +/- 0.15
    face sharp   +/- 0.05
    expression   +/- 0.10
So a mid-blink frame (eyes ~1) loses ~0.12 to an open-eyes frame (eyes ~9) with
the same quality — enough to decide a tie, not enough to rescue a bad frame.
"""

from __future__ import annotations

from dataclasses import dataclass

NEUTRAL = 5.0
EYES_W = 0.030        # ±0.15 over the 0..10 range
FACE_SHARP_W = 0.010  # ±0.05
SMILE_W = 0.020       # ±0.10


@dataclass
class SelectionSignals:
    photo_id: int
    quality: float
    eyes_open: float | None = None        # 0..10, worst usable face
    face_sharpness: float | None = None   # 0..100 axis score
    smile: float | None = None            # 0..10
    reason: str | None = None             # technical rejection, if any
    captured_at: float | None = None
    order: int = 0


def lead_score(sig: SelectionSignals) -> float:
    score = float(sig.quality)
    if sig.eyes_open is not None:
        score += (float(sig.eyes_open) - NEUTRAL) * EYES_W
    if sig.face_sharpness is not None:
        score += (float(sig.face_sharpness) - NEUTRAL) * FACE_SHARP_W
    if sig.smile is not None:
        score += (float(sig.smile) - NEUTRAL) * SMILE_W
    return score


def rank(signals: list[SelectionSignals]) -> list[SelectionSignals]:
    """Best first; ties resolved by earlier capture (the decisive moment)."""
    return sorted(
        signals,
        key=lambda s: (
            -(lead_score(s)),
            s.captured_at if s.captured_at is not None else float("inf"),
            s.order,
        ),
    )


@dataclass
class GroupSelection:
    lead_id: int
    keeper_ids: list[int]
    duplicate_ids: list[int]      # acceptable frames demoted because a better one exists


def select(group_ids: list[int], signals: list[SelectionSignals],
           keep_min: int = 1) -> GroupSelection:
    """Pick the lead plus any extra keepers; the rest become duplicate rejects.

    Technically-rejected frames are not "duplicates" — they keep their own
    reason, which is more actionable in the UI. If no frame in the group is
    technically acceptable, the group yields no keeper and nothing is marked as
    a duplicate (there is nothing to prefer).
    """
    if not signals:
        raise ValueError("select() needs at least one signal")
    ordered = rank(signals)
    acceptable = [s for s in ordered if s.reason is None]
    keep_target = max(1, keep_min)
    keepers = acceptable[:keep_target] if acceptable else []

    keeper_ids = [s.photo_id for s in keepers]
    duplicates: list[int] = []
    if keeper_ids:
        lead_id = keeper_ids[0]
        duplicates = [s.photo_id for s in ordered
                      if s.photo_id not in keeper_ids and s.reason is None]
    else:
        lead_id = ordered[0].photo_id
    return GroupSelection(lead_id=lead_id, keeper_ids=keeper_ids,
                          duplicate_ids=duplicates)
