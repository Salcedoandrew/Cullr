"""Measure duplicate thresholds instead of guessing them.

Ported from facet `utils/duplicate.py` (``evaluate_dedup_thresholds`` /
``report_dedup_thresholds``, MIT): a labelled set of hash pairs is swept over
hash distance x embedding cosine, and each threshold pair is scored for
precision, recall and F1. facet ships this so its numbers came from data; cullr
has no labelled set yet, so this is the tool to run on the first shoot where the
photographer marks which frames really are the same shot.

Usage, once such a set exists::

    from culler.dedup_tuning import LabelledPair, report_dedup_thresholds
    pairs = [
        LabelledPair(distance=3, cosine=0.96, same=True),
        LabelledPair(distance=9, cosine=0.71, same=False),
    ]
    print(report_dedup_thresholds(pairs))

Pure standard library: no numpy, no model, no files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# facet sweeps these ranges; cullr's shipped defaults (6 strict, 14 loose,
# 0.90 cosine) sit inside them, so the sweep can confirm or replace them.
DEFAULT_DISTANCES = tuple(range(4, 15))
DEFAULT_COSINES: tuple[float | None, ...] = (
    None, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96,
)


@dataclass(frozen=True)
class LabelledPair:
    """One pair a human has judged, with the evidence the gate sees."""

    distance: int                 # Hamming distance between the pHashes
    same: bool                    # ground truth: really the same frame
    cosine: float | None = None   # scene-embedding cosine, when one exists


@dataclass(frozen=True)
class ThresholdScore:
    distance: int
    cosine: float | None
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def _accepts(pair: LabelledPair, distance: int, cosine: float | None) -> bool:
    """Would this gate merge the pair?

    Mirrors ``similarity.find_groups``: the hash gate decides first, and a
    cosine threshold only ever removes recall. ``cosine=None`` is the hash-only
    gate, and a pair with no vector falls back to the hash decision.
    """
    if pair.distance > distance:
        return False
    if cosine is None or pair.cosine is None:
        return True
    return pair.cosine >= cosine


def evaluate_dedup_thresholds(
    pairs: Iterable[LabelledPair],
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    cosines: Sequence[float | None] = DEFAULT_COSINES,
) -> list[ThresholdScore]:
    """Sweep every (distance, cosine) pair; best F1 first."""
    labelled = list(pairs)
    rows: list[ThresholdScore] = []
    for distance in distances:
        for cosine in cosines:
            true_pos = false_pos = false_neg = 0
            for pair in labelled:
                accepted = _accepts(pair, distance, cosine)
                if accepted and pair.same:
                    true_pos += 1
                elif accepted:
                    false_pos += 1
                elif pair.same:
                    false_neg += 1
            precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
            recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            rows.append(ThresholdScore(
                distance=distance, cosine=cosine,
                true_positives=true_pos, false_positives=false_pos,
                false_negatives=false_neg,
                precision=precision, recall=recall, f1=f1,
            ))
    rows.sort(key=lambda row: (-row.f1, -row.precision, row.distance,
                               -1.0 if row.cosine is None else row.cosine))
    return rows


def report_dedup_thresholds(
    pairs: Iterable[LabelledPair],
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    cosines: Sequence[float | None] = DEFAULT_COSINES,
    top: int = 10,
) -> str:
    """A plain-text table of the best thresholds, for a terminal or a report."""
    rows = evaluate_dedup_thresholds(pairs, distances=distances, cosines=cosines)[:top]
    lines = [
        f"{'dist':>4}  {'cosine':>6}  {'tp':>4} {'fp':>4} {'fn':>4}  "
        f"{'precision':>9} {'recall':>6} {'f1':>5}",
    ]
    for row in rows:
        cosine = "  hash" if row.cosine is None else f"{row.cosine:6.2f}"
        lines.append(
            f"{row.distance:4d}  {cosine}  {row.true_positives:4d} "
            f"{row.false_positives:4d} {row.false_negatives:4d}  "
            f"{row.precision:9.3f} {row.recall:6.3f} {row.f1:5.3f}"
        )
    return "\n".join(lines)
