from __future__ import annotations

from culler.dedup_tuning import (
    LabelledPair, evaluate_dedup_thresholds, report_dedup_thresholds,
)


def labelled_set() -> list[LabelledPair]:
    """Twenty true pairs inside the gate and twenty outside it.

    The labels are made by construction, not measured on a real shoot: this
    tests the sweep, not the thresholds cullr ships.
    """
    pairs = [LabelledPair(distance=d, cosine=0.95, same=True) for d in range(0, 11)]
    pairs += [LabelledPair(distance=d, cosine=0.70, same=False) for d in range(14, 34)]
    return pairs


def test_sweep_finds_the_separating_thresholds():
    rows = evaluate_dedup_thresholds(labelled_set())
    assert rows, "the sweep must return rows"
    best = rows[0]
    assert best.f1 == 1.0
    assert best.distance >= 10
    # Sorted best-first, and every row is a real measurement.
    assert all(rows[i].f1 >= rows[i + 1].f1 for i in range(len(rows) - 1))
    assert all(0.0 <= row.f1 <= 1.0 for row in rows)

    # The embedding gate reaches the same ceiling, at or below the positives'
    # own cosine -- above that it would start dropping true pairs.
    perfect_with_cosine = [row for row in rows
                           if row.f1 == 1.0 and row.cosine is not None]
    assert perfect_with_cosine, "the cosine gate must reach F1 1.0 too"
    assert max(row.cosine for row in perfect_with_cosine) <= 0.95


def test_a_loose_threshold_trades_precision_for_recall():
    rows = evaluate_dedup_thresholds(labelled_set(), distances=[6, 25],
                                     cosines=[None])
    tight = next(row for row in rows if row.distance == 6)
    loose = next(row for row in rows if row.distance == 25)
    assert tight.recall == 7 / 11         # four true pairs sit past distance 6
    assert loose.recall == 1.0
    assert loose.precision < 1.0          # it now merges the negative pairs too
    assert loose.false_positives == 12     # the negatives at distance 14..25


def test_hash_only_rows_are_available_alongside_embedding_rows():
    rows = evaluate_dedup_thresholds(labelled_set())
    assert any(row.cosine is None for row in rows)
    assert any(row.cosine is not None for row in rows)


def test_report_renders_a_readable_table():
    text = report_dedup_thresholds(labelled_set(), top=5)
    assert "precision" in text and "recall" in text and "f1" in text
    assert len(text.splitlines()) == 6      # header + 5 rows
