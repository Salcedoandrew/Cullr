from __future__ import annotations

import pytest

from culler.config import CullingSettings
from culler.imaging import dhash, hashes_to_hex, phash
from culler.similarity import SimilarityItem, find_groups
from conftest import blur, make_scene, shift_image


def item(photo_id: int, captured_at: float | None, image) -> SimilarityItem:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return SimilarityItem(
        photo_id=photo_id,
        captured_at=captured_at,
        phash=hashes_to_hex(phash(gray)),
        dhash=hashes_to_hex(dhash(gray)),
        order=photo_id,
    )


def test_identical_images_group_regardless_of_time():
    base = make_scene(1)
    items = [item(1, 0.0, base), item(2, 3600.0, base)]
    groups = find_groups(items, CullingSettings())
    assert len(groups) == 1
    assert sorted(groups[0].member_ids) == [1, 2]


def test_burst_groups_on_time_even_when_content_moves():
    items = [
        item(1, 100.0, make_scene(1)),
        item(2, 100.4, make_scene(2)),
        item(3, 100.8, make_scene(3)),
    ]
    groups = find_groups(items, CullingSettings(burst_window_seconds=2.0))
    assert len(groups) == 1
    assert groups[0].kind == "burst"
    assert sorted(groups[0].member_ids) == [1, 2, 3]


def test_distant_different_scenes_do_not_group():
    items = [item(1, 0.0, make_scene(1)), item(2, 600.0, make_scene(2))]
    assert find_groups(items, CullingSettings()) == []


def test_grouping_can_be_disabled():
    base = make_scene(1)
    items = [item(1, 0.0, base), item(2, 0.2, base)]
    assert find_groups(items, CullingSettings(group_duplicates=False)) == []


def test_blurred_near_duplicate_still_groups():
    base = make_scene(3)
    items = [item(1, 0.0, base), item(2, 3.0, blur(shift_image(base, 2, 0), 1.2))]
    groups = find_groups(items, CullingSettings())
    assert len(groups) == 1


def test_long_burst_is_chunked_not_chained():
    """Single-linkage chaining must not turn a whole shoot into one group."""
    from culler.config import ANALYSIS_VERSION  # noqa: F401  (documented import)

    items = [item(i, i * 0.2, shift_image(make_scene(5), i, 0)) for i in range(1, 60)]
    settings = CullingSettings(burst_window_seconds=2.0)
    groups = find_groups(items, settings)
    assert groups, "expected at least one burst group"
    assert all(len(g.member_ids) <= 24 for g in groups)
    total = sum(len(g.member_ids) for g in groups)
    assert total >= 40, f"chunking dropped too many members: {total}"


def test_similar_window_requires_hash_agreement():
    """Two frames 6 s apart with unrelated content must not merge."""
    items = [item(1, 0.0, make_scene(1)), item(2, 6.0, make_scene(9))]
    assert find_groups(items, CullingSettings(similar_window_seconds=8.0)) == []


def spread_hash(base: int, count: int) -> int:
    """Flip ``count`` bits across *both* 32-bit halves of a hash.

    Deliberately defeats cullr's top/bottom-half blocking, which is how the
    anywhere-in-the-shoot pass finds candidates: a pair like this is invisible
    to that pass and only reachable through a looser comparison.
    """
    top = [40, 44, 48, 52, 56, 60]
    bottom = [4, 8, 12, 16, 20, 24]
    other = base
    for k in range(count):
        other ^= 1 << (top[k // 2] if k % 2 == 0 else bottom[k // 2])
    return other


def test_two_stage_gate_finds_a_loose_match_the_time_window_hides():
    """facet's stage 1 + 2: Hamming 10 spread over both halves, 30 s apart.

    Today's gate needs a shared 32-bit hash bucket or the 8 s similar window, so
    this pair is missed. The loose prefilter (<= 12) plus a confirming cosine
    admits it -- the recall the two-stage design exists for.
    """
    import cv2

    from culler.imaging import hamming

    base_hash = phash(cv2.cvtColor(make_scene(6), cv2.COLOR_BGR2GRAY))
    other_hash = spread_hash(base_hash, 10)
    assert hamming(base_hash, other_hash) == 10

    items = [
        SimilarityItem(1, 0.0, hashes_to_hex(base_hash), None, 1),
        SimilarityItem(2, 30.0, hashes_to_hex(other_hash), None, 2),
    ]
    settings = CullingSettings()          # burst 2 s, similar window 8 s
    assert find_groups(items, settings) == []

    embeddings = {1: [1.0, 0.0], 2: [0.99, 0.01]}   # cosine 0.99
    groups = find_groups(items, settings, embeddings=embeddings)
    assert len(groups) == 1
    assert sorted(groups[0].member_ids) == [1, 2]


def test_two_stage_gate_refuses_a_loose_match_the_embedding_denies():
    import cv2

    base_hash = phash(cv2.cvtColor(make_scene(6), cv2.COLOR_BGR2GRAY))
    items = [
        SimilarityItem(1, 0.0, hashes_to_hex(base_hash), None, 1),
        SimilarityItem(2, 30.0, hashes_to_hex(spread_hash(base_hash, 10)), None, 2),
    ]
    denied = {1: [1.0, 0.0], 2: [0.2, 0.98]}        # cosine 0.20
    assert find_groups(items, CullingSettings(), embeddings=denied) == []


def test_two_stage_gate_needs_a_vector_to_widen_recall():
    """No embedder, no widening: the loose pair stays missed, as today."""
    import cv2

    base_hash = phash(cv2.cvtColor(make_scene(6), cv2.COLOR_BGR2GRAY))
    items = [
        SimilarityItem(1, 0.0, hashes_to_hex(base_hash), None, 1),
        SimilarityItem(2, 30.0, hashes_to_hex(spread_hash(base_hash, 10)), None, 2),
    ]
    assert find_groups(items, CullingSettings()) == []
    # A half-populated embedding map must not open the gate either.
    only_one = {1: [1.0, 0.0]}
    assert find_groups(items, CullingSettings(), embeddings=only_one) == []


def test_embedding_gate_can_block_a_hash_match():
    base = make_scene(1)
    items = [item(1, 0.0, base), item(2, 0.2, base)]
    # Orthogonal embeddings: the second stage refuses the merge.
    embeddings = {1: [1.0, 0.0], 2: [0.0, 1.0]}
    groups = find_groups(items, CullingSettings(), embeddings=embeddings,
                         embedding_threshold=0.9)
    assert groups == []


def test_embedding_gate_allows_a_confident_match():
    base = make_scene(1)
    items = [item(1, 0.0, base), item(2, 0.2, base)]
    embeddings = {1: [1.0, 0.0], 2: [0.98, 0.02]}
    groups = find_groups(items, CullingSettings(), embeddings=embeddings)
    assert len(groups) == 1


@pytest.mark.parametrize("distance,expected", [(0, 1), (6, 1), (30, 0)])
def test_duplicate_distance_gate(distance, expected):
    import numpy as np

    from culler.imaging import hamming

    base = make_scene(4)
    base_hash = phash(__import__("cv2").cvtColor(base, __import__("cv2").COLOR_BGR2GRAY))
    other = base_hash
    for bit in range(distance):
        other ^= 1 << (bit + 1)
    assert hamming(base_hash, other) == distance

    items = [
        SimilarityItem(1, 0.0, hashes_to_hex(base_hash), None, 1),
        SimilarityItem(2, 3600.0, hashes_to_hex(other), None, 2),
    ]
    groups = find_groups(items, CullingSettings(duplicate_distance=6))
    assert len(groups) == expected
    assert np is not None
