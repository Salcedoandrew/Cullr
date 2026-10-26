from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from culler.config import CullingSettings
from culler.db import Store
from culler.imaging import ImageMeta
from culler.metrics.faceid import (
    DESCRIPTOR_DIM, align_face, describe_aligned, embedding_bytes,
    embedding_from_bytes, face_landmarks_to_points,
)
from culler.metrics.faces import Face, FaceMetrics
from culler.metrics.technical import TechnicalMetrics
from culler.persons import (
    FaceObservation, PersonCluster, apply_merge_groups, cluster_faces,
    clusters_from_rows, coverage_promotions, person_rows, propose_merges,
    uncovered_persons,
)
from culler.selection import SelectionSignals

FIXTURES = Path(__file__).resolve().parents[2] / ".research" / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "catalog.db")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def unit_vector(seed: int = 0, dim: int = DESCRIPTOR_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def vector_at(base: np.ndarray, similarity: float, seed: int = 0) -> np.ndarray:
    """A unit vector whose cosine similarity to ``base`` is ``similarity``."""
    rng = np.random.default_rng(seed)
    orthogonal = rng.normal(size=base.shape[0]).astype(np.float32)
    orthogonal -= base * float(base @ orthogonal)
    orthogonal /= np.linalg.norm(orthogonal)
    vector = similarity * base + np.sqrt(1.0 - similarity ** 2) * orthogonal
    return vector / np.linalg.norm(vector)


def observation(face_id: int, photo_id: int, vector: np.ndarray,
                *, area: float = 0.06, captured_at: float | None = 1000.0) -> FaceObservation:
    return FaceObservation(face_id=face_id, photo_id=photo_id,
                           captured_at=captured_at, vector=vector,
                           area_ratio=area, sharpness=400.0, eyes_open=9.0)


def signal(photo_id: int, *, quality: float = 0.7, reason: str | None = None,
           captured_at: float | None = None) -> SelectionSignals:
    return SelectionSignals(photo_id=photo_id, quality=quality, reason=reason,
                            captured_at=captured_at, order=photo_id)


def technical(sharpness: float = 320.0) -> TechnicalMetrics:
    return TechnicalMetrics(sharpness=sharpness, tenengrad=40.0, clip_hi=0.0,
                            clip_lo=0.0, mean_luma=0.48, p95_luma=0.8, noise=3.0,
                            contrast=55.0)


def person(index: int, photo_ids: list[int], *, area: float = 0.06) -> PersonCluster:
    return PersonCluster(
        index=index,
        face_ids=[index * 100 + offset for offset in range(len(photo_ids))],
        photo_ids=photo_ids,
        centroid=unit_vector(index),
        max_area_ratio=area,
        representative_face_id=index * 100,
        representative_photo_id=photo_ids[0],
    )


def textured_image(size: int = 120, seed: int = 0) -> np.ndarray:
    """Deterministic textured BGR patch, for descriptor tests without a camera."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.stack([
        90 + 70 * (xx / size) + 30 * np.sin(xx / 7.0),
        80 + 60 * (yy / size) + 25 * np.cos(yy / 9.0),
        100 + 50 * ((xx + yy) / (2 * size)),
    ], axis=-1)
    for _ in range(30):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        radius = int(rng.integers(4, 18))
        color = tuple(float(v) for v in rng.integers(30, 230, size=3))
        cv2_circle(base, cx, cy, radius, color)
    base += rng.normal(0, 2.0, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def cv2_circle(image: np.ndarray, cx: int, cy: int, radius: int, color) -> None:
    import cv2

    cv2.circle(image, (cx, cy), radius, color, -1)


def save_jpeg(path: Path, image: np.ndarray, captured_at: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.ascontiguousarray(image[..., ::-1])
    exif = Image.Exif()
    exif[36867] = captured_at
    Image.fromarray(rgb).save(path, format="JPEG", quality=92, exif=exif)
    return path


# --------------------------------------------------------------------------- #
# descriptor
# --------------------------------------------------------------------------- #

def test_descriptor_is_deterministic_and_unit_length():
    crop = textured_image(112, seed=1)
    first = describe_aligned(crop)
    second = describe_aligned(crop)
    assert first.shape == (DESCRIPTOR_DIM,)
    assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(first, second)


def test_descriptor_survives_exposure_changes():
    crop = textured_image(112, seed=2)
    base = describe_aligned(crop)
    other = describe_aligned(textured_image(112, seed=3))
    brighter = describe_aligned(np.clip(crop.astype(np.float32) * 1.12, 0, 255).astype(np.uint8))
    darker = describe_aligned(np.clip(crop.astype(np.float32) * 0.85, 0, 255).astype(np.uint8))
    assert float(base @ brighter) > 0.90
    assert float(base @ darker) > 0.90
    assert float(base @ other) < float(base @ brighter)


def test_descriptor_never_emits_nan_on_a_flat_crop():
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)
    vector = describe_aligned(flat)
    assert vector.shape == (DESCRIPTOR_DIM,)
    assert np.all(np.isfinite(vector))


def test_embedding_round_trips_through_a_blob():
    vector = unit_vector(3)
    blob = embedding_bytes(vector)
    assert isinstance(blob, bytes) and len(blob) == DESCRIPTOR_DIM * 4
    restored = embedding_from_bytes(blob)
    assert restored is not None and np.allclose(restored, vector, atol=1e-6)
    assert embedding_from_bytes(None) is None
    assert embedding_from_bytes(b"short") is None


def test_alignment_rejects_degenerate_input():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert align_face(image, None) is None
    assert align_face(image, np.zeros((4, 2), dtype=np.float32)) is None
    points = np.array([[10, 10], [50, 10], [30, 30], [15, 45], [45, 45]], dtype=np.float32)
    assert align_face(image, points) is not None


def test_landmark_points_reject_short_or_collapsed_meshes():
    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    assert face_landmarks_to_points([], 100, 100) is None
    collapsed = [Point(0.5, 0.5) for _ in range(478)]
    assert face_landmarks_to_points(collapsed, 100, 100) is None

    for index, (x, y) in {33: (0.30, 0.40), 133: (0.35, 0.40), 362: (0.60, 0.40),
                          263: (0.65, 0.40), 1: (0.50, 0.55), 61: (0.42, 0.70),
                          291: (0.58, 0.70)}.items():
        collapsed[index] = Point(x, y)
    points = face_landmarks_to_points(collapsed, 100, 100)
    assert points is not None and points.shape == (5, 2)


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #

def test_clusters_separate_two_identities():
    alice, bob = unit_vector(1), unit_vector(2)
    observations = [
        observation(1, 10, vector_at(alice, 0.97, seed=11)),
        observation(2, 10, vector_at(bob, 0.97, seed=12)),
        observation(3, 11, vector_at(alice, 0.95, seed=13)),
        observation(4, 12, vector_at(bob, 0.96, seed=14)),
        observation(5, 13, vector_at(alice, 0.94, seed=15)),
    ]
    clusters = cluster_faces(observations)
    assert [c.face_ids for c in clusters] == [[1, 3, 5], [2, 4]]
    assert [c.index for c in clusters] == [1, 2]
    assert clusters[0].photo_ids == [10, 11, 13]


def test_clustering_is_order_independent():
    alice, bob = unit_vector(4), unit_vector(5)
    observations = [
        observation(1, 10, vector_at(alice, 0.96, seed=1)),
        observation(2, 10, vector_at(bob, 0.96, seed=2)),
        observation(3, 11, vector_at(alice, 0.95, seed=3)),
        observation(4, 12, vector_at(bob, 0.97, seed=4)),
    ]
    forwards = [c.face_ids for c in cluster_faces(observations)]
    backwards = [c.face_ids for c in cluster_faces(list(reversed(observations)))]
    assert forwards == backwards


def test_single_face_makes_a_person_and_empty_input_is_empty():
    assert cluster_faces([]) == []
    clusters = cluster_faces([observation(7, 70, unit_vector(9))])
    assert len(clusters) == 1
    assert clusters[0].face_ids == [7]
    assert clusters[0].representative_photo_id == 70


def test_faces_below_the_match_threshold_stay_separate():
    base = unit_vector(6)
    clusters = cluster_faces([
        observation(1, 10, base),
        observation(2, 11, vector_at(base, 0.85, seed=7)),
    ])
    assert len(clusters) == 2


def test_two_faces_in_one_photo_are_never_one_person():
    """A hard must-not-link constraint: same photo, therefore different people."""
    base = unit_vector(8)
    clusters = cluster_faces([
        observation(1, 10, base),
        observation(2, 10, vector_at(base, 0.995, seed=9)),
    ])
    assert len(clusters) == 2
    assert [[c.face_ids for c in clusters]] == [[[1], [2]]]


def test_confirmed_merge_group_rejoins_split_people():
    base = unit_vector(10)
    first = cluster_faces([observation(1, 10, base)])
    second = cluster_faces([observation(5, 14, vector_at(base, 0.86, seed=11))])
    merged = apply_merge_groups(first + second, [[1, 5]])
    assert len(merged) == 1
    assert merged[0].face_ids == [1, 5]
    assert merged[0].photo_ids == [10, 14]
    assert merged[0].index == 1


def test_merge_group_with_a_vanished_face_still_applies():
    base = unit_vector(12)
    clusters = cluster_faces([observation(1, 10, base),
                              observation(2, 11, vector_at(base, 0.86, seed=13))])
    assert len(clusters) == 2
    assert len(apply_merge_groups(clusters, [[1, 999]])) == 2
    assert len(apply_merge_groups(clusters, [[1, 2, 999]])) == 1


def test_proposals_require_similarity_and_no_shared_photo():
    near = [person(1, [10]), person(2, [20])]
    same_photo = [person(1, [10]), person(2, [10])]
    for cluster_a, cluster_b in (near, same_photo):
        cluster_a.centroid = unit_vector(20)
        cluster_b.centroid = vector_at(cluster_a.centroid, 0.88, seed=21)

    proposals = propose_merges(near)
    assert [(p.a, p.b) for p in proposals] == [(1, 2)]
    assert proposals[0].similarity == pytest.approx(0.88, abs=0.01)
    assert propose_merges(same_photo) == []

    far = [person(1, [10]), person(2, [20])]
    far[1].centroid = vector_at(far[0].centroid, 0.70, seed=22)
    assert propose_merges(far, min_similarity=0.84) == []


def test_proposals_skip_a_pair_marked_as_different():
    left, right = person(1, [10]), person(2, [20])
    left.centroid = unit_vector(23)
    right.centroid = vector_at(left.centroid, 0.89, seed=24)
    assert propose_merges([left, right])
    assert propose_merges([left, right], dismissed={(100, 200)}) == []


def test_proposals_are_ranked_and_capped():
    base = unit_vector(25)
    clusters = []
    for index in range(1, 5):
        cluster = person(index, [index * 10])
        cluster.centroid = vector_at(base, 0.95, seed=index)
        clusters.append(cluster)
    proposals = propose_merges(clusters, limit=3)
    assert len(proposals) == 3
    scores = [p.similarity for p in proposals]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #

def test_uncovered_person_gets_their_best_frame_promoted():
    people = [person(1, [10, 11]), person(2, [20, 21, 22])]
    signals = {pid: signal(pid, quality=q) for pid, q in
               [(10, 0.9), (11, 0.8), (20, 0.7), (21, 0.75), (22, 0.6)]}
    promotions = coverage_promotions(people, signals, keeper_ids={10})
    assert [(p.photo_id, p.person, p.forced) for p in promotions] == [(21, 2, False)]


def test_covered_person_is_not_promoted_again():
    people = [person(1, [10, 11])]
    signals = {pid: signal(pid) for pid in (10, 11)}
    assert coverage_promotions(people, signals, keeper_ids={11}) == []


def test_one_frame_is_not_enough_to_require_coverage():
    people = [person(1, [10]), person(2, [20, 21])]
    signals = {pid: signal(pid) for pid in (10, 20, 21)}
    assert coverage_promotions(people, signals, keeper_ids={20}) == []


def test_background_sized_faces_do_not_force_keepers():
    people = [person(1, [10, 11], area=0.004)]
    signals = {pid: signal(pid) for pid in (10, 11)}
    assert coverage_promotions(people, signals, keeper_ids=set()) == []


def test_all_flawed_frames_promote_the_best_and_are_marked_forced():
    people = [person(1, [10, 11, 12])]
    signals = {
        10: signal(10, quality=0.5, reason="blurry"),
        11: signal(11, quality=0.7, reason="closed_eyes"),
        12: signal(12, quality=0.2, reason="blurry"),
    }
    promotions = coverage_promotions(people, signals, keeper_ids=set())
    assert [(p.photo_id, p.forced) for p in promotions] == [(11, True)]


def test_unreadable_frames_are_never_promoted():
    people = [person(1, [10, 11])]
    signals = {10: signal(10, reason="unreadable"), 11: signal(11, reason="unreadable")}
    assert coverage_promotions(people, signals, keeper_ids=set()) == []


def test_one_wide_frame_covers_two_people():
    shared = person(1, [10, 11])
    other = person(2, [11, 12])
    signals = {pid: signal(pid, quality=q) for pid, q in [(10, 0.4), (11, 0.9), (12, 0.5)]}
    promotions = coverage_promotions([shared, other], signals, keeper_ids=set())
    assert [(p.photo_id, p.person) for p in promotions] == [(11, 1)]
    assert uncovered_persons([shared, other], keeper_ids={11}) == []


def test_person_rows_report_counts_and_namespaced_ids():
    people = [person(1, [10, 11]), person(2, [20])]
    rows = person_rows(people, keeper_ids={10}, offset=2_000_000)
    assert [r["id"] for r in rows] == [2_000_001, 2_000_002]
    assert [r["label"] for r in rows] == ["Person 1", "Person 2"]
    assert rows[0]["photo_count"] == 2 and rows[0]["keep_count"] == 1
    assert rows[0]["covered"] == 1 and rows[1]["covered"] == 0
    # A person seen once is not required to have a keeper.
    assert rows[0]["required"] == 1 and rows[1]["required"] == 0


# --------------------------------------------------------------------------- #
# catalog persistence
# --------------------------------------------------------------------------- #

def test_embeddings_and_person_ids_persist(store, tmp_path):
    library = store.create_or_get_library(str(tmp_path / "shoot"))
    photo_id = store.upsert_photo(library["id"], tmp_path / "shoot" / "a.jpg", "key", 10, 1.0)
    vector = unit_vector(21)
    store.update_analysis(photo_id, {
        "faces": [{"x": 1, "y": 2, "w": 30, "h": 30, "area_ratio": 0.05,
                   "eyes_open": 9.0, "embedding": vector}],
        "phash": "00ff00ff00ff00ff", "dhash": "00ff00ff00ff00ff",
    }, version=2)

    rows = store.faces_with_embeddings(library["id"])
    assert len(rows) == 1
    restored = embedding_from_bytes(rows[0]["embedding"])
    assert restored is not None and np.allclose(restored, vector, atol=1e-6)

    face_id = rows[0]["face_id"]
    person_id = library["id"] * store.PERSON_ID_STRIDE + 1
    store.replace_persons(library["id"], [{
        "id": person_id, "label": "Person 1", "face_count": 1, "photo_count": 1,
        "keep_count": 1, "covered": 1, "required": 0,
        "representative_photo_id": photo_id, "representative_face_id": face_id,
        "max_area_ratio": 0.05,
    }], [(person_id, face_id)])

    face = store.faces_for([photo_id])[photo_id][0]
    assert face["person_id"] == person_id
    assert "embedding" not in face        # descriptors never reach the API
    assert [p["id"] for p in store.persons(library["id"])] == [person_id]
    assert [r["id"] for r in store.photos(library["id"], person_id=person_id)] == [photo_id]
    assert store.photos(library["id"], person_id=person_id + 1) == []
    assert store.stats(library["id"])["persons"] == 1


def _seed_person(store: Store, library_id: int, root: Path, name: str,
                 vector: np.ndarray, *, phash: str = "00ff00ff00ff00ff"):
    """One photo with one described face, returning ``(photo_id, face_id)``."""
    photo_id = store.upsert_photo(library_id, root / name, name, 10, 1.0)
    store.update_analysis(photo_id, {
        "faces": [{"x": 1, "y": 2, "w": 40, "h": 40, "area_ratio": 0.05,
                   "eyes_open": 9.0, "sharpness": 300.0, "embedding": vector}],
        "phash": phash, "dhash": phash,
    }, version=2)
    face_id = store.faces_for([photo_id])[photo_id][0]["id"]
    return photo_id, face_id


def _two_people(store: Store, library_id: int, root: Path):
    base = unit_vector(31)
    photo_a, face_a = _seed_person(store, library_id, root, "a.jpg", base)
    photo_b, face_b = _seed_person(store, library_id, root, "b.jpg",
                                   vector_at(base, 0.87, seed=32))
    offset = library_id * store.PERSON_ID_STRIDE
    people = [
        {"id": offset + 1, "label": "Person 1", "face_count": 1, "photo_count": 1,
         "keep_count": 1, "covered": 1, "required": 0,
         "representative_photo_id": photo_a, "representative_face_id": face_a,
         "max_area_ratio": 0.05},
        {"id": offset + 2, "label": "Person 2", "face_count": 1, "photo_count": 1,
         "keep_count": 0, "covered": 0, "required": 0,
         "representative_photo_id": photo_b, "representative_face_id": face_b,
         "max_area_ratio": 0.05},
    ]
    store.replace_persons(library_id, people,
                          [(offset + 1, face_a), (offset + 2, face_b)])
    return (photo_a, face_a), (photo_b, face_b), offset


def test_merge_persons_moves_faces_photos_and_coverage(store, tmp_path):
    library = store.create_or_get_library(str(tmp_path / "merge"))
    (photo_a, _face_a), (photo_b, face_b), offset = _two_people(store, library["id"], tmp_path / "merge")

    # Pretend photo B was promoted to cover Person 2.
    store.update_decisions([(None, "reject", "duplicate", None, 0, offset + 2, photo_b)])

    target = store.merge_persons(library["id"], [offset + 2, offset + 1])
    assert target == offset + 1                     # tie on photos -> lower id wins

    people = store.persons(library["id"])
    assert [p["id"] for p in people] == [target]
    assert people[0]["face_count"] == 2 and people[0]["photo_count"] == 2
    assert people[0]["keep_count"] == 1 and people[0]["covered"] == 1

    face = store.faces_for([photo_b])[photo_b][0]
    assert face["person_id"] == target
    assert store.photo_row(photo_b)["covered_person"] == target

    groups = store.person_links(library["id"], "merge")
    assert len(groups) == 1 and len(groups[0]) == 2 and face_b in groups[0]


def test_person_rows_rebuild_from_the_catalog(store, tmp_path):
    library = store.create_or_get_library(str(tmp_path / "rows"))
    _two_people(store, library["id"], tmp_path / "rows")
    clusters = clusters_from_rows(store.persons(library["id"]),
                                  store.person_faces(library["id"]))
    assert [c.index for c in clusters] == sorted(p["id"] for p in store.persons(library["id"]))
    assert all(c.photo_ids for c in clusters)
    assert float(clusters[0].centroid @ clusters[1].centroid) == pytest.approx(0.87, abs=0.02)
    assert len(propose_merges(clusters)) == 1


def test_distinct_links_persist_and_a_merge_clears_them(store, tmp_path):
    library = store.create_or_get_library(str(tmp_path / "distinct"))
    (_photo_a, _face_a), (_photo_b, face_b), offset = _two_people(
        store, library["id"], tmp_path / "distinct")

    store.mark_persons_distinct(library["id"], [offset + 1, offset + 2])
    dismissed = store.person_links(library["id"], "distinct")
    assert len(dismissed) == 1 and face_b in dismissed[0]

    clusters = clusters_from_rows(store.persons(library["id"]),
                                  store.person_faces(library["id"]))
    assert propose_merges(clusters, dismissed={tuple(dismissed[0])}) == []

    store.merge_persons(library["id"], [offset + 1, offset + 2])
    assert store.person_links(library["id"], "distinct") == []


# --------------------------------------------------------------------------- #
# end-to-end scan with synthetic identities (no MediaPipe needed)
# --------------------------------------------------------------------------- #

def test_scan_promotes_a_keeper_for_a_second_person(store, tmp_path, monkeypatch):
    """One lead with person A, two duplicates with only person B: B must survive."""
    from culler import pipeline

    shoot = tmp_path / "people"
    shoot.mkdir()
    names = ["IMG_5001.jpg", "IMG_5002.jpg", "IMG_5003.jpg"]
    scene = textured_image(160, seed=31)
    for index, name in enumerate(names):
        save_jpeg(shoot / name, scene, f"2024:06:01 09:00:0{index}")

    person_a, person_b = unit_vector(31), unit_vector(32)
    identities = {
        "IMG_5001.jpg": [person_a],
        "IMG_5002.jpg": [vector_at(person_b, 0.97, seed=1)],
        "IMG_5003.jpg": [vector_at(person_b, 0.96, seed=2)],
    }

    def fake_analyse(path, analyzer, long_edge=None):
        faces = [Face(x=100, y=80, w=220, h=220, eyes_open=9.0, smile=5.0,
                      yaw=2.0, pitch=1.0, sharpness=420.0, area_ratio=0.06,
                      clip_hi=0.0, clip_lo=0.0, mean_luma=0.5, embedding=vector)
                 for vector in identities[path.name]]
        meta = ImageMeta(width=1600, height=1067,
                         captured_at=1_750_000_000.0 + names.index(path.name))
        return (meta, technical(), FaceMetrics(faces=faces), None,
                "1234567890abcdef", "1234567890abcdef")

    monkeypatch.setattr(pipeline, "analyse_file", fake_analyse)

    library = store.create_or_get_library(str(shoot))
    result = pipeline.run_scan(store, library["id"], CullingSettings(preset="wedding"))

    assert result.persons == 2
    assert result.uncovered == 0
    rows = store.photos(library["id"])
    by_name = {r["filename"]: r for r in rows}

    assert by_name["IMG_5001.jpg"]["verdict"] == "keep"
    assert by_name["IMG_5001.jpg"]["is_lead"] == 1

    promoted = [r for r in rows if r["covered_person"] is not None]
    assert len(promoted) == 1
    assert promoted[0]["filename"] in ("IMG_5002.jpg", "IMG_5003.jpg")
    assert promoted[0]["verdict"] == "keep"
    assert promoted[0]["reason"] is None
    assert promoted[0]["is_lead"] == 0

    person_row = next(p for p in store.persons(library["id"])
                      if p["id"] == promoted[0]["covered_person"])
    assert person_row["covered"] == 1
    assert store.stats(library["id"])["uncovered_persons"] == 0


def test_confirmed_merge_survives_a_rescan(store, tmp_path, monkeypatch):
    """A curated merge must not be undone by the next scan."""
    from culler import pipeline

    shoot = tmp_path / "curated"
    shoot.mkdir()
    for index in range(2):
        save_jpeg(shoot / f"IMG_700{index}.jpg", textured_image(200, seed=70 + index),
                  f"2024:08:01 12:00:0{index}")

    base = unit_vector(71)
    vectors = {
        "IMG_7000.jpg": [base],
        "IMG_7001.jpg": [vector_at(base, 0.87, seed=72)],
    }

    def fake_analyse(path, analyzer, long_edge=None):
        faces = [Face(x=100, y=80, w=220, h=220, eyes_open=9.0, area_ratio=0.06,
                      sharpness=420.0, embedding=v) for v in vectors[path.name]]
        return (ImageMeta(width=1600, height=1067,
                          captured_at=1_760_000_000.0 + list(vectors).index(path.name)),
                technical(), FaceMetrics(faces=faces), None,
                f"{list(vectors).index(path.name):016x}", f"{list(vectors).index(path.name):016x}")

    monkeypatch.setattr(pipeline, "analyse_file", fake_analyse)
    library = store.create_or_get_library(str(shoot))
    first = pipeline.run_scan(store, library["id"], CullingSettings())
    assert first.persons == 2, "0.87 must stay split under the conservative gate"

    ids = [p["id"] for p in store.persons(library["id"])]
    merged = store.merge_persons(library["id"], ids)
    assert len(store.persons(library["id"])) == 1

    second = pipeline.run_scan(store, library["id"], CullingSettings())
    assert second.reused == second.total
    people = store.persons(library["id"])
    assert len(people) == 1
    assert people[0]["id"] == merged
    assert people[0]["photo_count"] == 2 and people[0]["face_count"] == 2


def test_people_can_be_disabled(store, tmp_path, monkeypatch):
    from culler import pipeline

    shoot = tmp_path / "nopeople"
    shoot.mkdir()
    for index in range(2):
        save_jpeg(shoot / f"IMG_600{index}.jpg", textured_image(160, seed=41),
                  f"2024:06:01 10:00:0{index}")

    def fake_analyse(path, analyzer, long_edge=None):
        faces = [Face(x=100, y=80, w=220, h=220, eyes_open=9.0, area_ratio=0.06,
                      sharpness=420.0, embedding=unit_vector(41))]
        return (ImageMeta(width=1600, height=1067, captured_at=1_750_000_000.0),
                technical(), FaceMetrics(faces=faces), None,
                "ffff0000ffff0000", "ffff0000ffff0000")

    monkeypatch.setattr(pipeline, "analyse_file", fake_analyse)
    library = store.create_or_get_library(str(shoot))
    result = pipeline.run_scan(store, library["id"],
                               CullingSettings(person_coverage=False))
    assert result.persons == 0
    assert store.persons(library["id"]) == []
    assert all(r["covered_person"] is None for r in store.photos(library["id"]))


# --------------------------------------------------------------------------- #
# real faces (needs MediaPipe + the model)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (FIXTURES / "portrait_group.jpg").exists(),
                    reason="Wikimedia fixtures not downloaded")
def test_real_faces_cluster_by_identity():
    import cv2

    from culler.imaging import load_image
    from culler.metrics.faces import FaceAnalyzer

    analyzer = FaceAnalyzer()
    if not analyzer.available:
        pytest.skip(f"face model unavailable: {analyzer.load_error}")

    def face_of(image_or_path):
        if isinstance(image_or_path, Path):
            bgr, _meta, _err = load_image(image_or_path)
        else:
            bgr = image_or_path
        metrics = analyzer.analyze(bgr)
        assert metrics.faces and metrics.faces[0].embedding is not None
        return metrics.faces[0].embedding

    bgr, _meta, _err = load_image(FIXTURES / "portrait_open_eyes.jpg")
    same_person = face_of(bgr)
    # The same face, underexposed by a stop and a half.
    darker = np.clip(bgr.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    same_person_dark = face_of(darker)
    other_person = face_of(FIXTURES / "portrait_group.jpg")

    assert float(same_person @ same_person_dark) > 0.90
    assert float(same_person @ other_person) < float(same_person @ same_person_dark)

    observations = [
        FaceObservation(1, 10, 1000.0, same_person, 0.056),
        FaceObservation(2, 11, 1001.0, same_person_dark, 0.056),
        FaceObservation(3, 12, 1002.0, other_person, 0.058),
    ]
    clusters = cluster_faces(observations)
    assert [c.face_ids for c in clusters] == [[1, 2], [3]]
