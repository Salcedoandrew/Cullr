"""People across a shoot: identity clustering plus the keeper-coverage guarantee.

Wedding and event photographers do not just want *the* best frame; they want
everyone who mattered to survive the cut. Aftershoot-style "key face" culling
therefore does two things this module reproduces:

1. **Cluster faces into people.** Face descriptors (``metrics/faceid.py``) are
   grouped with a greedy centroid match followed by a merge pass. The merge pass
   matters: a single pass is order dependent, and two clusters of the same
   person are much worse than one cluster of two people for the guarantee below.

2. **Guarantee coverage.** For every person seen in at least ``min_photos`` frames
   with at least one face big enough to matter, if no current keeper contains
   them, the best frame that does contain them is promoted to keeper. Promotion
   only ever *adds* keepers, never demotes one, and it prefers frames that passed
   the technical gates. When a person only appears in technically flawed frames
   (soft, blink), the best of those is still promoted and flagged ``forced`` so
   the review UI can say why the frame is worth keeping.

Clustering errors are benign for the guarantee, which is why a classical
descriptor is good enough here: over-merging produces a keeper that contains both
people, and under-merging produces one extra keeper. Neither loses a person.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .metrics.faceid import DESCRIPTOR_DIM, embedding_from_bytes
from .selection import SelectionSignals, rank


@dataclass
class FaceObservation:
    """One detected face, as needed for identity clustering."""

    face_id: int
    photo_id: int
    captured_at: float | None
    vector: np.ndarray
    area_ratio: float = 0.0
    sharpness: float = 0.0
    eyes_open: float | None = None


@dataclass
class PersonCluster:
    index: int                       # 1-based, stable within one clustering run
    face_ids: list[int]
    photo_ids: list[int]
    centroid: np.ndarray
    max_area_ratio: float = 0.0
    representative_face_id: int = 0
    representative_photo_id: int = 0

    def __len__(self) -> int:
        return len(self.face_ids)


def cluster_faces(
    observations: list[FaceObservation],
    *,
    match: float = 0.90,
    merge: float = 0.92,
    max_merge_rounds: int = 6,
) -> list[PersonCluster]:
    """Group face descriptors into people.

    ``match`` is the cosine similarity needed to join an existing person;
    ``merge`` is the similarity between two finished cluster centroids needed to
    merge them. Measured on real portraits the descriptor separates identities by
    a clear margin (same person 0.94-1.00, different people up to 0.88), but the
    two distributions are close enough on similar-looking subjects that the
    threshold is set high on purpose: a false merge can silently defeat the
    coverage guarantee, while a split only costs one extra keeper.

    Input order does not affect the result: observations are sorted by capture
    time and face id first, and the merge pass removes most of the remaining
    order dependence.

    Two faces detected in the same photo cannot be the same person, so that is a
    hard must-not-link constraint: it is enforced both when a face joins a
    cluster and when two clusters would merge. It matters in group shots, where
    two similar-looking guests would otherwise collapse into one identity.
    """
    usable = [o for o in observations
              if o.vector is not None and len(o.vector) > 0]
    if not usable:
        return []

    ordered = sorted(
        usable,
        key=lambda o: (o.captured_at is None, o.captured_at or 0.0, o.face_id),
    )
    vectors = np.stack([np.asarray(o.vector, dtype=np.float32) for o in ordered])

    # ---- greedy centroid clustering, O(n * people) ------------------------ #
    groups: list[list[int]] = []
    sums: list[np.ndarray] = []
    group_photos: list[set[int]] = []
    for position, vector in enumerate(vectors):
        photo = ordered[position].photo_id
        if groups:
            centroids = np.stack([s / len(g) for s, g in zip(sums, groups)])
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / np.maximum(norms, 1e-9)
            similarities = centroids @ vector
            for index, photos in enumerate(group_photos):
                if photo in photos:
                    similarities[index] = -np.inf
            best = int(np.argmax(similarities))
            if np.isfinite(similarities[best]) and float(similarities[best]) >= match:
                groups[best].append(position)
                sums[best] = sums[best] + vector
                group_photos[best].add(photo)
                continue
        groups.append([position])
        sums.append(vector.copy())
        group_photos.append({photo})

    # ---- merge near-identical clusters ----------------------------------- #
    def centroid_of(indices: list[int]) -> np.ndarray:
        mean = vectors[indices].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-9 else mean

    centroids = [centroid_of(g) for g in groups]
    for _ in range(max_merge_rounds):
        merged = False
        for a in range(len(groups)):
            if not groups[a]:
                continue
            for b in range(a + 1, len(groups)):
                if not groups[b]:
                    continue
                if group_photos[a] & group_photos[b]:
                    continue  # a shared photo proves these are different people
                if float(np.dot(centroids[a], centroids[b])) < merge:
                    continue
                groups[a] = groups[a] + groups[b]
                groups[b] = []
                group_photos[a] |= group_photos[b]
                group_photos[b] = set()
                centroids[a] = centroid_of(groups[a])
                merged = True
        if not merged:
            break

    # ---- stable numbering: order of first appearance ---------------------- #
    kept = [(min(g), g) for g in groups if g]
    kept.sort(key=lambda item: item[0])

    clusters: list[PersonCluster] = []
    for index, (_first, member_positions) in enumerate(kept, start=1):
        members = [ordered[p] for p in member_positions]
        best_face = max(members, key=lambda o: (o.area_ratio, o.sharpness))
        clusters.append(PersonCluster(
            index=index,
            face_ids=sorted(o.face_id for o in members),
            photo_ids=sorted({o.photo_id for o in members}),
            centroid=centroid_of(member_positions),
            max_area_ratio=max(o.area_ratio for o in members),
            representative_face_id=best_face.face_id,
            representative_photo_id=best_face.photo_id,
        ))
    return clusters


@dataclass
class MergeProposal:
    """Two people that look alike but cannot be proven to be one."""

    a: int
    b: int
    similarity: float


def clusters_from_rows(person_rows: list[dict], face_rows: list[dict]) -> list[PersonCluster]:
    """Rebuild clusters from the catalog, using person ids as cluster indices.

    Centroids are recomputed from the stored descriptors, so proposals can be
    made without re-running the analysis.
    """
    by_person: dict[int, list[dict]] = defaultdict(list)
    for face in face_rows:
        person_id = face.get("person_id")
        if person_id is None:
            continue
        by_person[int(person_id)].append(face)

    clusters: list[PersonCluster] = []
    for row in person_rows:
        person_id = int(row["id"])
        faces = by_person.get(person_id, [])
        vectors = [v for v in (embedding_from_bytes(f.get("embedding")) for f in faces)
                   if v is not None]
        if vectors:
            centroid = np.mean(np.stack(vectors), axis=0)
            norm = float(np.linalg.norm(centroid))
            centroid = centroid / norm if norm > 1e-9 else centroid
        else:
            centroid = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
        best = max(faces, key=lambda f: (f.get("area_ratio") or 0.0), default=None)
        clusters.append(PersonCluster(
            index=person_id,
            face_ids=sorted(int(f["face_id"]) for f in faces),
            photo_ids=sorted({int(f["photo_id"]) for f in faces}),
            centroid=centroid,
            max_area_ratio=max((f.get("area_ratio") or 0.0) for f in faces) if faces else 0.0,
            representative_face_id=int(best["face_id"]) if best else 0,
            representative_photo_id=int(best["photo_id"]) if best else 0,
        ))
    clusters.sort(key=lambda c: c.index)
    return clusters


def apply_merge_groups(clusters: list[PersonCluster],
                       groups: list[list[int]]) -> list[PersonCluster]:
    """Force user-confirmed merges back together after a rescan.

    ``groups`` are lists of face ids: whatever clusters those faces landed in are
    the same person. Face ids survive rescans as long as the file was not
    re-analysed, so a group normally re-applies in full; if some ids are gone the
    remaining ones still link the clusters.
    """
    if not clusters or not groups:
        return clusters

    index_of_face = {fid: c.index for c in clusters for fid in c.face_ids}
    parent = {c.index: c.index for c in clusters}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for group in groups:
        members = sorted({index_of_face[fid] for fid in group if fid in index_of_face})
        for other in members[1:]:
            root, node = find(members[0]), find(other)
            if root != node:
                parent[max(root, node)] = min(root, node)

    buckets: dict[int, list[PersonCluster]] = defaultdict(list)
    for cluster in clusters:
        buckets[find(cluster.index)].append(cluster)

    merged: list[PersonCluster] = []
    for members in buckets.values():
        members.sort(key=lambda c: c.index)
        weights = np.array([max(1, len(c.face_ids)) for c in members], dtype=np.float32)
        centroid = np.average(np.stack([c.centroid for c in members]), axis=0,
                              weights=weights)
        norm = float(np.linalg.norm(centroid))
        best = max(members, key=lambda c: (c.max_area_ratio, -c.index))
        merged.append(PersonCluster(
            index=min(c.index for c in members),
            face_ids=sorted({fid for c in members for fid in c.face_ids}),
            photo_ids=sorted({pid for c in members for pid in c.photo_ids}),
            centroid=centroid / norm if norm > 1e-9 else centroid,
            max_area_ratio=max(c.max_area_ratio for c in members),
            representative_face_id=best.representative_face_id,
            representative_photo_id=best.representative_photo_id,
        ))

    merged.sort(key=lambda c: c.index)
    for index, cluster in enumerate(merged, start=1):
        cluster.index = index
    return merged


def propose_merges(clusters: list[PersonCluster],
                   *, min_similarity: float = 0.84, limit: int = 8,
                   dismissed: set[tuple[int, int]] | None = None
                   ) -> list[MergeProposal]:
    """Suggest pairs of people that are probably the same person.

    Auto-clustering is deliberately conservative, so the confirmable cases are
    the ones it refused to join. Two hard rules keep the list trustworthy:

    * a pair never appears in the same photo (that would be two people);
    * a pair the user already marked as different is not suggested again.
    """
    dismissed = dismissed or set()
    if len(clusters) < 2:
        return []

    photos = {c.index: set(c.photo_ids) for c in clusters}
    index_of_face = {fid: c.index for c in clusters for fid in c.face_ids}
    blocked: set[tuple[int, int]] = set()
    for face_a, face_b in dismissed:
        left, right = index_of_face.get(face_a), index_of_face.get(face_b)
        if left is None or right is None or left == right:
            continue
        blocked.add((min(left, right), max(left, right)))

    proposals: list[MergeProposal] = []
    for i, left in enumerate(clusters):
        for right in clusters[i + 1:]:
            if photos[left.index] & photos[right.index]:
                continue
            if (left.index, right.index) in blocked:
                continue
            similarity = float(np.dot(left.centroid, right.centroid))
            if similarity < min_similarity:
                continue
            proposals.append(MergeProposal(left.index, right.index, similarity))

    proposals.sort(key=lambda p: (-p.similarity, p.a, p.b))
    return proposals[:max(1, limit)]


@dataclass
class Promotion:
    photo_id: int
    person: int
    forced: bool = False             # no technically acceptable frame for them


def coverage_promotions(
    clusters: list[PersonCluster],
    signals: dict[int, SelectionSignals],
    keeper_ids: set[int] | list[int],
    *,
    min_photos: int = 2,
    min_face_ratio: float = 0.015,
) -> list[Promotion]:
    """Promote the best frame for any person missing from the keepers.

    A person is *required* when they appear in at least ``min_photos`` distinct
    photos and at least one of their faces is large enough to be a subject
    (``min_face_ratio``). A crowd of background bystanders therefore does not
    force keepers, but a guest photographed twice does.

    Returns promotions in person order. The same frame is never promoted twice,
    and a promoted frame counts as a keeper when the next person is checked, so
    one wide group shot can cover several people at once.
    """
    keepers = set(keeper_ids)
    promotions: list[Promotion] = []

    required = [c for c in clusters
                if len(c.photo_ids) >= max(1, min_photos)
                and c.max_area_ratio >= min_face_ratio]
    for cluster in sorted(required, key=lambda c: c.index):
        candidates = [pid for pid in cluster.photo_ids
                      if pid in signals and signals[pid].reason != "unreadable"]
        if not candidates:
            continue
        if any(pid in keepers for pid in candidates):
            continue

        acceptable = [pid for pid in candidates if signals[pid].reason is None]
        pool = acceptable or candidates
        best = rank([signals[pid] for pid in pool])[0].photo_id
        promotions.append(Promotion(photo_id=best, person=cluster.index,
                                    forced=not acceptable))
        keepers.add(best)

    return promotions


def uncovered_persons(clusters: list[PersonCluster],
                      keeper_ids: set[int] | list[int], *,
                      min_photos: int = 2,
                      min_face_ratio: float = 0.015) -> list[int]:
    """Required people with no keeper of their own (expected to be empty)."""
    keepers = set(keeper_ids)
    return [c.index for c in clusters
            if len(c.photo_ids) >= max(1, min_photos)
            and c.max_area_ratio >= min_face_ratio
            and not any(pid in keepers for pid in c.photo_ids)]


def person_rows(clusters: list[PersonCluster], keeper_ids: set[int] | list[int],
                *, offset: int = 0, min_photos: int = 2,
                min_face_ratio: float = 0.015) -> list[dict]:
    """Catalog rows for the ``persons`` table, plus per-face assignments.

    ``required`` distinguishes a person the guarantee applies to from a face seen
    once in a crowd: only the former is a coverage failure when uncovered.
    """
    keepers = set(keeper_ids)
    rows: list[dict] = []
    for cluster in clusters:
        kept = sum(1 for pid in cluster.photo_ids if pid in keepers)
        required = (len(cluster.photo_ids) >= max(1, min_photos)
                    and cluster.max_area_ratio >= min_face_ratio)
        rows.append({
            "id": offset + cluster.index,
            "index": cluster.index,
            "label": f"Person {cluster.index}",
            "face_count": len(cluster.face_ids),
            "photo_count": len(cluster.photo_ids),
            "keep_count": kept,
            "covered": int(kept > 0),
            "required": int(required),
            "representative_photo_id": cluster.representative_photo_id,
            "representative_face_id": cluster.representative_face_id,
            "max_area_ratio": round(float(cluster.max_area_ratio), 4),
            "face_ids": cluster.face_ids,
        })
    return rows
