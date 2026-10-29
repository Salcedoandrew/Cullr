"""Similar-shot detection: perceptual hashes, capture-time windows, union-find.

Two independent recall signals are combined because each misses what the other
catches:

* **pHash/dHash** catches the same frame re-encoded, resized or copied, and
  frames that are pixel-wise close. It misses pose changes and blinks
  (photo-pipeline's own README says so, which is why it added DINOv2).
* **EXIF capture-time proximity** catches bursts where the content moved too
  much for hashing, which is the common case for people.

Candidates are confirmed by an embedding cosine gate when an embedder is
available (facet's two-stage design: loose Hamming recall, cosine precision).
Stage 1 admits a pair up to ``LOOSE_PREFILTER_DISTANCE`` bits apart and stage 2
requires a cosine at or above ``settings.embedding_threshold``; with an embedder
the candidate loop also stops truncating at the similar-time window, because a
loose hash match the photographer would call the same shot can be minutes
apart. Without an embedder the Hamming gate stands alone and every rule is
exactly what it was before embeddings existed, so the feature works with zero
model downloads.

Complexity is O(n·K) rather than O(n²): photos are ordered by capture time and
only compared against the next K within the similarity time window, plus an
exact-hash map for duplicates that appear anywhere in the shoot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from .config import CullingSettings
from .imaging import hamming, hex_to_hash

MAX_NEIGHBOURS = 60
# facet `utils/duplicate.py` stage 1: the loose Hamming prefilter, confirmed by
# stage 2 (the scene-embedding cosine). Only consulted when embeddings exist.
#
# Measured on the 72-frame Teasers shoot with DINOv2-small vectors: no pair
# inside 12 bits is one the hash rules miss, so this arm changed nothing there.
# The pairs a photographer would call duplicates sat 16-30 bits apart (9 pairs
# within 30 bits at cosine >= 0.88, 6 of those >= 0.90). Widening this constant
# therefore widens recall -- it is a grouping change, not a free tune, so it
# stays at facet's 12 until the labelled set says what the extra merges are.
LOOSE_PREFILTER_DISTANCE = 12


@dataclass
class SimilarityItem:
    photo_id: int
    captured_at: float | None
    phash: str | None
    dhash: str | None
    order: int = 0


@dataclass
class Group:
    member_ids: list[int]
    kind: str = "similar"          # "burst" | "similar"
    lead_id: int | None = None

    def __len__(self) -> int:
        return len(self.member_ids)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class _Edge:
    a: int
    b: int
    strong: bool          # an exact/very close hash match, or a rapid burst
    dt: float | None


@dataclass
class _Pair:
    """A stage-1 candidate: the hash rules' verdict on one pair of frames."""

    a: int
    b: int
    dist_ph: int
    dt: float | None
    burst: bool
    near_dup: bool
    similar: bool
    same_scene: bool
    loose: bool           # only the loose Hamming prefilter admits it

    @property
    def strong(self) -> bool:
        return self.near_dup or self.burst


def _time_ordered(items: list[SimilarityItem]) -> list[SimilarityItem]:
    """Order by capture time; untimed files keep their (path) order at the end."""
    timed = [it for it in items if it.captured_at is not None]
    untimed = [it for it in items if it.captured_at is None]
    timed.sort(key=lambda it: (it.captured_at, it.order))
    untimed.sort(key=lambda it: it.order)
    return timed + untimed


def find_groups(
    items: list[SimilarityItem],
    settings: CullingSettings,
    embeddings: dict[int, "object"] | None = None,
    *,
    embedding_threshold: float | None = None,
) -> list[Group]:
    """Cluster near-identical and same-burst photos.

    ``embeddings`` maps photo_id -> normalised vector supporting ``@`` or, for
    plain sequences, float dot products. It is optional; pass None to use the
    hash-only path.
    """
    if not items:
        return []

    ordered = _time_ordered(items)
    index_of = {it.photo_id: i for i, it in enumerate(ordered)}
    uf = UnionFind(len(ordered))
    edges: list[_Edge] = []

    hashes = [(hex_to_hash(it.phash), hex_to_hash(it.dhash)) for it in ordered]
    cos_threshold = (settings.embedding_threshold
                     if embedding_threshold is None else embedding_threshold)

    has_embeddings = bool(embeddings)

    def cosine_of(a: int, b: int) -> float | None:
        """Cosine between two photos' vectors; None when either is missing."""
        if not embeddings:
            return None
        va, vb = embeddings.get(ordered[a].photo_id), embeddings.get(ordered[b].photo_id)
        if va is None or vb is None:
            return None
        try:
            return float(sum(x * y for x, y in zip(va, vb)))
        except TypeError:
            return None

    def embedding_ok(a: int, b: int) -> bool:
        sim = cosine_of(a, b)
        if sim is None:
            return True  # no evidence against: keep the hash decision
        return sim >= cos_threshold

    for pair in _stage_one_pairs(ordered, hashes, settings,
                                 truncate_at_window=not has_embeddings):
        if not pair.loose:
            if not embedding_ok(pair.a, pair.b):
                continue
        else:
            # An edge the hash rules would not stand up alone needs a real
            # confirmation: no vector means no edge, not "keep the hash".
            sim = cosine_of(pair.a, pair.b)
            if sim is None or sim < cos_threshold:
                continue
        edges.append(_Edge(pair.a, pair.b, strong=pair.strong, dt=pair.dt))

    min_size = max(2, settings.min_group_size)
    if not settings.group_duplicates:
        return []

    for edge in edges:
        uf.union(edge.a, edge.b)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ordered)):
        members[uf.find(i)].append(i)

    edge_by_root: dict[int, list[_Edge]] = defaultdict(list)
    for edge in edges:
        edge_by_root[uf.find(edge.a)].append(edge)

    groups: list[Group] = []
    for root, idxs in members.items():
        if len(idxs) < min_size:
            continue
        root_edges = edge_by_root.get(root, [])
        dts = [e.dt for e in root_edges if e.dt is not None]
        rapid = bool(dts) and max(dts) <= settings.burst_window_seconds
        kind = "burst" if rapid else "similar"
        ids = sorted((ordered[i].photo_id for i in idxs),
                     key=lambda pid: index_of[pid])
        for chunk in _split_oversized(ids, ordered, index_of, settings):
            if len(chunk) >= min_size:
                groups.append(Group(member_ids=chunk, kind=kind))

    groups.sort(key=lambda g: index_of.get(g.member_ids[0], 0))
    return groups


def embedding_candidates(items: list[SimilarityItem],
                         settings: CullingSettings) -> set[int]:
    """Photos whose vector stage 2 can actually consult.

    Only a frame the hash rules already put in a candidate pair can be admitted
    or vetoed by the cosine, so a frame with no such neighbour never needs a
    vector. Encoded here rather than in the scan so the gate has one definition:
    this walks the same stage-1 pairs ``find_groups`` walks, with the loop bound
    embeddings unlock (no truncation at the similar-time window), so the set is
    a superset of the pairs a partially embedded library can decide.
    """
    if not items or not settings.group_duplicates:
        return set()
    ordered = _time_ordered(items)
    hashes = [(hex_to_hash(it.phash), hex_to_hash(it.dhash)) for it in ordered]
    ids: set[int] = set()
    for pair in _stage_one_pairs(ordered, hashes, settings, truncate_at_window=False):
        ids.add(ordered[pair.a].photo_id)
        ids.add(ordered[pair.b].photo_id)
    return ids


def _stage_one_pairs(
    ordered: list[SimilarityItem],
    hashes: list[tuple[int | None, int | None]],
    settings: CullingSettings,
    *,
    truncate_at_window: bool,
) -> Iterator[_Pair]:
    """Every pair the hash rules call a candidate, in the order they are tried.

    Two independent arms, both O(n):

    * **Anywhere in the shoot.** Files can be imported twice with different EXIF
      times (or no EXIF at all), so the time window alone is not enough.
      Blocking on the top and bottom 32 bits of the 64-bit hash keeps this O(n)
      while still finding ~99.9% of pairs within a 6-bit Hamming distance: a
      hash pair within 6 bits differs in the top half with probability < 2%,
      and in both halves with probability < 0.1%.
    * **Neighbour window.** Bursts and same-scene sequences, bounded by
      MAX_NEIGHBOURS. ``truncate_at_window`` is the hash-only bound: once the
      gap exceeds the similar-time window no further pair is worth a look.
      Embeddings lift it, because a loose hash match the photographer would call
      the same shot can be minutes away.
    """
    for a, b in _blocked_pairs(hashes):
        ph_a, ph_b = hashes[a][0], hashes[b][0]
        if ph_a is None or ph_b is None:
            continue
        dist_ph = hamming(ph_a, ph_b)
        if dist_ph > settings.similar_distance:
            continue
        yield _Pair(a, b, dist_ph, None, burst=False,
                    near_dup=dist_ph <= settings.duplicate_distance,
                    similar=False, same_scene=False, loose=False)

    for i, item in enumerate(ordered):
        compared = 0
        for j in range(i + 1, len(ordered)):
            if compared >= MAX_NEIGHBOURS:
                break
            other = ordered[j]
            dt: float | None = None
            if item.captured_at is not None and other.captured_at is not None:
                dt = abs(other.captured_at - item.captured_at)
                if truncate_at_window and dt > settings.similar_window_seconds:
                    break  # ordered by time: everything further is further away
            else:
                # Untimed files: compare against the next few in path order only.
                if j - i > 6:
                    break

            ph_a, dh_a = hashes[i]
            ph_b, dh_b = hashes[j]
            compared += 1
            if ph_a is None or ph_b is None:
                continue

            dist_ph = hamming(ph_a, ph_b)
            dist_dh = hamming(dh_a, dh_b) if dh_a is not None and dh_b is not None else 64

            burst = dt is not None and dt <= settings.burst_window_seconds
            near_dup = dist_ph <= settings.duplicate_distance
            similar = (dist_ph <= settings.similar_distance
                       and dist_dh <= settings.similar_distance + 4)
            same_scene = (dt is not None and dt <= settings.similar_window_seconds
                          and dist_ph <= settings.similar_distance + 6)
            # Stage 1, loose arm: a hash pair the strict rules reject is still a
            # candidate -- stage 2 decides it.
            loose = (dist_ph <= LOOSE_PREFILTER_DISTANCE
                     and not (near_dup or similar or same_scene or burst))
            if not (near_dup or similar or same_scene or burst or loose):
                continue
            yield _Pair(i, j, dist_ph, dt, burst=burst, near_dup=near_dup,
                        similar=similar, same_scene=same_scene, loose=loose)


def _blocked_pairs(hashes: list[tuple[int | None, int | None]]) -> set[tuple[int, int]]:
    """Candidate pairs that share either half of the perceptual hash."""
    pairs: set[tuple[int, int]] = set()
    for shift in (32, 0):
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, (ph, _dh) in enumerate(hashes):
            if ph is None:
                continue
            buckets[(ph >> shift) & 0xFFFFFFFF].append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # A very common bucket (a long burst shares its top bits) would be
            # quadratic; 48 gives 1128 comparisons per bucket and still covers
            # the neighbours that matter.
            members = members[:48]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
    return pairs


def _split_oversized(
    ids: list[int],
    ordered: list[SimilarityItem],
    index_of: dict[int, int],
    settings: CullingSettings,
) -> list[list[int]]:
    """Break single-linkage chaining.

    Union-find over time-adjacent frames will happily chain a 30-minute ceremony
    into one 400-photo "group", which makes best-of-group meaningless. Groups are
    therefore chunked by capture time as well as by member count (facet caps its
    similarity groups at 50; 24 keeps the compare view usable and matches a long
    burst at 10 fps).
    """
    max_size = 24
    span = max(6.0, settings.burst_window_seconds * 4.0) * 4

    if len(ids) <= 2:
        return [ids]

    times = [ordered[index_of[pid]].captured_at for pid in ids]
    timed = [t for t in times if t is not None]
    if len(timed) < 2:
        return [ids[i:i + max_size] for i in range(0, len(ids), max_size)]

    chunks: list[list[int]] = []
    current: list[int] = []
    start = timed[0]
    for pid, ts in zip(ids, times):
        if ts is None:
            current.append(pid)
            continue
        if current and (ts - start > span or len(current) >= max_size):
            chunks.append(current)
            current = []
            start = ts
        current.append(pid)
    if current:
        chunks.append(current)
    return chunks
