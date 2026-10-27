"""Scan pipeline: ingest -> analyse -> group -> score -> decide.

The expensive per-file work (decode, technical metrics, faces, hashes) runs in a
thread pool because OpenCV/numpy release the GIL for these operations, which is
the same reason photo-pipeline's `technical_screen` uses a ThreadPool. MediaPipe
is stateful, so each worker thread gets its own landmarker (see
`metrics.faces.FaceAnalyzer`).

Everything after analysis is pure: `plan()` takes photo records and settings and
returns groups, verdicts and leads. That makes the culling logic testable
without decoding a single image.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import ANALYSIS_LONG_EDGE, ANALYSIS_VERSION, CullingSettings
from .db import Store, utcnow
from .imaging import (
    ImageMeta, content_key, dhash, hashes_to_hex, iter_images, load_image, phash,
)
from .metrics.faces import Face, FaceAnalyzer, FaceMetrics
from .metrics.faceid import embedding_from_bytes
from .metrics.technical import TechnicalMetrics
from .persons import (
    FaceObservation, PersonCluster, apply_merge_groups, cluster_faces,
    coverage_promotions, person_rows as build_person_rows,
)
from .scoring import PhotoScore, group_median_sharpness, score_photo
from .selection import GroupSelection, SelectionSignals, select
from .similarity import Group, SimilarityItem, find_groups

ProgressCb = Callable[[str, int, int, str], None]


@dataclass
class AnalysedPhoto:
    photo_id: int
    path: str
    filename: str
    meta: ImageMeta
    tech: TechnicalMetrics | None
    faces: FaceMetrics
    phash: str | None = None
    dhash: str | None = None
    error: str | None = None
    content_key: str = ""

    def analysis_payload(self) -> dict:
        tech = self.tech
        faces = self.faces
        return {
            "width": self.meta.width,
            "height": self.meta.height,
            "captured_at": self.meta.captured_at,
            "camera": self.meta.camera,
            "lens": self.meta.lens,
            "iso": self.meta.iso,
            "shutter": self.meta.shutter,
            "aperture": self.meta.aperture,
            "focal_len": self.meta.focal_len,
            "sharpness": tech.sharpness if tech else None,
            "tenengrad": tech.tenengrad if tech else None,
            "clip_hi": tech.clip_hi if tech else None,
            "clip_lo": tech.clip_lo if tech else None,
            "mean_luma": tech.mean_luma if tech else None,
            "noise": tech.noise if tech else None,
            "contrast": tech.contrast if tech else None,
            "face_count": faces.count,
            "eyes_open": faces.eyes_open,
            "worst_eyes_open": faces.worst_eyes_open,
            "blink": int(bool(faces.usable) and (faces.worst_eyes_open or 10) <= 4.0),
            "smile": faces.smile,
            "face_sharpness": faces.face_sharpness,
            "frontal": faces.frontal,
            "face_ratio": faces.area_ratio,
            "phash": self.phash,
            "dhash": self.dhash,
            "error": self.error,
            # Descriptors travel as arrays into the catalog, never to the API.
            "faces": [
                {**face.to_dict(), "embedding": face.embedding}
                for face in faces.faces
            ],
        }


def analyse_file(path: Path, analyzer: FaceAnalyzer,
                 long_edge: int = ANALYSIS_LONG_EDGE) -> tuple[ImageMeta, TechnicalMetrics | None,
                                                               FaceMetrics, str | None,
                                                               str | None, str | None]:
    """Decode and measure one file. Never raises."""
    from .metrics.technical import analyse as analyse_technical

    bgr, meta, error = load_image(path, long_edge)
    if bgr is None:
        return meta, None, FaceMetrics(faces=[], available=analyzer.available), error, None, None

    tech = analyse_technical(bgr)
    faces = analyzer.analyze(bgr)
    from .imaging import gray_of

    gray = gray_of(bgr)
    p_hash = hashes_to_hex(phash(gray))
    d_hash = hashes_to_hex(dhash(gray))
    return meta, tech, faces, None, p_hash, d_hash


@dataclass
class PlanResult:
    """Everything the culling pass decided, before person coverage runs."""

    groups: list[dict]
    scores: dict[int, PhotoScore]
    signals: dict[int, SelectionSignals]      # every photo, for coverage ranking
    selections: dict[int, GroupSelection]
    membership: dict[int, list[int]]

    def keeper_ids(self) -> set[int]:
        keepers: set[int] = set()
        for selection in self.selections.values():
            keepers.update(selection.keeper_ids)
        return keepers


def _selection_signals(photo: AnalysedPhoto, score: PhotoScore) -> SelectionSignals:
    face_sharp = score.axes.get("face_sharpness")
    return SelectionSignals(
        photo_id=photo.photo_id,
        quality=score.quality,
        eyes_open=photo.faces.worst_eyes_open,
        face_sharpness=None if face_sharp is None else face_sharp * 100.0,
        smile=photo.faces.smile,
        reason=score.reason,
        captured_at=photo.meta.captured_at,
        order=photo.photo_id,
    )


def plan(photos: list[AnalysedPhoto], settings: CullingSettings) -> PlanResult:
    """Group, score and select in one pure pass.

    ``PlanResult.groups`` are the DB-ready group records with stable 1-based
    ids; ``signals`` covers every photo so person coverage can rank frames that
    are not group leads.
    """
    items = [
        SimilarityItem(
            photo_id=p.photo_id,
            captured_at=p.meta.captured_at,
            phash=p.phash,
            dhash=p.dhash,
            order=p.photo_id,
        )
        for p in photos
    ]
    groups: list[Group] = find_groups(items, settings)
    by_id = {p.photo_id: p for p in photos}

    scores: dict[int, PhotoScore] = {}
    signals_all: dict[int, SelectionSignals] = {}
    selections: dict[int, GroupSelection] = {}
    group_rows: list[dict] = []
    membership: dict[int, list[int]] = {}

    grouped_ids: set[int] = set()
    for idx, group in enumerate(groups, start=1):
        members = [by_id[pid] for pid in group.member_ids if pid in by_id]
        if len(members) < 2:
            continue
        median = group_median_sharpness([m.tech.sharpness for m in members if m.tech])
        signals: list[SelectionSignals] = []
        for member in members:
            score = score_photo(
                member.tech, member.faces, settings, median,
                relative_to_group=True,
            ) if member.tech else PhotoScore(
                verdict="reject", reason="unreadable")
            scores[member.photo_id] = score
            signalled = _selection_signals(member, score)
            signals.append(signalled)
            signals_all[member.photo_id] = signalled
        gates = settings.gates()
        selection = select([s.photo_id for s in signals], signals,
                           gates.keep_min_per_group)
        selections[idx] = selection
        membership[idx] = [m.photo_id for m in members]
        grouped_ids.update(group.member_ids)
        lead_score = max((s.quality for s in signals
                          if s.photo_id == selection.lead_id), default=0.0)
        group_rows.append({
            "id": idx,
            "kind": group.kind,
            "size": len(members),
            "lead_id": selection.lead_id,
            "lead_score": round(lead_score, 4),
        })

    # Ungrouped photos are scored on their own (no relative blur gate).
    for photo in photos:
        if photo.photo_id in grouped_ids:
            continue
        if photo.tech is None:
            scores[photo.photo_id] = PhotoScore(verdict="reject", reason="unreadable")
        else:
            scores[photo.photo_id] = score_photo(photo.tech, photo.faces, settings)
        signals_all[photo.photo_id] = _selection_signals(photo,
                                                         scores[photo.photo_id])

    return PlanResult(groups=group_rows, scores=scores, signals=signals_all,
                      selections=selections, membership=membership)


# --------------------------------------------------------------------------- #
# People across the shoot
# --------------------------------------------------------------------------- #

def cluster_people(store: Store, library_id: int,
                   settings: CullingSettings) -> list[PersonCluster]:
    """Cluster every describable face in the library into people.

    Reads the catalog rather than the in-memory photos so reused analysis (files
    that did not change since the last scan) contributes identities too.
    """
    if not settings.person_coverage:
        return []
    observations: list[FaceObservation] = []
    for row in store.faces_with_embeddings(library_id):
        vector = embedding_from_bytes(row.get("embedding"))
        if vector is None:
            continue
        observations.append(FaceObservation(
            face_id=int(row["face_id"]),
            photo_id=int(row["photo_id"]),
            captured_at=row.get("captured_at"),
            vector=vector,
            area_ratio=float(row.get("area_ratio") or 0.0),
            sharpness=float(row.get("sharpness") or 0.0),
            eyes_open=row.get("eyes_open"),
        ))
    clusters = cluster_faces(observations, match=settings.person_match,
                             merge=settings.person_merge)
    # User-confirmed merges are hard constraints, not suggestions: re-apply them
    # so a rescan never splits a person the photographer has curated.
    return apply_merge_groups(clusters, store.person_links(library_id, "merge"))


# --------------------------------------------------------------------------- #
# Scan orchestration
# --------------------------------------------------------------------------- #

@dataclass
class ScanResult:
    library_id: int
    total: int = 0
    analysed: int = 0
    reused: int = 0
    failed: int = 0
    groups: int = 0
    keepers: int = 0
    persons: int = 0
    uncovered: int = 0
    ingested: int = 0       # files copied into the working copy / backup
    ingest_reused: int = 0  # files already there, left alone
    cancelled: bool = False
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)


def ingest_copy(source: Path, dest: Path, *, recursive: bool, extensions: set[str],
                progress: ProgressCb | None = None,
                cancel_event: threading.Event | None = None) -> tuple[int, int]:
    """Copy the import set from ``source`` into ``dest``, preserving layout.

    Only files the import admits (the extension filter, the recursive flag)
    are copied, and only when the destination is missing one or the source
    changed (size or mtime) — a rescan reuses the working copy instead of
    copying it again. ``copy2`` preserves mtime, which is what the content key
    is built from, so a copied file is still recognised as unchanged. The
    source is only ever read: originals are never modified.
    """
    from .imaging import iter_images

    progress = progress or (lambda *_: None)
    cancel_event = cancel_event or threading.Event()
    files = iter_images(source, recursive=recursive, extensions=extensions)
    copied = reused = 0
    for index, path in enumerate(files, start=1):
        if cancel_event.is_set():
            break
        target = dest / path.relative_to(source)
        try:
            stat = path.stat()
            if (target.exists() and target.is_file()
                    and target.stat().st_size == stat.st_size
                    and int(target.stat().st_mtime) == int(stat.st_mtime)):
                reused += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
        except OSError as exc:
            raise ValueError(f"cannot copy {path.name}: {exc}") from exc
        if index % 100 == 0 or index == len(files):
            progress("ingesting", index, len(files),
                     f"Ingested {index}/{len(files)} photos")
    return copied, reused


def _worker_count() -> int:
    return max(1, min(8, (os.cpu_count() or 4)))


def cull_to_target(updates: list[tuple], ratio: float,
                   protected: set[int]) -> list[tuple]:
    """Keep roughly ``ratio`` of the shoot, ranked by measured quality.

    "Cull to target" asks a different question from the gates: not "is this
    frame acceptable" but "how many frames do I want". Keepers are ranked by
    quality and everything past the target is rejected with reason ``target``,
    so the sheet can still say why a frame went.

    Coverage beats the target: a frame that is someone's only keeper is never
    dropped, which is why the achieved count can sit slightly above it.
    """
    if not updates:
        return updates
    limit = max(1, round(max(0.0, min(1.0, float(ratio))) * len(updates)))
    keeper_ids = {row[6] for row in updates if row[1] == "keep"}
    if len(keeper_ids) <= limit:
        return updates
    ranked = sorted((row for row in updates if row[1] == "keep"),
                    key=lambda row: (-row[0], row[6]))
    winners = {row[6] for row in ranked[:limit]} | (protected & keeper_ids)
    capped: list[tuple] = []
    for quality, verdict, reason, gid, is_lead, covered_for, photo_id in updates:
        if verdict == "keep" and photo_id not in winners:
            verdict, reason = "reject", "target"
        capped.append((quality, verdict, reason, gid, is_lead, covered_for, photo_id))
    return capped


def run_scan(
    store: Store,
    library_id: int,
    settings: CullingSettings,
    progress: ProgressCb | None = None,
    cancel_event: threading.Event | None = None,
    force: bool = False,
) -> ScanResult:
    started = time.monotonic()
    progress = progress or (lambda *_: None)
    cancel_event = cancel_event or threading.Event()
    library = store.get_library(library_id)
    if not library:
        raise ValueError(f"library {library_id} not found")

    root = Path(library["root_path"])
    result = ScanResult(library_id=library_id)

    # ---- 0. ingest: work on a copy, never on the originals --------------- #
    # Import options are sticky per library (they live in settings_json), so
    # changing ingest mode on an existing library is not a supported flow —
    # the working copy is the library once ingest is on.
    effective_root = root
    if settings.ingest:
        working = store.ingest_root(library_id)
        extensions = settings.import_extensions()
        if root.is_dir():
            copied, reused = ingest_copy(root, working, recursive=settings.recursive,
                                         extensions=extensions, progress=progress,
                                         cancel_event=cancel_event)
            result.ingested += copied
            result.ingest_reused += reused
            if settings.backup_dir:
                backup = Path(settings.backup_dir).expanduser()
                if not backup.is_dir():
                    raise ValueError(f"backup folder is not available: {backup}")
                if backup.resolve() == root.resolve() or root.resolve() in backup.resolve().parents:
                    raise ValueError("backup folder must not be inside the source folder")
                copied, reused = ingest_copy(root, backup, recursive=settings.recursive,
                                             extensions=extensions, progress=progress,
                                             cancel_event=cancel_event)
                result.ingested += copied
                result.ingest_reused += reused
            effective_root = working
        elif working.is_dir() and any(working.iterdir()):
            # The source is gone (a reformatted card) but the working copy is
            # still here: scan that. A rescan reuses the same working copy.
            progress("ingesting", 0, 0, "Source folder is gone — using the working copy")
            effective_root = working
        if cancel_event.is_set():
            result.cancelled = True
            result.elapsed = time.monotonic() - started
            return result

    progress("discovering", 0, 0, "Finding images")

    files = iter_images(effective_root, recursive=settings.recursive,
                        extensions=settings.import_extensions())
    result.total = len(files)
    if not files:
        progress("done", 0, 0, "No images found")
        return result

    # ---- 1. register files, decide what needs analysis -------------------- #
    pending: list[tuple[Path, int, str]] = []
    photos: list[AnalysedPhoto] = []
    stored_by_path = {row["path"]: row for row in store.stored_analysis(library_id)}
    faces_by_photo = store.faces_for([r["id"] for r in stored_by_path.values()])

    for index, path in enumerate(files, start=1):
        try:
            st = path.stat()
            key = content_key(path)
        except OSError:
            continue
        photo_id = store.upsert_photo(library_id, path, key, st.st_size, st.st_mtime)
        row = store.stored_analysis_for(photo_id)
        reuse = (not force and row is not None and row["content_key"] == key
                 and row["analysis_version"] == ANALYSIS_VERSION
                 and row["error"] is None and row["sharpness"] is not None)
        if reuse:
            photos.append(_from_row(row, faces_by_photo.get(photo_id, [])))
            result.reused += 1
        else:
            pending.append((path, photo_id, key))
        if index % 200 == 0:
            progress("discovering", index, len(files), "Registering files")

    # ---- 2. analyse pending files ----------------------------------------- #
    analyzer = FaceAnalyzer()
    total_pending = len(pending)
    progress("analyzing", 0, total_pending,
             f"Analysing {total_pending} photos" + ("" if analyzer.available
                                                    else " (faces unavailable)"))
    done = 0
    lock = threading.Lock()

    def analyse_one(item: tuple[Path, int, str]) -> AnalysedPhoto:
        path, photo_id, key = item
        meta, tech, faces, error, p_hash, d_hash = analyse_file(path, analyzer)
        return AnalysedPhoto(
            photo_id=photo_id, path=str(path), filename=path.name, meta=meta,
            tech=tech, faces=faces, phash=p_hash, dhash=d_hash, error=error,
            content_key=key,
        )

    def handle(future, item) -> None:
        nonlocal done
        path, photo_id, key = item
        try:
            photo = future.result()
        except Exception as exc:  # noqa: BLE001 - a single bad file must not abort
            photo = AnalysedPhoto(
                photo_id=photo_id, path=str(path), filename=path.name,
                meta=ImageMeta(width=0, height=0), tech=None,
                faces=FaceMetrics(faces=[], available=False),
                error=f"{type(exc).__name__}: {exc}", content_key=key,
            )
        store.update_analysis(photo_id, photo.analysis_payload(), ANALYSIS_VERSION)
        if photo.error:
            result.failed += 1
            if len(result.errors) < 25:
                result.errors.append(f"{photo.filename}: {photo.error}")
        with lock:
            done += 1
            result.analysed += 1
            if done % 10 == 0 or done == total_pending:
                progress("analyzing", done, total_pending,
                         f"Analysed {done}/{total_pending}")
        photos.append(photo)

    if pending:
        with ThreadPoolExecutor(max_workers=_worker_count()) as pool:
            futures = {pool.submit(analyse_one, item): item for item in pending}
            for future in as_completed(futures):
                if cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    break
                handle(future, futures[future])
        if cancel_event.is_set():
            result.cancelled = True
            result.elapsed = time.monotonic() - started
            return result

    # ---- 3. group, score, select ------------------------------------------ #
    progress("grouping", 0, len(photos), "Grouping similar shots")
    planned = plan(photos, settings)
    group_rows, scores = planned.groups, planned.scores
    selections, membership = planned.selections, planned.membership

    # ---- 4. people: cluster identities, then guarantee coverage ----------- #
    progress("persons", 0, len(photos), "Clustering people")
    clusters = cluster_people(store, library_id, settings)
    group_keepers = planned.keeper_ids()
    promotions = coverage_promotions(
        clusters, planned.signals, group_keepers,
        min_photos=settings.person_min_photos,
        min_face_ratio=settings.person_min_face_ratio,
    ) if clusters else []
    # Person ids are namespaced per library, exactly like group ids, so the
    # photo-level pointer must use the same global id the persons rows get.
    person_offset = library_id * Store.PERSON_ID_STRIDE
    covered = {p.photo_id: person_offset + p.person for p in promotions}

    # Group ids live in a per-library namespace: `groups.id` is a global primary
    # key, so library 2 must not reuse library 1's group 1.
    offset = library_id * Store.GROUP_ID_STRIDE
    updates: list[tuple] = []
    group_of: dict[int, int] = {}
    lead_of: dict[int, int] = {}
    for row in group_rows:
        local_id = row["id"]
        row["id"] = local_id + offset
        for member_id in membership[local_id]:
            group_of[member_id] = row["id"]
        lead_of[row["id"]] = selections[local_id].lead_id

    keepers = 0
    final_keepers: set[int] = set()
    for photo in photos:
        score = scores.get(photo.photo_id)
        if score is None:
            continue
        reason = score.reason
        verdict = score.verdict
        gid = group_of.get(photo.photo_id)
        is_lead = 0
        if gid is not None:
            is_lead = 1 if lead_of.get(gid) == photo.photo_id else 0
            if photo.photo_id in group_keepers:
                verdict, reason = "keep", None
            elif reason is None:
                verdict, reason = "reject", "duplicate"
        covered_for = covered.get(photo.photo_id)
        if covered_for is not None:
            # Coverage only ever adds a keeper. A technical reason is preserved
            # so the UI can explain the compromise; "duplicate" is not, because
            # this frame is now the keeper for that person.
            verdict = "keep"
            if reason == "duplicate":
                reason = None
        if verdict == "keep":
            keepers += 1
            final_keepers.add(photo.photo_id)
        updates.append((score.quality, verdict, reason, gid, is_lead,
                        covered_for, photo.photo_id))

    # ---- 5. optional target cap ------------------------------------------ #
    if settings.target_ratio:
        updates = cull_to_target(updates, settings.target_ratio, set(covered))
        keepers = sum(1 for row in updates if row[1] == "keep")
        final_keepers = {row[6] for row in updates if row[1] == "keep"}

    store.replace_groups(library_id, group_rows)
    store.update_decisions(updates)
    store.update_library_settings(library_id, settings.to_dict())

    if clusters:
        rows = build_person_rows(
            clusters, final_keepers, offset=person_offset,
            min_photos=settings.person_min_photos,
            min_face_ratio=settings.person_min_face_ratio,
        )
        store.replace_persons(
            library_id, rows,
            [(row["id"], face_id) for row in rows for face_id in row["face_ids"]],
        )
        result.persons = len(rows)
        result.uncovered = sum(1 for row in rows
                                if row["required"] and not row["covered"])
    else:
        store.replace_persons(library_id, [])

    result.groups = len(group_rows)
    result.keepers = keepers
    result.elapsed = time.monotonic() - started
    people = f" · {result.persons} people" if result.persons else ""
    ingested = ""
    if result.ingested or result.ingest_reused:
        ingested = (f" · ingested {result.ingested} photos "
                    f"({result.ingest_reused} already in the working copy)")
    progress("done", len(photos), len(photos),
             f"{keepers} keepers · {len(group_rows)} groups{people} · "
             f"{result.elapsed:.1f}s{ingested}")
    return result


def _from_row(row: dict, faces: list[dict]) -> AnalysedPhoto:
    tech = TechnicalMetrics(
        sharpness=row.get("sharpness") or 0.0,
        tenengrad=row.get("tenengrad") or 0.0,
        clip_hi=row.get("clip_hi") or 0.0,
        clip_lo=row.get("clip_lo") or 0.0,
        mean_luma=row.get("mean_luma") or 0.0,
        p95_luma=0.0,
        noise=row.get("noise") or 0.0,
        contrast=row.get("contrast") or 0.0,
    )
    face_objs = [
        Face(
            x=f.get("x") or 0, y=f.get("y") or 0, w=f.get("w") or 0, h=f.get("h") or 0,
            eyes_open=f.get("eyes_open"), smile=f.get("smile"),
            yaw=f.get("yaw"), pitch=f.get("pitch"),
            sharpness=f.get("sharpness") or 0.0,
            area_ratio=f.get("area_ratio") or 0.0,
            clip_hi=f.get("clip_hi"),
            clip_lo=f.get("clip_lo"),
            mean_luma=f.get("mean_luma"),
        )
        for f in faces
    ]
    meta = ImageMeta(
        width=row.get("width") or 0, height=row.get("height") or 0,
        captured_at=row.get("captured_at"), camera=row.get("camera"),
        lens=row.get("lens"), iso=row.get("iso"), shutter=row.get("shutter"),
        aperture=row.get("aperture"), focal_len=row.get("focal_len"),
    )
    return AnalysedPhoto(
        photo_id=row["id"], path=row["path"], filename=row["filename"],
        meta=meta, tech=tech, faces=FaceMetrics(faces=face_objs, available=True),
        phash=row.get("phash"), dhash=row.get("dhash"), error=row.get("error"),
        content_key=row.get("content_key") or "",
    )
