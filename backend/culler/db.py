"""SQLite catalog: libraries, jobs, photos, faces and groups."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .metrics.faceid import embedding_bytes

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS libraries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    settings_json TEXT,
    -- Last export of this shoot, so the panel can prefill it. In the catalog
    -- rather than the browser: the destination belongs to the shoot.
    export_mode TEXT,
    export_dest TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    message TEXT,
    total INTEGER NOT NULL DEFAULT 0,
    done INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    -- Files the scan could not read at all, and the reason from the first of
    -- them. Distinct from `error`, which means the scan itself failed: a shoot
    -- where two frames are unreadable is a finished cull that needs explaining,
    -- not a crash.
    error_count INTEGER NOT NULL DEFAULT 0,
    error_sample TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER,
    mtime REAL,
    content_key TEXT,
    width INTEGER, height INTEGER,
    captured_at REAL,
    camera TEXT, lens TEXT,
    iso INTEGER, shutter REAL, aperture REAL, focal_len REAL,
    sharpness REAL, tenengrad REAL, clip_hi REAL, clip_lo REAL,
    mean_luma REAL, noise REAL, contrast REAL,
    subject_sharpness REAL, bg_separation REAL,
    bimodality REAL, shadow_clipped INTEGER, highlight_clipped INTEGER,
    color_harmony REAL,
    face_count INTEGER DEFAULT 0,
    eyes_open REAL, worst_eyes_open REAL, blink INTEGER DEFAULT 0,
    smile REAL, face_sharpness REAL, frontal REAL, face_ratio REAL,
    phash TEXT, dhash TEXT,
    scene_embedding BLOB,
    quality REAL,
    verdict TEXT DEFAULT 'keep',
    reason TEXT,
    user_decision TEXT,
    star INTEGER DEFAULT 0,
    group_id INTEGER,
    is_lead INTEGER DEFAULT 0,
    covered_person INTEGER,
    error TEXT,
    analysis_version INTEGER DEFAULT 0,
    analyzed_at TEXT,
    UNIQUE(library_id, path)
);
CREATE INDEX IF NOT EXISTS idx_photos_library ON photos(library_id);
CREATE INDEX IF NOT EXISTS idx_photos_group ON photos(group_id);
CREATE INDEX IF NOT EXISTS idx_photos_verdict ON photos(library_id, verdict);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    eyes_open REAL, smile REAL, yaw REAL, pitch REAL,
    sharpness REAL, area_ratio REAL,
    clip_hi REAL, clip_lo REAL, mean_luma REAL,
    embedding BLOB,
    person_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    face_count INTEGER NOT NULL DEFAULT 0,
    photo_count INTEGER NOT NULL DEFAULT 0,
    keep_count INTEGER NOT NULL DEFAULT 0,
    covered INTEGER NOT NULL DEFAULT 0,
    required INTEGER NOT NULL DEFAULT 0,
    representative_photo_id INTEGER,
    representative_face_id INTEGER,
    max_area_ratio REAL
);
CREATE INDEX IF NOT EXISTS idx_persons_library ON persons(library_id);

-- Curated identity decisions that must survive a rescan:
--   kind='merge'    : face ids the photographer confirmed as one person
--   kind='distinct' : a face pair confirmed as two different people
CREATE TABLE IF NOT EXISTS person_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    face_ids TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_links_library ON person_links(library_id, kind);

CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id INTEGER NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    lead_id INTEGER,
    lead_score REAL
);
CREATE INDEX IF NOT EXISTS idx_groups_library ON groups(library_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_data_dir() -> Path:
    override = os.environ.get("CULLR_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if getattr(sys, "frozen", False):
        # A bundled app must not write inside its own (read-only) bundle.
        return Path.home() / "Library" / "Application Support" / "Cullr"
    return Path.home() / ".cullr"


class Store:
    """Thin synchronous wrapper. The API is single-process and low-concurrency."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive migrations for catalogs created by an older build."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(faces)")}
        for column in ("clip_hi", "clip_lo", "mean_luma"):
            if column not in existing:
                conn.execute(f"ALTER TABLE faces ADD COLUMN {column} REAL")
        for column, kind in (("embedding", "BLOB"), ("person_id", "INTEGER")):
            if column not in existing:
                conn.execute(f"ALTER TABLE faces ADD COLUMN {column} {kind}")
        photos = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
        if "covered_person" not in photos:
            conn.execute("ALTER TABLE photos ADD COLUMN covered_person INTEGER")
        for column, kind in (("subject_sharpness", "REAL"), ("bg_separation", "REAL"),
                             ("bimodality", "REAL"), ("shadow_clipped", "INTEGER"),
                             ("highlight_clipped", "INTEGER"),
                             ("color_harmony", "REAL"),
                             ("scene_embedding", "BLOB")):
            if column not in photos:
                conn.execute(f"ALTER TABLE photos ADD COLUMN {column} {kind}")
        people = {row["name"] for row in conn.execute("PRAGMA table_info(persons)")}
        if "required" not in people:
            conn.execute("ALTER TABLE persons ADD COLUMN required INTEGER NOT NULL DEFAULT 0")
        libraries = {row["name"] for row in conn.execute("PRAGMA table_info(libraries)")}
        for column in ("export_mode", "export_dest"):
            if column not in libraries:
                conn.execute(f"ALTER TABLE libraries ADD COLUMN {column} TEXT")
        jobs = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column, kind in (("error_count", "INTEGER NOT NULL DEFAULT 0"),
                             ("error_sample", "TEXT")):
            if column not in jobs:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {kind}")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Libraries
    # ------------------------------------------------------------------ #
    def create_or_get_library(self, root_path: str, settings: dict | None = None) -> dict:
        root = str(Path(root_path).expanduser().resolve())
        name = Path(root).name or root
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM libraries WHERE root_path = ?", (root,)
            ).fetchone()
            if row:
                return dict(row)
            cur = conn.execute(
                "INSERT INTO libraries (root_path, name, created_at, settings_json) "
                "VALUES (?, ?, ?, ?)",
                (root, name, utcnow(), json.dumps(settings or {})),
            )
            return dict(conn.execute(
                "SELECT * FROM libraries WHERE id = ?", (cur.lastrowid,)
            ).fetchone())

    def get_library(self, library_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM libraries WHERE id = ?",
                               (library_id,)).fetchone()
        return dict(row) if row else None

    def list_libraries(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT l.*, (SELECT COUNT(*) FROM photos p WHERE p.library_id = l.id) "
                "AS photo_count FROM libraries l ORDER BY l.id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def ingest_root(self, library_id: int) -> Path:
        """Where an ingested library's working copy lives.

        Deterministic from the catalog location, so the library remembers its
        working copy without another column: source folder is ``root_path``,
        working copy is ``<data dir>/ingest/<library id>``. A rescan reuses it.
        """
        return Path(self.db_path).parent / "ingest" / str(library_id)

    def update_library_settings(self, library_id: int, settings: dict) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE libraries SET settings_json = ? WHERE id = ?",
                         (json.dumps(settings), library_id))

    def update_library_export(self, library_id: int, mode: str, dest: str = "") -> None:
        """Remember where this shoot's last export went.

        A sidecar export has no destination, so it moves the mode on without
        forgetting the folder the photographer delivered to last time.
        """
        cleaned = dest.strip()
        with self.connect() as conn:
            if cleaned:
                conn.execute(
                    "UPDATE libraries SET export_mode = ?, export_dest = ? WHERE id = ?",
                    (mode, cleaned, library_id),
                )
            else:
                conn.execute("UPDATE libraries SET export_mode = ? WHERE id = ?",
                             (mode, library_id))

    def delete_library(self, library_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM libraries WHERE id = ?", (library_id,))

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #
    def create_job(self, library_id: int) -> dict:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (library_id, status, stage, started_at) VALUES (?, 'running', 'starting', ?)",
                (library_id, utcnow()),
            )
            return dict(conn.execute("SELECT * FROM jobs WHERE id = ?",
                                     (cur.lastrowid,)).fetchone())

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"status", "stage", "message", "total", "done", "error",
                   "error_count", "error_sample", "started_at", "finished_at"}
        cols = [k for k in fields if k in allowed]
        sets = ", ".join(f"{c} = ?" for c in cols)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?",
                         [fields[c] for c in cols] + [job_id])

    def get_job(self, job_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def latest_job(self, library_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE library_id = ? ORDER BY id DESC LIMIT 1",
                (library_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Photos
    # ------------------------------------------------------------------ #
    def upsert_photo(self, library_id: int, path: Path, content_key: str,
                     size_bytes: int, mtime: float) -> int:
        rel = path.name
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM photos WHERE library_id = ? AND path = ?",
                (library_id, str(path)),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE photos SET content_key = ?, size_bytes = ?, mtime = ?, filename = ? "
                    "WHERE id = ?",
                    (content_key, size_bytes, mtime, rel, row["id"]),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO photos (library_id, path, filename, size_bytes, mtime, content_key) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (library_id, str(path), rel, size_bytes, mtime, content_key),
            )
            return int(cur.lastrowid)

    def stored_analysis_for(self, photo_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()
        return dict(row) if row else None

    def photo_row(self, photo_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return dict(row) if row else None

    def update_analysis(self, photo_id: int, data: dict, version: int) -> None:
        cols = [
            "width", "height", "captured_at", "camera", "lens", "iso", "shutter",
            "aperture", "focal_len", "sharpness", "tenengrad", "clip_hi", "clip_lo",
            "mean_luma", "noise", "contrast", "subject_sharpness",
            "bg_separation", "bimodality", "shadow_clipped",
            "highlight_clipped", "color_harmony", "face_count", "eyes_open",
            "worst_eyes_open", "blink", "smile", "face_sharpness", "frontal",
            "face_ratio", "phash", "dhash", "scene_embedding", "error",
        ]
        payload = {k: data.get(k) for k in cols}
        payload["analysis_version"] = version
        payload["analyzed_at"] = utcnow()
        sets = ", ".join(f"{k} = ?" for k in payload)
        with self.connect() as conn:
            conn.execute(f"UPDATE photos SET {sets} WHERE id = ?",
                         list(payload.values()) + [photo_id])
            conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            for face in data.get("faces", []):
                conn.execute(
                    "INSERT INTO faces (photo_id, x, y, w, h, eyes_open, smile, yaw, "
                    "pitch, sharpness, area_ratio, clip_hi, clip_lo, mean_luma, "
                    "embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (photo_id, face.get("x"), face.get("y"), face.get("w"),
                     face.get("h"), face.get("eyes_open"), face.get("smile"),
                     face.get("yaw"), face.get("pitch"), face.get("sharpness"),
                     face.get("area_ratio"), face.get("clip_hi"),
                     face.get("clip_lo"), face.get("mean_luma"),
                     embedding_bytes(face.get("embedding"))),
                )

    def stored_analysis(self, library_id: int) -> list[dict]:
        """Rows needed for grouping/scoring without re-decoding the files."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, path, filename, captured_at, phash, dhash, sharpness, "
                "tenengrad, clip_hi, clip_lo, mean_luma, noise, contrast, "
                "subject_sharpness, bg_separation, bimodality, shadow_clipped, "
                "highlight_clipped, color_harmony, "
                "face_count, eyes_open, worst_eyes_open, smile, face_sharpness, "
                "frontal, face_ratio, scene_embedding, error, analysis_version, "
                "content_key "
                "FROM photos WHERE library_id = ? ORDER BY path", (library_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_decisions(self, rows: Iterable[tuple]) -> None:
        """rows: (quality, verdict, reason, group_id, is_lead, covered_person, photo_id)."""
        with self.connect() as conn:
            conn.executemany(
                "UPDATE photos SET quality = ?, verdict = ?, reason = ?, "
                "group_id = ?, is_lead = ?, covered_person = ? WHERE id = ?",
                list(rows),
            )

    # Group ids are globally unique (they are the primary key of `groups`), but
    # the planner numbers groups from 1 per library. The library id provides the
    # namespace, so two libraries never collide.
    GROUP_ID_STRIDE = 1_000_000

    def replace_groups(self, library_id: int, groups: list[dict]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE photos SET group_id = NULL, is_lead = 0 WHERE library_id = ?",
                         (library_id,))
            conn.execute("DELETE FROM groups WHERE library_id = ?", (library_id,))
            for g in groups:
                conn.execute(
                    "INSERT INTO groups (id, library_id, kind, size, lead_id, lead_score) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (g["id"], library_id, g["kind"], g["size"], g["lead_id"],
                     g.get("lead_score")),
                )

    # Person ids are also globally unique primary keys, numbered from 1 per
    # library (same namespace trick as groups).
    PERSON_ID_STRIDE = 1_000_000

    def replace_persons(self, library_id: int, persons: list[dict],
                        face_persons: Iterable[tuple[int, int]] = ()) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE faces SET person_id = NULL WHERE photo_id IN "
                "(SELECT id FROM photos WHERE library_id = ?)", (library_id,))
            conn.execute("DELETE FROM persons WHERE library_id = ?", (library_id,))
            conn.executemany(
                "INSERT INTO persons (id, library_id, label, face_count, photo_count, "
                "keep_count, covered, required, representative_photo_id, "
                "representative_face_id, max_area_ratio) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(p["id"], library_id, p["label"], p["face_count"], p["photo_count"],
                  p["keep_count"], p["covered"], p["required"],
                  p["representative_photo_id"], p["representative_face_id"],
                  p["max_area_ratio"]) for p in persons],
            )
            assignments = list(face_persons)
            if assignments:
                conn.executemany("UPDATE faces SET person_id = ? WHERE id = ?",
                                 assignments)

    def persons(self, library_id: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM persons WHERE library_id = ? ORDER BY id", (library_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def person_faces(self, library_id: int) -> list[dict]:
        """Described faces that already belong to a person, for merge proposals."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT f.id AS face_id, f.photo_id, f.person_id, f.area_ratio, "
                "f.embedding FROM faces f JOIN photos p ON p.id = f.photo_id "
                "WHERE p.library_id = ? AND f.person_id IS NOT NULL ORDER BY f.id",
                (library_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def person_links(self, library_id: int, kind: str) -> list[list[int]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT face_ids FROM person_links WHERE library_id = ? AND kind = ? "
                "ORDER BY id", (library_id, kind),
            ).fetchall()
        groups: list[list[int]] = []
        for row in rows:
            try:
                groups.append([int(value) for value in json.loads(row["face_ids"])])
            except (TypeError, ValueError):
                continue
        return groups

    def merge_persons(self, library_id: int, person_ids: list[int], *,
                      min_photos: int = 2, min_face_ratio: float = 0.015) -> int:
        """Fold several people into one and remember the decision.

        The surviving row keeps the identity with more photos. The confirmed
        group is stored as face ids so the next scan rebuilds it instead of
        splitting the person again. Returns the surviving person id.
        """
        ids = sorted({int(pid) for pid in person_ids})
        if len(ids) < 2:
            raise ValueError("merging needs at least two people")
        marks = ",".join("?" for _ in ids)

        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM persons WHERE library_id = ? AND id IN ({marks})",
                [library_id, *ids],
            ).fetchall()
            if len(rows) != len(ids):
                raise ValueError("person not found in this library")

            target = int(max(rows, key=lambda r: (r["photo_count"] or 0, -r["id"]))["id"])
            others = [pid for pid in ids if pid != target]
            other_marks = ",".join("?" for _ in others)
            conn.execute(f"UPDATE faces SET person_id = ? WHERE person_id IN ({other_marks})",
                         [target, *others])
            conn.execute(
                f"UPDATE photos SET covered_person = ? WHERE covered_person IN ({other_marks})",
                [target, *others],
            )
            conn.execute(f"DELETE FROM persons WHERE id IN ({other_marks})", others)

            stats = conn.execute(
                "SELECT COUNT(*) AS faces, COUNT(DISTINCT photo_id) AS photos, "
                "MAX(area_ratio) AS area FROM faces WHERE person_id = ?", (target,),
            ).fetchone()
            kept = conn.execute(
                "SELECT COUNT(DISTINCT f.photo_id) AS n FROM faces f "
                "JOIN photos p ON p.id = f.photo_id WHERE f.person_id = ? "
                "AND COALESCE(p.user_decision, p.verdict) IN ('keep', 'pick')", (target,),
            ).fetchone()
            best = conn.execute(
                "SELECT id, photo_id FROM faces WHERE person_id = ? "
                "ORDER BY area_ratio DESC, sharpness DESC LIMIT 1", (target,),
            ).fetchone()

            photo_count = int(stats["photos"] or 0)
            area = float(stats["area"] or 0.0)
            keep_count = int(kept["n"] or 0)
            required = int(photo_count >= max(1, min_photos) and area >= min_face_ratio)
            conn.execute(
                "UPDATE persons SET face_count = ?, photo_count = ?, keep_count = ?, "
                "covered = ?, required = ?, representative_face_id = ?, "
                "representative_photo_id = ?, max_area_ratio = ? WHERE id = ?",
                (int(stats["faces"] or 0), photo_count, keep_count,
                 int(keep_count > 0), required,
                 best["id"] if best else None, best["photo_id"] if best else None,
                 round(area, 4), target),
            )

            group = {int(r[0]) for r in conn.execute(
                "SELECT id FROM faces WHERE person_id = ?", (target,))}
            merges = conn.execute(
                "SELECT id, face_ids FROM person_links WHERE library_id = ? "
                "AND kind = 'merge'", (library_id,),
            ).fetchall()
            absorbed: list[int] = []
            for row in merges:
                members = {int(value) for value in json.loads(row["face_ids"])}
                if members & group:
                    group |= members
                    absorbed.append(int(row["id"]))
            if absorbed:
                conn.execute(
                    f"DELETE FROM person_links WHERE id IN ({','.join('?' for _ in absorbed)})",
                    absorbed,
                )
            conn.execute(
                "INSERT INTO person_links (library_id, kind, face_ids, created_at) "
                "VALUES (?, 'merge', ?, ?)",
                (library_id, json.dumps(sorted(group)), utcnow()),
            )

            # A confirmed merge overrules any earlier "different people" note.
            distinct = conn.execute(
                "SELECT id, face_ids FROM person_links WHERE library_id = ? "
                "AND kind = 'distinct'", (library_id,),
            ).fetchall()
            stale = [int(row["id"]) for row in distinct
                     if {int(v) for v in json.loads(row["face_ids"])} <= group]
            if stale:
                conn.execute(
                    f"DELETE FROM person_links WHERE id IN ({','.join('?' for _ in stale)})",
                    stale,
                )
        return target

    def mark_persons_distinct(self, library_id: int, person_ids: list[int]) -> None:
        """Record that two people are different, using their largest faces.

        Stored as a face-id pair: proposals skip it, and a later confirmed merge
        of the same two people removes it.
        """
        ids = sorted({int(pid) for pid in person_ids})
        if len(ids) < 2:
            raise ValueError("need two people")
        marks = ",".join("?" for _ in ids)
        with self.connect() as conn:
            found = conn.execute(
                f"SELECT id FROM persons WHERE library_id = ? AND id IN ({marks})",
                [library_id, *ids],
            ).fetchall()
            if len(found) != len(ids):
                raise ValueError("person not found in this library")
            faces: list[int] = []
            for person_id in ids:
                row = conn.execute(
                    "SELECT id FROM faces WHERE person_id = ? "
                    "ORDER BY area_ratio DESC, sharpness DESC LIMIT 1", (person_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("person has no faces to compare")
                faces.append(int(row["id"]))
            conn.execute(
                "DELETE FROM person_links WHERE library_id = ? AND kind = 'distinct' "
                "AND face_ids = ?", (library_id, json.dumps(sorted(faces))),
            )
            conn.execute(
                "INSERT INTO person_links (library_id, kind, face_ids, created_at) "
                "VALUES (?, 'distinct', ?, ?)",
                (library_id, json.dumps(sorted(faces)), utcnow()),
            )

    def set_user_decision(self, photo_id: int, decision: str | None,
                          star: int | None) -> dict | None:
        with self.connect() as conn:
            if decision is not None:
                conn.execute("UPDATE photos SET user_decision = ? WHERE id = ?",
                             (decision or None, photo_id))
            if star is not None:
                conn.execute("UPDATE photos SET star = ? WHERE id = ?", (star, photo_id))
        return self.photo_row(photo_id)

    def reset_decisions(self, library_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE photos SET user_decision = NULL, star = 0 WHERE library_id = ?",
                         (library_id,))

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def photos(self, library_id: int, filter_key: str = "all",
               group_id: int | None = None, sort: str = "group",
               person_id: int | None = None) -> list[dict]:
        where = ["p.library_id = ?"]
        params: list[Any] = [library_id]
        if person_id is not None:
            where.append("EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id "
                         "AND f.person_id = ?)")
            params.append(person_id)
        if filter_key == "keep":
            where.append("COALESCE(p.user_decision, p.verdict) IN ('keep', 'pick')")
        elif filter_key == "reject":
            where.append("COALESCE(p.user_decision, p.verdict) = 'reject'")
        elif filter_key == "pick":
            where.append("p.user_decision = 'pick'")
        elif filter_key == "blur":
            where.append("p.reason = 'blurry'")
        elif filter_key == "blink":
            where.append("p.reason = 'closed_eyes'")
        elif filter_key == "exposure":
            where.append("p.reason IN ('overexposed', 'underexposed')")
        elif filter_key == "duplicates":
            where.append("p.group_id IS NOT NULL AND p.is_lead = 0")
        elif filter_key == "groups":
            where.append("p.group_id IS NOT NULL")
        elif filter_key == "errors":
            where.append("p.error IS NOT NULL")
        if group_id is not None:
            where.append("p.group_id = ?")
            params.append(group_id)

        order = {
            "group": "p.group_id IS NULL, p.group_id, p.is_lead DESC, p.quality DESC",
            "quality": "p.quality DESC",
            "time": "p.captured_at IS NULL, p.captured_at, p.path",
            "name": "p.path",
            "reason": "p.reason IS NULL, p.reason, p.quality DESC",
        }.get(sort, "p.group_id IS NULL, p.group_id, p.is_lead DESC, p.quality DESC")

        sql = f"SELECT p.* FROM photos p WHERE {' AND '.join(where)} ORDER BY {order}"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def faces_for(self, photo_ids: list[int]) -> dict[int, list[dict]]:
        """Face rows for the API. Embeddings are deliberately not selected."""
        if not photo_ids:
            return {}
        placeholders = ",".join("?" for _ in photo_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT id, photo_id, x, y, w, h, eyes_open, smile, yaw, pitch, "
                f"sharpness, area_ratio, clip_hi, clip_lo, mean_luma, person_id "
                f"FROM faces WHERE photo_id IN ({placeholders}) "
                f"ORDER BY area_ratio DESC",
                photo_ids,
            ).fetchall()
        out: dict[int, list[dict]] = {}
        for r in rows:
            out.setdefault(r["photo_id"], []).append(dict(r))
        return out

    def faces_with_embeddings(self, library_id: int) -> list[dict]:
        """Every describable face in a library, for identity clustering."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT f.id AS face_id, f.photo_id, f.area_ratio, f.sharpness, "
                "f.eyes_open, f.embedding, p.captured_at "
                "FROM faces f JOIN photos p ON p.id = f.photo_id "
                "WHERE p.library_id = ? AND f.embedding IS NOT NULL "
                "ORDER BY f.id", (library_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, library_id: int) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT CASE WHEN COALESCE(user_decision, verdict) IN ('keep','pick') "
                "THEN 'keep' ELSE 'reject' END AS v, COUNT(*) AS n, "
                "SUM(CASE WHEN reason = 'blurry' THEN 1 ELSE 0 END) AS blurry, "
                "SUM(CASE WHEN reason = 'closed_eyes' THEN 1 ELSE 0 END) AS blink, "
                "SUM(CASE WHEN reason = 'duplicate' THEN 1 ELSE 0 END) AS duplicates, "
                "SUM(CASE WHEN reason IN ('overexposed','underexposed') THEN 1 ELSE 0 END) AS exposure, "
                # Files the scan could not read at all. Counted apart from the
                # reasons on purpose: a frame rejected for soft focus was read
                # and judged, one with an error never got judged, and only the
                # second kind means something is wrong with the setup.
                "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors "
                "FROM photos WHERE library_id = ? GROUP BY v", (library_id,)
            ).fetchall()
            groups = conn.execute(
                "SELECT COUNT(*) AS n FROM groups WHERE library_id = ?", (library_id,)
            ).fetchone()
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM photos WHERE library_id = ?", (library_id,)
            ).fetchone()
            people = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(CASE WHEN covered = 0 AND "
                "required = 1 THEN 1 ELSE 0 END), 0) AS uncovered "
                "FROM persons WHERE library_id = ?",
                (library_id,),
            ).fetchone()
        out = {"total": total["n"] if total else 0, "keep": 0, "reject": 0,
               "blurry": 0, "blink": 0, "duplicates": 0, "exposure": 0,
               "errors": 0,
               "groups": groups["n"] if groups else 0,
               "persons": people["n"] if people else 0,
               "uncovered_persons": people["uncovered"] if people else 0}
        for r in rows:
            if r["v"] == "keep":
                out["keep"] = r["n"]
            elif r["v"] == "reject":
                out["reject"] = r["n"]
            out["blurry"] += r["blurry"] or 0
            out["blink"] += r["blink"] or 0
            out["duplicates"] += r["duplicates"] or 0
            out["exposure"] += r["exposure"] or 0
            out["errors"] += r["errors"] or 0
        return out


def serialize_photo(row: dict, faces: list[dict] | None = None) -> dict:
    """API representation. The user decision, when present, is the effective one."""
    raw = row.get("user_decision") or row.get("verdict") or "keep"
    effective = "keep" if raw in ("keep", "pick") else "reject"
    return {
        "id": row["id"],
        "filename": row.get("filename"),
        "path": row.get("path"),
        "width": row.get("width"),
        "height": row.get("height"),
        "captured_at": row.get("captured_at"),
        "camera": row.get("camera"),
        "lens": row.get("lens"),
        "iso": row.get("iso"),
        "shutter": row.get("shutter"),
        "aperture": row.get("aperture"),
        "focal_len": row.get("focal_len"),
        "quality": row.get("quality"),
        "verdict": effective,
        "ai_verdict": row.get("verdict"),
        "reason": row.get("reason"),
        "user_decision": row.get("user_decision"),
        "star": row.get("star") or 0,
        "group_id": row.get("group_id"),
        "is_lead": bool(row.get("is_lead")),
        "covered_person": row.get("covered_person"),
        "sharpness": row.get("sharpness"),
        "subject_sharpness": row.get("subject_sharpness"),
        "bg_separation": row.get("bg_separation"),
        "exposure": row.get("mean_luma"),
        "clip_hi": row.get("clip_hi"),
        "clip_lo": row.get("clip_lo"),
        "bimodality": row.get("bimodality"),
        "shadow_clipped": bool(row.get("shadow_clipped")),
        "highlight_clipped": bool(row.get("highlight_clipped")),
        "color_harmony": row.get("color_harmony"),
        "noise": row.get("noise"),
        "face_count": row.get("face_count") or 0,
        "eyes_open": row.get("worst_eyes_open"),
        "blink": bool((row.get("worst_eyes_open") or 10) <= 4.0
                      and (row.get("face_count") or 0) > 0),
        "error": row.get("error"),
        "faces": faces or [],
    }
