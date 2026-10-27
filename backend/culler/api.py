"""FastAPI service: scan control, review data, preview images and export."""

from __future__ import annotations

import hashlib
import os
import plistlib
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from .config import (
    ANALYSIS_LONG_EDGE, IMAGE_EXTENSIONS, IMPORT_TYPE_EXTENSIONS, IMPORT_TYPE_KEYS,
    IMPORT_TYPE_LABELS, PRESETS, THUMB_LONG_EDGE, CullingSettings, resource_path,
    strictness_stops,
)
from .db import Store, default_data_dir, serialize_photo
from .imaging import iter_images, render_jpeg
from .metrics.faces import FaceAnalyzer
from .metrics.scene import (
    download_model, model_present, model_verified, runtime_available,
    semantic_available,
)
from .persons import clusters_from_rows, propose_merges
from .pipeline import run_scan

FRONTEND_DIST = resource_path("frontend", "dist")
FACE_CROP_EDGE = 192

# The folder picker is a Finder-style window: a sidebar of places, a grid of
# folders and files, and a preview tile per image. Counts and dimensions are
# read with a small thread pool because each one is a separate stat/open and a
# folder on a slow external disk would otherwise stall the listing.
PICKER_ENTRY_LIMIT = 400
PICKER_FAVORITES = ("Desktop", "Documents", "Downloads", "Pictures", "Movies", "Applications")
NETWORK_FILESYSTEMS = {"smbfs", "nfs", "afpfs", "webdav"}
_MOUNT_CACHE: tuple[float, dict[str, list[dict]]] = (0.0, {})
_BOOT_VOLUME = ""


def _boot_volume_name() -> str:
    """The startup disk's label, which is what Finder puts beside its icon."""
    global _BOOT_VOLUME
    if not _BOOT_VOLUME:
        # Only macOS has a Finder to imitate; elsewhere the root is just root.
        name = "Macintosh HD" if sys.platform == "darwin" else "/"
        try:
            info = subprocess.run(["/usr/sbin/diskutil", "info", "-plist", "/"],
                                  capture_output=True, timeout=5, check=False)
            label = plistlib.loads(info.stdout).get("VolumeName")
            if isinstance(label, str) and label.strip():
                name = label.strip()
        except (OSError, subprocess.SubprocessError, ValueError, plistlib.InvalidFileException):
            pass
        _BOOT_VOLUME = name
    return _BOOT_VOLUME


def _parse_mounts() -> list[dict]:
    """Mounted volumes, from ``mount`` (macOS has no /etc/mtab).

    Format: ``<device> on <path> (<fstype>, <options>, ...)``. System mounts are
    ``nobrowse`` and are skipped, so only what Finder would show survives.
    """
    try:
        listing = subprocess.run(["/sbin/mount"], capture_output=True, text=True,
                                 timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mounts: list[dict] = []
    for line in listing.splitlines():
        match = re.match(r"^(?P<dev>.+?) on (?P<path>.+) \((?P<opts>[^)]*)\)$", line)
        if not match:
            continue
        options = {option.strip() for option in match.group("opts").split(",")}
        if "nobrowse" in options:
            continue
        path = match.group("path")
        fstype = next(iter(options), "")
        if fstype in ("devfs", "autofs"):
            continue
        mounts.append({"path": path, "fstype": fstype})
    return mounts


def _sidebar() -> dict[str, list[dict]]:
    """Favourites, network shares and volumes, cached because it changes slowly."""
    global _MOUNT_CACHE
    cached_at, cached = _MOUNT_CACHE
    if cached and time.monotonic() - cached_at < 5.0:
        return cached

    home = Path.home()
    favorites = []
    for name in ("Home", *PICKER_FAVORITES):
        path = home if name == "Home" else home / name
        if name == "Applications":  # Finder's row is the system folder, not ~/Applications
            path = Path("/Applications")
        try:
            if path.is_dir():
                favorites.append({"name": name, "path": str(path)})
        except OSError:
            continue

    mounts = _parse_mounts()
    shared = []
    locations = []
    for mount in mounts:
        path = Path(mount["path"])
        if mount["fstype"] in NETWORK_FILESYSTEMS:
            shared.append({"name": path.name or mount["fstype"], "path": str(path)})
        if path == Path("/"):
            locations.append({"name": _boot_volume_name(), "path": "/"})
        elif str(path).startswith("/Volumes/"):
            locations.append({"name": path.name, "path": str(path)})
    if not locations:  # `mount` unavailable: still offer whatever /Volumes holds
        locations = [{"name": _boot_volume_name(), "path": "/"}]
        try:
            locations += [{"name": child.name, "path": str(child)}
                          for child in sorted(Path("/Volumes").iterdir())
                          if child.is_dir()]
        except OSError:
            pass

    sidebar = {"favorites": favorites, "shared": shared, "locations": locations}
    _MOUNT_CACHE = (time.monotonic(), sidebar)
    return sidebar


def _folder_meta(path: str) -> dict:
    """What Finder prints under a folder: how much is in it, and can we write."""
    try:
        with os.scandir(path) as entries:
            count: int | None = sum(1 for e in entries if not e.name.startswith("."))
    except OSError:
        count = None
    return {"items": count, "writable": os.access(path, os.W_OK)}


def _image_size(path: str) -> tuple[int, int] | None:
    """Dimensions from the file header only — no decode, no thumbnail."""
    import warnings

    try:
        with warnings.catch_warnings():
            # A 100 MP scan in a browsed folder is not a reason to log a warning
            # every time the picker opens.
            warnings.simplefilter("ignore")
            with PILImage.open(path) as image:
                return image.width, image.height
    except Exception:  # noqa: BLE001 - a broken or exotic file is just a plain icon
        return None


def _picker_entries(path: Path) -> tuple[list[dict], bool]:
    """Folders first, then files, with the metadata Finder prints underneath."""
    try:
        with os.scandir(path) as scanned:
            raw = [entry for entry in scanned if not entry.name.startswith(".")]
    except PermissionError:
        raise HTTPException(403, f"permission denied: {path}") from None

    def is_dir(entry: os.DirEntry) -> bool:
        try:
            return entry.is_dir()
        except OSError:
            return False

    directories = sorted((e for e in raw if is_dir(e)), key=lambda e: e.name.lower())
    files = sorted((e for e in raw if not is_dir(e)), key=lambda e: e.name.lower())
    truncated = len(directories) > PICKER_ENTRY_LIMIT or len(files) > PICKER_ENTRY_LIMIT
    directories, files = directories[:PICKER_ENTRY_LIMIT], files[:PICKER_ENTRY_LIMIT]
    images = [e for e in files if e.name.lower().endswith(tuple(IMAGE_EXTENSIONS))]

    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = list(pool.map(_folder_meta, [e.path for e in directories]))
        sizes = dict(zip([e.path for e in images], pool.map(_image_size, [e.path for e in images])))

    entries = [{"name": e.name, "path": e.path, "kind": "folder", **meta}
               for e, meta in zip(directories, metas)]
    for entry in files:
        info: dict = {"name": entry.name, "path": entry.path, "kind": "file"}
        try:
            info["size"] = entry.stat().st_size
        except OSError:
            info["size"] = None
        if entry.path in sizes:
            info["kind"] = "image"
            dimensions = sizes[entry.path]
            if dimensions:
                info["width"], info["height"] = dimensions
        entries.append(info)
    return entries, truncated


def _cached_preview(cache_dir: Path, photo_id: int, row: dict,
                    edge: int) -> Path | None:
    """Render (and memoise) the preview at ``edge`` — the analysis geometry."""
    cached = cache_dir / f"{photo_id}_{edge}.jpg"
    stale = cached.exists() and cached.stat().st_mtime < (row.get("mtime") or 0)
    if not cached.exists() or stale:
        data = render_jpeg(Path(row["path"]), edge)
        if data is None:
            return None
        cached.write_bytes(data)
    return cached


def _crop_face(preview: Path, face: dict, size: int = FACE_CROP_EDGE) -> bytes | None:
    """Square JPEG crop around a face box, for person thumbnails."""
    import cv2

    image = cv2.imread(str(preview))
    if image is None:
        return None
    height, width = image.shape[:2]
    fx = int(face.get("x") or 0)
    fy = int(face.get("y") or 0)
    fw = int(face.get("w") or 0)
    fh = int(face.get("h") or 0)
    if fw < 8 or fh < 8:
        return None
    pad = int(max(fw, fh) * 0.28)
    x0, y0 = max(0, fx - pad), max(0, fy - pad)
    x1, y1 = min(width, fx + fw + pad), min(height, fy + fh + pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    side = min(crop.shape[:2])
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    crop = crop[cy - side // 2:cy + side // 2, cx - side // 2:cx + side // 2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buffer.tobytes() if ok else None


class ScanRequest(BaseModel):
    settings: dict[str, Any] | None = None
    force: bool = False


class FolderCreateRequest(BaseModel):
    parent: str
    name: str = Field(min_length=1, max_length=120)


class LibraryRequest(BaseModel):
    path: str
    settings: dict[str, Any] | None = None


class DecisionRequest(BaseModel):
    decision: str | None = Field(default=None, description="pick | reject | null")
    star: int | None = Field(default=None, ge=0, le=5)


class PersonLinkRequest(BaseModel):
    ids: list[int] = Field(min_length=2, max_length=10)


class ExportRequest(BaseModel):
    mode: str = Field(default="sidecar", description="sidecar | sidecar_all | copy | csv")
    dest: str | None = None
    include_rejects: bool = True


class JobRunner:
    """One scan at a time per library, in a daemon thread."""

    def __init__(self, store: Store):
        self.store = store
        self._threads: dict[int, threading.Thread] = {}
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self.faces = FaceAnalyzer()

    def start(self, library_id: int, settings: CullingSettings, force: bool) -> dict:
        with self._lock:
            for job_id, thread in list(self._threads.items()):
                if thread.is_alive():
                    job = self.store.get_job(job_id)
                    if job and job["library_id"] == library_id:
                        raise HTTPException(409, f"a scan is already running (job {job_id})")
            job = self.store.create_job(library_id)
            event = threading.Event()
            self._events[job["id"]] = event

            def progress(stage: str, done: int, total: int, message: str) -> None:
                self.store.update_job(job["id"], stage=stage, done=done,
                                      total=total, message=message)

            def work() -> None:
                try:
                    result = run_scan(self.store, library_id, settings, progress, event,
                                      force=force)
                    if result.cancelled:
                        self.store.update_job(job["id"], status="cancelled",
                                              stage="cancelled", message="Cancelled")
                    else:
                        message = (f"{result.keepers} keepers · {result.groups} groups · "
                                   f"{result.total} photos in {result.elapsed:.1f}s")
                        if result.ingested or result.ingest_reused:
                            message += (f" · ingested {result.ingested} photos "
                                        f"({result.ingest_reused} already in the "
                                        f"working copy)")
                        self.store.update_job(
                            job["id"], status="done", stage="done", message=message,
                            total=result.total, done=result.total,
                        )
                except Exception as exc:  # noqa: BLE001
                    self.store.update_job(job["id"], status="error", stage="error",
                                          error=f"{type(exc).__name__}: {exc}",
                                          message="Scan failed")
                finally:
                    with self._lock:
                        self._events.pop(job["id"], None)

            thread = threading.Thread(target=work, name=f"scan-{library_id}", daemon=True)
            self._threads[job["id"]] = thread
            thread.start()
            return self.store.get_job(job["id"]) or job

    def cancel(self, job_id: int) -> bool:
        event = self._events.get(job_id)
        if event is None:
            return False
        event.set()
        return True


class SceneModelInstaller:
    """One scene-model download at a time, in a daemon thread.

    The app is the only client, so the state is a few counters rather than a
    job row: the cull dialog polls `/api/semantic` while its toggle is waiting.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done = 0
        self._total = 0
        self._error: str | None = None

    def _snapshot(self) -> dict:
        running = self._thread is not None and self._thread.is_alive()
        return {
            "runtime": runtime_available(),
            "model": model_present(),
            "available": semantic_available(),
            "downloading": running,
            "done": self._done,
            "total": self._total,
            "error": self._error,
        }

    def status(self) -> dict:
        with self._lock:
            return self._snapshot()

    def start(self) -> dict:
        if not runtime_available():
            raise HTTPException(
                409, "this build has no scene runtime; install the semantic extra"
            )
        # Hash outside the lock: 42 MB should not block the progress polls.
        if model_present() and model_verified():
            return self.status()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._snapshot()
            self._done = 0
            self._total = 0
            self._error = None
            self._thread = threading.Thread(
                target=self._work, name="scene-model", daemon=True
            )
            self._thread.start()
            return self._snapshot()

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self._done = done
            self._total = total

    def _work(self) -> None:
        try:
            download_model(progress=self._progress)
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/semantic
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "catalog.db")
    runner = JobRunner(store)
    scene_model = SceneModelInstaller()
    cache_dir = data_dir / "previews"
    cache_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Cullr — AI photo culling", version="0.6.0")
    app.state.store = store
    app.state.runner = runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/capabilities")
    def capabilities() -> dict:
        return {
            "faces": runner.faces.available,
            "faces_error": runner.faces.load_error,
            # Optional: the DINOv2 scene runtime ships with the build, but the
            # weights are a first-use download. The UI offers the toggle only
            # when the runtime is there, and shows download progress until the
            # model is too.
            "semantic": semantic_available(),
            "semantic_runtime": runtime_available(),
            "presets": [
                {"key": p.key, "label": p.label, "description": p.description,
                 "strictness": p.strictness}
                for p in PRESETS.values()
            ],
            "default_settings": CullingSettings().to_dict(),
            # The Auto Target stop scale, single-sourced from config.py so the
            # dialog cannot invent a stop the gates do not know.
            "strictness_stops": strictness_stops(),
            "analysis_long_edge": ANALYSIS_LONG_EDGE,
            # The import screen's photo-type checkboxes, single-sourced from
            # config.py so the panel can never offer a type the scan rejects.
            "import_types": [
                {"key": key, "label": IMPORT_TYPE_LABELS[key],
                 "extensions": sorted(IMPORT_TYPE_EXTENSIONS[key])}
                for key in IMPORT_TYPE_KEYS
            ],
        }

    # ---- semantic duplicate model -------------------------------------- #
    @app.get("/api/semantic")
    def semantic_status() -> dict:
        """Whether the scene runtime and model are ready, and download progress."""
        return scene_model.status()

    @app.post("/api/semantic/model")
    def download_semantic_model() -> dict:
        """Start the one-time scene-model download (idempotent)."""
        return scene_model.start()

    # ---- folders ------------------------------------------------------- #
    @app.get("/api/folders")
    def folders(path: str | None = None) -> dict:
        """One Finder-style listing: entries, plus the sidebar places.

        ``folders`` stays directory-only (it is the contract the export flow was
        built on); ``entries`` adds files so the picker can show the folder the
        way the OS does.
        """
        start = Path(path).expanduser() if path else Path.home()
        try:
            resolved = start.resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(400, f"cannot open {start}") from None
        if not resolved.is_dir():
            raise HTTPException(400, "not a folder")
        entries, truncated = _picker_entries(resolved)
        parent = resolved.parent
        return {
            "path": str(resolved),
            "parent": str(parent) if parent != resolved else None,
            "home": str(Path.home()),
            "writable": os.access(resolved, os.W_OK),
            "folders": [{"name": entry["name"], "path": entry["path"]}
                        for entry in entries if entry["kind"] == "folder"][:500],
            "entries": entries,
            "truncated": truncated,
            "sidebar": _sidebar(),
        }

    @app.get("/api/folders/preview")
    def folder_preview(path: str, size: int = Query(256, ge=64, le=512)) -> Response:
        """A tile for one image in the picker, decoded on demand and cached.

        Separate from the library previews on purpose: these are arbitrary paths
        chosen by the user, not rows in the catalog.
        """
        target = Path(path).expanduser()
        if target.suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(400, "not a supported image")
        try:
            stat = target.stat()
        except OSError:
            raise HTTPException(404, "cannot open file") from None
        if not target.is_file():
            raise HTTPException(404, "not a file")
        key = hashlib.sha1(
            f"{target.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{size}".encode()
        ).hexdigest()
        tile_dir = cache_dir / "picker"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile = tile_dir / f"{key}.jpg"
        if not tile.exists():
            data = render_jpeg(target, size, quality=78)
            if data is None:
                raise HTTPException(404, "cannot decode image")
            tile.write_bytes(data)
        return FileResponse(tile, media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=600"})

    @app.post("/api/folders", status_code=201)
    def create_folder(body: FolderCreateRequest) -> dict:
        """Create one folder inside ``parent``, for a delivery destination.

        Flat by design: a name with a separator would silently create nested
        folders the picker never showed, and a leading dot would produce a
        folder the listing then hides.
        """
        name = body.name.strip()
        if (not name or name in (".", "..") or name.startswith(".")
                or "/" in name or "\\" in name or "\x00" in name):
            raise HTTPException(400, "invalid folder name")
        parent = Path(body.parent).expanduser()
        if not parent.is_dir():
            raise HTTPException(400, f"cannot open {parent}")
        target = parent / name
        if target.exists():
            raise HTTPException(409, "a folder with that name already exists")
        try:
            target.mkdir()
            resolved = target.resolve()
        except PermissionError:
            raise HTTPException(403, f"permission denied: {parent}") from None
        except OSError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"path": str(resolved), "name": resolved.name}

    # ---- libraries ----------------------------------------------------- #
    @app.get("/api/libraries")
    def list_libraries() -> list[dict]:
        return [_decorate_library(row, store) for row in store.list_libraries()]

    @app.post("/api/libraries")
    def create_library(body: LibraryRequest) -> dict:
        root = Path(body.path).expanduser()
        if not root.exists() or not root.is_dir():
            raise HTTPException(400, f"not a directory: {root}")
        library = store.create_or_get_library(str(root), body.settings)
        if body.settings:
            library["settings"] = CullingSettings.from_dict(body.settings).to_dict()
        return _decorate_library(library, store)


    @app.get("/api/libraries/{library_id}")
    def get_library(library_id: int) -> dict:
        library = store.get_library(library_id)
        if not library:
            raise HTTPException(404, "library not found")
        return _decorate_library(library, store)

    @app.delete("/api/libraries/{library_id}")
    def delete_library(library_id: int) -> dict:
        store.delete_library(library_id)
        return {"deleted": library_id}

    # ---- scanning ------------------------------------------------------ #
    @app.post("/api/libraries/{library_id}/scan")
    def scan(library_id: int, body: ScanRequest) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        settings = CullingSettings.from_dict(body.settings)
        job = runner.start(library_id, settings, body.force)
        return job

    @app.post("/api/libraries/{library_id}/discover")
    def discover(library_id: int) -> dict:
        """Count the photos the import will process, before any scan runs.

        A names-only directory walk with the library's stored import options
        (recursive flag, type filter) — no file contents are read, nothing is
        analysed or copied, and the originals are never touched. The pre-cull
        window shows this total so the target reads "~20% of 5,745 photos"
        instead of 0. When ingest is on and the source is gone but a working
        copy exists, count the working copy, the same fallback the scan uses.
        """
        library = store.get_library(library_id)
        if not library:
            raise HTTPException(404, "library not found")
        settings = CullingSettings.from_dict(
            _json_or_none(library.get("settings_json")))
        root = Path(library["root_path"])
        effective_root = root
        if settings.ingest:
            working = store.ingest_root(library_id)
            if root.is_dir():
                effective_root = root
            elif working.is_dir() and any(working.iterdir()):
                effective_root = working
            else:
                raise HTTPException(400, f"not a directory: {root}")
        elif not root.is_dir():
            raise HTTPException(400, f"not a directory: {root}")
        try:
            total = len(iter_images(effective_root, recursive=settings.recursive,
                                    extensions=settings.import_extensions()))
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"total": total}

    @app.get("/api/libraries/{library_id}/job")
    def latest_job(library_id: int) -> dict:
        return store.latest_job(library_id) or {}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int) -> dict:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int) -> dict:
        if not runner.cancel(job_id):
            raise HTTPException(409, "job is not cancellable (already finished)")
        return {"cancelling": job_id}

    # ---- review data --------------------------------------------------- #
    @app.get("/api/libraries/{library_id}/persons")
    def persons(library_id: int) -> list[dict]:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        return store.persons(library_id)

    @app.get("/api/libraries/{library_id}/persons/suggestions")
    def person_suggestions(library_id: int) -> list[dict]:
        """Probable duplicates among the people, for the user to confirm."""
        library = store.get_library(library_id)
        if not library:
            raise HTTPException(404, "library not found")
        settings = CullingSettings.from_dict(
            _json_or_none(library.get("settings_json")))
        roster = store.persons(library_id)
        clusters = clusters_from_rows(roster, store.person_faces(library_id))
        dismissed = {tuple(pair) for pair in store.person_links(library_id, "distinct")
                     if len(pair) == 2}
        proposals = propose_merges(clusters, min_similarity=settings.person_propose,
                                   dismissed=dismissed)
        labels = {row["id"]: row["label"] for row in roster}
        return [
            {"a": p.a, "b": p.b, "a_label": labels.get(p.a), "b_label": labels.get(p.b),
             "similarity": round(p.similarity, 3)}
            for p in proposals
        ]

    @app.post("/api/libraries/{library_id}/persons/merge")
    def merge_people(library_id: int, body: PersonLinkRequest) -> dict:
        """Confirm that several people are one person."""
        library = store.get_library(library_id)
        if not library:
            raise HTTPException(404, "library not found")
        settings = CullingSettings.from_dict(
            _json_or_none(library.get("settings_json")))
        try:
            target = store.merge_persons(
                library_id, body.ids,
                min_photos=settings.person_min_photos,
                min_face_ratio=settings.person_min_face_ratio,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"merged_into": target, "persons": store.persons(library_id)}

    @app.post("/api/libraries/{library_id}/persons/distinct")
    def distinct_people(library_id: int, body: PersonLinkRequest) -> dict:
        """Confirm that two people are different, and stop suggesting them."""
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        try:
            store.mark_persons_distinct(library_id, body.ids)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"dismissed": True}

    @app.get("/api/libraries/{library_id}/photos")
    def photos(library_id: int,
               filter: str = Query("all"),
               sort: str = Query("group"),
               group_id: int | None = None,
               person_id: int | None = None,
               limit: int = Query(10000, ge=1, le=50000),
               offset: int = 0) -> dict:
        if not store.get_library(library_id):
            raise HTTPException(404, "library not found")
        rows = store.photos(library_id, filter, group_id, sort, person_id)
        total = len(rows)
        page = rows[offset:offset + limit]
        faces = store.faces_for([r["id"] for r in page])
        return {
            "total": total,
            "photos": [serialize_photo(r, faces.get(r["id"], [])) for r in page],
            "stats": store.stats(library_id),
        }

    @app.get("/api/photos/{photo_id}")
    def photo_detail(photo_id: int) -> dict:
        row = store.photo_row(photo_id)
        if not row:
            raise HTTPException(404, "photo not found")
        faces = store.faces_for([photo_id]).get(photo_id, [])
        return serialize_photo(row, faces)

    @app.get("/api/photos/{photo_id}/image")
    def photo_image(photo_id: int, size: str = Query("thumb", pattern="^(thumb|preview)$"),
                    v: str | None = None) -> Response:
        row = store.photo_row(photo_id)
        if not row:
            raise HTTPException(404, "photo not found")
        edge = THUMB_LONG_EDGE if size == "thumb" else ANALYSIS_LONG_EDGE
        cached = _cached_preview(cache_dir, photo_id, row, edge)
        if cached is None:
            raise HTTPException(415, "preview unavailable for this file")
        return FileResponse(cached, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/photos/{photo_id}/face/{face_id}")
    def face_image(photo_id: int, face_id: int) -> Response:
        row = store.photo_row(photo_id)
        if not row:
            raise HTTPException(404, "photo not found")
        face = next((f for f in store.faces_for([photo_id]).get(photo_id, [])
                     if f["id"] == face_id), None)
        if face is None:
            raise HTTPException(404, "face not found")
        cached = cache_dir / f"{photo_id}_{face_id}_face.jpg"
        if not cached.exists():
            preview = _cached_preview(cache_dir, photo_id, row, ANALYSIS_LONG_EDGE)
            data = _crop_face(preview, face) if preview else None
            if data is None:
                raise HTTPException(415, "face crop unavailable for this file")
            cached.write_bytes(data)
        return FileResponse(cached, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.patch("/api/photos/{photo_id}")
    def update_photo(photo_id: int, body: DecisionRequest) -> dict:
        row = store.photo_row(photo_id)
        if not row:
            raise HTTPException(404, "photo not found")
        if body.decision not in (None, "", "pick", "reject"):
            raise HTTPException(400, "decision must be pick, reject or null")
        updated = store.set_user_decision(photo_id, body.decision, body.star)
        return serialize_photo(updated or row)

    @app.post("/api/libraries/{library_id}/reset-decisions")
    def reset_decisions(library_id: int) -> dict:
        store.reset_decisions(library_id)
        return {"reset": True}

    @app.get("/api/libraries/{library_id}/stats")
    def stats(library_id: int) -> dict:
        return store.stats(library_id)

    # ---- export -------------------------------------------------------- #
    @app.post("/api/libraries/{library_id}/export")
    def export(library_id: int, body: ExportRequest) -> dict:
        from . import export as export_mod

        library = store.get_library(library_id)
        if not library:
            raise HTTPException(404, "library not found")
        # The pre-cull window decides whether an existing sidecar is replaced;
        # export is where that opinion is acted on.
        settings = CullingSettings.from_dict(_json_or_none(library.get("settings_json")))

        if body.mode == "sidecar":
            result = export_mod.write_sidecars(
                store, library_id, only_raw=True, include_rejects=body.include_rejects,
                overwrite=settings.overwrite_xmp)
        elif body.mode == "sidecar_all":
            result = export_mod.write_sidecars(
                store, library_id, only_raw=False, include_rejects=body.include_rejects,
                overwrite=settings.overwrite_xmp)
        elif body.mode == "copy":
            if not body.dest:
                raise HTTPException(400, "dest is required for copy export")
            result = export_mod.copy_keepers(store, library_id, body.dest)
        elif body.mode == "csv":
            if not body.dest:
                raise HTTPException(400, "dest is required for csv export")
            result = export_mod.write_csv(store, library_id, body.dest)
        else:
            raise HTTPException(400, f"unknown export mode: {body.mode}")
        # Remember the mode and the path the caller gave, not the resolved file:
        # the panel's field means "folder", even for the CSV report.
        store.update_library_export(library_id, body.mode, body.dest or "")
        return {"written": result.written, "copied": result.copied,
                "skipped": result.skipped, "dest": result.dest,
                "errors": result.errors[:20]}

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")

    return app


def _decorate_library(library: dict, store: Store) -> dict:
    """Add the parsed settings, the remembered export and, for an ingested
    library, where its working copy lives to a library row."""
    library.setdefault(
        "settings",
        CullingSettings.from_dict(_json_or_none(library.get("settings_json"))).to_dict(),
    )
    mode = library.get("export_mode")
    library["export"] = (
        {"mode": mode, "dest": library.get("export_dest") or ""} if mode else None
    )
    settings = library["settings"]
    library["ingest_root"] = (
        str(store.ingest_root(library["id"]))
        if isinstance(settings, dict) and settings.get("ingest")
        else None
    )
    return library


def _json_or_none(value: Any) -> dict | None:
    import json

    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


# No module-level app instance: importing this module must not create a catalog
# directory or load the face model. `uvicorn culler.api:create_app --factory`
# builds it, honouring CULLR_DATA_DIR.
