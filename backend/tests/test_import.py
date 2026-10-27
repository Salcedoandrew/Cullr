"""Import options: sub-folders, photo types, ingest-and-backup.

The import screen's settings travel with the library (settings_json), so a
rescan reuses them — including the working copy an ingest created. Originals
are never modified: ingest only reads the source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from culler.api import create_app
from culler.config import RAW_EXTENSIONS, CullingSettings, extensions_for_types
from culler.db import Store
from culler.pipeline import run_scan
from tests.conftest import make_scene, save_jpeg


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "catalog.db")


def _save_raster(path: Path, bgr, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format=fmt)
    return path


@pytest.fixture
def mixed_shoot(tmp_path: Path) -> Path:
    """JPEGs at the top, one nested JPEG, a TIFF and a PNG."""
    root = tmp_path / "shoot"
    scene = make_scene(21)
    save_jpeg(root / "IMG_0001.jpg", scene, "2024:05:01 10:00:00")
    save_jpeg(root / "IMG_0002.jpg", make_scene(22), "2024:05:01 10:00:01")
    save_jpeg(root / "sub" / "IMG_0003.jpg", make_scene(23), "2024:05:01 10:01:00")
    _save_raster(root / "IMG_0004.tif", make_scene(24), "TIFF")
    _save_raster(root / "IMG_0005.png", make_scene(25), "PNG")
    return root


def _scan(store: Store, folder: Path, settings: CullingSettings | None = None,
          force: bool = False):
    library = store.create_or_get_library(str(folder))
    return library["id"], run_scan(store, library["id"],
                                   settings or CullingSettings(), force=force)


def _filenames(store: Store, library_id: int) -> set[str]:
    return {row["filename"] for row in store.photos(library_id)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ types ---

def test_extensions_for_types_maps_the_four_checkboxes():
    assert extensions_for_types(["raw"]) == RAW_EXTENSIONS
    assert extensions_for_types(["jpeg"]) == {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    assert extensions_for_types(["tiff"]) == {".tif", ".tiff"}
    assert extensions_for_types(["heic"]) == {".heic", ".heif", ".avif"}
    assert extensions_for_types([]) == set()
    assert extensions_for_types(["jpeg", "bogus"]) == {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def test_extension_filter_admits_only_checked_types(store, mixed_shoot, tmp_path):
    _id, result = _scan(store, mixed_shoot, CullingSettings(import_types=["jpeg"]))
    assert result.total == 4  # 2 top JPEGs + nested JPEG + PNG; TIFF excluded
    assert "IMG_0004.tif" not in _filenames(store, _id)

    # A fresh library for the TIFF-only scan: a scan never removes rows for
    # files it no longer discovers, so reusing this library would keep the
    # JPEG rows from the first scan.
    import shutil

    second_shoot = tmp_path / "shoot2"
    shutil.copytree(mixed_shoot, second_shoot)
    _id2, result2 = _scan(store, second_shoot,
                          CullingSettings(import_types=["tiff"]))
    assert result2.total == 1
    assert _filenames(store, _id2) == {"IMG_0004.tif"}


def test_recursive_off_ignores_nested_files(store, mixed_shoot):
    _id, result = _scan(store, mixed_shoot, CullingSettings(recursive=False))
    assert result.total == 4  # nested IMG_0003.jpg is not discovered
    assert "IMG_0003.jpg" not in _filenames(store, _id)

    _id, result = _scan(store, mixed_shoot, CullingSettings(recursive=True), force=True)
    assert result.total == 5
    assert "IMG_0003.jpg" in _filenames(store, _id)


# ----------------------------------------------------------------- ingest ---

def test_ingest_copies_and_leaves_originals_untouched(store, mixed_shoot, tmp_path):
    before = {str(p.relative_to(mixed_shoot)): _sha(p)
              for p in mixed_shoot.rglob("*") if p.is_file()}
    library_id, result = _scan(store, mixed_shoot, CullingSettings(ingest=True))
    assert result.total == 5
    assert result.ingested == 5
    assert result.ingest_reused == 0

    working = store.ingest_root(library_id)
    assert working.is_dir()
    copied = {str(p.relative_to(working)): _sha(p)
              for p in working.rglob("*") if p.is_file()}
    assert copied == before  # same layout, same bytes

    # Originals are byte-identical and nothing new appeared beside them.
    after = {str(p.relative_to(mixed_shoot)): _sha(p)
             for p in mixed_shoot.rglob("*") if p.is_file()}
    assert after == before

    # The catalog points at the working copy, not the source.
    rows = store.photos(library_id)
    assert rows and all(str(working) in row["path"] for row in rows)


def test_ingest_rescan_reuses_working_copy(store, mixed_shoot, tmp_path):
    library_id, first = _scan(store, mixed_shoot, CullingSettings(ingest=True))
    assert first.ingested == 5

    second_id, second = _scan(store, mixed_shoot, CullingSettings(ingest=True))
    assert second_id == library_id
    assert second.ingested == 0
    assert second.ingest_reused == 5
    assert second.reused == second.total == 5


def test_ingest_rescan_works_after_the_source_is_gone(store, mixed_shoot, tmp_path):
    library_id, first = _scan(store, mixed_shoot, CullingSettings(ingest=True))
    assert first.total == 5

    import shutil

    shutil.rmtree(mixed_shoot)
    _id, second = _scan(store, mixed_shoot, CullingSettings(ingest=True))
    assert second.total == 5
    assert second.reused == 5
    assert second.ingested == 0
    assert _filenames(store, library_id) == {
        "IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.tif", "IMG_0005.png",
    }


def test_ingest_backup_copies_alongside(store, mixed_shoot, tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    settings = CullingSettings(ingest=True, backup_dir=str(backup))
    library_id, result = _scan(store, mixed_shoot, settings)
    assert result.total == 5
    assert result.ingested == 10  # working copy + backup
    backed_up = {p.name for p in backup.rglob("*") if p.is_file()}
    assert backed_up == {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg",
                         "IMG_0004.tif", "IMG_0005.png"}
    assert (backup / "sub" / "IMG_0003.jpg").is_file()


def test_ingest_rejects_a_backup_inside_the_source(store, mixed_shoot, tmp_path):
    inside = mixed_shoot / "backup"
    inside.mkdir()
    settings = CullingSettings(ingest=True, backup_dir=str(inside))
    library_id = store.create_or_get_library(str(mixed_shoot))["id"]
    with pytest.raises(ValueError, match="must not be inside the source"):
        run_scan(store, library_id, settings)


def test_ingest_rejects_a_missing_backup_folder(store, mixed_shoot, tmp_path):
    settings = CullingSettings(ingest=True, backup_dir=str(tmp_path / "gone"))
    library_id = store.create_or_get_library(str(mixed_shoot))["id"]
    with pytest.raises(ValueError, match="backup folder is not available"):
        run_scan(store, library_id, settings)


# -------------------------------------------------------------------- api ---

@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as test_client:
        yield test_client


def test_capabilities_lists_import_types(client):
    body = client.get("/api/capabilities").json()
    keys = [item["key"] for item in body["import_types"]]
    assert keys == ["raw", "jpeg", "tiff", "heic"]
    jpeg = next(item for item in body["import_types"] if item["key"] == "jpeg")
    assert ".jpg" in jpeg["extensions"] and ".png" in jpeg["extensions"]
    assert ".tif" in next(
        item for item in body["import_types"] if item["key"] == "tiff")["extensions"]


def test_library_persists_import_options_and_exposes_the_working_copy(
        client, mixed_shoot):
    settings = {"recursive": False, "import_types": ["jpeg"], "ingest": True}
    library = client.post("/api/libraries",
                          json={"path": str(mixed_shoot), "settings": settings}).json()
    assert library["settings"]["recursive"] is False
    assert library["settings"]["import_types"] == ["jpeg"]
    assert library["settings"]["ingest"] is True
    assert library["ingest_root"] is not None
    assert library["ingest_root"].endswith(f"ingest/{library['id']}")

    started = client.post(f"/api/libraries/{library['id']}/scan",
                          json={"settings": settings})
    assert started.status_code == 200
    import time

    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/libraries/{library['id']}/job").json()
        if job.get("status") in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job

    photos = client.get(f"/api/libraries/{library['id']}/photos").json()
    assert photos["total"] == 3  # top-level JPEGs + PNG; nested JPEG and TIFF out
    assert all(library["ingest_root"] in photo["path"] for photo in photos["photos"])
