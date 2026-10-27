"""Discover endpoint: count the import set before any scan runs.

The pre-cull window opens before the first scan, so it has no stats yet. This
endpoint is the count it shows: a names-only walk with the library's stored
import options (recursive flag, type filter), the same admitted set the scan
would discover. No file contents are read, nothing is analysed or copied.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from culler.api import create_app
from tests.conftest import make_scene, save_jpeg


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mixed_shoot(tmp_path: Path) -> Path:
    """JPEGs at the top, one nested JPEG, a TIFF and a PNG."""
    root = tmp_path / "shoot"
    save_jpeg(root / "IMG_0001.jpg", make_scene(21), "2024:05:01 10:00:00")
    save_jpeg(root / "IMG_0002.jpg", make_scene(22), "2024:05:01 10:00:01")
    save_jpeg(root / "sub" / "IMG_0003.jpg", make_scene(23), "2024:05:01 10:01:00")
    for name, fmt in (("IMG_0004.tif", "TIFF"), ("IMG_0005.png", "PNG")):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(make_scene(24), cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, format=fmt)
    return root


def _create_library(client: TestClient, folder: Path, settings: dict | None = None) -> int:
    body: dict = {"path": str(folder)}
    if settings:
        body["settings"] = settings
    response = client.post("/api/libraries", json=body)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _wait_for_scan(client: TestClient, library_id: int, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/libraries/{library_id}/job").json()
        if job.get("status") in ("done", "error", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError("scan did not finish in time")


def test_discover_counts_the_import_set(client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot)
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 5}


def test_discover_honours_recursive_off(client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot, {"recursive": False})
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 4}  # nested JPEG excluded


def test_discover_honours_the_type_filter(client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot, {"import_types": ["jpeg"]})
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 4}  # 2 top JPEGs + nested JPEG + PNG; TIFF excluded


def test_discover_empty_folder_returns_zero(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    library_id = _create_library(client, empty)
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 0}


def test_discover_unknown_library_404(client):
    response = client.post("/api/libraries/999/discover")
    assert response.status_code == 404


def test_discover_missing_folder_errors(client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot)
    shutil.rmtree(mixed_shoot)
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_discover_counts_the_source_when_ingest_is_on(client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot, {"ingest": True})
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 5}  # the source set, not the working copy


def test_discover_matches_what_the_scan_discovers(client, mixed_shoot):
    settings = {"recursive": False, "import_types": ["jpeg"]}
    library_id = _create_library(client, mixed_shoot, settings)
    discovered = client.post(f"/api/libraries/{library_id}/discover").json()
    assert discovered == {"total": 3}  # 2 top JPEGs + PNG; nested + TIFF excluded
    response = client.post(f"/api/libraries/{library_id}/scan",
                           json={"settings": settings})
    assert response.status_code == 200
    job = _wait_for_scan(client, library_id)
    assert job["status"] == "done", job
    assert job["total"] == discovered["total"]


def test_discover_falls_back_to_the_working_copy_when_the_source_is_gone(
        client, mixed_shoot):
    library_id = _create_library(client, mixed_shoot, {"ingest": True})
    response = client.post(f"/api/libraries/{library_id}/scan",
                           json={"settings": {"ingest": True}})
    assert response.status_code == 200
    job = _wait_for_scan(client, library_id)
    assert job["status"] == "done", job
    shutil.rmtree(mixed_shoot)
    response = client.post(f"/api/libraries/{library_id}/discover")
    assert response.status_code == 200
    assert response.json() == {"total": 5}  # the working copy, same admitted set
