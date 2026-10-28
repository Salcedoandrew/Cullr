from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from culler.api import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as test_client:
        yield test_client


def wait_for_scan(client: TestClient, library_id: int, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/libraries/{library_id}/job").json()
        if job.get("status") in ("done", "error", "cancelled"):
            return job
        time.sleep(0.2)
    raise AssertionError("scan did not finish in time")


def test_capabilities_lists_presets(client):
    body = client.get("/api/capabilities").json()
    keys = {p["key"] for p in body["presets"]}
    assert {"portrait", "wedding", "sports", "landscape"} <= keys
    assert "faces" in body


def test_create_library_rejects_missing_folder(client):
    response = client.post("/api/libraries", json={"path": "/definitely/not/here"})
    assert response.status_code == 400


def test_scan_then_review_then_decide_then_export(client, photo_dir, tmp_path):
    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    library_id = library["id"]

    started = client.post(f"/api/libraries/{library_id}/scan",
                          json={"settings": {"preset": "wedding"}})
    assert started.status_code == 200
    job = wait_for_scan(client, library_id)
    assert job["status"] == "done", job

    listing = client.get(f"/api/libraries/{library_id}/photos").json()
    assert listing["total"] == 6
    photos = listing["photos"]
    assert all(p["quality"] is not None for p in photos)
    assert any(p["verdict"] == "keep" for p in photos)
    assert any(p["reason"] == "duplicate" for p in photos)
    assert listing["stats"]["total"] == 6

    target = next(p for p in photos if p["verdict"] == "reject")
    patched = client.patch(f"/api/photos/{target['id']}",
                           json={"decision": "pick", "star": 4}).json()
    assert patched["verdict"] == "keep"
    assert patched["user_decision"] == "pick"
    assert patched["star"] == 4

    filtered = client.get(f"/api/libraries/{library_id}/photos",
                          params={"filter": "pick"}).json()
    assert [p["id"] for p in filtered["photos"]] == [target["id"]]

    image = client.get(f"/api/photos/{target['id']}/image", params={"size": "thumb"})
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert len(image.content) > 500
    # Second request is served from the cache: exercises the staleness check.
    again = client.get(f"/api/photos/{target['id']}/image", params={"size": "thumb"})
    assert again.status_code == 200 and again.content == image.content

    csv_path = tmp_path / "out.csv"
    exported = client.post(f"/api/libraries/{library_id}/export",
                           json={"mode": "csv", "dest": str(csv_path)}).json()
    assert exported["written"] == 6
    assert csv_path.exists()

    sidecars = client.post(f"/api/libraries/{library_id}/export",
                           json={"mode": "sidecar_all"}).json()
    assert sidecars["written"] == 6


def test_a_scan_that_cannot_read_anything_fails_instead_of_keeping_zero(client, tmp_path):
    """"0 keepers · 0 groups" is a verdict on the shoot. Nothing was read, so
    nothing was judged, and the job has to say that instead."""
    root = tmp_path / "unreadable"
    root.mkdir()
    (root / "IMG_0001.jpg").write_bytes(b"not a jpeg at all")
    (root / "IMG_0002.jpg").write_bytes(b"nor is this one")
    library = client.post("/api/libraries", json={"path": str(root)}).json()

    client.post(f"/api/libraries/{library['id']}/scan", json={})
    job = wait_for_scan(client, library["id"])

    assert job["status"] == "error"
    assert job["error_count"] == 2
    assert job["error_sample"] and job["error_sample"].startswith("UnidentifiedImageError")
    assert "none of the 2 files could be read" in (job["error"] or "")
    assert "could be evaluated" in job["message"]

    stats = client.get(f"/api/libraries/{library['id']}/stats").json()
    assert stats["errors"] == 2
    unreadable = client.get(f"/api/libraries/{library['id']}/photos",
                            params={"filter": "errors"}).json()
    assert unreadable["total"] == 2


def test_a_partly_unreadable_shoot_still_finishes_and_says_so(client, photo_dir):
    (photo_dir / "IMG_9999.jpg").write_bytes(b"not a jpeg at all")
    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()

    client.post(f"/api/libraries/{library['id']}/scan", json={})
    job = wait_for_scan(client, library["id"])

    assert job["status"] == "done", "the rest of the shoot was culled"
    assert job["error_count"] == 1
    assert job["message"].endswith("1 could not be read"), job["message"]
    assert job["error_sample"].startswith("UnidentifiedImageError")
    # The scan itself did not fail, so the job's error field stays empty.
    assert job["error"] is None


def test_a_clean_scan_carries_no_read_failures(client, photo_dir):
    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    client.post(f"/api/libraries/{library['id']}/scan", json={})
    job = wait_for_scan(client, library["id"])

    assert job["status"] == "done"
    assert job["error_count"] == 0
    assert job["error_sample"] is None
    assert job["message"].endswith("s"), "the old message format, unchanged"


def test_second_scan_is_rejected_while_one_is_running(client, photo_dir, monkeypatch):
    """The runner refuses a concurrent scan of the same library."""
    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    library_id = library["id"]
    app = client.app
    runner = app.state.runner

    import culler.pipeline as pipeline_mod

    original = pipeline_mod.run_scan

    def slow_scan(*args, **kwargs):
        time.sleep(0.8)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "run_scan", slow_scan)
    first = client.post(f"/api/libraries/{library_id}/scan", json={})
    assert first.status_code == 200
    second = client.post(f"/api/libraries/{library_id}/scan", json={})
    assert second.status_code == 409
    wait_for_scan(client, library_id)


def _seed_faces(store, library_id: int, photo_ids: list[int], vectors) -> list[int]:
    """Attach one described face to each photo, returning the face ids."""
    keys = ("width", "height", "captured_at", "camera", "lens", "iso", "shutter",
            "aperture", "focal_len", "sharpness", "tenengrad", "clip_hi", "clip_lo",
            "mean_luma", "noise", "contrast", "phash", "dhash")
    face_ids = []
    for photo_id, vector in zip(photo_ids, vectors):
        row = store.photo_row(photo_id)
        payload = {key: row.get(key) for key in keys}
        payload["faces"] = [{"x": 200, "y": 150, "w": 300, "h": 300,
                             "area_ratio": 0.05, "eyes_open": 9.0,
                             "sharpness": 400.0, "embedding": vector}]
        store.update_analysis(photo_id, payload, version=2)
        face_ids.append(store.faces_for([photo_id])[photo_id][0]["id"])
    return face_ids


def test_merge_suggestions_are_confirmed_not_automatic(client, photo_dir):
    import numpy as np

    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    library_id = library["id"]
    client.post(f"/api/libraries/{library_id}/scan", json={})
    wait_for_scan(client, library_id)

    store = client.app.state.store
    photos = client.get(f"/api/libraries/{library_id}/photos").json()["photos"][:2]
    rng = np.random.default_rng(0)
    base = rng.normal(size=117).astype(np.float32)
    base /= np.linalg.norm(base)
    orthogonal = rng.normal(size=117).astype(np.float32)
    orthogonal -= base * float(base @ orthogonal)
    orthogonal /= np.linalg.norm(orthogonal)
    close = 0.87 * base + np.sqrt(1 - 0.87 ** 2) * orthogonal

    face_ids = _seed_faces(store, library_id, [p["id"] for p in photos], [base, close])
    offset = library_id * store.PERSON_ID_STRIDE
    store.replace_persons(library_id, [
        {"id": offset + 1, "label": "Person 1", "face_count": 1, "photo_count": 1,
         "keep_count": 1, "covered": 1, "required": 1,
         "representative_photo_id": photos[0]["id"],
         "representative_face_id": face_ids[0], "max_area_ratio": 0.05},
        {"id": offset + 2, "label": "Person 2", "face_count": 1, "photo_count": 1,
         "keep_count": 1, "covered": 1, "required": 1,
         "representative_photo_id": photos[1]["id"],
         "representative_face_id": face_ids[1], "max_area_ratio": 0.05},
    ], list(zip([offset + 1, offset + 2], face_ids)))

    suggestions = client.get(f"/api/libraries/{library_id}/persons/suggestions").json()
    assert len(suggestions) == 1
    assert (suggestions[0]["a"], suggestions[0]["b"]) == (offset + 1, offset + 2)
    assert suggestions[0]["a_label"] == "Person 1"
    assert suggestions[0]["similarity"] == pytest.approx(0.87, abs=0.02)

    # "Different people" stops the suggestion coming back.
    assert client.post(f"/api/libraries/{library_id}/persons/distinct",
                       json={"ids": [offset + 1, offset + 2]}).status_code == 200
    assert client.get(f"/api/libraries/{library_id}/persons/suggestions").json() == []

    # "Same person" folds them together and clears the dismissal.
    merged = client.post(f"/api/libraries/{library_id}/persons/merge",
                         json={"ids": [offset + 2, offset + 1]}).json()
    assert merged["merged_into"] == offset + 1
    assert len(merged["persons"]) == 1
    assert merged["persons"][0]["photo_count"] == 2
    assert store.person_links(library_id, "distinct") == []
    assert client.get(f"/api/libraries/{library_id}/persons/suggestions").json() == []

    unknown = client.post(f"/api/libraries/{library_id}/persons/merge",
                          json={"ids": [offset + 1, offset + 999]})
    assert unknown.status_code == 400


def test_export_remembers_where_it_went_per_shoot(client, photo_dir, tmp_path):
    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    library_id = library["id"]
    (tmp_path / "other").mkdir()
    other_id = client.post("/api/libraries",
                           json={"path": str(tmp_path / "other")}).json()["id"]

    destination = tmp_path / "delivery"
    client.post(f"/api/libraries/{library_id}/export",
                json={"mode": "copy", "dest": str(destination)})

    memory = client.get(f"/api/libraries/{library_id}").json()["export"]
    assert memory == {"mode": "copy", "dest": str(destination)}
    assert destination.is_dir()

    # Another shoot never inherits it.
    assert client.get(f"/api/libraries/{other_id}").json()["export"] is None
    listed = {row["id"]: row for row in client.get("/api/libraries").json()}
    assert listed[library_id]["export"]["dest"] == str(destination)
    assert listed[other_id]["export"] is None

    # A rejected export does not overwrite what worked last time.
    failed = client.post(f"/api/libraries/{library_id}/export", json={"mode": "copy"})
    assert failed.status_code == 400
    assert client.get(f"/api/libraries/{library_id}").json()["export"] == memory

    # Sidecars have no destination, and they do not wipe the remembered folder.
    client.post(f"/api/libraries/{library_id}/export", json={"mode": "sidecar_all"})
    after = client.get(f"/api/libraries/{library_id}").json()["export"]
    assert after == {"mode": "sidecar_all", "dest": str(destination)}


def test_folder_listing_backs_the_export_picker(client, tmp_path):
    root = tmp_path / "delivery"
    (root / "b" / "nested").mkdir(parents=True)
    (root / "A").mkdir()
    (root / ".hidden").mkdir()
    (root / "file.jpg").write_bytes(b"not really a jpeg")

    body = client.get("/api/folders", params={"path": str(root)}).json()
    assert body["path"] == str(root)
    # Folders only, hidden ones skipped, case-insensitive order.
    assert [f["name"] for f in body["folders"]] == ["A", "b"]
    assert body["folders"][1]["path"] == str(root / "b")
    assert body["parent"] == str(tmp_path)
    assert body["writable"] is True

    assert client.get("/api/folders", params={"path": str(tmp_path / "nope")}).status_code == 400
    assert client.get("/api/folders",
                      params={"path": str(root / "file.jpg")}).status_code == 400


def test_folder_listing_is_finder_shaped(client, tmp_path):
    root = tmp_path / "browse"
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "a.jpg").write_bytes(b"x")
    (root / "notes.txt").write_text("hi")
    (root / ".hidden.txt").write_text("no")
    # A file that claims to be a JPEG but cannot be read: still a file, no size.
    (root / "frame.jpg").write_bytes(b"not really a jpeg")

    body = client.get("/api/folders", params={"path": str(root)}).json()
    entries = {entry["name"]: entry for entry in body["entries"]}

    assert body["entries"][0]["kind"] == "folder"      # folders come first
    assert entries["keep"]["items"] == 1                # hidden children not counted
    assert entries["keep"]["writable"] is True
    assert entries["notes.txt"]["kind"] == "file"
    assert entries["notes.txt"]["size"] == 2
    assert entries["frame.jpg"]["kind"] == "image"
    assert "width" not in entries["frame.jpg"]
    assert ".hidden.txt" not in entries
    # The old contract still holds: directories only, in name order.
    assert [folder["name"] for folder in body["folders"]] == ["keep"]

    sidebar = body["sidebar"]
    assert sidebar["favorites"][0]["name"] == "Home"
    assert any(place["path"] == "/" for place in sidebar["locations"])
    assert isinstance(sidebar["shared"], list)


def test_folder_preview_serves_a_tile_only_for_images(client, photo_dir):
    tile = client.get("/api/folders/preview", params={"path": str(photo_dir / "IMG_0001.jpg")})
    assert tile.status_code == 200
    assert tile.headers["content-type"] == "image/jpeg"
    assert tile.content[:2] == b"\xff\xd8"                 # JPEG SOI
    # Second call comes from the cache and is byte-identical.
    again = client.get("/api/folders/preview", params={"path": str(photo_dir / "IMG_0001.jpg")})
    assert again.content == tile.content

    notes = photo_dir / "notes.txt"
    notes.write_text("not an image")
    assert client.get("/api/folders/preview", params={"path": str(notes)}).status_code == 400
    missing = client.get("/api/folders/preview", params={"path": str(photo_dir / "nope.jpg")})
    assert missing.status_code == 404


def test_folder_can_be_created_while_browsing(client, tmp_path):
    root = tmp_path / "delivery"
    root.mkdir()

    created = client.post("/api/folders", json={"parent": str(root), "name": " 2026-06-15 Wedding "})
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == " 2026-06-15 Wedding".strip()
    assert (root / body["name"]).is_dir()
    # It shows up the next time the picker lists that folder.
    names = [f["name"] for f in client.get("/api/folders", params={"path": str(root)}).json()["folders"]]
    assert body["name"] in names

    again = client.post("/api/folders", json={"parent": str(root), "name": body["name"]})
    assert again.status_code == 409

    for bad in ("", "   ", ".", "..", ".cache", "nested/inside", "back\\slash"):
        response = client.post("/api/folders", json={"parent": str(root), "name": bad})
        assert response.status_code in (400, 422), bad

    missing = client.post("/api/folders", json={"parent": str(tmp_path / "nope"), "name": "x"})
    assert missing.status_code == 400
    assert not (tmp_path / "nope").exists()


def test_unknown_photo_is_404(client):
    assert client.get("/api/photos/9999").status_code == 404
    assert client.get("/api/photos/9999/image").status_code == 404


def test_people_are_listed_filterable_and_have_thumbnails(client, photo_dir):
    """The persons API plus face crops, seeded directly so no face model is needed."""
    import numpy as np

    library = client.post("/api/libraries", json={"path": str(photo_dir)}).json()
    library_id = library["id"]
    client.post(f"/api/libraries/{library_id}/scan", json={})
    wait_for_scan(client, library_id)

    store = client.app.state.store
    photo = client.get(f"/api/libraries/{library_id}/photos").json()["photos"][0]
    row = store.photo_row(photo["id"])
    technical = {key: row.get(key) for key in (
        "width", "height", "captured_at", "camera", "lens", "iso", "shutter",
        "aperture", "focal_len", "sharpness", "tenengrad", "clip_hi", "clip_lo",
        "mean_luma", "noise", "contrast", "phash", "dhash")}
    vector = np.random.default_rng(0).normal(size=117).astype(np.float32)
    technical["faces"] = [{"x": 200, "y": 150, "w": 300, "h": 300,
                           "area_ratio": 0.05, "eyes_open": 9.0,
                           "embedding": vector / np.linalg.norm(vector)}]
    store.update_analysis(photo["id"], technical, version=2)

    face = store.faces_for([photo["id"]])[photo["id"]][0]
    person_id = library_id * store.PERSON_ID_STRIDE + 1
    store.replace_persons(library_id, [{
        "id": person_id, "label": "Person 1", "face_count": 1, "photo_count": 1,
        "keep_count": 1, "covered": 1, "required": 1,
        "representative_photo_id": photo["id"],
        "representative_face_id": face["id"], "max_area_ratio": 0.05,
    }], [(person_id, face["id"])])

    people = client.get(f"/api/libraries/{library_id}/persons").json()
    assert [p["id"] for p in people] == [person_id]
    assert people[0]["label"] == "Person 1"
    assert people[0]["covered"] == 1

    filtered = client.get(f"/api/libraries/{library_id}/photos",
                          params={"person_id": person_id}).json()
    assert [p["id"] for p in filtered["photos"]] == [photo["id"]]
    assert filtered["photos"][0]["faces"][0]["person_id"] == person_id

    crop = client.get(f"/api/photos/{photo['id']}/face/{face['id']}")
    assert crop.status_code == 200
    assert crop.headers["content-type"] == "image/jpeg"
    assert len(crop.content) > 500
    assert client.get(f"/api/photos/{photo['id']}/face/999999").status_code == 404


def test_capabilities_report_whether_the_scene_model_can_be_used(client):
    body = client.get("/api/capabilities").json()
    assert isinstance(body["semantic_runtime"], bool)
    assert isinstance(body["semantic"], bool)
    # A downloaded model without the runtime is not usable, so the UI is told
    # about the runtime separately and only offers the toggle when it is there.
    assert not body["semantic"] or body["semantic_runtime"]


def test_semantic_status_reports_model_state(client):
    body = client.get("/api/semantic").json()
    assert {"runtime", "model", "available", "downloading", "done", "total",
            "error"} <= set(body)
    assert body["downloading"] is False
    assert body["error"] is None


def test_scene_model_download_is_refused_without_the_runtime(client, monkeypatch):
    monkeypatch.setattr("culler.api.runtime_available", lambda: False)
    response = client.post("/api/semantic/model")
    assert response.status_code == 409
    assert "semantic" in response.json()["detail"]


def test_scene_model_download_runs_once_and_reports_progress(client, monkeypatch):
    """Clicking the toggle twice must not start two downloads."""
    import threading

    import culler.api as api_module

    calls: list[int] = []
    started = threading.Event()
    release = threading.Event()

    def fake_download(progress=None):
        calls.append(1)
        if progress:
            progress(10, 42)
        started.set()
        release.wait(5.0)

    monkeypatch.setattr(api_module, "runtime_available", lambda: True)
    monkeypatch.setattr(api_module, "model_present", lambda: False)
    monkeypatch.setattr(api_module, "model_verified", lambda: False)
    monkeypatch.setattr(api_module, "download_model", fake_download)

    first = client.post("/api/semantic/model").json()
    assert first["runtime"] is True
    assert started.wait(5.0), "the download thread never started"

    # The second click lands while the first download is still in flight.
    second = client.post("/api/semantic/model").json()
    assert second["downloading"] is True
    release.set()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        body = client.get("/api/semantic").json()
        if not body["downloading"]:
            break
        time.sleep(0.02)
    assert body["downloading"] is False
    assert body["error"] is None
    assert calls == [1]
    assert (body["done"], body["total"]) == (10, 42)


def test_semantic_dedup_round_trips_with_the_library(tmp_path):
    root = tmp_path / "shoot"
    root.mkdir()
    app = create_app(tmp_path / "data")
    with TestClient(app) as client:
        created = client.post("/api/libraries", json={
            "path": str(root),
            "settings": {"semantic_dedup": True, "embedding_threshold": 0.9},
        }).json()
        assert created["settings"]["semantic_dedup"] is True
        assert created["settings"]["embedding_threshold"] == 0.9

        listed = client.get("/api/libraries").json()[0]
        assert listed["settings"]["semantic_dedup"] is True

        # Off again: a library that never opts in keeps today's hash-only path.
        assert client.get("/api/capabilities").json()["default_settings"][
            "semantic_dedup"] is False


def test_the_picker_finds_a_folder_past_the_400_entry_window(client, tmp_path):
    """The shoot folder in the video held ~6,000 entries. The grid renders the
    first 400, so a destination created inside it still has to be reachable:
    search asks the server about the whole folder, not about the window."""
    root = tmp_path / "shoot"
    root.mkdir()
    for index in range(405):
        (root / f"shoot_{index:04d}").mkdir()
    (root / "zzz-delivery").mkdir()      # sorts last, past the window

    listing = client.get("/api/folders", params={"path": str(root)}).json()
    assert listing["truncated"] is True
    assert listing["limit"] == 400
    shown = [entry["name"] for entry in listing["entries"]]
    assert len(shown) == 400
    assert "zzz-delivery" not in shown

    found = client.get("/api/folders",
                       params={"path": str(root), "q": "delivery"}).json()
    assert [entry["name"] for entry in found["entries"]] == ["zzz-delivery"]
    assert found["truncated"] is False


def test_the_picker_lists_folders_before_files(client, tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "delivery").mkdir()
    for index in range(30):
        (root / f"IMG_{index:04d}.jpg").write_bytes(b"")

    kinds = [entry["kind"] for entry in
             client.get("/api/folders", params={"path": str(root)}).json()["entries"]]
    assert kinds[:1] == ["folder"]
    assert "folder" not in kinds[1:]
