from __future__ import annotations

from pathlib import Path

import pytest

from culler.config import RATING_CATEGORIES, CullingSettings, default_rating_scheme
from culler.db import Store
from culler.export import copy_keepers, write_csv, write_sidecars
from culler.pipeline import run_scan


@pytest.fixture
def library(store_and_folder):
    store, folder = store_and_folder
    library = store.create_or_get_library(str(folder))
    run_scan(store, library["id"], CullingSettings(), force=True)
    return store, library["id"], folder


@pytest.fixture
def store_and_folder(tmp_path: Path, photo_dir):
    return Store(tmp_path / "catalog.db"), photo_dir


def test_sidecars_carry_rating_and_label(library):
    store, library_id, folder = library
    result = write_sidecars(store, library_id, only_raw=False)
    assert result.written == 6
    assert result.errors == []

    sidecars = sorted(folder.glob("*.xmp"))
    assert len(sidecars) == 6

    keepers = [r for r in store.photos(library_id) if r["verdict"] == "keep"]
    rejects = [r for r in store.photos(library_id) if r["verdict"] == "reject"]
    assert keepers and rejects

    keep_text = (folder / f"{keepers[0]['filename']}.xmp").read_text()
    assert 'xmp:Rating="1"' in keep_text
    assert 'xmp:Label="Green"' in keep_text

    reject_text = (folder / f"{rejects[0]['filename']}.xmp").read_text()
    assert 'xmp:Rating="-1"' in reject_text
    assert 'xmp:Label="Red"' in reject_text


def test_sidecar_only_raw_by_default(library):
    store, library_id, _folder = library
    result = write_sidecars(store, library_id, only_raw=True)
    assert result.written == 0
    assert result.skipped == 6


def test_sidecars_can_leave_existing_files_alone(library):
    store, library_id, folder = library
    keeper = next(r for r in store.photos(library_id) if r["verdict"] == "keep")
    sidecar = folder / f"{keeper['filename']}.xmp"
    sidecar.write_text("hand-rated in Lightroom", encoding="utf-8")

    result = write_sidecars(store, library_id, only_raw=False, overwrite=False)
    assert result.skipped >= 1
    assert sidecar.read_text(encoding="utf-8") == "hand-rated in Lightroom"

    # The default still replaces it, which is what an export is expected to do.
    write_sidecars(store, library_id, only_raw=False)
    assert "xmp:Rating" in sidecar.read_text(encoding="utf-8")


def test_star_overrides_rating(library):
    store, library_id, folder = library
    keeper = next(r for r in store.photos(library_id) if r["verdict"] == "keep")
    store.set_user_decision(keeper["id"], "pick", star=5)
    write_sidecars(store, library_id, only_raw=False)
    text = (folder / f"{keeper['filename']}.xmp").read_text()
    assert 'xmp:Rating="5"' in text


def test_copy_keepers_creates_one_file_per_keeper(library, tmp_path):
    store, library_id, _folder = library
    dest = tmp_path / "delivery"
    result = copy_keepers(store, library_id, str(dest))
    keepers = [r for r in store.photos(library_id) if r["verdict"] == "keep"]
    assert result.copied == len(keepers)
    assert len(list(dest.iterdir())) == len(keepers)


def test_csv_report_matches_photo_count(library, tmp_path):
    store, library_id, _folder = library
    target = tmp_path / "report.csv"
    result = write_csv(store, library_id, str(target))
    lines = target.read_text().strip().splitlines()
    assert result.written == 6
    assert len(lines) == 7  # header + 6 photos
    assert lines[0].startswith("filename,path,decision,decided_by,reason")


def test_a_custom_scheme_is_written_to_the_sidecar(library):
    store, library_id, folder = library
    keeper = next(r for r in store.photos(library_id) if r["verdict"] == "keep")
    scheme = default_rating_scheme()
    scheme["selected"] = {"stars": 2, "color": "Blue"}

    write_sidecars(store, library_id, only_raw=False, scheme=scheme)

    text = (folder / f"{keeper['filename']}.xmp").read_text()
    assert 'xmp:Rating="2"' in text
    assert 'xmp:Label="Blue"' in text


def test_each_reject_reason_reads_its_own_row(library):
    store, library_id, folder = library
    scheme = default_rating_scheme()
    scheme["duplicate"] = {"stars": 0, "color": "Yellow"}
    scheme["blurry"] = {"stars": 3, "color": "Purple"}

    write_sidecars(store, library_id, only_raw=False, scheme=scheme)

    rows = {r["filename"]: r for r in store.photos(library_id)}
    duplicate = next(r for r in rows.values() if r["reason"] == "duplicate")
    assert 'xmp:Label="Yellow"' in (folder / f"{duplicate['filename']}.xmp").read_text()

    blurred = rows["IMG_0102.jpg"]       # the deliberately soft burst frame
    assert blurred["reason"] == "blurry"
    blurred_text = (folder / "IMG_0102.jpg.xmp").read_text()
    assert 'xmp:Rating="3"' in blurred_text
    assert 'xmp:Label="Purple"' in blurred_text


def test_a_chosen_star_still_wins_over_the_scheme(library):
    store, library_id, folder = library
    keeper = next(r for r in store.photos(library_id) if r["verdict"] == "keep")
    store.set_user_decision(keeper["id"], "pick", star=5)
    scheme = default_rating_scheme()
    scheme["selected"] = {"stars": 2, "color": "Blue"}

    write_sidecars(store, library_id, only_raw=False, scheme=scheme)

    text = (folder / f"{keeper['filename']}.xmp").read_text()
    assert 'xmp:Rating="5"' in text
    assert 'xmp:Label="Blue"' in text


def test_the_untouched_scheme_is_the_old_hardcoded_pair():
    """The default is not a new default: it is exactly what shipped before."""
    scheme = CullingSettings().rating_scheme
    assert scheme["selected"] == {"stars": 1, "color": "Green"}
    for key in ("highlights", "blurry", "closed_eyes", "duplicate", "unrated"):
        assert scheme[key] == {"stars": -1, "color": "Red"}
