from __future__ import annotations

from pathlib import Path

import pytest

from culler.config import CullingSettings
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
