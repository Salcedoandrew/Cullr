"""RAW decoding, and the packaging contract that keeps it in the build.

0.8.0 shipped without the `raw` extra, and the effect was not subtle: the
import screen counted 3,016 `.CR3` files, the dialog promised a cull of all
3,016, and the scan rejected every one of them as unreadable — 0 keepers, 0
groups, no previews. `rawpy` was an optional extra (`pip install .[raw]`) that
`release.yml` never installed, so a developer venv with it installed read RAW
perfectly while every released build could not read a single RAW file.

The numbers below were measured in this environment, not guessed:

* Baseline (the release's extras — no rawpy): `load_image()` on the fixture
  returns `UnidentifiedImageError: cannot identify image file '…ORF'`, the
  preview renders 0 bytes, and the scan reports the file as
  `reject / reason="unreadable"`.
* With rawpy installed: the fixture decodes (2576x1926 RAW, 160x120 embedded
  JPEG preview), `render_jpeg()` returns 19,335 bytes, the scan reports no
  errors, and the frame is scored on its merits (`quality=0.47`, rejected for
  `blurry` — a 160x120 preview upscaled to the 1600 px analysis frame *is*
  soft, which is the app's documented preview-based RAW behaviour).

The fixture is a CC0 sample from raw.pixls.us (see NOTICE.md), committed so
that both the decode path and the release build's own smoke test can be
verified without a camera attached.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from culler import imaging
from culler.api import create_app
from culler.config import RAW_EXTENSIONS, CullingSettings
from culler.db import Store
from culler.pipeline import run_scan

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "olympus_c5050z.orf"
NEEDS_RAWPY = pytest.mark.skipif(
    not imaging.RAWPY_AVAILABLE,
    reason="the `raw` extra is not installed in this environment",
)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as test_client:
        yield test_client


def only_raw_folder(tmp_path: Path) -> Path:
    """A shoot folder holding exactly the RAW fixture."""
    folder = tmp_path / "raw-shoot"
    folder.mkdir()
    (folder / FIXTURE.name).write_bytes(FIXTURE.read_bytes())
    return folder


def without_rawpy(monkeypatch) -> None:
    """Simulate a build that shipped without the extra, as 0.8.0 did."""
    monkeypatch.setattr(imaging, "rawpy", None)
    monkeypatch.setattr(imaging, "RAWPY_AVAILABLE", False)


def test_the_fixture_is_a_real_raw_file():
    assert FIXTURE.is_file(), "the RAW fixture is missing from the repository"
    assert FIXTURE.suffix.lower() in RAW_EXTENSIONS
    # An ORF is a little-endian TIFF-family container carrying Olympus's own
    # magic ("IIRS"): a truncated or placeholder file would keep the extension
    # but could never exercise decoding.
    assert FIXTURE.read_bytes()[:4] == b"IIRS"
    assert FIXTURE.stat().st_size > 1_000_000


@NEEDS_RAWPY
def test_a_raw_frame_decodes_and_renders_a_preview():
    bgr, meta, error = imaging.load_image(FIXTURE)
    assert error is None, f"rawpy is installed but the RAW did not decode: {error}"
    assert bgr is not None
    assert meta.width > 0 and meta.height > 0
    preview = imaging.render_jpeg(FIXTURE)
    assert preview is not None and preview[:2] == b"\xff\xd8"


@NEEDS_RAWPY
def test_a_scan_reads_every_raw_frame_instead_of_failing_them(tmp_path: Path):
    """The 0.8.0 symptom, inverted: a RAW shoot scans with no read errors."""
    store = Store(tmp_path / "catalog.db")
    folder = only_raw_folder(tmp_path)
    library = store.create_or_get_library(str(folder))
    result = run_scan(store, library["id"], CullingSettings(), force=True)

    assert result.total == 1
    assert result.errors == [], f"the RAW frame was not read: {result.errors}"
    row = store.photos(library["id"])[0]
    assert row["error"] is None
    assert row["reason"] != "unreadable"
    assert row["quality"] is not None, "the frame was never scored"


def test_a_build_without_rawpy_reports_the_file_as_unreadable(tmp_path: Path,
                                                             monkeypatch):
    """The 0.8.0 behaviour, pinned: this is what the missing extra looked like.

    Runs in every environment — including one that has rawpy installed — so the
    failure the released build showed stays covered without needing a second
    virtualenv.
    """
    without_rawpy(monkeypatch)
    bgr, meta, error = imaging.load_image(FIXTURE)
    assert bgr is None
    assert meta.width == 0 and meta.height == 0
    assert error is not None
    assert error.startswith("UnidentifiedImageError: cannot identify image file")
    assert imaging.render_jpeg(FIXTURE) is None

    store = Store(tmp_path / "catalog.db")
    folder = only_raw_folder(tmp_path)
    library = store.create_or_get_library(str(folder))
    result = run_scan(store, library["id"], CullingSettings(), force=True)
    assert result.keepers == 0
    row = store.photos(library["id"])[0]
    assert row["reason"] == "unreadable"
    assert row["error"] is not None


def test_capabilities_report_whether_this_build_can_decode_raw(client):
    body = client.get("/api/capabilities").json()
    assert body["raw"] is imaging.RAWPY_AVAILABLE


@pytest.mark.parametrize("workflow", ["release.yml", "ci.yml"])
def test_the_build_installs_the_raw_extra(workflow: str):
    """The bug was a packaging omission, so the packaging is asserted.

    The frozen-bundle smoke test in `desktop/build_macos.sh` is the real gate;
    this catches the omission at the point it is written, on every CI run,
    rather than at tag time.
    """
    text = (ROOT / ".github" / "workflows" / workflow).read_text()
    extras = re.findall(r"backend\[([^\]]+)\]", text)
    assert extras, f"{workflow} no longer installs the backend package"
    assert any("raw" in part.split(",") for part in extras), (
        f"{workflow} installs {extras}, which cannot decode RAW files")
