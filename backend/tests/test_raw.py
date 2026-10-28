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

The fixtures are CC0 samples from raw.pixls.us (see NOTICE.md), committed so
that both the decode path and the release build's own smoke test can be
verified without a camera attached.

Both containers are kept because they are not the same test. The ORF proves
the rawpy path works at all; the CR3 proves it works on the container that
actually failed, Canon's CRX (an ISO-BMFF box structure, `ftypcrx`, not a
TIFF like every other format here). A build can carry a working rawpy and
still fail on CR3 alone — LibRaw has to be compiled with the CR3 decoder — so
pinning the ORF would have let 0.8.0 through even with the extra installed.
Measured with rawpy 0.27.1: the CR3 decodes to 3888x2592 and renders a
37,067-byte preview.
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
CR3_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "canon_eos_rp.cr3"

# Each fixture's own container magic, at the offset that format puts it: an ORF
# is a TIFF-family file starting with Olympus's "IIRS"; a CR3 is an ISO-BMFF
# file whose brand sits after the box length, at byte 4.
SAMPLES = [
    pytest.param(FIXTURE, id="orf"),
    pytest.param(CR3_FIXTURE, id="cr3"),
]
CONTAINER_MAGIC = {
    FIXTURE.name: (0, b"IIRS"),
    CR3_FIXTURE.name: (4, b"ftypcrx"),
}
NEEDS_RAWPY = pytest.mark.skipif(
    not imaging.RAWPY_AVAILABLE,
    reason="the `raw` extra is not installed in this environment",
)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as test_client:
        yield test_client


def only_raw_folder(tmp_path: Path, sample: Path = FIXTURE) -> Path:
    """A shoot folder holding exactly one RAW fixture."""
    folder = tmp_path / "raw-shoot"
    folder.mkdir()
    (folder / sample.name).write_bytes(sample.read_bytes())
    return folder


def without_rawpy(monkeypatch) -> None:
    """Simulate a build that shipped without the extra, as 0.8.0 did."""
    monkeypatch.setattr(imaging, "rawpy", None)
    monkeypatch.setattr(imaging, "RAWPY_AVAILABLE", False)


@pytest.mark.parametrize("sample", SAMPLES)
def test_each_fixture_is_a_real_raw_file(sample: Path):
    assert sample.is_file(), f"the RAW fixture is missing: {sample.name}"
    assert sample.suffix.lower() in RAW_EXTENSIONS
    # A truncated or placeholder file would keep the extension but could never
    # exercise decoding, so the container's own magic is asserted.
    offset, magic = CONTAINER_MAGIC[sample.name]
    assert sample.read_bytes()[offset:offset + len(magic)] == magic
    assert sample.stat().st_size > 1_000_000


@NEEDS_RAWPY
@pytest.mark.parametrize("sample", SAMPLES)
def test_a_raw_frame_decodes_and_renders_a_preview(sample: Path):
    bgr, meta, error = imaging.load_image(sample)
    assert error is None, f"rawpy is installed but the RAW did not decode: {error}"
    assert bgr is not None
    assert meta.width > 0 and meta.height > 0
    preview = imaging.render_jpeg(sample)
    assert preview is not None and preview[:2] == b"\xff\xd8"


@NEEDS_RAWPY
def test_a_cr3_parses_its_own_metadata():
    """The CR3 is decoded by LibRaw's CRX path, not by the TIFF one.

    A build whose LibRaw lacks CR3 support fails here and nowhere else: the ORF
    above still decodes, which is exactly how a "RAW works on my machine" venv
    can hide a shipped build that cannot read Canon's files at all.
    """
    _bgr, meta, error = imaging.load_image(CR3_FIXTURE)
    assert error is None, f"the CR3 did not decode: {error}"
    assert (meta.width, meta.height) == (3888, 2592)
    assert meta.camera and "EOS RP" in meta.camera
    assert imaging.render_jpeg(CR3_FIXTURE) is not None


@NEEDS_RAWPY
@pytest.mark.parametrize("sample", SAMPLES)
def test_a_scan_reads_every_raw_frame_instead_of_failing_them(tmp_path: Path,
                                                             sample: Path):
    """The 0.8.0 symptom, inverted: a RAW shoot scans with no read errors."""
    store = Store(tmp_path / "catalog.db")
    folder = only_raw_folder(tmp_path, sample)
    library = store.create_or_get_library(str(folder))
    result = run_scan(store, library["id"], CullingSettings(), force=True)

    assert result.total == 1
    assert result.errors == [], f"the RAW frame was not read: {result.errors}"
    row = store.photos(library["id"])[0]
    assert row["error"] is None
    assert row["reason"] != "unreadable"
    assert row["quality"] is not None, "the frame was never scored"


@pytest.mark.parametrize("sample", SAMPLES)
def test_a_build_without_rawpy_reports_the_file_as_unreadable(tmp_path: Path,
                                                             monkeypatch,
                                                             sample: Path):
    """The 0.8.0 behaviour, pinned: this is what the missing extra looked like.

    Runs in every environment — including one that has rawpy installed — so the
    failure the released build showed stays covered without needing a second
    virtualenv.
    """
    without_rawpy(monkeypatch)
    bgr, meta, error = imaging.load_image(sample)
    assert bgr is None
    assert meta.width == 0 and meta.height == 0
    assert error is not None
    assert error.startswith("UnidentifiedImageError: cannot identify image file")
    assert imaging.render_jpeg(sample) is None

    store = Store(tmp_path / "catalog.db")
    folder = only_raw_folder(tmp_path, sample)
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


def test_the_bundle_smoke_test_decodes_a_cr3():
    """The build gate must exercise the container that actually failed.

    A smoke test that previews only the ORF passes on a LibRaw without the CR3
    decoder, which is the shape of the 0.8.0 failure: RAW is "supported" right
    up until the shoot is Canon.
    """
    script = (ROOT / "desktop" / "build_macos.sh").read_text()
    assert CR3_FIXTURE.name in script, (
        "the bundle smoke test no longer decodes a CR3, so a build could ship "
        "without CR3 support and still pass")
    assert "filter=errors" in script, (
        "the bundle smoke test no longer runs a cull over the CR3, so it only "
        "proves the preview path")
