from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from culler.config import CullingSettings
from culler.db import Store
from culler.metrics.faces import FaceMetrics
from culler.metrics.technical import TechnicalMetrics
from culler.imaging import ImageMeta
from culler.pipeline import AnalysedPhoto, plan, run_scan


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "catalog.db")


def scan(store: Store, folder: Path, settings: CullingSettings | None = None,
         force: bool = False):
    library = store.create_or_get_library(str(folder))
    return library["id"], run_scan(store, library["id"], settings or CullingSettings(),
                                   force=force)


def test_scan_scores_and_groups_a_whole_shoot(store, photo_dir):
    library_id, result = scan(store, photo_dir)
    assert result.total == 6
    assert result.analysed == 6
    assert result.failed == 0
    assert result.groups >= 2, "expected the duplicate pair and the burst trio to group"

    rows = store.photos(library_id)
    assert len(rows) == 6
    assert all(r["quality"] is not None for r in rows)

    keepers = [r for r in rows if r["verdict"] == "keep"]
    assert keepers, "a shoot with sharp frames must produce keepers"

    # Every group has exactly one lead, and the duplicates are explained.
    leads: dict[int, int] = {}
    for row in rows:
        if row["group_id"] is not None:
            leads[row["group_id"]] = leads.get(row["group_id"], 0) + int(row["is_lead"])
    assert all(count == 1 for count in leads.values())
    assert any(r["reason"] == "duplicate" for r in rows)

    # The heavily blurred burst frame must be rejected for blur, not for being a dupe.
    blurred = next(r for r in rows if r["filename"] == "IMG_0102.jpg")
    assert blurred["verdict"] == "reject"


def test_rescan_reuses_analysis_and_keeps_human_decisions(store, photo_dir):
    library_id, first = scan(store, photo_dir)
    rows = store.photos(library_id)
    target = rows[0]

    store.set_user_decision(target["id"], "pick", star=5)
    _library_id, second = scan(store, photo_dir)

    assert second.reused == second.total
    assert second.analysed == 0

    again = store.photo_row(target["id"])
    assert again["user_decision"] == "pick"
    assert again["star"] == 5


def test_force_reanalyses_everything(store, photo_dir):
    library_id, _ = scan(store, photo_dir)
    _library_id, forced = scan(store, photo_dir, force=True)
    assert forced.analysed == forced.total
    assert forced.reused == 0


def test_empty_folder_is_not_an_error(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    _library_id, result = scan(store, empty)
    assert result.total == 0
    assert result.groups == 0


def test_keep_min_per_group_is_respected(store, photo_dir):
    library_id, _ = scan(store, photo_dir,
                         CullingSettings(preset="wedding", keep_min_per_group=2))
    rows = store.photos(library_id)
    keepers_per_group: dict[int, int] = {}
    for row in rows:
        if row["group_id"] is not None and row["verdict"] == "keep":
            keepers_per_group[row["group_id"]] = keepers_per_group.get(row["group_id"], 0) + 1
    assert keepers_per_group, "expected keepers inside groups"
    assert all(count <= 2 for count in keepers_per_group.values())


def test_target_ratio_caps_keepers_and_says_why(store, photo_dir):
    library_id, _ = scan(store, photo_dir)
    before = {r["id"] for r in store.photos(library_id) if r["verdict"] == "keep"}
    assert len(before) > 1, "the cap needs more than one keeper to bite"

    # 6 frames at 17% => 1 keeper, so a quality-ranked cap is the only way.
    _library_id, result = scan(store, photo_dir,
                               CullingSettings(preset="wedding", target_ratio=0.17))
    rows = store.photos(library_id)
    kept = [r for r in rows if r["verdict"] == "keep"]
    dropped = [r for r in rows if r["reason"] == "target"]

    assert len(kept) == 1, kept
    assert result.keepers == 1
    # The cap ranks by measured quality: the survivor is the best frame.
    assert kept[0]["quality"] == max(r["quality"] for r in rows)
    # Only frames that were keepers can lose to the target, and nothing else is
    # relabelled: a soft frame stays "blurry", it does not become "target".
    assert {r["id"] for r in dropped} == before - {kept[0]["id"]}
    assert all(r["verdict"] == "reject" for r in dropped)


def test_target_above_actual_keepers_changes_nothing(store, photo_dir):
    base_id, base = scan(store, photo_dir)
    baseline = {r["id"] for r in store.photos(base_id) if r["verdict"] == "keep"}

    generous_id, _ = scan(store, photo_dir, CullingSettings(preset="wedding", target_ratio=1.0))
    still_kept = {r["id"] for r in store.photos(generous_id) if r["verdict"] == "keep"}
    assert still_kept == baseline
    assert base.keepers == len(baseline)


def test_unreadable_file_is_recorded_not_fatal(store, photo_dir, tmp_path):
    broken = photo_dir / "IMG_9999.jpg"
    broken.write_bytes(b"not a jpeg at all")
    library_id, result = scan(store, photo_dir)
    assert result.failed == 1
    row = next(r for r in store.photos(library_id) if r["filename"] == "IMG_9999.jpg")
    assert row["error"]
    assert row["verdict"] == "reject"


def test_stats_match_photo_rows(store, photo_dir):
    library_id, _ = scan(store, photo_dir)
    stats = store.stats(library_id)
    rows = store.photos(library_id)
    assert stats["total"] == len(rows)
    assert stats["keep"] == sum(1 for r in rows if r["verdict"] == "keep")
    assert stats["reject"] == sum(1 for r in rows if r["verdict"] == "reject")
    assert stats["errors"] == sum(1 for r in rows if r["error"])


def test_files_the_scan_could_not_read_are_their_own_filter(store, photo_dir):
    """A frame rejected for soft focus was read and judged; a frame with an
    error never was. Both end up in Rejected, and telling them apart is the
    difference between "this shoot is soft" and "this build cannot read my
    camera" -- which is what the sidebar's Unreadable row is for."""
    broken = photo_dir / "IMG_9999.jpg"
    broken.write_bytes(b"not a jpeg at all")
    library_id, result = scan(store, photo_dir)

    assert result.failed == 1
    stats = store.stats(library_id)
    assert stats["errors"] == 1
    assert stats["reject"] == stats["total"] - stats["keep"]
    assert stats["errors"] < stats["reject"], "the unreadable file is not a reason"

    unreadable = store.photos(library_id, "errors")
    assert [row["filename"] for row in unreadable] == ["IMG_9999.jpg"]
    assert unreadable[0]["error"]


def test_two_libraries_can_both_have_groups(store, photo_dir, tmp_path):
    """groups.id is a global primary key; each library numbers from 1."""
    import shutil

    second = tmp_path / "second_shoot"
    shutil.copytree(photo_dir, second)

    first_id, first = scan(store, photo_dir)
    second_id, second_result = scan(store, second)

    assert first.groups >= 1 and second_result.groups >= 1
    first_groups = {r["group_id"] for r in store.photos(first_id) if r["group_id"]}
    second_groups = {r["group_id"] for r in store.photos(second_id) if r["group_id"]}
    assert first_groups and second_groups
    assert first_groups.isdisjoint(second_groups)


def measured_frame(photo_id: int, sharpness: float, phash: str | None = None,
                   captured_at: float | None = None) -> AnalysedPhoto:
    """A stand-in analysed frame: sharpness is the only thing varying."""
    return AnalysedPhoto(
        photo_id=photo_id,
        path=f"/shoot/{photo_id}.jpg",
        filename=f"{photo_id}.jpg",
        meta=ImageMeta(width=1200, height=800, captured_at=captured_at),
        tech=TechnicalMetrics(
            sharpness=sharpness, tenengrad=sharpness * 60, clip_hi=0.0,
            clip_lo=0.02, mean_luma=0.5, p95_luma=0.9, noise=4.0, contrast=60.0,
        ),
        faces=FaceMetrics(faces=[]),
        phash=phash or hashlib.sha1(str(photo_id).encode()).hexdigest()[:16],
        dhash=phash or hashlib.sha1(str(photo_id).encode()).hexdigest()[:16],
    )


def test_the_auto_target_stop_sets_how_many_frames_survive():
    """The floor is what makes the stop's name true -- not the gate scaling.

    Gates alone move this set by nothing measurable (every frame is clean, only
    its sharpness -- and so its composite -- differs), which is exactly what the
    real-shoot measurement showed: 19.9% kept at EXTREME versus 21.5% at MORE
    across a 5,745-frame wedding. The keep fraction turns the same set into
    2/4/7/10 survivors.
    """
    photos = [measured_frame(i + 1, sharp)
              for i, sharp in enumerate((20, 40, 60, 80, 100, 140, 180, 240, 320, 400))]

    def survivors(value: int) -> set[int]:
        settings = CullingSettings(preset="wedding", selection_strictness=value)
        result = plan(photos, settings)
        return {pid for pid, score in result.scores.items() if score.reason is None}

    assert tuple(len(survivors(v)) for v in (85, 60, 35, 15)) == (2, 4, 7, 10)
    # The survivors are the strongest frames, not an arbitrary subset.
    best = max(photos, key=lambda p: p.tech.sharpness).photo_id
    assert survivors(85) == {best - 1, best}


def test_the_floor_empties_a_group_whose_best_frame_is_below_it():
    """A group with no strong frame yields nothing, which is what drops the count."""
    weak = [measured_frame(1, 30, phash="a" * 16, captured_at=1_000.0),
            measured_frame(2, 32, phash="a" * 16, captured_at=1_000.4)]
    strong = [measured_frame(3, 380, phash="b" * 16),
              measured_frame(4, 400, phash="c" * 16)]
    photos = weak + strong

    strict = plan(photos, CullingSettings(preset="wedding", selection_strictness=85))
    assert {pid for pid, s in strict.scores.items() if s.reason is None} == {4}
    assert {pid for pid, s in strict.scores.items() if s.reason == "target"} == {1, 2, 3}
    # The weak pair is a group, and with nothing left to choose from it keeps
    # no one -- which is the mechanism that lowers the shoot's keeper count.
    assert strict.selections[1].keeper_ids == []

    # With no floor the same shoot keeps the weak group's sharper frame too.
    loose = plan(photos, CullingSettings(preset="wedding", selection_strictness=15))
    assert loose.selections[1].keeper_ids == [2]
    assert {pid for pid, s in loose.scores.items() if s.reason is None} == {1, 2, 3, 4}


def test_a_library_without_a_stop_scores_exactly_as_before(store, photo_dir):
    """The floor is opt-in: the preset alone must not change any verdict."""
    library_id, _ = scan(store, photo_dir, CullingSettings(preset="wedding"))
    baseline = {r["id"]: (r["verdict"], r["reason"])
                for r in store.photos(library_id)}
    again = scan(store, photo_dir, CullingSettings(preset="wedding"))[0]
    assert {r["id"]: (r["verdict"], r["reason"])
            for r in store.photos(again)} == baseline
    assert not any(reason == "target" for _v, reason in baseline.values())


def test_target_cut_rank_handles_missing_tiebreak_signals():
    """No eyes / smile / face data: the cap still orders by quality, as before."""
    from culler.pipeline import cull_to_target
    from culler.selection import SelectionSignals

    updates = [
        (0.50, "keep", None, None, 0, None, 1),
        (0.90, "keep", None, None, 0, None, 2),
    ]
    signals = {1: SelectionSignals(photo_id=1, quality=0.50),
               2: SelectionSignals(photo_id=2, quality=0.90)}
    capped = cull_to_target(updates, 0.5, set(), signals)
    assert [row[6] for row in capped if row[1] == "keep"] == [2]


def test_target_cut_uses_the_same_ranking_as_best_of_group():
    """One score for both paths: the group tie-break decides the cap too.

    Two keepers of identical quality; the open-eyes frame survives, which is
    the tie-break `selection.rank()` already applies inside a group.
    """
    from culler.pipeline import cull_to_target
    from culler.selection import SelectionSignals

    updates = [
        (0.70, "keep", None, None, 0, None, 1),   # blinking
        (0.70, "keep", None, None, 0, None, 2),   # open eyes
    ]
    signals = {
        1: SelectionSignals(photo_id=1, quality=0.70, eyes_open=1.0),
        2: SelectionSignals(photo_id=2, quality=0.70, eyes_open=9.5),
    }
    capped = cull_to_target(updates, 0.5, set(), signals)
    assert [row[6] for row in capped if row[1] == "keep"] == [2]
    assert any(row[2] == "target" for row in capped)
