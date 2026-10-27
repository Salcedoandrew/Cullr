from __future__ import annotations

from pathlib import Path

import pytest

from culler.config import CullingSettings
from culler.db import Store
from culler.pipeline import run_scan


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
