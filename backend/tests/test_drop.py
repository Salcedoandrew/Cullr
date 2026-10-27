"""Drop resolution: a dropped folder must come back as a real absolute path.

The event dicts here are what pywebview 5.1 delivers to a Python-side ``drop``
handler: files carry ``pywebviewFullPath`` when their name matched a pasteboard
URL, and ``webkitRelativePath`` (``"Folder/IMG_0001.jpg"``) for files inside a
dropped folder. ``dnd_paths`` is the pasteboard record — ``(basename, absolute
path)`` pairs, one per item dropped from Finder.
"""

from __future__ import annotations

from pathlib import Path

from culler.drop import resolve_dropped_path


def _file(name: str, **extra) -> dict:
    entry = {"name": name, "size": 100, "type": "image/jpeg"}
    entry.update(extra)
    return entry


def test_single_file_drop_resolves_to_its_parent(tmp_path: Path):
    photo = tmp_path / "IMG_0001.jpg"
    photo.write_bytes(b"fake")
    event = {"dataTransfer": {"files": [
        _file("IMG_0001.jpg", pywebviewFullPath=str(photo))]}}
    path, consumed = resolve_dropped_path(event, [("IMG_0001.jpg", str(photo))])
    assert path == str(tmp_path)
    assert consumed == [("IMG_0001.jpg", str(photo))]


def test_folder_drop_resolves_via_relative_path(tmp_path: Path):
    folder = tmp_path / "2026-06-15 Wedding"
    folder.mkdir()
    (folder / "IMG_0001.jpg").write_bytes(b"fake")
    event = {"dataTransfer": {"files": [
        _file("IMG_0001.jpg", webkitRelativePath="2026-06-15 Wedding/IMG_0001.jpg")]}}
    path, consumed = resolve_dropped_path(event, [("2026-06-15 Wedding", str(folder))])
    assert path == str(folder)
    assert consumed == [("2026-06-15 Wedding", str(folder))]


def test_folder_drop_without_relative_path_uses_the_only_entry(tmp_path: Path):
    folder = tmp_path / "Shoot"
    folder.mkdir()
    event = {"dataTransfer": {"files": [_file("IMG_0001.jpg")]}}
    path, consumed = resolve_dropped_path(event, [("Shoot", str(folder))])
    assert path == str(folder)
    assert consumed == [("Shoot", str(folder))]


def test_folder_drop_matches_case_insensitively(tmp_path: Path):
    folder = tmp_path / "Shoot"
    folder.mkdir()
    event = {"dataTransfer": {"files": [
        _file("IMG_0001.jpg", webkitRelativePath="shoot/IMG_0001.jpg")]}}
    path, _ = resolve_dropped_path(event, [("Shoot", str(folder))])
    assert path == str(folder)


def test_first_matching_folder_wins_when_several_are_dropped(tmp_path: Path):
    first = tmp_path / "A"
    second = tmp_path / "B"
    first.mkdir()
    second.mkdir()
    event = {"dataTransfer": {"files": [
        _file("IMG_0001.jpg", webkitRelativePath="A/IMG_0001.jpg"),
        _file("IMG_0002.jpg", webkitRelativePath="B/IMG_0002.jpg")]}}
    path, consumed = resolve_dropped_path(
        event, [("A", str(first)), ("B", str(second))])
    assert path == str(first)
    assert consumed == [("A", str(first))]


def test_unresolvable_drop_returns_none(tmp_path: Path):
    event = {"dataTransfer": {"files": [
        _file("IMG_0001.jpg", webkitRelativePath="Nowhere/IMG_0001.jpg")]}}
    path, consumed = resolve_dropped_path(event, [("Elsewhere", str(tmp_path))])
    assert path is None
    assert consumed == []


def test_empty_drop_returns_none():
    assert resolve_dropped_path({"dataTransfer": {"files": []}}, []) == (None, [])
    assert resolve_dropped_path({}, []) == (None, [])
