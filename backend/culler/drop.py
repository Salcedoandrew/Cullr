"""Resolve a dropped folder to a real absolute path.

Pure logic, no pywebview dependency, so it can be unit-tested. The desktop
wrapper feeds it the drop event pywebview delivers and the pasteboard paths
pywebview recorded; it returns the folder to import, or ``None`` when the drop
cannot be resolved (the UI then falls back to the folder picker).

How a drop resolves, from pywebview 5.1's source
(``platforms/cocoa.py`` ``WebKitHost.performDragOperation_`` and ``util.py``
``js_bridge_call``):

* On macOS, when something is dropped from Finder onto the window *and* a
  Python-side ``drop`` listener is registered, the pasteboard's file URLs are
  recorded as ``(basename, absolute path)`` pairs — a dropped *folder* is one
  such pair, with the folder's own path.
* The JavaScript ``drop`` event is forwarded to Python, and every dropped file
  whose name matches a recorded pair gets ``pywebviewFullPath`` attached.
  JavaScript itself never sees an absolute path (WebKit does not expose one);
  the full path exists only on Python's side.

A dropped folder therefore resolves in one of two ways: a file dropped on its
own carries ``pywebviewFullPath`` directly (its parent folder is the import),
or a folder's name is recovered from the files inside it — their
``webkitRelativePath`` starts with the folder name — and matched against the
recorded pasteboard pairs.
"""

from __future__ import annotations

from pathlib import Path

# (basename, absolute path) as recorded from the pasteboard.
DndPaths = list[tuple[str, str]]


def resolve_dropped_path(event: dict, dnd_paths: DndPaths) -> tuple[str | None, DndPaths]:
    """Return ``(folder path, consumed pasteboard entries)`` for a drop event.

    ``event`` is the pywebview DOM ``drop`` event: ``event["dataTransfer"]
    ["files"]`` is a list of file dicts with ``name`` and, for files inside a
    dropped folder, ``webkitRelativePath`` (``"Folder/IMG_0001.jpg"``); files
    dropped on their own also carry ``pywebviewFullPath``. Consumed entries are
    returned so the caller can drop them from its pasteboard record — a stale
    entry could otherwise resolve a later drop of a same-named folder to the
    wrong parent.
    """
    files = (event.get("dataTransfer") or {}).get("files") or []
    if not isinstance(files, list):
        return None, []

    # 1. A file dropped on its own carries its absolute path already.
    for entry in files:
        full = entry.get("pywebviewFullPath") if isinstance(entry, dict) else None
        if not full:
            continue
        path = Path(str(full))
        folder = path if path.is_dir() else path.parent
        consumed = [pair for pair in dnd_paths if pair[1] == str(full)]
        return str(folder), consumed

    # 2. A folder drop: the top-level name is the first segment of the
    #    relative path of any file inside it. Without a relative path there is
    #    nothing to match a name against, so a single recorded entry is the
    #    only unambiguous answer.
    top_name: str | None = None
    for entry in files:
        relative = entry.get("webkitRelativePath") if isinstance(entry, dict) else None
        if isinstance(relative, str) and "/" in relative:
            top_name = relative.split("/", 1)[0]
            break
    if top_name is None and len(dnd_paths) == 1:
        return _as_folder(dnd_paths[0]), dnd_paths
    if top_name is None:
        return None, []

    for pair in dnd_paths:
        if pair[0] == top_name:
            return _as_folder(pair), [pair]
    lowered = top_name.lower()
    for pair in dnd_paths:
        if pair[0].lower() == lowered:
            return _as_folder(pair), [pair]
    return None, []


def _as_folder(pair: tuple[str, str]) -> str:
    """A pasteboard entry as the folder to import (a file means its parent)."""
    path = Path(pair[1])
    return str(path if path.is_dir() else path.parent)
