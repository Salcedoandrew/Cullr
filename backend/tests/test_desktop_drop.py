"""The desktop wrapper's native drop wiring, with a stub window.

The real WKWebView cannot run on Linux, so the window is faked: the only
thing under test is that attaching the drop listener registers a handler that
resolves a drop to a real path and hands it back to the page with
``evaluate_js`` — and that an unresolvable drop reports an error instead of
silently doing nothing. Skipped entirely where pywebview is not installed
(the desktop extra is not part of the CI test environment).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

webview = pytest.importorskip("webview")

ROOT = Path(__file__).resolve().parents[2]


def _load_desktop():
    spec = importlib.util.spec_from_file_location(
        "cullr_desktop", ROOT / "desktop" / "cullr_desktop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DropSlot:
    """Stands in for ``element.events.drop``: records what was attached."""

    def __init__(self) -> None:
        self.handlers: list = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _StubWindow:
    def __init__(self) -> None:
        self.js_calls: list[str] = []
        drop = _DropSlot()
        events = type("Events", (), {"drop": drop})()
        document = type("Document", (), {"events": events})()
        self.dom = type("Dom", (), {"document": document})()
        self._drop_slot = drop

    def evaluate_js(self, code: str):
        self.js_calls.append(code)
        return None


@pytest.fixture
def desktop():
    return _load_desktop()


def test_attach_delivers_a_dropped_folder_path(desktop, tmp_path: Path):
    from webview.dom import _dnd_state

    folder = tmp_path / "2026-06-15 Wedding"
    folder.mkdir()
    (folder / "IMG_0001.jpg").write_bytes(b"fake")
    _dnd_state["paths"] = [("2026-06-15 Wedding", str(folder))]

    window = _StubWindow()
    assert desktop._attach_native_drop(window, "http://127.0.0.1:8000",
                                       logging.getLogger("test"))
    assert "window.__cullrNativeDrop = true" in window.js_calls
    assert len(window._drop_slot.handlers) == 1

    handler = window._drop_slot.handlers[0]
    handler.callback({"dataTransfer": {"files": [
        {"name": "IMG_0001.jpg",
         "webkitRelativePath": "2026-06-15 Wedding/IMG_0001.jpg"}]}})
    assert window.js_calls[-1] == (
        f"window.__cullrDrop && window.__cullrDrop({json.dumps(str(folder))})")
    assert _dnd_state["paths"] == []  # consumed, so it cannot mis-resolve later


def test_attach_reports_an_unresolvable_drop(desktop, tmp_path: Path):
    from webview.dom import _dnd_state

    _dnd_state["paths"] = []
    window = _StubWindow()
    assert desktop._attach_native_drop(window, "http://127.0.0.1:8000",
                                       logging.getLogger("test"))
    handler = window._drop_slot.handlers[0]
    handler.callback({"dataTransfer": {"files": [
        {"name": "IMG_0001.jpg", "webkitRelativePath": "Nowhere/IMG_0001.jpg"}]}})
    assert window.js_calls[-1].startswith("window.__cullrDropError")


def test_attach_delivers_a_dropped_files_parent(desktop, tmp_path: Path):
    from webview.dom import _dnd_state

    photo = tmp_path / "IMG_0001.jpg"
    photo.write_bytes(b"fake")
    _dnd_state["paths"] = [("IMG_0001.jpg", str(photo))]
    window = _StubWindow()
    assert desktop._attach_native_drop(window, "http://127.0.0.1:8000",
                                       logging.getLogger("test"))
    handler = window._drop_slot.handlers[0]
    handler.callback({"dataTransfer": {"files": [
        {"name": "IMG_0001.jpg", "pywebviewFullPath": str(photo)}]}})
    assert window.js_calls[-1] == (
        f"window.__cullrDrop && window.__cullrDrop({json.dumps(str(tmp_path))})")
