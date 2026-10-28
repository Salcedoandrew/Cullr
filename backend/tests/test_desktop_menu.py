"""The desktop wrapper's app-menu entry for Check for Updates, with a stub window.

Same shape as ``test_desktop_drop.py``: the real WKWebView cannot run here, so
the only things under test are that the item is built in the macOS application
menu and that clicking it reaches the UI's own panel through ``evaluate_js`` —
never a second update implementation. Skipped where pywebview is not installed
(the desktop extra is not part of the CI test environment).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

webview = pytest.importorskip("webview")

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_CALL = "window.__cullrCheckUpdates && window.__cullrCheckUpdates()"


def _load_desktop():
    spec = importlib.util.spec_from_file_location(
        "cullr_desktop", ROOT / "desktop" / "cullr_desktop.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _StubWindow:
    def __init__(self, fail: bool = False) -> None:
        self.js_calls: list[str] = []
        self.fail = fail

    def evaluate_js(self, code: str):
        self.js_calls.append(code)
        if self.fail:
            raise RuntimeError("no document loaded")
        return None


@pytest.fixture
def desktop():
    return _load_desktop()


def test_the_menu_item_lives_in_the_application_menu(desktop):
    window = _StubWindow()
    menu = desktop.build_menu(lambda: desktop._open_update_panel(
        window, logging.getLogger("test")))

    assert len(menu) == 1
    # "__app__" is pywebview's application menu: About, Hide and Quit live there,
    # which is where a Mac user looks for Check for Updates.
    assert menu[0].title == "__app__"
    actions = [item for item in menu[0].items if hasattr(item, "function")]
    assert [action.title for action in actions] == ["Check for Updates…"]


def test_the_menu_action_opens_the_ui_panel(desktop):
    window = _StubWindow()
    menu = desktop.build_menu(lambda: desktop._open_update_panel(
        window, logging.getLogger("test")))
    action = next(item for item in menu[0].items if hasattr(item, "function"))

    action.function()

    assert window.js_calls == [BRIDGE_CALL]


def test_a_menu_click_never_raises_into_the_app(desktop, caplog):
    # Before the page has loaded, evaluate_js can fail; a menu click must not.
    window = _StubWindow(fail=True)
    with caplog.at_level(logging.ERROR):
        desktop._open_update_panel(window, logging.getLogger("test"))

    assert window.js_calls == [BRIDGE_CALL]
    assert "could not open the update panel" in caplog.text


def test_no_menu_rather_than_a_crash_without_the_menu_api(desktop, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "webview.menu":
            raise ImportError("old pywebview")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # The footer button still opens the panel; only the menu item is missing.
    assert desktop.build_menu(lambda: None) == []
