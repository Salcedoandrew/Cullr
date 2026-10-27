"""Desktop entry point: the local culling server inside a native macOS window.

The app is a FastAPI service plus a compiled React UI. On the desktop it runs as
a normal Mac application: a WKWebView window (via pywebview, which drives the
system WebKit rather than bundling a browser) pointed at a loopback server this
process owns. Closing the window stops the server; the catalog and preview cache
live in ``~/Library/Application Support/Cullr`` so the bundle itself can stay
read-only and signed.

Startup is not instant — importing MediaPipe costs a few seconds — so the window
opens immediately with a splash and swaps to the app once the service answers.

No data leaves the machine: the server binds to 127.0.0.1 on an ephemeral port.

Run from source with ``python desktop/cullr_desktop.py``; ``--server-only``
starts just the service (used by the packaging smoke test, where CI has no
window server).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "Cullr"
READY_TIMEOUT = 120.0
SPLASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Cullr</title>
<style>
  html, body { height: 100%; margin: 0; background: #14110d; color: #ece2d2;
               font-family: -apple-system, "Avenir Next", "Segoe UI", sans-serif;
               -webkit-user-select: none; cursor: default; }
  .mark { height: 100%; display: grid; align-content: center; justify-items: center; gap: 14px; }
  h1 { margin: 0; font-size: 27px; font-weight: 400; letter-spacing: 0.01em; }
  .bar { width: 190px; height: 2px; background: rgba(236, 226, 210, 0.1); overflow: hidden; }
  .bar span { display: block; height: 100%; width: 38%; background: #d9a95f;
              animation: slide 1.2s ease-in-out infinite; }
  p { margin: 0; font-size: 12.5px; letter-spacing: 0.02em; color: #7c7264; }
  @keyframes slide { from { transform: translateX(-110%); } to { transform: translateX(300%); } }
</style></head>
<body><div class="mark"><h1>Cullr</h1>
<div class="bar"><span></span></div><p>Starting the local service…</p></div></body></html>"""


def _ensure_importable() -> None:
    """Running this file from the repository root must work without PYTHONPATH."""
    try:
        import culler  # noqa: F401
    except ModuleNotFoundError:
        backend = Path(__file__).resolve().parents[1] / "backend"
        if backend.is_dir():
            sys.path.insert(0, str(backend))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(base_url: str, timeout: float = READY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/api/capabilities", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.15)
    return False


def _configure_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cullr.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(handler)
    return log_path


def _start_service(data_dir: Path, port: int):
    """Build the app and run uvicorn on a worker thread."""
    import uvicorn

    from culler.api import create_app

    application = create_app(data_dir)
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port,
                                           log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="cullr-server", daemon=True)
    thread.start()
    return server, thread


def _attach_native_drop(window, base_url: str, log: logging.Logger) -> bool:
    """Attach the Python-side drop listener; True when a dropped folder can
    now resolve to a real path.

    How it works (pywebview 5.0+, verified against its source): on macOS, when
    something is dropped from Finder onto the window *and* a Python-side
    ``drop`` listener is registered, the pasteboard's file URLs are recorded
    as ``(basename, absolute path)`` pairs — a dropped folder is one such
    pair. The JavaScript ``drop`` event is then forwarded to Python, and
    ``culler.drop.resolve_dropped_path`` turns it into the folder to import.
    JavaScript itself never sees an absolute path; the full path exists only
    on Python's side, and it is handed back to the page with ``evaluate_js``
    (``window.__cullrDrop``), never by uploading a file.

    Anything that fails here — an older pywebview, the listener not attaching
    — only loses the native drop. The import screen still works: a drop then
    opens the folder picker pre-navigated to the last-used parent with the
    dropped folder's name shown.
    """
    try:
        from webview.dom import DOMEventHandler, _dnd_state
    except ImportError as exc:
        log.warning("native drag-and-drop is unavailable (%s); drops use the "
                    "folder picker", exc)
        return False
    from culler.drop import resolve_dropped_path

    def on_drop(event: dict) -> None:
        # The bridge dispatches drop events on its own thread; evaluate_js is
        # queue-based, so calling it here is the designed pattern.
        try:
            paths = list(_dnd_state.get("paths", []))
            resolved, consumed = resolve_dropped_path(event, paths)
            for pair in consumed:
                try:
                    _dnd_state["paths"].remove(pair)
                except ValueError:
                    pass
            if resolved:
                window.evaluate_js(
                    "window.__cullrDrop && window.__cullrDrop("
                    + json.dumps(resolved) + ")")
            else:
                window.evaluate_js(
                    "window.__cullrDropError && window.__cullrDropError("
                    + json.dumps("could not resolve the dropped folder") + ")")
        except Exception as exc:  # noqa: BLE001 - a drop must never take the window down
            log.exception("drop failed")
            try:
                window.evaluate_js(
                    "window.__cullrDropError && window.__cullrDropError("
                    + json.dumps(str(exc)) + ")")
            except Exception:
                pass

    try:
        window.dom.document.events.drop += DOMEventHandler(on_drop, prevent_default=True)
        window.evaluate_js("window.__cullrNativeDrop = true")
    except Exception:
        log.exception("could not attach the drop listener")
        return False
    log.info("native drag-and-drop attached")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.add_argument("--server-only", action="store_true",
                        help="run the service without opening a window (packaging tests)")
    parser.add_argument("--port", type=int, default=None)
    args, _unknown = parser.parse_known_args(argv)

    _ensure_importable()

    from culler.db import default_data_dir

    data_dir = default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    # MediaPipe imports matplotlib, which rebuilds a font cache in an unwritable
    # default location on every launch if left alone. Keep it with our data so
    # only the first run pays for it.
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / "matplotlib"))
    log_path = _configure_logging(data_dir)
    log = logging.getLogger("cullr")

    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log.info("starting on %s (data %s)", base_url, data_dir)

    if args.server_only:
        server, service = _start_service(data_dir, port)
        if not _wait_until_ready(base_url):
            print(f"{APP_NAME}: the local service did not start; see {log_path}", file=sys.stderr)
            return 1
        print(f"ready {base_url}", flush=True)
        try:
            while service.is_alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        server.should_exit = True
        service.join(timeout=10)
        return 0

    import webview

    window = webview.create_window(
        APP_NAME,
        html=SPLASH_HTML,
        width=1440,
        height=900,
        min_size=(1024, 680),
        background_color="#14110d",
    )
    assert window is not None
    state: dict[str, object] = {}

    # Native drag-and-drop: attach the Python-side drop listener once the app
    # itself is showing. The splash is a different document, so attaching
    # earlier would listen on a page that is about to be replaced.
    drop_attached = {"done": False}

    def on_loaded() -> None:
        if drop_attached["done"]:
            return
        try:
            current = window.get_current_url() or ""
        except Exception:
            return
        if not current.startswith(base_url):
            return
        if _attach_native_drop(window, base_url, log):
            drop_attached["done"] = True

    window.events.loaded += on_loaded

    def boot() -> None:
        try:
            server, service = _start_service(data_dir, port)
            state["server"] = server
            state["service"] = service
            if not _wait_until_ready(base_url):
                raise RuntimeError("the local service did not answer in time")
            window.load_url(base_url)
            log.info("UI loaded from %s", base_url)
        except Exception as exc:  # surfaced in the window, not silently in a log
            log.exception("startup failed")
            message = (
                "<!doctype html><html><body style=\"margin:0;height:100%;display:grid;"
                "place-content:center;gap:10px;background:#14110d;color:#ece2d2;"
                "font-family:-apple-system,sans-serif;text-align:center\">"
                f"<h1 style=\"font-weight:400;font-size:22px;margin:0\">{APP_NAME} could not start</h1>"
                f"<p style=\"color:#a99c88;font-size:13px;max-width:38ch\">{exc}</p>"
                f"<p style=\"color:#7c7264;font-size:11.5px\">{log_path}</p></body></html>"
            )
            window.load_html(message)

    threading.Thread(target=boot, name="cullr-boot", daemon=True).start()
    try:
        webview.start()
    finally:
        log.info("window closed, stopping server")
        server = state.get("server")
        service = state.get("service")
        if server is not None:
            server.should_exit = True  # type: ignore[union-attr]
        if service is not None:
            service.join(timeout=10)  # type: ignore[union-attr]
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
