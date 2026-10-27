#!/usr/bin/env python3
"""Boot a built Cullr headlessly and check the count the cull options show.

The number the "AI Automated Cull" window shows before any scan — "~20% of 72
photos" — needs two things to be right: the walk an import runs
(``POST /api/libraries/{id}/discover``) has to return the real count, and the
photo list has to report that no scan has happened yet. This script proves a
*built* app does both, without installing it:

    python3 scripts/check_build.py --folder ~/Pictures/shoot --expect 72
    python3 scripts/check_build.py --dmg dist/Cullr-0.3.0-arm64.dmg --folder ~/Pictures/shoot
    python3 scripts/check_build.py --app dist/Cullr.app --folder ~/Pictures/shoot \\
        --ui-from frontend/dist

It runs the app with ``--server-only`` on a loopback port against a throwaway
data directory, so the real library in ``~/Library/Application Support/Cullr``
is never opened, and it deletes that directory again unless ``--keep`` is
given. Every check prints one line; the exit status is 0 only if all pass.

What it cannot see is the rendering: the dialog's own expression is pinned by
the frontend suite (``frontend/src/App.test.tsx``). Between them the chain is
covered — this script says the build produces the right numbers before a scan,
that suite says the UI puts them on screen.

Standard library only, so it runs with any python3 on a Mac, with or without
the project venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

APP_NAME = "Cullr"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = ROOT / "dist" / f"{APP_NAME}.app"
READY_TIMEOUT = 180.0


class CheckError(Exception):
    """Something the script cannot work around: no app, no macOS, no folder."""


class Report:
    """One line per check, and a count of what went wrong."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'✓' if ok else '✗'} {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures += 1
        return ok


class Service:
    """The app's own HTTP API, over loopback."""

    def __init__(self, base: str, log: Path) -> None:
        self.base = base
        self.log = log

    def raw(self, method: str, path: str, body: dict | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    def json(self, method: str, path: str, body: dict | None = None) -> dict:
        return json.loads(self.raw(method, path, body))

    def text(self, method: str, path: str) -> str:
        return self.raw(method, path).decode("utf-8", "replace")

    def wait_ready(self, process: subprocess.Popen, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise CheckError(f"the app exited before it was ready; it logged to {self.log}")
            try:
                return self.json("GET", "/api/capabilities")
            except Exception:  # noqa: BLE001 - not up yet, or still booting
                time.sleep(0.25)
        raise CheckError(f"the app never answered /api/capabilities in {timeout:.0f}s; "
                         f"it logged to {self.log}")

    def log_tail(self, lines: int = 12) -> str:
        try:
            tail = self.log.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return ""
        return "\n".join(tail)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def binary_of(app: Path) -> Path:
    """The executable inside a .app, or the path itself when given a binary."""
    if app.is_dir() and app.suffix == ".app":
        return app / "Contents" / "MacOS" / APP_NAME
    return app


def version_of(app: Path) -> str:
    """The bundle's version, which a manual (non-tag) build leaves at 0.1.0."""
    if not (app.is_dir() and app.suffix == ".app"):
        return "unknown"
    try:
        with (app / "Contents" / "Info.plist").open("rb") as handle:
            return str(plistlib.load(handle).get("CFBundleShortVersionString", "unknown"))
    except (OSError, plistlib.InvalidFileException):
        return "unknown"


def quarantined(app: Path) -> bool:
    """A browser-downloaded .app carries com.apple.quarantine, which stops an
    ad-hoc signed bundle from launching; curl and gh do not set it."""
    if not app.is_dir():
        return False
    try:
        result = subprocess.run(["xattr", str(app)], capture_output=True, text=True)
    except OSError:
        return False
    return "com.apple.quarantine" in result.stdout


def attach_dmg(dmg: Path) -> tuple[Path, Path]:
    """Mount a .dmg read-only; returns (the .app inside it, the mount point)."""
    result = subprocess.run(["hdiutil", "attach", str(dmg), "-nobrowse", "-readonly"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise CheckError(f"could not mount {dmg}: {result.stderr.strip()}")
    mount: Path | None = None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split("\t")]
        if fields and fields[-1].startswith("/Volumes/"):
            mount = Path(fields[-1])
    if mount is None:
        raise CheckError(f"could not find where {dmg} was mounted")
    apps = sorted(mount.glob("*.app"))
    if not apps:
        raise CheckError(f"no .app inside {dmg}")
    return apps[0], mount


def detach(mount: Path) -> None:
    subprocess.run(["hdiutil", "detach", str(mount)], capture_output=True, text=True)


def count_photos(folder: Path, extensions: set[str], recursive: bool) -> int:
    """The same set the app counts: a names-only walk, no file ever opened."""
    paths = folder.rglob("*") if recursive else folder.glob("*")
    return sum(1 for path in paths if path.is_file() and path.suffix.lower() in extensions)


def ui_asset(index_html: str) -> str | None:
    match = re.search(r'src="([^"]+\.js)"', index_html)
    return match.group(1) if match else None


def check_ui(service: Service, report: Report, ui_dir: Path | None) -> None:
    """The frozen app serves the UI it shipped with. Compare that copy with a
    local build when one is named, so a stale UI inside a fresh .app shows."""
    index_html = service.text("GET", "/")
    served = ui_asset(index_html)
    if not report.check("the app serves its UI", served is not None,
                        served or "no <script src=...> in the served index.html"):
        return
    assert served is not None
    served_bytes = service.raw("GET", served)
    report.check("the UI bundle downloads", len(served_bytes) > 10_000,
                 f"{served} ({len(served_bytes):,} bytes)")

    if ui_dir is None:
        return

    local_index = ui_dir / "index.html"
    if not local_index.is_file():
        report.check(f"a local UI build exists at {ui_dir}", False,
                     "run `npm run build` in frontend/ first")
        return
    report.check(f"a local UI build exists at {ui_dir}", True)
    local_asset = ui_asset(local_index.read_text())
    if not report.check("the local UI references a bundle", local_asset is not None):
        return
    assert local_asset is not None
    local_bytes = (ui_dir / local_asset.lstrip("/")).read_bytes()
    same_bytes = hashlib.sha256(local_bytes).hexdigest() == hashlib.sha256(served_bytes).hexdigest()
    report.check(f"the app embeds the UI built from this tree", same_bytes,
                 "identical" if same_bytes
                 else f"app has {served}, this tree built {local_asset}")


def check_count(service: Service, report: Report, folder: Path, types: list[str],
                expect: int | None, extensions: set[str]) -> None:
    """The pre-cull count: an import counts the folder, and until a scan runs
    the photo list reports no total at all. The dialog has to prefer the count
    it was given in exactly that state."""
    library = service.json("POST", "/api/libraries", {
        "path": str(folder),
        "settings": {"recursive": True, "import_types": types, "ingest": False,
                     "backup_dir": None},
    })
    library_id = library["id"]
    # What Recent folders shows: a library that exists and holds no analysed
    # photos yet, because nothing has been scanned.
    listed = [entry for entry in service.json("GET", "/api/libraries")
              if entry["id"] == library_id]
    report.check("importing the folder creates a library with no scan yet",
                 bool(listed) and not listed[0].get("photo_count"),
                 f"photo_count={listed[0].get('photo_count') if listed else 'no library'}")

    latest = service.json("GET", f"/api/libraries/{library_id}/job")
    report.check("no cull has run", not latest,
                 f"a job is already recorded ({latest.get('status')})" if latest
                 else "this is the pre-cull window")

    discovered = service.json("POST", f"/api/libraries/{library_id}/discover")["total"]
    walked = count_photos(folder, extensions, recursive=True)
    report.check("discovery counts the folder", discovered == walked,
                 f"discover={discovered}, a walk finds {walked}")
    if expect is not None:
        report.check(f"the count is the {expect} photos expected", discovered == expect,
                     f"discover={discovered}")

    listing = service.json("GET", f"/api/libraries/{library_id}/photos?filter=all&sort=group")
    stats_total = (listing.get("stats") or {}).get("total")
    report.check("the photo list reports no total before a scan",
                 not listing.get("total") and not stats_total,
                 f"total={listing.get('total')}, stats.total={stats_total} — the 0 the "
                 f"dialog must not treat as a known total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_build.py",
        description="Boot a built Cullr headlessly and check the pre-cull photo count.")
    parser.add_argument("--folder", type=Path, required=True,
                        help="a shoot to import (the script never modifies it)")
    parser.add_argument("--expect", type=int, default=None,
                        help="the count you know the folder has, e.g. 72 for the test shoot")
    parser.add_argument("--app", type=Path, default=None,
                        help=f"a built {APP_NAME}.app or its binary (default: {DEFAULT_APP})")
    parser.add_argument("--dmg", type=Path, default=None,
                        help="a .dmg to mount and check instead of a local build")
    parser.add_argument("--ui-from", type=Path, default=None,
                        help="a built frontend (frontend/dist) the app's UI must match")
    parser.add_argument("--keep", action="store_true",
                        help="keep the throwaway data directory for inspection")
    parser.add_argument("--timeout", type=float, default=READY_TIMEOUT,
                        help=f"seconds to wait for the app to boot (default {READY_TIMEOUT:.0f})")
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        print("this checks a built macOS app; run it on a Mac", file=sys.stderr)
        return 2
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2

    mount: Path | None = None
    workdir: Path | None = None
    process: subprocess.Popen | None = None
    log = Path(tempfile.gettempdir()) / "cullr-check-build.log"
    try:
        if args.dmg is not None:
            if not args.dmg.is_file():
                raise CheckError(f"no such .dmg: {args.dmg}")
            app, mount = attach_dmg(args.dmg)
        else:
            app = (args.app or DEFAULT_APP).expanduser().resolve()
            if not app.exists():
                raise CheckError(f"no app at {app} — build one with desktop/build_macos.sh, "
                                 f"or point --app/--dmg at one")
        binary = binary_of(app)
        if not binary.is_file():
            raise CheckError(f"no executable at {binary}")

        workdir = Path(tempfile.mkdtemp(prefix="cullr-check-"))
        data_dir = workdir / "data"
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        service = Service(base, log)

        print(f"{APP_NAME} {version_of(app)} — {app}")
        print(f"folder: {args.folder}")
        print(f"data:   {data_dir}")

        environment = {**os.environ, "CULLR_DATA_DIR": str(data_dir)}
        with log.open("w") as handle:
            process = subprocess.Popen([str(binary), "--server-only", "--port", str(port)],
                                       stdout=handle, stderr=subprocess.STDOUT, env=environment)

        report = Report()
        try:
            capabilities = service.wait_ready(process, args.timeout)
            report.check("the service boots and answers", True, base)
            report.check("the face model loads inside the build", bool(capabilities.get("faces")),
                         capabilities.get("faces_error") or "faces available")

            types = [entry["key"] for entry in capabilities.get("import_types", [])]
            extensions = {ext.lower() for entry in capabilities.get("import_types", [])
                          for ext in entry.get("extensions", [])}

            check_ui(service, report, args.ui_from.expanduser().resolve()
                     if args.ui_from else None)
            check_count(service, report, args.folder, types, args.expect, extensions)
        except CheckError as error:
            report.check("the check could run", False, str(error))
            print(f"\n--- {log} ---\n{service.log_tail()}", file=sys.stderr)
        except urllib.error.HTTPError as error:
            report.check("the API accepted every call", False,
                         f"{error.code} {error.reason}: {error.read().decode('utf-8', 'replace')}")
        except urllib.error.URLError as error:
            # The app went away mid-run — a crash inside the frozen bundle.
            report.check("the app stayed up for the whole check", False,
                         f"{error.reason}; it logged to {log}")
            print(f"\n--- {log} ---\n{service.log_tail()}", file=sys.stderr)

        if report.failures:
            print(f"\nFAIL — {report.failures} check(s) failed")
            if quarantined(app):
                print(f"note: {app} carries com.apple.quarantine; a browser-downloaded "
                      f"build needs `xattr -dr com.apple.quarantine \"{app}\"` before it runs")
            return 1
        print("\nPASS — this build counts the shoot before any cull")
        return 0
    except CheckError as error:
        print(f"cannot check this build: {error}", file=sys.stderr)
        return 2
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if mount is not None:
            detach(mount)
        if workdir is not None:
            if args.keep:
                print(f"kept {workdir}")
            else:
                shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
