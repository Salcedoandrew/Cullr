"""The in-app update: version comparison, the manifest, the download, the install.

Two layers are covered here. The first is the update itself — a manifest that is
untrusted input, a download that must hash-check before it counts, and a detached
``/bin/sh`` helper whose whole job is to leave the installed app alone until the
new one is staged and verified. The second is the real thing: on macOS the test
at the end of this file builds a throwaway ``.dmg``, runs the generated helper
against a fake installed bundle, and checks that the old bundle is gone, the new
one is in its place, and the relaunch actually happened. It exercises
``hdiutil``, ``ditto`` and ``open`` for real rather than mocking them, which is
the part of this feature that cannot be reasoned about — only run.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import plistlib
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from culler import update
from culler.api import create_app

GOOD_URL = ("https://github.com/Salcedoandrew/cullr/releases/download/"
            "v0.9.0/Cullr-0.9.0-arm64.dmg")
GOOD_SHA = "a" * 64


def manifest(**overrides) -> dict:
    payload = {
        "version": "0.9.0",
        "url": GOOD_URL,
        "filename": "Cullr-0.9.0-arm64.dmg",
        "sha256": GOOD_SHA,
        "size": 3_000_000,
        "notes_url": "https://github.com/Salcedoandrew/cullr/releases/tag/v0.9.0",
    }
    payload.update(overrides)
    return payload


# ---- version comparison -------------------------------------------------- #


def test_parse_version_accepts_only_x_y_z():
    assert update.parse_version("0.9.0") == (0, 9, 0)
    assert update.parse_version("v1.12.3") == (1, 12, 3)
    for bad in ("0.9", "0.9.0.1", "0.9.0-rc1", "later", ""):
        with pytest.raises(update.UpdateError):
            update.parse_version(bad)


def test_comparison_is_semantic_not_alphabetical():
    """The trap: "0.10.0" < "0.9.0" as strings, and 0.10.0 is the newer build."""
    assert update.is_newer("0.10.0", "0.9.0") is True
    assert update.is_newer("0.9.0", "0.9.0") is False
    assert update.is_newer("0.9.0", "0.10.0") is False
    assert update.is_newer("1.0.0", "0.99.99") is True


# ---- the manifest is untrusted input ------------------------------------- #


def test_validate_manifest_reads_a_release_manifest():
    parsed = update.validate_manifest(manifest())
    assert parsed["version"] == "0.9.0"
    assert parsed["filename"] == "Cullr-0.9.0-arm64.dmg"
    assert parsed["sha256"] == GOOD_SHA
    assert parsed["notes_url"].endswith("/v0.9.0")


def test_validate_manifest_fills_in_the_filename_and_notes_link():
    parsed = update.validate_manifest({"version": "0.9.0", "url": GOOD_URL,
                                       "sha256": GOOD_SHA})
    assert parsed["filename"] == "Cullr-0.9.0-arm64.dmg"
    assert parsed["notes_url"] == "https://github.com/Salcedoandrew/cullr/releases/tag/v0.9.0"


@pytest.mark.parametrize("payload, reason", [
    ([], "not an object"),
    ({"url": GOOD_URL, "sha256": GOOD_SHA}, "a missing version"),
    (manifest(version="0.9"), "a version that is not x.y.z"),
    (manifest(url=GOOD_URL.replace("https://", "http://")), "plain HTTP"),
    (manifest(url="https://evil.example.com/Cullr-0.9.0-arm64.dmg"), "another host"),
    (manifest(url="https://github.com/x/y/releases/download/v0.9.0/Cullr-0.9.0.zip"),
     "something that is not a .dmg"),
    (manifest(filename="Cullr-0.9.0-arm64.dmg", url=GOOD_URL.replace("arm64", "x64")),
     "a filename that disagrees with the URL"),
    (manifest(filename="../../Library/LaunchAgents/evil.dmg"), "a path in the filename"),
    (manifest(filename="evil.dmg\nrm -rf /"), "a newline in the filename"),
    (manifest(sha256="deadbeef"), "a truncated hash"),
    (manifest(sha256="z" * 64), "a hash that is not hex"),
    (manifest(notes_url="http://github.com/x"), "a plain-HTTP notes link"),
])
def test_validate_manifest_refuses_anything_else(payload, reason):
    with pytest.raises(update.UpdateError):
        update.validate_manifest(payload)


def test_validate_manifest_normalises_an_uppercase_hash():
    assert update.validate_manifest(manifest(sha256=GOOD_SHA.upper()))["sha256"] == GOOD_SHA


def test_fetch_manifest_refuses_a_non_https_url():
    with pytest.raises(update.UpdateError):
        update.fetch_manifest("http://127.0.0.1:8000/update.json")


def test_fetch_manifest_refuses_an_unexpected_host():
    with pytest.raises(update.UpdateError):
        update.fetch_manifest("https://raw.githubusercontent.com/x/y/update.json")


# ---- downloading --------------------------------------------------------- #


@pytest.fixture
def served(tmp_path):
    """A tiny loopback HTTP server, so the download path is exercised for real."""
    payload = bytes(range(256)) * 4096          # 1 MB

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                        # noqa: N802 - http.server's API
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):            # keep the test output clean
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", payload
    finally:
        server.shutdown()
        server.server_close()


def test_download_verifies_then_installs(served, tmp_path):
    base, payload = served
    digest = hashlib.sha256(payload).hexdigest()
    seen: list[tuple[int, int]] = []
    path = update.download_update(f"{base}/Cullr-0.9.0-arm64.dmg", digest,
                                  "Cullr-0.9.0-arm64.dmg", tmp_path / "updates",
                                  progress=lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == payload
    assert not path.with_name(path.name + ".part").exists()
    assert seen and seen[-1] == (len(payload), len(payload))


def test_download_rejects_a_tampered_payload(served, tmp_path):
    base, payload = served
    wrong = hashlib.sha256(payload + b"!").hexdigest()
    with pytest.raises(update.UpdateError, match="SHA-256"):
        update.download_update(f"{base}/Cullr-0.9.0-arm64.dmg", wrong,
                               "Cullr-0.9.0-arm64.dmg", tmp_path / "updates")
    # Neither the file nor the partial may be left behind for install to find.
    assert list((tmp_path / "updates").iterdir()) == []


def test_download_cancel_removes_the_partial_file(served, tmp_path):
    base, payload = served
    cancel = threading.Event()

    def progress(done: int, _total: int) -> None:
        cancel.set()                     # as if the user clicked Cancel

    with pytest.raises(update.UpdateCancelled):
        update.download_update(f"{base}/Cullr-0.9.0-arm64.dmg",
                               hashlib.sha256(payload).hexdigest(),
                               "Cullr-0.9.0-arm64.dmg", tmp_path / "updates",
                               progress=progress, cancel=cancel)
    assert list((tmp_path / "updates").iterdir()) == []


def test_download_never_writes_outside_the_destination(served, tmp_path):
    """A filename from the manifest cannot escape the updates folder."""
    base, payload = served
    dest = tmp_path / "updates"
    path = update.download_update(f"{base}/x.dmg", hashlib.sha256(payload).hexdigest(),
                                 "../../escaped.dmg", dest)
    assert path == dest / "escaped.dmg"
    assert not (tmp_path.parent / "escaped.dmg").exists()


# ---- which bundle would be replaced -------------------------------------- #


def _fake_bundle(root: Path, name: str = "Cullr.app") -> Path:
    app = root / name
    (app / "Contents" / "MacOS").mkdir(parents=True)
    binary = app / "Contents" / "MacOS" / name.removesuffix(".app")
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    with (app / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump({"CFBundleExecutable": name.removesuffix(".app"),
                       "CFBundleShortVersionString": "0.9.0"}, handle)
    return app


def test_bundle_of_walks_up_from_the_running_executable(tmp_path):
    app = _fake_bundle(tmp_path / "Somewhere")
    binary = app / "Contents" / "MacOS" / "Cullr"
    assert update.bundle_of(binary) == app


def test_bundle_of_is_none_for_a_source_checkout(tmp_path):
    assert update.bundle_of(Path(sys.executable)) is None


def test_resolve_app_path_prefers_the_running_bundle(tmp_path):
    app = _fake_bundle(tmp_path / "Elsewhere")
    found = update.resolve_app_path(app / "Contents" / "MacOS" / "Cullr",
                                    app_dir=tmp_path / "Applications")
    assert found == app


def test_resolve_app_path_falls_back_when_the_app_is_translocated(tmp_path):
    """A quarantined app runs from a read-only random path: update the real one."""
    installed = _fake_bundle(tmp_path / "Applications")
    translocated = _fake_bundle(
        tmp_path / "AppTranslocation" / "8F6E-1C9A" / "d")
    found = update.resolve_app_path(translocated / "Contents" / "MacOS" / "Cullr",
                                    app_dir=tmp_path / "Applications")
    assert found == installed


def test_resolve_app_path_is_none_when_nothing_installed(tmp_path):
    assert update.resolve_app_path(Path(sys.executable),
                                   app_dir=tmp_path / "Applications") is None


# ---- the helper script --------------------------------------------------- #


def test_helper_script_quotes_awkward_paths_into_valid_shell(tmp_path):
    target = tmp_path / "My Apps" / "Cul'lr.app"
    dmg = tmp_path / "down loads" / "Cullr-0.9.0-arm64.dmg"
    script = tmp_path / "install.sh"
    script.write_text(update.helper_script(dmg=dmg, target=target,
                                           work=tmp_path / "work dir",
                                           log_file=tmp_path / "u'pdate.log",
                                           version="0.9.0", pid=4321))
    # The generated source must be valid /bin/sh, and the quote in the log path
    # must be escaped rather than ending the string early.
    assert subprocess.run(["/bin/sh", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "'\\''" in text
    assert "'/bin/ps'" in text or "/bin/ps" in text


def test_helper_script_stages_before_it_deletes(tmp_path):
    """The ordering guarantee, asserted on the source itself."""
    text = update.helper_script(dmg=tmp_path / "x.dmg", target=tmp_path / "Cullr.app",
                                work=tmp_path / "work", log_file=tmp_path / "log",
                                version="0.9.0", pid=1)
    staged = text.index('STAGED="$TARGET.new"')
    verify = text.index('the staged copy is incomplete')
    delete = text.index('rm -rf "$TARGET"')
    assert staged < verify < delete


# ---- the state machine --------------------------------------------------- #


def _manager(tmp_path, **kwargs):
    return update.UpdateManager(tmp_path / "data", current="0.9.0", **kwargs)


def test_check_reports_no_update_for_the_running_version(tmp_path):
    manager = _manager(tmp_path, fetcher=lambda url: update.validate_manifest(manifest()))
    state = manager.check()
    assert state["phase"] == "up_to_date"
    assert state["latest"] == "0.9.0"
    assert state["error"] is None


def test_check_reports_a_newer_version(tmp_path):
    manager = _manager(tmp_path,
                       fetcher=lambda url: update.validate_manifest(manifest(version="0.10.0")))
    state = manager.check()
    assert state["phase"] == "available"
    assert state["latest"] == "0.10.0"
    assert state["filename"] == "Cullr-0.9.0-arm64.dmg"


def test_check_reports_why_it_could_not_check(tmp_path):
    def boom(url):
        raise update.UpdateError("nothing published yet (HTTP 404)")

    state = _manager(tmp_path, fetcher=boom).check()
    assert state["phase"] == "error"
    assert "404" in state["error"]


def test_download_without_a_check_does_nothing(tmp_path):
    state = _manager(tmp_path).start_download()
    assert state["phase"] == "error"
    assert "no newer release" in state["error"]


def test_download_then_cancel_deletes_the_file(tmp_path):
    def downloader(url, sha, filename, dest_dir, progress, cancel):
        path = Path(dest_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dmg")
        progress(3, 3)
        return path

    manager = _manager(tmp_path, fetcher=lambda url: update.validate_manifest(
        manifest(version="0.10.0")), downloader=downloader)
    manager.check()
    manager.start_download()
    for _ in range(100):                        # the download runs on a thread
        if manager.status()["phase"] != "downloading":
            break
        time.sleep(0.05)
    assert manager.status()["phase"] == "ready"
    assert manager.status()["percent"] == 100
    state = manager.cancel()
    assert state["phase"] == "cancelled"
    assert list((tmp_path / "data" / "updates").glob("*.dmg")) == []


def test_install_without_a_verified_download_installs_nothing(tmp_path, monkeypatch):
    spawned: list = []
    monkeypatch.setattr(update.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    manager = _manager(tmp_path, app_path=tmp_path / "Applications" / "Cullr.app")
    state = manager.install()
    assert state["phase"] == "error"
    assert spawned == []


def test_install_launches_a_detached_helper_and_then_quits(tmp_path, monkeypatch):
    payload = b"a disk image"
    digest = hashlib.sha256(payload).hexdigest()

    def downloader(url, sha, filename, dest_dir, progress, cancel):
        path = Path(dest_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        progress(len(payload), len(payload))
        return path

    spawned: list[tuple] = []
    monkeypatch.setattr(update.subprocess, "Popen",
                        lambda args, **kwargs: spawned.append((args, kwargs)))
    target = _fake_bundle(tmp_path / "Applications")
    manager = _manager(tmp_path, fetcher=lambda url: update.validate_manifest(
        manifest(version="0.10.0", sha256=digest)), downloader=downloader, app_path=target)
    manager.check()
    manager.start_download()
    for _ in range(100):
        if manager.status()["phase"] != "downloading":
            break
        time.sleep(0.05)

    quit_called = threading.Event()
    state = manager.install(on_quit=quit_called.set)
    assert state["phase"] == "installing"
    assert state["installed_at"] == str(target)
    assert len(spawned) == 1
    args, kwargs = spawned[0]
    assert args[0] == "/bin/sh"
    script = Path(args[1])
    assert script.is_file() and script.stat().st_mode & 0o111
    assert kwargs["start_new_session"] is True
    assert f"'{target}'" in script.read_text()
    assert quit_called.wait(timeout=5)


def test_install_deletes_a_file_that_changed_after_verification(tmp_path, monkeypatch):
    digest = hashlib.sha256(b"the real thing").hexdigest()

    def downloader(url, sha, filename, dest_dir, progress, cancel):
        path = Path(dest_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"the real thing")
        return path

    monkeypatch.setattr(update.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("nothing may be installed"))
    manager = _manager(tmp_path, fetcher=lambda url: update.validate_manifest(
        manifest(version="0.10.0", sha256=digest)), downloader=downloader,
        app_path=tmp_path / "Applications")
    manager.check()
    manager.start_download()
    for _ in range(100):
        if manager.status()["phase"] != "downloading":
            break
        time.sleep(0.05)
    dmg = next((tmp_path / "data" / "updates").glob("*.dmg"))
    dmg.write_bytes(b"swapped afterwards")

    state = manager.install()
    assert state["phase"] == "error"
    assert "changed on disk" in state["error"]
    assert not dmg.exists()


# ---- the API the panel talks to ------------------------------------------ #


def test_update_endpoints_drive_the_state_machine(tmp_path, photo_dir):
    app = create_app(tmp_path / "data")
    updates = app.state.updates
    updates._fetcher = lambda url: update.validate_manifest(manifest(version="0.10.0"))

    def downloader(url, sha, filename, dest_dir, progress, cancel):
        path = Path(dest_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dmg")
        return path

    updates._downloader = downloader
    with TestClient(app) as client:
        assert client.get("/api/update").json()["phase"] == "idle"
        assert client.get("/api/capabilities").json()["version"] == update.__version__

        checked = client.post("/api/update/check").json()
        assert checked["phase"] == "available"
        assert checked["latest"] == "0.10.0"

        # The stub download finishes instantly, so the first answer is either
        # "downloading" or already "ready"; the state the feature promises is
        # the one it settles on, and that the UI polls its way to.
        assert client.post("/api/update/download").json()["phase"] in ("downloading", "ready")
        for _ in range(100):
            state = client.get("/api/update").json()
            if state["phase"] != "downloading":
                break
            time.sleep(0.05)
        assert state["phase"] == "ready"
        assert client.post("/api/update/cancel").json()["phase"] == "cancelled"


def test_install_over_the_api_asks_the_desktop_app_to_quit(tmp_path):
    asked = threading.Event()
    app = create_app(tmp_path / "data", on_quit=asked.set)
    updates = app.state.updates
    payload = b"a disk image"
    digest = hashlib.sha256(payload).hexdigest()
    updates._fetcher = lambda url: update.validate_manifest(
        manifest(version="0.10.0", sha256=digest))
    updates._app_path = _fake_bundle(tmp_path / "Applications")
    updates._app_resolved = True

    def downloader(url, sha, filename, dest_dir, progress, cancel):
        path = Path(dest_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    updates._downloader = downloader
    launched: list = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(update.subprocess, "Popen", lambda args, **kwargs: launched.append(args))
    try:
        with TestClient(app) as client:
            client.post("/api/update/check")
            client.post("/api/update/download")
            for _ in range(100):
                if client.get("/api/update").json()["phase"] == "ready":
                    break
                time.sleep(0.05)
            state = client.post("/api/update/install").json()
            assert state["phase"] == "installing"
    finally:
        monkey.undo()
    assert len(launched) == 1
    assert asked.wait(timeout=5)          # the app quits so the helper can work


# ---- the real round trip, on macOS --------------------------------------- #

BUNDLE_PLIST = {
    "CFBundleExecutable": "Cullr",
    "CFBundleIdentifier": "com.cullrapp.fake",
    "CFBundleName": "Cullr",
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": "0.0.0",
}


def _bundle(root: Path, version: str, marker: Path, name: str = "Cullr.app") -> Path:
    """A real (if tiny) .app whose executable records that it was launched."""
    app = root / name
    (app / "Contents" / "MacOS").mkdir(parents=True)
    plist = dict(BUNDLE_PLIST, CFBundleShortVersionString=version)
    with (app / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(plist, handle)
    binary = app / "Contents" / "MacOS" / "Cullr"
    binary.write_text(f"#!/bin/sh\nprintf '{version}\\n' >> '{marker}'\n")
    binary.chmod(0o755)
    return app


def _dmg_with(stage: Path, dmg: Path, volume: str) -> Path:
    """Pack a directory into a .dmg, retrying the host's intermittent failure.

    ``hdiutil create`` answers "Resource busy" now and then on this machine when
    a volume was attached or detached moments earlier. That is the environment
    settling, not the code under test, so it is retried a few times and the real
    error is reported if it keeps happening. Note that ``-quiet`` swallows the
    reason, which is why the retry runs without it.
    """
    last = ""
    for attempt in range(4):
        result = subprocess.run(
            ["hdiutil", "create", "-volname", volume, "-srcfolder", str(stage),
             "-fs", "HFS+", "-format", "UDZO", "-ov", str(dmg)],
            capture_output=True, text=True)
        if result.returncode == 0:
            return dmg
        last = result.stderr.strip() or result.stdout.strip()
        time.sleep(1.0 + attempt)
    pytest.fail(f"hdiutil create failed after 4 attempts: {last}")
    raise AssertionError("unreachable")


def _dead_pid() -> int:
    """A pid that has already been reaped, so the helper's wait loop exits."""
    process = subprocess.Popen(["/bin/sh", "-c", "true"])
    process.wait()
    return process.pid


@pytest.mark.skipif(sys.platform != "darwin", reason="installs a macOS .app")
def test_helper_replaces_the_bundle_and_relaunches_it(tmp_path):
    marker = tmp_path / "launched.txt"
    install_dir = tmp_path / "Applications"
    install_dir.mkdir()
    target = _bundle(install_dir, "0.8.0", marker)

    stage = tmp_path / "stage"
    stage.mkdir()
    _bundle(stage, "0.9.0", marker)
    dmg = tmp_path / "Cullr-0.9.0-arm64.dmg"
    _dmg_with(stage, dmg, "Cullr 0.9.0")

    script = tmp_path / "install-0.9.0.sh"
    script.write_text(update.helper_script(
        dmg=dmg, target=target, work=tmp_path / "work",
        log_file=tmp_path / "update.log", version="0.9.0", pid=_dead_pid()))
    script.chmod(0o755)
    result = subprocess.run(["/bin/sh", str(script)], capture_output=True, text=True,
                            timeout=300)
    log = (tmp_path / "update.log").read_text()

    assert result.returncode == 0, f"{result.stderr}\n{log}"
    assert "installed" in log and "done" in log
    # Exactly one app, and it is the new one.
    assert sorted(path.name for path in install_dir.iterdir()) == ["Cullr.app"]
    assert "0.9.0" in (target / "Contents" / "Info.plist").read_text(
        errors="replace") or plistlib.loads(
        (target / "Contents" / "Info.plist").read_bytes())["CFBundleShortVersionString"] == "0.9.0"
    # The download, the mount point and the helper itself are gone.
    assert not dmg.exists()
    assert not script.exists()
    assert not (tmp_path / "work" / "mnt").exists()
    # And the new app was really reopened: its executable appends to the marker.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.2)
    assert marker.exists(), log
    assert marker.read_text().split()[-1] == "0.9.0"


@pytest.mark.skipif(sys.platform != "darwin", reason="mounts a macOS .dmg")
def test_helper_leaves_the_old_app_alone_when_the_image_is_bad(tmp_path):
    marker = tmp_path / "launched.txt"
    install_dir = tmp_path / "Applications"
    install_dir.mkdir()
    target = _bundle(install_dir, "0.8.0", marker)
    broken = tmp_path / "Cullr-0.9.0-arm64.dmg"
    broken.write_bytes(b"not a disk image at all")

    script = tmp_path / "install-0.9.0.sh"
    script.write_text(update.helper_script(
        dmg=broken, target=target, work=tmp_path / "work",
        log_file=tmp_path / "update.log", version="0.9.0", pid=_dead_pid()))
    result = subprocess.run(["/bin/sh", str(script)], capture_output=True, text=True,
                            timeout=300)
    log = (tmp_path / "update.log").read_text()

    assert result.returncode != 0
    assert "FAIL" in log                       # never a silent success
    assert sorted(path.name for path in install_dir.iterdir()) == ["Cullr.app"]
    assert plistlib.loads((target / "Contents" / "Info.plist").read_bytes())[
        "CFBundleShortVersionString"] == "0.8.0"
    assert not (install_dir / "Cullr.app.new").exists()
    assert not marker.exists()                 # nothing was launched
