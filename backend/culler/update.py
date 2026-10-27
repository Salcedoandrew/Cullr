"""In-app update: check the latest release, download it, verify it, replace in place.

The flow is deliberately manual. Nothing is fetched on launch and the app makes
no network call until the user asks it to check, which is the only way this
feature can be honest about Cullr not phoning home.

The server does all four steps, because the loopback server is the part of the
app that is allowed to touch the filesystem and run a process:

1. ``GET .../releases/latest/download/update.json`` — a manifest published as a
   release asset (see ``scripts/update_manifest.py``). Semantic version
   comparison decides whether it is newer than the running build.
2. Download the ``.dmg`` it names into the data directory, with progress, into a
   ``.part`` file that is renamed only after the SHA-256 matches the one in
   ``SHA256SUMS``.
3. Quit Cullr and hand over to a detached ``/bin/sh`` helper (see
   :func:`helper_script`). The helper waits for this process to exit, mounts the
   verified ``.dmg``, stages the new ``.app``, verifies the staged copy, and only
   then deletes the old bundle and moves the new one into its place.
4. The helper unmounts, cleans up and relaunches Cullr on the new version.

What the SHA-256 actually proves: the ``.dmg`` is the file the release published,
matching ``SHA256SUMS`` byte for byte. It does *not* prove the release itself was
not tampered with — a manifest and an asset can both be replaced by whoever
controls the repository. The host allowlist and the HTTPS-only rule below keep a
tampered *manifest* from pointing the download somewhere else; they do not turn
this into a signature scheme. Cullr's builds are ad-hoc signed, so nothing here
claims Gatekeeper or notarisation behaviour.

Every field of the manifest is treated as untrusted input: the version must be
``x.y.z``, the URL must be HTTPS on GitHub's release hosts, the filename must be
a bare name ending in ``.dmg``, and the hash must be 64 hex characters. Nothing
from the manifest is ever interpolated into a shell command unquoted — the
helper is generated from values that have already been validated here, and every
path in it is single-quoted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import __version__

log = logging.getLogger(__name__)

APP_NAME = "Cullr"
# Where the app expects to live, and the only place the helper falls back to
# when it cannot work out where the running bundle is.
STANDARD_APP_DIR = Path("/Applications")
# A manifest asset on the latest release is a stable URL: it always resolves to
# the newest `v*` release, so the app never scrapes a release web page.
MANIFEST_URL = (
    f"https://github.com/Salcedoandrew/cullr/releases/latest/download/update.json"
)
# GitHub serves release assets from github.com and redirects to
# objects.githubusercontent.com. Both are HTTPS-only, and nothing else is
# allowed to appear in a manifest.
RELEASE_HOSTS = frozenset({"github.com", "objects.githubusercontent.com"})
MANIFEST_LIMIT = 256 * 1024
VERSION_RE = re.compile(r"^(\d{1,4})\.(\d{1,4})\.(\d{1,4})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.dmg$")

# One phase per state the UI shows. "checking" is the UI's own state while it
# waits for /api/update/check to answer; every other state is real here.
PHASES = (
    "idle",          # nothing has been checked yet
    "up_to_date",    # the running build is the latest release
    "available",     # a newer release exists; it can be downloaded
    "downloading",   # transfer in flight, with progress
    "ready",         # downloaded and hash-verified; not installed yet
    "cancelled",     # the user stopped the download or deleted it
    "installing",    # the helper has been launched and Cullr is quitting
    "error",         # nothing was changed; the reason is in "error"
)
# Downloading is allowed from here: a cancelled or failed attempt can be retried
# without asking GitHub for the manifest again.
RETRYABLE = ("available", "cancelled", "error")
CHUNK = 1 << 20


class UpdateError(RuntimeError):
    """Anything the user should see as an update error instead of a traceback."""


class UpdateCancelled(UpdateError):
    """The user stopped the download; the partial file has been removed."""


def parse_version(text: str) -> tuple[int, int, int]:
    """``"v0.9.0"`` -> ``(0, 9, 0)``. Raises on anything else."""
    match = VERSION_RE.match(str(text).strip().lstrip("vV"))
    if not match:
        raise UpdateError(f"not a version number: {text!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_newer(candidate: str, current: str) -> bool:
    """Semantic comparison, so 0.10.0 is newer than 0.9.0 (string order is not)."""
    return parse_version(candidate) > parse_version(current)


def _require_https(url: str) -> urllib.parse.SplitResult:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise UpdateError(f"refusing a non-HTTPS URL: {url}")
    if parts.hostname not in RELEASE_HOSTS:
        raise UpdateError(f"refusing a URL on an unexpected host: {parts.hostname}")
    return parts


def validate_manifest(raw: Any) -> dict[str, Any]:
    """Check an untrusted manifest and return the fields the app will use.

    Raises :class:`UpdateError` for anything that is not exactly what a Cullr
    release publishes. Nothing is coerced or defaulted into shape: a manifest
    that is nearly right is a manifest that cannot be trusted to name the file
    the release signed a hash for.
    """
    if not isinstance(raw, dict):
        raise UpdateError("the update manifest is not a JSON object")
    version = str(raw.get("version", "")).strip()
    parse_version(version)                      # raises when it is not x.y.z
    url = str(raw.get("url", "")).strip()
    parts = _require_https(url)
    if not parts.path.endswith(".dmg"):
        raise UpdateError("the manifest does not point at a .dmg")
    filename = str(raw.get("filename") or Path(parts.path).name).strip()
    if not FILENAME_RE.match(filename):
        raise UpdateError(f"the manifest names an unusable file: {filename!r}")
    if filename != Path(parts.path).name:
        raise UpdateError("the manifest's filename and URL disagree")
    sha256 = str(raw.get("sha256", "")).strip().lower()
    if not SHA256_RE.match(sha256):
        raise UpdateError("the manifest's sha256 is not a SHA-256 digest")
    notes_url = str(raw.get("notes_url") or "").strip()
    if notes_url:
        _require_https(notes_url)
    else:
        notes_url = f"https://github.com/Salcedoandrew/cullr/releases/tag/v{version}"
    size = raw.get("size")
    try:
        size = int(size) if size is not None else 0
    except (TypeError, ValueError):
        size = 0
    return {
        "version": version,
        "url": url,
        "filename": filename,
        "sha256": sha256,
        "notes_url": notes_url,
        "size": max(0, size),
        "released_at": str(raw.get("released_at") or ""),
    }


def _transport_message(exc: Exception, url: str) -> str:
    """A sentence for a failed request, since the user sees this text.

    A 404 on the manifest is the ordinary state before the first release carries
    one, and it should read that way rather than as a raw HTTPError.
    """
    host = urllib.parse.urlsplit(url).hostname or url
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return (f"the release manifest is not published yet (HTTP 404 from {host}); "
                    f"releases before this feature do not carry one")
        if exc.code in (403, 429):
            return f"{host} refused the request (HTTP {exc.code}); try again a little later"
        return f"{host} answered HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return f"could not reach {host} ({exc.reason})"
    return f"{type(exc).__name__}: {exc}"


def fetch_manifest(url: str = MANIFEST_URL, timeout: float = 15.0) -> dict[str, Any]:
    """Read and validate the update manifest. HTTPS on a release host only."""
    _require_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MANIFEST_LIMIT + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise UpdateError(_transport_message(exc, url)) from exc
    if len(body) > MANIFEST_LIMIT:
        raise UpdateError("the update manifest is implausibly large")
    try:
        raw = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError(f"the update manifest is not readable JSON ({exc})") from exc
    return validate_manifest(raw)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_update(
    url: str,
    sha256: str,
    filename: str,
    dest_dir: Path | str,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Stream the ``.dmg`` to ``dest_dir`` and verify it before it counts.

    The bytes land in ``<name>.part`` and are renamed only once the SHA-256
    matches, so a truncated, tampered or interrupted transfer can never be
    installed. Any failure — including a checksum mismatch — deletes the partial
    file and raises; the installed app is not touched by any of this.
    """
    target = Path(dest_dir) / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.unlink(missing_ok=True)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": f"{APP_NAME}/{__version__}"})
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise UpdateError(_transport_message(exc, url)) from exc
        with response, tmp.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            if progress is not None:
                progress(0, total)
            while chunk := response.read(CHUNK):
                if cancel is not None and cancel.is_set():
                    raise UpdateCancelled("the download was cancelled")
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        if cancel is not None and cancel.is_set():
            raise UpdateCancelled("the download was cancelled")
        digest = _digest(tmp)
        if digest != sha256:
            raise UpdateError(
                f"the download does not match the release's SHA-256 "
                f"(expected {sha256}, got {digest})")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return target


# ---- the installed bundle ------------------------------------------------ #


def bundle_of(executable: str | Path | None = None) -> Path | None:
    """The ``.app`` the running process lives in, or None from a source checkout."""
    start = Path(executable or sys.executable)
    try:
        start = start.resolve()
    except OSError:                      # a deleted or unreadable path
        start = Path(executable or sys.executable)
    for candidate in (start, *start.parents):
        if candidate.suffix == ".app":
            return candidate
    return None


def unusable_reason(path: Path | str) -> str | None:
    """Why this bundle cannot be replaced in place, if it cannot.

    Both cases are read-only copies of the app, and both quit-then-fail rather
    than anything useful, so an update has to decide before it quits:

    * macOS App Translocation — a quarantined app runs from a random path macOS
      owns, so replacing "itself" there is impossible;
    * a mounted disk image — Cullr was launched straight out of the ``.dmg``.

    The returned text is user-facing: it names what to do about it.
    """
    text = str(path)
    if "/AppTranslocation/" in text:
        return ("Cullr is running from the read-only copy macOS makes of a"
                " quarantined download")
    if text.startswith("/Volumes/"):
        return "Cullr is running from a disk image"
    return None


def resolve_app_path(executable: str | Path | None = None,
                     app_dir: Path | str = STANDARD_APP_DIR) -> Path | None:
    """Which bundle an update would replace.

    Detection walks up from the running executable, so an app kept outside
    ``/Applications`` updates itself where it actually is. When that answer
    cannot be replaced (:func:`unusable_reason`) or there is no bundle at all
    (a source checkout), the standard install location is tried instead — which
    is the right answer for an app run from a disk image, since the copy in
    Applications is the one being kept up to date.

    Returns None when neither answer exists; the caller reports that instead of
    installing somewhere unexpected.
    """
    found = bundle_of(executable)
    if found is not None and unusable_reason(found) is None:
        return found
    standard = Path(app_dir) / f"{APP_NAME}.app"
    return standard if standard.is_dir() else None


def _q(value: Any) -> str:
    """Single-quote a value for /bin/sh, closing and reopening around quotes."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def helper_script(*, dmg: Path | str, target: Path | str, work: Path | str,
                  log_file: Path | str, version: str, pid: int | None = None) -> str:
    """The detached installer, as ``/bin/sh`` source.

    Generated, single-use, and written to be read: the order is the guarantee.
    The new bundle is fully staged in the target's own parent directory and
    checked for a plist and an executable *before* the old one is deleted, so a
    copy that fails part-way leaves the working app in place rather than an empty
    Applications folder. Every failure logs one ``FAIL`` line and exits non-zero;
    nothing pretends an update happened that did not.
    """
    pid = os.getpid() if pid is None else pid
    name = Path(str(target)).name or f"{APP_NAME}.app"
    return f"""#!/bin/sh
# Cullr {version} update helper. Generated on {time.strftime('%Y-%m-%d %H:%M:%S')};
# it deletes itself when it is done. Do not edit: rerun "Check for updates".
#
# Order matters. The new app is copied next to the old one and verified there;
# only then is the old bundle deleted and the new one moved into its place. If
# anything before that step fails, the installed app is untouched.
set -u

LOG={_q(log_file)}
DMG={_q(dmg)}
TARGET={_q(target)}
WORK={_q(work)}
PID={pid}
NAME={_q(name)}
STAGED="$TARGET.new"
APP="{APP_NAME}"

log() {{ printf '%s %s\\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }}

cleanup() {{
  /usr/bin/hdiutil detach "$WORK/mnt" >/dev/null 2>&1 || true
  rm -rf "$STAGED" "$WORK/mnt"
  rm -f "$0"
}}

fail() {{
  log "FAIL $*"
  cleanup
  exit 1
}}

# A zombie still answers kill -0, so the command line decides whether the app is
# really still up. Waiting here is what makes replacing the bundle safe.
alive() {{
  /bin/ps -p "$PID" -o command= 2>/dev/null | /usr/bin/grep -qF "$TARGET/Contents/MacOS/"
}}

i=0
while [ "$i" -lt 1200 ] && alive; do
  i=$((i + 1))
  sleep 0.5
done
if alive; then
  fail "Cullr (pid $PID) is still running after 10 minutes; nothing was changed"
fi
log "Cullr exited; installing $APP {version} over $TARGET"

mkdir -p "$WORK/mnt" || fail "cannot create the mount point $WORK/mnt"
# hdiutil answers "Resource busy" now and then when a volume was attached or
# detached moments earlier, so a failed mount is retried twice before the
# update is called off.
mounted=0
attempt=1
while [ "$attempt" -le 3 ]; do
  if /usr/bin/hdiutil attach "$DMG" -nobrowse -readonly -mountpoint "$WORK/mnt" >>"$LOG" 2>&1; then
    mounted=1
    break
  fi
  log "WARN mount attempt $attempt of 3 failed"
  attempt=$((attempt + 1))
  sleep 2
done
if [ "$mounted" != "1" ]; then
  fail "cannot mount $DMG"
fi
log "mounted $DMG"

SOURCE="$WORK/mnt/$NAME"
if [ ! -d "$SOURCE" ]; then
  COUNT="$(ls -d "$WORK"/mnt/*.app 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$COUNT" = "1" ]; then
    SOURCE="$(ls -d "$WORK"/mnt/*.app)"
  else
    fail "the disk image does not contain $NAME (found $COUNT apps); nothing was changed"
  fi
fi

rm -rf "$STAGED"
if ! /usr/bin/ditto "$SOURCE" "$STAGED" >>"$LOG" 2>&1; then
  fail "could not copy $SOURCE into $STAGED; the installed app was left alone"
fi
if [ ! -f "$STAGED/Contents/Info.plist" ] || [ ! -x "$STAGED/Contents/MacOS/$APP" ]; then
  fail "the staged copy is incomplete; the installed app was left alone"
fi
log "staged and verified $STAGED"

if ! rm -rf "$TARGET"; then
  fail "could not delete $TARGET; the new version is staged at $STAGED"
fi
if ! mv "$STAGED" "$TARGET"; then
  fail "could not move the new version into $TARGET; it is still at $STAGED"
fi
log "installed $TARGET"

/usr/bin/hdiutil detach "$WORK/mnt" >>"$LOG" 2>&1 || log "WARN could not unmount $WORK/mnt"
rm -f "$DMG"
cleanup
log "removed the download and the mount point"

if /usr/bin/open "$TARGET" >>"$LOG" 2>&1; then
  log "relaunched $TARGET on $APP {version}"
else
  log "WARN $APP {version} is installed but could not be relaunched; open it from Applications"
fi
log "done: $APP {version} installed at $TARGET"
"""


# ---- the state machine --------------------------------------------------- #


class UpdateManager:
    """One manual update flow, kept in memory for the life of the process.

    The UI polls :meth:`status` while a download runs; every transition is also
    written to the app's log, so "no update", "downloaded", "installed" and
    "failed" stay distinguishable after the fact.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        current: str = __version__,
        manifest_url: str = MANIFEST_URL,
        app_path: Path | str | None = None,
        fetcher: Callable[[str], dict[str, Any]] = fetch_manifest,
        downloader: Callable[..., Path] = download_update,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.current = current
        self.manifest_url = manifest_url
        self._fetcher = fetcher
        self._downloader = downloader
        self._app_path = Path(app_path) if app_path is not None else None
        self._app_resolved = app_path is not None
        self.work_dir = self.data_dir / "updates"
        self.log_file = self.data_dir / "logs" / "update.log"

        self._lock = threading.Lock()
        self._phase = "idle"
        self._error: str | None = None
        self._manifest: dict[str, Any] | None = None
        self._dmg: Path | None = None
        self._done = 0
        self._total = 0
        self._checked_at: float | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- reading state --------------------------------------------------- #

    def app_path(self) -> Path | None:
        """Resolve once: the bundle cannot move while the app is running."""
        if not self._app_resolved:
            self._app_path = resolve_app_path()
            self._app_resolved = True
        return self._app_path

    def status(self) -> dict[str, Any]:
        with self._lock:
            manifest = self._manifest or {}
            percent = int(self._done * 100 / self._total) if self._total else None
            return {
                "current": self.current,
                "phase": self._phase,
                "error": self._error,
                "latest": manifest.get("version"),
                "notes_url": manifest.get("notes_url"),
                "filename": manifest.get("filename"),
                "download_url": manifest.get("url"),
                "size": manifest.get("size") or 0,
                "done": self._done,
                "total": self._total,
                "percent": percent,
                "checked_at": self._checked_at,
                "installed_at": str(self.app_path()) if self.app_path() else None,
                "update_log": str(self.log_file),
            }

    def _set(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, f"_{key}", value)

    def _fail(self, message: str, exc: BaseException | None = None) -> dict[str, Any]:
        self._set(phase="error", error=message, done=0, total=0)
        log.warning("update: %s%s", message, f" ({exc})" if exc else "")
        return self.status()

    # ---- 1. check -------------------------------------------------------- #

    def check(self) -> dict[str, Any]:
        """Fetch the manifest and compare it with the running version.

        Synchronous on purpose: the frontend's "Checking for updates" state *is*
        this call, and a check that cannot answer is an error the user sees
        immediately rather than a job to poll.
        """
        try:
            manifest = self._fetcher(self.manifest_url)
        except Exception as exc:  # noqa: BLE001 - network, TLS, JSON, validation
            return self._fail(f"could not check for updates: {exc}", exc)

        newer = is_newer(manifest["version"], self.current)
        self._set(manifest=manifest, error=None, checked_at=time.time(),
                  phase="available" if newer else "up_to_date", done=0, total=0)
        if newer:
            log.info("update: %s is available (running %s)", manifest["version"], self.current)
        else:
            log.info("update: %s is the latest release (running %s)",
                     manifest["version"], self.current)
        return self.status()

    # ---- 2. download ----------------------------------------------------- #

    def start_download(self) -> dict[str, Any]:
        with self._lock:
            phase, manifest = self._phase, self._manifest
        if phase == "downloading":
            return self.status()
        if manifest is None or phase not in RETRYABLE:
            return self._fail("there is no newer release to download")
        self._cancel.clear()
        self._set(phase="downloading", error=None, done=0, total=manifest.get("size") or 0)
        log.info("update: downloading %s", manifest["filename"])
        self._thread = threading.Thread(target=self._download, name="cullr-update",
                                        args=(manifest,), daemon=True)
        self._thread.start()
        return self.status()

    def _download(self, manifest: dict[str, Any]) -> None:
        def progress(done: int, total: int) -> None:
            with self._lock:
                self._done = done
                if total:
                    self._total = total

        try:
            path = self._downloader(manifest["url"], manifest["sha256"],
                                    manifest["filename"], self.work_dir,
                                    progress, self._cancel)
        except UpdateCancelled:
            log.info("update: download cancelled; the installed app is unchanged")
            self._set(phase="cancelled", error=None,
                      dmg=None, done=0, total=0)
        except Exception as exc:  # noqa: BLE001 - reported, never a traceback
            self._fail(f"the download failed: {exc}", exc)
        else:
            self._set(phase="ready", dmg=path, error=None)
            log.info("update: %s downloaded and verified", path)

    def cancel(self) -> dict[str, Any]:
        """Stop before installation: the partial or verified file is deleted."""
        with self._lock:
            phase = self._phase
        if phase == "downloading":
            self._cancel.set()
            return self.status()
        if phase == "ready" and self._dmg is not None:
            self._dmg.unlink(missing_ok=True)
            self._set(phase="cancelled", dmg=None, done=0, total=0)
            log.info("update: the verified download was deleted; nothing was installed")
            return self.status()
        return self.status()

    # ---- 3. install ------------------------------------------------------ #

    def install(self, on_quit: Callable[[], None] | None = None) -> dict[str, Any]:
        """Hand over to the detached helper and ask the app to quit.

        Re-hashes the downloaded file first: between the download and this click
        it sat on disk with nothing watching it. A mismatch deletes it and
        installs nothing.
        """
        with self._lock:
            phase, manifest, dmg = self._phase, self._manifest, self._dmg
        if phase != "ready" or manifest is None or dmg is None:
            return self._fail("there is no verified download to install")
        target = self.app_path()
        if target is None:
            running = bundle_of()
            reason = unusable_reason(running) if running is not None else None
            if reason:
                return self._fail(f"{reason}; move it into Applications and update "
                                  f"from there")
            return self._fail(
                "could not work out which installed copy of Cullr to replace; "
                "move Cullr into Applications and try again")
        try:
            digest = _digest(dmg)
        except OSError as exc:
            return self._fail(f"the downloaded file is no longer readable: {exc}", exc)
        if digest != manifest["sha256"]:
            dmg.unlink(missing_ok=True)
            self._set(dmg=None)
            return self._fail(
                "the downloaded file changed on disk since it was verified; "
                "nothing was installed")

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        script = self.work_dir / f"install-{manifest['version']}.sh"
        try:
            script.write_text(helper_script(dmg=dmg, target=target, work=self.work_dir,
                                            log_file=self.log_file,
                                            version=manifest["version"]),
                              encoding="utf-8")
            script.chmod(0o755)
            subprocess.Popen(
                ["/bin/sh", str(script)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,       # survives this process quitting
                close_fds=True,
            )
        except OSError as exc:
            return self._fail(f"could not start the installer: {exc}", exc)

        self._set(phase="installing", error=None)
        log.info("update: installer launched for %s (replacing %s); quitting",
                 manifest["version"], target)
        if on_quit is not None:
            # Let the HTTP response that announces "installing" reach the UI
            # before the window goes away.
            threading.Thread(target=self._quit, name="cullr-update-quit",
                             args=(on_quit,), daemon=True).start()
        return self.status()

    @staticmethod
    def _quit(on_quit: Callable[[], None], delay: float = 1.0) -> None:
        time.sleep(delay)
        try:
            on_quit()
        except Exception:  # noqa: BLE001 - the helper will not wait for a traceback
            log.exception("update: quitting for the update failed")
