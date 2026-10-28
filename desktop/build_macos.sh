#!/usr/bin/env bash
# Build Cullr.app and a drag-to-Applications .dmg on macOS.
#
# Everything the app needs — the built UI, the MediaPipe face model, the Python
# runtime — is baked into the bundle, which is why the result is a large but
# self-contained .app. Signing is ad-hoc: notarisation needs an Apple Developer
# ID, so a downloaded build needs one manual "Open" in Gatekeeper.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="Cullr"
VERSION="${CULLR_VERSION:-0.14.0}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ARCH="$(uname -m)"
DMG="$ROOT/dist/${APP_NAME}-${VERSION}-${ARCH}.dmg"

step() { printf '\n\033[1m› %s\033[0m\n' "$1"; }

if [ ! -x "$PYTHON" ]; then
  echo "No interpreter at $PYTHON — set PYTHON=/path/to/python (with the desktop extra installed)." >&2
  exit 1
fi

step "Frontend"
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  npm --prefix "$ROOT/frontend" ci
fi
npm --prefix "$ROOT/frontend" run build

step "Face model"
if [ ! -s "$ROOT/models/face_landmarker.task" ]; then
  (cd "$ROOT/backend" && "$PYTHON" -m culler.cli fetch-model)
fi
test -s "$ROOT/models/face_landmarker.task"

step "Icon"
"$PYTHON" "$ROOT/desktop/make_icon.py"

step "Bundle"
rm -rf "$ROOT/build" "$ROOT/dist"
CULLR_VERSION="$VERSION" "$PYTHON" -m PyInstaller \
  --noconfirm --clean --log-level WARN \
  --distpath "$ROOT/dist" --workpath "$ROOT/build" \
  "$ROOT/desktop/Cullr.spec"

APP="$ROOT/dist/${APP_NAME}.app"
test -d "$APP"

step "Smoke test the bundle"
LOG="$(mktemp)"
# The smoke test opens a library and runs a cull in it. Point the bundled app
# at a throwaway data directory so a release build never leaves a test library
# in the developer's real catalog, and so the run is repeatable.
SMOKE_DATA="$(mktemp -d)"
CULLR_DATA_DIR="$SMOKE_DATA" "$APP/Contents/MacOS/$APP_NAME" --server-only >"$LOG" 2>&1 &
SERVICE_PID=$!
cleanup() { kill "$SERVICE_PID" 2>/dev/null || true; rm -rf "$SMOKE_DATA"; }
trap cleanup EXIT
BASE=""
for _ in $(seq 1 120); do
  BASE="$(sed -n 's/^ready \(http:\/\/[0-9.]*:[0-9]*\)$/\1/p' "$LOG" | head -1)"
  [ -n "$BASE" ] && break
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "bundled app exited during startup:" >&2; cat "$LOG" >&2; exit 1
  fi
  sleep 0.5
done
[ -n "$BASE" ] || { echo "bundled app never reported ready:" >&2; cat "$LOG" >&2; exit 1; }

CAPS="$(curl -fsS "$BASE/api/capabilities")"
case "$CAPS" in
  *'"faces":true'*) echo "faces: available" ;;
  *) echo "face model did not load inside the bundle: $CAPS" >&2; exit 1 ;;
esac
# The scene model's weights are a first-use download, but the runtime has to be
# inside the bundle or the cull-options toggle could never fetch them. A frozen
# onnxruntime that lost its dylib is exactly the failure this catches.
case "$CAPS" in
  *'"semantic_runtime":true'*) echo "scene runtime: available" ;;
  *) echo "onnxruntime missing from the bundle (install backend[semantic]): $CAPS" >&2
     exit 1 ;;
esac
# RAW decoding is not optional in a shipped build. Without rawpy the app lists
# a shoot's RAW files, counts them (so the dialog promises a cull of 3,016
# frames), and then rejects every one of them as unreadable. 0.8.0 shipped
# exactly that way; this is the assertion that would have caught it.
case "$CAPS" in
  *'"raw":true'*) echo "raw decoding: available" ;;
  *) echo "rawpy missing from the bundle (install backend[raw]): $CAPS" >&2
     exit 1 ;;
esac
# Importing rawpy is not the same as decoding with it: rawpy carries libraw as
# a native library, and a native library is what a frozen build loses. So hand
# the frozen server a real RAW file and insist on the JPEG the picker would
# show for it.
#
# Two containers, because they are separate decoders inside libraw. The ORF
# proves the rawpy path survived freezing; the CR3 is the one that failed in
# 0.8.0, and LibRaw needs its CRX decoder compiled in to read it. A build with
# a working rawpy and a CR3-less libraw passes the first and fails the second.
RAW_FIXTURES=(
  "$ROOT/backend/tests/fixtures/olympus_c5050z.orf"
  "$ROOT/backend/tests/fixtures/canon_eos_rp.cr3"
)
for RAW_FIXTURE in "${RAW_FIXTURES[@]}"; do
  if [ ! -s "$RAW_FIXTURE" ]; then
    echo "no RAW fixture to smoke-test with (expected $RAW_FIXTURE)" >&2
    exit 1
  fi
  RAW_TILE="$(mktemp)"
  curl -fsS -G --data-urlencode "path=$RAW_FIXTURE" -o "$RAW_TILE" \
    "$BASE/api/folders/preview?size=256"
  if [ "$(xxd -p -l 2 "$RAW_TILE")" != "ffd8" ]; then
    echo "the bundle could not decode $(basename "$RAW_FIXTURE") (no JPEG preview rendered)" >&2
    exit 1
  fi
  echo "raw decoding: rendered a $(wc -c <"$RAW_TILE" | tr -d ' ') byte preview from $(basename "$RAW_FIXTURE")"
  rm -f "$RAW_TILE"
done

# The preview is libraw's embedded thumbnail; the cull is the full decode, and
# the full decode is what rejected all 3,016 CR3 frames. So run a one-frame CR3
# shoot through the bundled server and insist the frame was evaluated: a clean
# job, and nothing in the Unreadable filter.
SHOOT="$(mktemp -d)"
cp "$ROOT/backend/tests/fixtures/canon_eos_rp.cr3" "$SHOOT/"
LIBRARY="$(curl -fsS -X POST -H 'Content-Type: application/json' \
  --data "{\"path\":\"$SHOOT\"}" "$BASE/api/libraries")"
LIBRARY_ID="$(printf '%s' "$LIBRARY" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')"
[ -n "$LIBRARY_ID" ] || { echo "the bundle would not open a library: $LIBRARY" >&2; exit 1; }
curl -fsS -X POST -H 'Content-Type: application/json' -d '{"force":true}' \
  "$BASE/api/libraries/$LIBRARY_ID/scan" >/dev/null
JOB=""
for _ in $(seq 1 240); do
  JOB="$(curl -fsS "$BASE/api/libraries/$LIBRARY_ID/job")"
  case "$JOB" in
    *'"status":"running"'*) sleep 0.5 ;;
    *) break ;;
  esac
done
case "$JOB" in
  *'"status":"done"'*) ;;
  *) echo "the bundled cull of a CR3 did not finish cleanly: $JOB" >&2; exit 1 ;;
esac
UNREADABLE="$(curl -fsS "$BASE/api/libraries/$LIBRARY_ID/photos?filter=errors")"
case "$UNREADABLE" in
  *'"total":0'*) echo "raw decoding: the bundle culled a CR3 frame instead of calling it unreadable" ;;
  *) echo "the bundle reported a CR3 as unreadable: $UNREADABLE" >&2; exit 1 ;;
esac
rm -rf "$SHOOT"
curl -fsS -o /dev/null "$BASE/" && echo "UI: served from the bundle"
curl -fsS -o /dev/null "$BASE/api/libraries" && echo "API: responding"
cleanup
trap - EXIT
sleep 1

step "Sign (ad-hoc)"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

step "DMG"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -quiet -volname "$APP_NAME $VERSION" -srcfolder "$STAGE" \
  -fs HFS+ -format UDZO -ov "$DMG"
rm -rf "$STAGE"

printf '\n%s\n' "$DMG"
shasum -a 256 "$DMG" | awk '{print "sha256 " $1}'
