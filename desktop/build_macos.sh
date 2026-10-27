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
VERSION="${CULLR_VERSION:-0.8.0}"
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
"$APP/Contents/MacOS/$APP_NAME" --server-only >"$LOG" 2>&1 &
SERVICE_PID=$!
cleanup() { kill "$SERVICE_PID" 2>/dev/null || true; }
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
RAW_FIXTURE="$ROOT/backend/tests/fixtures/olympus_c5050z.orf"
if [ ! -s "$RAW_FIXTURE" ]; then
  echo "no RAW fixture to smoke-test with (expected $RAW_FIXTURE)" >&2
  exit 1
fi
RAW_TILE="$(mktemp)"
curl -fsS -G --data-urlencode "path=$RAW_FIXTURE" -o "$RAW_TILE" \
  "$BASE/api/folders/preview?size=256"
if [ "$(xxd -p -l 2 "$RAW_TILE")" != "ffd8" ]; then
  echo "the bundle could not decode a RAW file (no JPEG preview rendered)" >&2
  exit 1
fi
echo "raw decoding: rendered a $(wc -c <"$RAW_TILE" | tr -d ' ') byte preview from a RAW"
rm -f "$RAW_TILE"
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
