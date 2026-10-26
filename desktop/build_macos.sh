#!/usr/bin/env bash
# Build Sharpen.app and a drag-to-Applications .dmg on macOS.
#
# Everything the app needs — the built UI, the MediaPipe face model, the Python
# runtime — is baked into the bundle, which is why the result is a large but
# self-contained .app. Signing is ad-hoc: notarisation needs an Apple Developer
# ID, so a downloaded build needs one manual "Open" in Gatekeeper.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="Sharpen"
VERSION="${SHARPEN_VERSION:-0.1.0}"
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
SHARPEN_VERSION="$VERSION" "$PYTHON" -m PyInstaller \
  --noconfirm --clean --log-level WARN \
  --distpath "$ROOT/dist" --workpath "$ROOT/build" \
  "$ROOT/desktop/Sharpen.spec"

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
