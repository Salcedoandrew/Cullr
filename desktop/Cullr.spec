# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the macOS app: one .app with the UI, model and service.

Everything read-only ships inside the bundle (frontend/dist, the face model and
MediaPipe's dylib); the catalog and caches live in Application Support at run
time. Building from the repository root with ``desktop/build_macos.sh`` keeps
the paths below true.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

APP_NAME = "Cullr"
ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is provided by PyInstaller
VERSION = os.environ.get("CULLR_VERSION", "0.2.0")

datas = [
    (str(ROOT / "frontend" / "dist"), "frontend/dist"),
    (str(ROOT / "models" / "face_landmarker.task"), "models"),
    # Shipped in Contents/Resources: the bundle redistributes MediaPipe and the
    # Apache-2.0 model, so the attribution travels with the binary.
    (str(ROOT / "NOTICE.md"), "."),
]
binaries = []
hiddenimports = [
    "webview.platforms.cocoa",
    # The app menu (Cullr ▸ Check for Updates…) is imported inside a function, so
    # name it explicitly: without it the window would open with no menu at all.
    "webview.menu",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

try:  # MediaPipe has no PyInstaller hook; ship its dylib and data explicitly.
    datas += collect_data_files("mediapipe")
    binaries += collect_dynamic_libs("mediapipe")
    # Tests and benchmarks are not shipped code: they drag in scipy, tensorflow
    # and protobuf probes that would bloat the bundle and add import noise.
    hiddenimports += collect_submodules(
        "mediapipe",
        filter=lambda name: ".test" not in name and ".benchmark" not in name,
    )
except Exception as exc:  # pragma: no cover - only breaks a packaging run
    raise SystemExit(f"mediapipe could not be collected for packaging: {exc}")

try:  # Optional: semantic duplicate confirmation (the `semantic` extra).
    # The scene weights are a user download, but the runtime has to be inside
    # the bundle or an installed copy could never use them. Absent from a dev
    # environment, the build simply ships without it and the gate stays
    # hash-only, which is the default anyway.
    datas += collect_data_files("onnxruntime")
    binaries += collect_dynamic_libs("onnxruntime")
    hiddenimports += collect_submodules("onnxruntime")
except Exception as exc:  # pragma: no cover - optional by design
    print(f"semantic dedup not bundled: onnxruntime not collectable ({exc})")

analysis = Analysis(  # noqa: F821
    [str(ROOT / "desktop" / "cullr_desktop.py")],
    pathex=[str(ROOT / "backend")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # matplotlib is not optional here: mediapipe's package __init__ reaches
    # drawing_utils, which imports it at module import time.
    excludes=["tkinter", "PyQt5", "PySide6", "IPython", "pytest", "PyInstaller"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(ROOT / "assets" / "Cullr.icns"),
)

collection = COLLECT(  # noqa: F821
    exe,
    analysis.binaries,
    analysis.datas,
    strip=True,
    upx=False,
    name=APP_NAME,
)

app = BUNDLE(  # noqa: F821
    collection,
    name=f"{APP_NAME}.app",
    icon=str(ROOT / "assets" / "Cullr.icns"),
    bundle_identifier="com.cullr.app",
    version=VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "CFBundleExecutable": APP_NAME,
        "LSApplicationCategoryType": "public.app-category.photography",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "NSHumanReadableCopyright": "Photo culling that stays on your machine.",
    },
)
