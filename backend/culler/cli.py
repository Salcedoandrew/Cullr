"""Command line entry point: scan, serve, export, fetch-model, fetch-scene-model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from .config import PRESETS, CullingSettings, resource_path
from .db import Store, default_data_dir
from .metrics.faces import MODEL_FILENAME, MODEL_URL
from .metrics.scene import (
    MODEL_URL as SCENE_MODEL_URL,
    default_model_path as scene_model_path,
    download_model as download_scene_model,
    model_verified as scene_model_verified,
)
from .pipeline import run_scan

# Pinned: the same SHA-256 facet verifies for its MediaPipe bundle.
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_DIR = resource_path("models")
# The scene model's URL and pinned SHA-256 live in metrics/scene.py, which is
# also what the app uses for its own first-use download.


def _store(args) -> Store:
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
    return Store(data_dir / "catalog.db")


def cmd_fetch_model(args) -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / MODEL_FILENAME
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == MODEL_SHA256:
            print(f"model already present: {target}")
            return 0
        print("existing model failed verification, re-downloading")
        target.unlink()
    print(f"downloading {MODEL_URL}")
    tmp = target.with_suffix(".download")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        print(f"checksum mismatch: {digest}", file=sys.stderr)
        return 1
    tmp.rename(target)
    print(f"saved {target}")
    return 0


def cmd_fetch_scene_model(args) -> int:
    """Download the optional scene model used for semantic dedup."""
    target = scene_model_path()
    if scene_model_verified(target):
        print(f"scene model already present: {target}")
        return 0
    if target.exists():
        print("existing scene model failed verification, re-downloading")
    print(f"downloading {SCENE_MODEL_URL}")
    try:
        download_scene_model(target)
    except Exception as exc:  # noqa: BLE001 - report it, do not crash the CLI
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"saved {target}")
    return 0


def _settings_from_args(args) -> CullingSettings:
    settings = CullingSettings(preset=args.preset)
    if args.strictness is not None:
        settings.strictness = args.strictness
    if getattr(args, "selection_strictness", None) is not None:
        settings.selection_strictness = args.selection_strictness
    if args.no_blinks:
        settings.detect_blinks = False
    if args.no_grouping:
        settings.group_duplicates = False
    if args.keep_per_group is not None:
        settings.keep_min_per_group = args.keep_per_group
    if getattr(args, "semantic", False):
        settings.semantic_dedup = True
    return settings


def cmd_scan(args) -> int:
    root = Path(args.folder).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    store = _store(args)
    settings = _settings_from_args(args)
    library = store.create_or_get_library(str(root), settings.to_dict())
    job = store.create_job(library["id"])

    last = {"stage": None}

    def progress(stage: str, done: int, total: int, message: str) -> None:
        if stage != last["stage"] or done == total:
            print(f"[{stage}] {done}/{total} {message}")
            last["stage"] = stage if done < total else None

    result = run_scan(store, library["id"], settings, progress, force=args.force)
    store.update_job(job["id"], status="done", stage="done", done=result.total,
                     total=result.total, message="finished")
    stats = store.stats(library["id"])
    print()
    print(f"photos analysed : {result.total} ({result.reused} reused, {result.failed} failed)")
    print(f"groups          : {result.groups}")
    print(f"people          : {result.persons}"
          + (f" ({result.uncovered} without a keeper)" if result.persons else ""))
    print(f"keepers         : {stats['keep']}")
    print(f"rejected        : {stats['reject']}  (blur {stats['blurry']}, "
          f"eyes {stats['blink']}, duplicates {stats['duplicates']}, "
          f"exposure {stats['exposure']})")
    print(f"elapsed         : {result.elapsed:.1f}s")
    if result.errors:
        print("errors:")
        for line in result.errors[:10]:
            print(f"  - {line}")
    if args.json:
        print(json.dumps({"library_id": library["id"], "stats": stats,
                          "groups": result.groups, "persons": result.persons,
                          "uncovered_persons": result.uncovered,
                          "elapsed": result.elapsed}))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
    app = create_app(data_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_export(args) -> int:
    from . import export

    root = Path(args.folder).expanduser().resolve()
    store = _store(args)
    library = store.create_or_get_library(str(root))
    if args.mode == "sidecar":
        result = export.write_sidecars(store, library["id"], only_raw=not args.all_files)
    elif args.mode == "copy":
        result = export.copy_keepers(store, library["id"], args.dest)
    else:
        result = export.write_csv(store, library["id"], args.dest)
    print(f"written={result.written} copied={result.copied} "
          f"skipped={result.skipped} dest={result.dest}")
    for error in result.errors[:10]:
        print(f"  - {error}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cull",
        description="Cullr — local AI photo culling (quality, blinks, duplicates)",
    )
    parser.add_argument("--data-dir", help="catalog/preview directory (default ~/.cullr)")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="analyse a folder and score every photo")
    scan.add_argument("folder")
    scan.add_argument("--preset", default="wedding", choices=sorted(PRESETS))
    scan.add_argument("--strictness", type=int,
                      help="0-100; blur-only strictness (the Customize slider)")
    scan.add_argument("--selection-strictness", type=int, dest="selection_strictness",
                      help="0-100; overall selection strictness (the Auto Target stops)")
    scan.add_argument("--keep-per-group", type=int, dest="keep_per_group")
    scan.add_argument("--no-blinks", action="store_true")
    scan.add_argument("--no-grouping", action="store_true")
    scan.add_argument("--force", action="store_true", help="re-analyse unchanged files")
    scan.add_argument("--semantic", action="store_true",
                      help="confirm loose duplicate matches with DINOv2 scene "
                           "embeddings (needs fetch-scene-model)")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=cmd_scan)

    serve = sub.add_parser("serve", help="run the review UI and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    export_cmd = sub.add_parser("export", help="write XMP sidecars, a CSV or copy keepers")
    export_cmd.add_argument("folder")
    export_cmd.add_argument("--mode", default="sidecar", choices=["sidecar", "copy", "csv"])
    export_cmd.add_argument("--dest", default=None)
    export_cmd.add_argument("--all-files", action="store_true",
                            help="write sidecars for JPEG/TIFF too, not only RAW")
    export_cmd.set_defaults(func=cmd_export)

    fetch = sub.add_parser("fetch-model", help="download the MediaPipe face model")
    fetch.set_defaults(func=cmd_fetch_model)

    fetch_scene = sub.add_parser(
        "fetch-scene-model",
        help="download the DINOv2 scene model (semantic duplicate detection)",
    )
    fetch_scene.set_defaults(func=cmd_fetch_scene_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
