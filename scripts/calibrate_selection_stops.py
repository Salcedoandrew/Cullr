#!/usr/bin/env python3
"""Measure the Auto Target stops on a real shoot, and check they mean what the dialog says.

The stop values in ``config.SELECTION_STOPS`` are only honest if they were
measured, so this is the tool that measures them. It runs the real pipeline over
a folder — the analysis is cached in a throwaway catalog, so the first run pays
the decode/face cost and every later run is a re-gate — then prints one line per
stop: how many frames it keeps, and what it does to the photographer's own
picks, if you point it at their selected set.

    python3 scripts/calibrate_selection_stops.py ~/Pictures/wedding \\
        --selects ~/Pictures/wedding-SELECTS/Selected
    python3 scripts/calibrate_selection_stops.py ~/Pictures/wedding --sweep

``--sweep`` adds the raw 0..100 curve. Read it with care: the four points that
happen to be stop values carry their keep fraction, every other point is gates
only -- which is what a hand-set ``--selection-strictness`` gets. That contrast
is the whole argument for the floor: on the wedding this was built against
(5,745 frames) the gates alone moved the keeper count 18.4%..21.6% across the
entire 0..100 range, while the stops take it from 5.0% to 21.5%.

The catalog lives under ``.calibration/`` by default (gitignored). The shoot is
only ever read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from culler.config import CullingSettings, stop_keep_fraction, strictness_stops  # noqa: E402
from culler.db import Store  # noqa: E402
from culler.pipeline import run_scan  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def measure(store: Store, library_id: int, settings: CullingSettings,
            picks: set[str], label: str) -> dict:
    started = time.monotonic()
    result = run_scan(store, library_id, settings)
    rows = store.photos(library_id)
    kept = [r for r in rows if r["verdict"] == "keep"]

    kept_groups: dict[int, int] = defaultdict(int)
    for row in kept:
        if row["group_id"] is not None:
            kept_groups[row["group_id"]] += 1

    picked = {"kept": 0, "same_group": 0, "lost": 0}
    for row in rows:
        if Path(row["path"]).stem not in picks:
            continue
        if row["verdict"] == "keep":
            picked["kept"] += 1
        elif row["group_id"] is not None and kept_groups.get(row["group_id"]):
            picked["same_group"] += 1
        else:
            picked["lost"] += 1

    summary = {
        "label": label,
        "strictness": settings.effective_strictness(),
        "stop_keep_fraction": stop_keep_fraction(settings.selection_strictness),
        "total": result.total,
        "kept": len(kept),
        "kept_pct": round(100.0 * len(kept) / max(1, result.total), 2),
        "picks": picked,
        "rejections": dict(sorted(Counter(
            r["reason"] for r in rows if r["verdict"] != "keep").items())),
        "seconds": round(time.monotonic() - started, 1),
    }
    tail = ""
    if picks:
        survived = picked["kept"] + picked["same_group"]
        tail = (f" · picks kept {picked['kept']}, same group {picked['same_group']},"
                f" moment lost {picked['lost']} ({survived}/{len(picks)} survive)")
    log(f"  {label}: kept {summary['kept']} ({summary['kept_pct']}%){tail}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shoot", type=Path, help="the folder to cull")
    parser.add_argument("--selects", type=Path, default=None,
                        help="the photographer's own picks (sidecar or RAW names)")
    parser.add_argument("--preset", default="wedding")
    parser.add_argument("--sweep", action="store_true",
                        help="also print the raw 0..100 strictness curve")
    parser.add_argument("--db", type=Path, default=ROOT / ".calibration" / "catalog.db")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the JSON (default: next to the catalog)")
    args = parser.parse_args()

    if not args.shoot.is_dir():
        parser.error(f"not a folder: {args.shoot}")
    picks = ({p.stem for p in args.selects.iterdir() if p.is_file()}
             if args.selects and args.selects.is_dir() else set())
    out = args.out or args.db.with_name("calibration.json")

    store = Store(args.db)
    library_id = store.create_or_get_library(str(args.shoot))["id"]
    log(f"{args.shoot} · preset {args.preset} · {len(picks)} picks for ground truth")

    results: dict = {"shoot": str(args.shoot), "preset": args.preset,
                     "picks": len(picks), "stops": {}, "sweep": {}}

    log("analysing (cached for the runs below)")
    results["preset_only"] = measure(
        store, library_id, CullingSettings(preset=args.preset), picks,
        f"{args.preset} preset, no stop")

    for stop in strictness_stops():
        settings = CullingSettings(preset=args.preset,
                                   selection_strictness=stop["value"])
        results["stops"][stop["key"]] = measure(
            store, library_id, settings, picks, stop["label"])

    if args.sweep:
        log("sweeping the gate strictness alone")
        for value in range(0, 101, 5):
            settings = CullingSettings(preset=args.preset, selection_strictness=value)
            results["sweep"][str(value)] = measure(store, library_id, settings,
                                                   picks, f"strictness {value}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
