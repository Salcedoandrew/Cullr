"""Export decisions: XMP sidecars, keeper copies, CSV report.

XMP is the interchange format every serious tool reads. Following Aftershoot's
RAW workflow (and photo-pipeline's), originals are never modified: a sidecar is
written next to the file.

* keep  -> ``xmp:Rating="1"`` and ``xmp:Label="Green"``
* reject -> ``xmp:Rating="-1"`` (the XMP "rejected" value) and ``xmp:Label="Red"``
* star overrides the rating when the photographer set one

The sidecar path is ``<filename>.xmp``, which is what Lightroom Classic and
Capture One expect for RAW files. For JPEG/TIFF, an external sidecar is only
read when the application is configured to use one, so ``mode="sidecar_all"``
is offered explicitly.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .config import RAW_EXTENSIONS
from .db import Store

XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Sharpen 0.1">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmp:Rating="{rating}"
    xmp:Label="{label}">
   <xmp:Rating>{rating}</xmp:Rating>
   <xmp:Label>{label}</xmp:Label>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


@dataclass
class ExportResult:
    written: int = 0
    copied: int = 0
    skipped: int = 0
    dest: str | None = None
    errors: list[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def _effective(row: dict) -> str:
    """Normalised decision: "pick" is a human keep, everything else is reject."""
    raw = row.get("user_decision") or row.get("verdict") or "keep"
    return "keep" if raw in ("keep", "pick") else "reject"


def _rating(row: dict) -> tuple[int, str]:
    decision = _effective(row)
    star = int(row.get("star") or 0)
    if decision == "keep":
        return (star if star > 0 else 1), "Green"
    return -1, "Red"


def write_sidecars(store: Store, library_id: int, only_raw: bool = True,
                   include_rejects: bool = True) -> ExportResult:
    result = ExportResult()
    for row in store.photos(library_id, "all", sort="name"):
        path = Path(row["path"])
        if row.get("error") and row.get("quality") is None:
            result.skipped += 1
            continue
        decision = _effective(row)
        if not include_rejects and decision != "keep":
            result.skipped += 1
            continue
        if only_raw and path.suffix.lower() not in RAW_EXTENSIONS:
            result.skipped += 1
            continue
        rating, label = _rating(row)
        sidecar = path.with_suffix(path.suffix + ".xmp")
        try:
            sidecar.write_text(
                XMP_TEMPLATE.format(rating=rating, label=escape(label)),
                encoding="utf-8",
            )
            result.written += 1
        except OSError as exc:
            result.errors.append(f"{sidecar.name}: {exc}")
    return result


def copy_keepers(store: Store, library_id: int, dest: str) -> ExportResult:
    target = Path(dest).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    result = ExportResult(dest=str(target))
    for row in store.photos(library_id, "all", sort="name"):
        if _effective(row) != "keep":
            continue
        src = Path(row["path"])
        if not src.exists():
            result.errors.append(f"missing: {src}")
            continue
        out = target / src.name
        if out.exists():
            out = target / f"{src.stem}_{row['id']}{src.suffix}"
        try:
            shutil.copy2(src, out)
            result.copied += 1
        except OSError as exc:
            result.errors.append(f"{src.name}: {exc}")
    return result


def write_csv(store: Store, library_id: int, dest: str) -> ExportResult:
    target = Path(dest).expanduser()
    if target.is_dir() or target.suffix == "":
        target.mkdir(parents=True, exist_ok=True)
        target = target / "culling-report.csv"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    rows = store.photos(library_id, "all", sort="name")
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "filename", "path", "decision", "decided_by", "reason", "quality",
            "sharpness", "mean_luma", "clip_hi", "clip_lo", "noise", "face_count",
            "eyes_open", "group_id", "is_lead", "star",
        ])
        for r in rows:
            writer.writerow([
                r.get("filename"), r.get("path"), _effective(r),
                "you" if r.get("user_decision") else "ai", r.get("reason") or "",
                f"{r.get('quality') or 0:.4f}", f"{r.get('sharpness') or 0:.1f}",
                f"{r.get('mean_luma') or 0:.4f}", f"{r.get('clip_hi') or 0:.4f}",
                f"{r.get('clip_lo') or 0:.4f}", f"{r.get('noise') or 0:.2f}",
                r.get("face_count") or 0,
                "" if r.get("worst_eyes_open") is None else f"{r['worst_eyes_open']:.2f}",
                r.get("group_id") or "", int(bool(r.get("is_lead"))), r.get("star") or 0,
            ])
    return ExportResult(written=len(rows), dest=str(target))
