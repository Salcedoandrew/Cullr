# Sharpen

Local-first AI photo culling. Point it at a folder, get back the keepers: sharp
frames, open eyes, and one best shot from every burst.

This is a working MVP modelled on [Aftershoot Select](https://aftershoot.com/selects/):
the same shape (ingest → analyse → group → auto-select → review → export), built
from scratch on permissively licensed components. The full research that informs
it — how Aftershoot works, a file-level analysis of four reference repositories,
the licensing map, the architecture and the stack rationale — is in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

```
 ingest ──► technical + face metrics ──► hashes + EXIF burst windows ──► groups
                    │                                                      │
                    └──► face descriptors ──► people ──► coverage ─────────┤
                                                                           │
              review UI ◄── verdicts & reasons ◄── scoring ◄── best-of-group
                  │
                  └──► XMP sidecars · keeper copies · CSV
```

## Get the app

The Mac build is a single `.app` in a drag-to-Applications `.dmg`, published on
the [releases page](../../releases). Download
`Sharpen-<version>-arm64.dmg`, open it, and drag **Sharpen** into Applications.

It is Apple-silicon only and signed ad-hoc, not notarised: the first launch needs
right-click → **Open** instead of a double-click. Everything — the UI, the face
model, the Python runtime — is inside the bundle, so there is nothing else to
install. Catalog, preview cache and logs live in
`~/Library/Application Support/Sharpen`, and the app serves itself on a loopback
port that closes with the window.

## What it does

* **Sharpness / blur** — Laplacian variance on a normalised 1600 px frame,
  judged three ways: a frame-wide floor, an absolute *face-crop* gate (the eyes
  must be sharp), and a relative gate inside each burst. Frame-wide Laplacian
  alone is not enough: it swings about 6× between a clean portrait and a detailed
  landscape, and drops sharply when the background is intentionally out of focus.
* **Exposure** — clipping and mean luminance, measured on the subject's face
  when one is present. A bright studio backdrop or blown sky no longer rejects a
  well-exposed portrait.
* **Closed eyes / blinks** — MediaPipe Face Landmarker blendshapes, mapped to a
  0–10 eyes-open score (`(1 − max(eyeBlinkL, eyeBlinkR)) × 10`); the 4.0 cutoff is
  the same one facet measured on real catalogues. Head pose gates the decision so
  a profile shot is never judged on unreliable eye geometry.
* **Duplicates and bursts** — pHash + dHash with Hamming gates, plus EXIF
  capture-time windows, merged with union-find. Near-duplicates are also found
  across the whole shoot (a re-import with a different timestamp), and groups are
  chunked so single-linkage cannot chain a whole ceremony into one group.
* **Best of group** — quality with *bounded* tie-breaks for open eyes, face
  sharpness and expression, so a blink fixes a near-tie but never rescues a bad
  frame. Every group keeps at least one frame, and nothing is auto-rejected
  without a stated reason.
* **People** — faces are described from the aligned crop and clustered into
  people across the folder. Anyone photographed twice gets a keeper even when all
  of their frames lost to a duplicate or a soft frame: the best one is promoted
  and the sheet says who it was kept for ("Kept for Person 2 · soft focus").
* **Review** — keyboard-first contact sheet grouped by burst, loupe with face
  boxes and every measured value, per-frame overrides.
* **Export** — XMP sidecars with `xmp:Rating`/`xmp:Label` (originals untouched),
  a keeper copy, or a CSV report. Destinations are chosen in a folder browser
  (`/api/folders`) that can create the delivery folder, and each shoot remembers
  the mode and folder of its last export (stored in the catalog, so it follows
  the shoot rather than the browser).

## Quick start

Requires Python 3.10–3.12 and Node 18+.

```bash
# 1. Python environment
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e "backend[dev]"
uv pip install --python .venv/bin/python -e "backend[faces]"   # face/blink detection

# 2. Face model (Apache-2.0, ~3.7 MB, checksum-pinned)
.venv/bin/python -m culler.cli fetch-model

# 3. Interface
cd frontend && npm install && npm run build && cd ..

# 4. Run
.venv/bin/python -m culler.cli serve --port 8000
```

Open <http://127.0.0.1:8000>, paste a folder path, and press **Open folder**.

Without the `[faces]` extra everything still works except closed-eye detection:
the UI says so and the rest of the pipeline (blur, exposure, duplicates, groups)
is unchanged.

### Front-end development

```bash
.venv/bin/python -m culler.cli serve --port 8000     # terminal 1
cd frontend && npm run dev                           # terminal 2 → :5173 (proxied)
```

### The Mac app from source

```bash
uv pip install --python .venv/bin/python -e "backend[desktop]"
.venv/bin/python desktop/sharpen_desktop.py          # native window over the local server
```

`desktop/build_macos.sh` does the whole package: builds the UI, draws the icon
(`desktop/make_icon.py`), freezes `Sharpen.app` with PyInstaller, smoke-tests the
frozen service (it must answer `/api/capabilities` with faces enabled), signs it
ad-hoc and writes `dist/Sharpen-<version>-arm64.dmg`. A `v*` tag runs the same
script on a macOS runner (see [`.github/workflows/release.yml`](.github/workflows/release.yml))
and attaches the `.dmg` plus `SHA256SUMS` to the release.

### Headless

```bash
.venv/bin/python -m culler.cli scan ~/Pictures/shoot --preset wedding --strictness 40
.venv/bin/python -m culler.cli export ~/Pictures/shoot --mode sidecar
.venv/bin/python -m culler.cli export ~/Pictures/shoot --mode csv --dest ~/Desktop/cull.csv
```

`scan` prints the counts per rejection reason and is idempotent: unchanged files
are reused, and human decisions survive a rescan. `--force` re-analyses.

## How a decision is made

**Composite score** — a weighted mean over the axes that actually apply
(`sharpness`, `exposure`, `noise`, `contrast`, `eyes_open`, `face_sharpness`,
`subject`). Missing axes are dropped from numerator *and* denominator, so a
landscape is not punished for having no eyes. Presets reweight the axes and set
the gates:

| Preset | Strictness | Face-blur gate | Closed eyes at | Keeps per group |
| --- | --- | --- | --- | --- |
| Portrait | 50 | 55 | 4.0 / 10 | 1 |
| Wedding / event | 35 | 45 | 5.0 / 10 | 1 |
| Event / documentary | 50 | 55 | 4.0 / 10 | 1 |
| Sports / action | 85 | 85 | 2.0 / 10 | 1 |
| Landscape / travel | 60 | — | — | 1 |

Raising strictness raises the face-blur gate, tightens the relative-blur ratio and
lowers the eyes-open cutoff. Every rejection stores a reason (`blurry`,
`overexposed`, `underexposed`, `closed_eyes`, `duplicate`, `unreadable`) that the
UI shows — the point of auto-culling is that you can audit it.

**Groups** — a pair is a candidate when either (a) pHash is within 6 bits (≈90 %
similar, facet's default), or (b) the capture times are within 2 s and hashes are
loosely similar, or (c) they are within 8 s and hashes agree. Candidates are
merged with union-find, then chunked at 24 frames or a bounded time span. An
optional embedding cosine gate (≥ 0.90) confirms candidates when an embedder is
installed; without one the hash gate stands alone.

**People** — every detected face above 40 px becomes a 117-float descriptor: a
similarity transform onto the five-point ArcFace template, then an 8×8 DCT block
(structure) plus a 3×3 grid of gradient-orientation histograms (texture), L2
normalised. Descriptors are clustered greedily at cosine ≥ 0.90 with a merge pass
at ≥ 0.92. Two faces in the *same* frame can never be the same person, so that
is a hard constraint: no frame-mate ever shares a person. A person is *required*
when they appear in two or more photos with a face covering ≥ 1.5 % of the frame;
if no keeper contains them, the best frame that does is promoted. Promotion never
demotes anything, and a person whose only frames are flawed still gets one,
flagged `forced` so the reason stays visible.

Measured on real portraits: same person 0.94–1.00 (1.5 stops underexposed: 0.94),
different people 0.65–0.88. The gate is above the overlap on purpose — a false
merge can hide someone from the guarantee, while a false split only costs one
extra keeper. Turn it off with **Keep a frame of everyone** in Culling settings.

The split therefore happens on purpose instead of being hidden by a looser
gate, and the People strip turns it into a decision: any two people at cosine
≥ 0.84 whose frames never overlap are offered as *same person* (fold them
together) or *two people* (stop suggesting it). Both answers are stored as face
ids and re-applied on every later scan, so a rescan never splits a person you
have already curated.

## Verified behaviour

`cd backend && ../.venv/bin/python -m pytest` — **108 tests**, covering the
metrics, gates, grouping, selection, identity clustering, coverage and the
confirmable-merge rules, export formats, the HTTP API, and full
scan-then-review-then-export runs on generated shoots.

Beyond the unit tests, the pipeline was exercised on real photographs:

| Check | Result |
| --- | --- |
| Open-eye studio portrait | eyes-open 9.6 / 10, kept |
| Sleeping newborn | eyes-open 2.2 / 10 → rejected as *eyes closed* (UI shows a red face box) |
| 4-frame burst (soft, sharp, shifted, dark) | 1 group; sharp frame is the lead, shifted frame is a *duplicate*, soft and dark frames are *blurry* |
| Same photo imported twice, one hour apart | grouped and deduplicated (global hash pass, not just the time window) |
| White studio backdrop, 38 % of frame clipped | not flagged as overexposed; the subject's face decides |
| 48-frame burst | split into 2 groups of 24 instead of one chain |
| Same face across 5 frames (one 1.5 stops dark) | one person at similarity 0.94–0.99 |
| Two different men, studio portraits | separated: 0.65–0.81 apart |
| 4-frame burst, one guest only in the soft frames | 2 people clustered; the soft frame promoted → *Kept for Person 2 · soft focus*, 0 people uncovered |
| 9-frame real shoot (three people) | 3 people, 0 uncovered, no false merge suggested (closest different-people pair 0.80, below the 0.84 proposal floor) |
| Same shoot with the auto-join gate raised to 0.97 | the subject splits into two people (0.952) and the strip offers the merge; confirming it in the UI folds them back to 4 frames, and a rescan at the same gate keeps them together |
| "Two people" on the split | suggestion dismissed, still absent after a rescan at the same gate |

Throughput on an M-series laptop, CPU only: **~4–8 photos/second** with face
analysis (8 threads), so a 1,000-frame shoot analyses in a few minutes.

## Repository layout

```
backend/culler/
  config.py        presets, weights, gates, analysis sizes
  imaging.py       decode (RAW/HEIC optional), EXIF, previews, pHash/dHash
  metrics/         technical.py (sharpness, exposure, noise), faces.py (MediaPipe),
                   faceid.py (alignment + identity descriptor)
  similarity.py    hash + capture-time candidates, union-find, chunking
  persons.py       identity clustering + keeper-coverage guarantee
  scoring.py       axis composite, subject exposure, rejection reasons
  selection.py     bounded best-of-group tie-breaks
  pipeline.py      scan orchestration (thread pool, resumable, cancellable)
  db.py            SQLite catalog
  api.py           FastAPI service
  export.py        XMP sidecars, copies, CSV
  cli.py           scan / serve / export / fetch-model
frontend/src/      React review UI (contact sheet, loupe, filters, export)
desktop/           native entry point (WKWebView window), PyInstaller spec,
                   icon generator, .app + .dmg build script
.github/           tag-driven macOS release workflow
docs/RESEARCH.md   Aftershoot analysis, repo analysis, licensing, architecture
NOTICE.md          attribution and model licences
```

## Not in this version

RAW files decode through `rawpy` only if you install `backend[raw]`; otherwise
the embedded JPEG preview is used when present. Semantic embeddings
(DINOv2/CLIP), a *learned* face embedder (the permissive options were measured
and deferred — see [`docs/RESEARCH.md`](docs/RESEARCH.md) §4.6), naming people,
per-person albums, learned per-photographer ranking, and Lightroom catalogue
round-trip are deliberately out of scope for the MVP. The desktop bundle is
Apple silicon only (MediaPipe ships no universal2 wheel) and unsigned, so there
is no notarised download or auto-update yet.

## Licence notes

The reference repositories are MIT except `AI-POWERED-PHOTO-CULLING`, which has
**no licence** and was therefore not reused. The face model is Apache-2.0
(MediaPipe); InsightFace's `buffalo_l` and PyIQA are non-commercial and are
deliberately absent. Details and the full model table are in
[`NOTICE.md`](NOTICE.md).
