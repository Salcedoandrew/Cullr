# AI Photo Culling — Research, Architecture & Stack

This document is the design record for **Cullr** (working name), an AI photo-culling
application modelled on [Aftershoot](https://aftershoot.com/selects/). It covers:

1. How Aftershoot actually works (what is publicly documented).
2. A file-level analysis of the four reference repositories.
3. What is reusable, what must be rebuilt, and the licensing constraints.
4. The architecture of this app and the stack it uses.

The MVP described in the last section is implemented in this repository
(`backend/` + `frontend/`).

---

## 1. How Aftershoot works

Aftershoot does not publish model weights, thresholds or its scoring formula. What is
documented is the *shape* of the product: signals, tunables, workflow, and where
processing happens.

### 1.1 Signals

| Signal | Publicly documented behaviour |
| --- | --- |
| Sharpness / focus | Flags soft frames. Exposes a **blur strictness** control (Lenient / Moderate / Strict) rather than a numeric threshold. For sports it accepts body-in-focus instead of face-in-focus. |
| Exposure | Flags "blown" exposures; exposure is listed among the analysed factors. No numeric cutoff is published. |
| Closed eyes / blinks | Detected and used as a review category. Genre-dependent: off by default for sports, recommended off for newborn work. |
| Expressions / key faces | "Key face" assessment and expression scoring; per-genre policies (wedding weights key faces and emotion). |
| Duplicates / similar | Groups near-identical frames. A separate **duplicate strictness** control changes how aggressively frames are merged. Guarantees **at least one keeper per duplicate group**. |
| Best of burst | Auto-selects a preferred frame inside a group and lets the user swap it. Sports mode groups bursts but the wedding profile is explicitly not tuned for fast bursts. |

The product markets "30+ factors", but only these are operationally visible. The
important lesson for an implementation: **the tunables are consumer-facing
strictness levels, not thresholds.** Internally they map onto a small set of
numeric gates.

### 1.2 Workflow

1. Create an album / ingest from a folder or card, with a working (and optional
   backup) directory.
2. Pick a mode: **AI-automated** (proposes keep/reject) or **AI-assisted**
   (scores and organises, human decides), plus a shoot genre and strictness.
3. Review: grid + loupe, category filters (selected / rejected / blur / closed
   eyes / duplicates), side-by-side comparison ("Survey Mode", up to 16 frames).
   Shortcuts documented are `X` exclude, `S` swap the preferred frame, `A` add,
   `Space` zoom, arrows to navigate, `Ctrl/Cmd+S` save.
4. Export: push selections to Lightroom Classic / Capture One, or copy to a
   folder. Ratings and colour labels travel with the files; for RAW workflows
   Aftershoot writes **XMP sidecars** instead of touching originals.

### 1.3 Where it runs

Culling runs **locally** (Windows/macOS), offline, with 8 GB RAM cited as the
minimum. Competitors differ: Imagen is cloud-based; FilterPixel sends *previews*
to the cloud while RAWs stay local. That split is the central product decision
for a clone — this project follows Aftershoot's local-first model, because it is
what makes a desktop culling tool viable for 3–10k image weddings without
per-image cost.

### 1.4 What competitors reveal about the hard part

The concrete signals (blur, exposure, eyes, near-duplicates) are tractable with
classical CV. The genuinely hard part is **choosing the emotionally/editorially
best frame** — FilterPixel markets genre-aware "moment" scoring, Narrative
exposes evidence and ranks within a scene, and independent commentary on
Aftershoot focuses on grouping and emotional nuance. Any architecture should
therefore separate:

* **objective/rejectable** signals (technical quality, blink) — deterministic,
  explainable, testable; and
* **subjective/rankable** signals (expression, moment, aesthetics) — pluggable,
  swappable, optionally learned per photographer.

This project is built so the first class is complete and reliable, and the
second class is a clearly-bounded extension point.

---

## 2. Reference repository analysis

All four repositories were cloned and read. Sizes below are working trees.

### 2.1 `viktordrukker/photo-pipeline`

**License:** MIT (code). **Size:** ~8.7 MB. **Stack:** Python 3, FastAPI + Jinja +
vanilla JS front end, Redis, MinIO/S3, Docker Compose, PyTorch worker.

Structure: `app/` (web app + `app/pipeline/core.py`, a 1,100-line legacy pipeline),
`worker/` (GPU inference + `worker/pipeline/intelligent.py`, a 2,400-line DAG with
16 stages), `shared/schemas.py`, pluggable storage backends (S3/GCS/Azure/local)
and a hook-plugin system.

Algorithms worth studying:

* **Technical screen** — `_screen_one()` computes `cv2.Laplacian(...).var()` and
  exposure stats, then assigns a verdict: `blurry` if `sharpness < blur_threshold`
  (default **80**, documented as 30 lenient / 100 strict / 250 very strict), then
  `overexposed` / `underexposed` from clipped-pixel percentages and mean
  luminance (defaults: `overexposed_pct=0.30`, `overexposed_mean=200`,
  `underexposed_pct=0.45`, `underexposed_mean=50`).
* **Deduplication** — `imagededup` pHash with `phash_max_distance=6` (≈91 %
  similarity), optional **multi-hash voting** (pHash + dHash + wHash, merge only
  on ≥ 2-of-3 agreement to kill false positives on flat textures), EXIF burst
  pre-clustering within `burst_window_seconds=3.0`, all merged with
  **union-find**.
* **Semantic dedup** — DINOv2-small (384-d, L2-normalised) cosine, default
  `smart_dedup_threshold=0.92` ("same scene"; 0.95 near-identical, 0.88 same
  event). Explicitly motivated by the fact that pHash misses *hand pose changes,
  blinks and micro-expressions*.
* **Scoring** — a 10-axis composite (`sharpness`, `eyes_open`, `subject_attention`,
  `composition`, `technical_quality`, `creativity`, `subject_coverage`, …) with
  **`None` axes dropped from numerator *and* denominator**, then a weighted mean.
  Plus multiplicative penalties: back-of-head (`frontal_score < 0.20 → ×0.50`),
  no-face (`face_count_floor_penalty`), and genre presets.
* **Face quality** — CR-FIQA (iResNet-100 + regression head) when the weights are
  present, otherwise the L2 norm of the ArcFace embedding, min-max normalised
  per job.
* **Coverage guarantee** — every face cluster gets at least one winner, and
  `apricot-select` facility-location picks a diverse winner set when a cap is hit.

Weaknesses: very heavy footprint (4 containers, GPU, MinIO, Redis), the logic is
split across a legacy pipeline and an "intelligent" one, and several modules are
documented as placeholders (`expression`, `scene_fit`, `moment_peak` return
`None`).

### 2.2 `ncoevoet/facet`

**License:** MIT (code), with an explicit **model-licence registry and
`FACET_LICENSING_MODE=commercial` gate** (`licensing.py`, `requirements.restricted.txt`).
**Size:** ~59 MB. **Stack:** Python, FastAPI, SQLite, Angular viewer, Lightroom
plugin. This is the most mature of the four and the best source of tuned
defaults.

Relevant modules:

* `analyzers/technical.py` — Laplacian sharpness normalised as
  `min(10, variance / 50)`; **ISO-compensated** sharpness
  (`var × (1 + 0.15·log2(ISO/100))`); histogram-based exposure score combining
  luminance penalty, spread bonus, bimodality penalty and clipping penalty;
  shadow/highlight clipping flags; Immerkaer noise kernel; monochrome detection.
* `analyzers/face.py` — EAR blink detection on InsightFace's 106-point mesh
  (explicit eye indices), `EAR_CLOSED=0.12`, `EAR_OPEN=0.28` mapped to a 0–10
  `eyes_open_score`, `EYES_CLOSED_MAX = 4.0`, and a **head-pose gate**
  (`|yaw|/|pitch| > 35°` → don't judge blinks) plus a landmark-asymmetry fallback.
* `analyzers/face_blendshapes.py` — MediaPipe Face Landmarker blendshapes replace
  the geometric score: `eyes_open = (1 − max(eyeBlinkLeft, eyeBlinkRight)) · 10`.
  This is the appearance-based approach that survives off-axis faces and subtle
  blinks, and degrades silently when the model bundle is absent.
* `utils/duplicate.py` — **two-stage duplicate gate**: a loose pHash Hamming
  prefilter (≤ 12 bits) *confirmed* by an embedding cosine gate (≥ 0.90) before
  union-find merges. Includes a precision/recall sweep helper for tuning.
  Similarity 90 % ⇒ `max_distance = floor(64 × 0.10) = 6`.
* `utils/selection.py` — `composite_lead_score`: aggregate dominates, with
  **bounded tie-breaks** for eyes (±0.15), expression (±0.075), sharpness
  (±0.05), learned ranker (±0.05), all centred on a neutral 5.0. The comment is
  the design rationale: *a meaningful aggregate gap is never overridden; the
  tie-break only decides near-ties such as a mid-blink frame.*
* `processing/burst_score.py` — fixed linear blend
  `0.40·aggregate + 0.25·aesthetic + 0.20·sharpness + 0.15·blink` (+ optional
  eyes/expression), shared by the auto-cull API and the learned keeper baseline.
* `config/scoring_config.default.json` — the tuned cull profiles. Notably:
  `duplicate_detection.similarity_threshold_percent=90`,
  `burst_detection.{similarity_threshold_percent=70, time_window_minutes=0.8,
  rapid_burst_seconds=0.4}`, `similarity_groups.default_threshold=0.85`,
  and per-genre profiles (`balanced` strictness 50 / `wedding` 35 /
  `sports` 85 / `concert` 55 / `wildlife` 70) with `eyes_closed_max`,
  `keep_min_per_group` and `similarity_threshold`.
* `models/saliency_scorer.py` — subject-vs-background Laplacian variance, i.e.
  *where* the frame is sharp, not just whether it is sharp. This is the piece
  that fixes the classic failure of naive blur detection on shallow depth of
  field.

Weaknesses: enormous surface area (dozens of optional models, 7-language docs),
optimised for a long-running catalogue rather than a single shoot, and its
best face/quality stacks are exactly the non-commercial ones it gates off.

### 2.3 `sundhars2823/AI-POWERED-PHOTO-CULLING`

**License: none.** No `LICENSE` file in the repository → default copyright, all
rights reserved. **Size:** 218 MB (mostly a bundled `data/` image dump).
**Stack:** Python, Streamlit (`app.py`), OpenCV, `imagehash`, CLIP.

The README claims a lot (bride/groom priority, scene classification, story-based
selection, learned ranking). The implementation is ~450 lines total:

* `src/quality_checker.py` (17 lines) — `cv2.Laplacian(...).var() < 100 → blurry`.
* `src/duplicate_remover.py` (40 lines) — O(n²) `imagehash.phash` with
  `abs(h1 - h2) <= 8`, copying "unique" files to an output folder.
* `src/pipeline.py` — a sequential orchestration script.
* `src/features/aesthetic_scorer.py`, `face_detector.py`, `scene_classifier.py`,
  `story_classifier.py` — thin wrappers; `src/features/image_clustering.py`,
  `src/features/category_sorter.py`, `src/ranking/{train,inference}.py` are
  **empty files**.

Verdict: a student/demo project, not a reference implementation. Only useful as a
baseline for thresholds (`Laplacian < 100`, `phash distance ≤ 8`) and as a
cautionary example of README-vs-code drift. **Do not copy** — no licence.

### 2.4 `joemalott/retoucan`

**License:** MIT. **Size:** 2.2 MB. **Stack:** Tauri (Rust) + Svelte + TypeScript,
plus a Lightroom Classic Lua plugin.

This is an **editing** tool, not a culling tool — it starts where culling ends. It
is still the best reference for the integration layer:

* `src-tauri/src/features.rs` — a fixed 46-dimension per-photo feature vector
  (luminance percentiles, clipping, 8-bin histogram, colour ratios, saturation,
  skin fraction, centre/edge luminance, `log_iso`, `log_shutter`, f-number, flash,
  `ev100`, aspect, focal length, white-balance ratios) computed from a ~256 px
  preview. A tidy example of cheap, interpretable features for a learned model.
* `src-tauri/src/catalog.rs` — reads a Lightroom catalogue's SQLite
  (`AgLibraryFile` / `Adobe_imageDevelopSettings`) for pick flags, ratings and
  labels. This is how a desktop app round-trips decisions without an SDK.
* `lightroom/Retoucan.lrplugin/*.lua` — the plugin bridge (socket, apply, undo).
* `src-tauri/src/raw.rs` — RAW decoding and shot info extraction in Rust.

Verdict: not reusable as a culling engine, but its catalogue/XMP round-trip and
its feature-vector discipline are directly applicable to the export layer.

---

## 3. Reusability and licensing

### 3.1 Licence map

| Repository | Code licence | Model/weight licences that matter | Can we copy code? |
| --- | --- | --- | --- |
| photo-pipeline | MIT | `buffalo_l` / InsightFace **non-commercial**; CR-FIQA **CC-BY-NC**; Real-ESRGAN BSD-3; DINOv2 Apache-2.0; CLIP MIT | Yes, with attribution. Avoid its restricted models. |
| facet | MIT | InsightFace non-commercial; PyIQA non-commercial; SAMP-Net/U2-Net-P + LAION aesthetic head "unclear" | Yes, with attribution. Its licensing gate is itself a pattern to copy. |
| AI-POWERED-PHOTO-CULLING | **none** | n/a | **No.** All rights reserved. Read for ideas only. |
| retoucan | MIT | n/a (no shipped models) | Yes, with attribution. |

### 3.2 Practical constraints for this project

1. **Do not import `insightface` / `buffalo_l` or `pyiqa` by default.** Both are
   non-commercial, and both reference projects ship them as the *default* path.
   A product that may be sold cannot do that.
2. **MediaPipe Face Landmarker is the permissive substitute.** Apache-2.0 model
   (pinned `float16/1` bundle, SHA-256 verified in facet), gives 478 landmarks
   *and* the `eyeBlink*` / `mouthSmile*` blendshapes that facet already proved
   better than landmark geometry. This is what the MVP uses; no InsightFace.
3. **DINOv2-small (Apache-2.0) is the permissive semantic embedding**, if/when
   semantic dedup beyond pHash is enabled. CLIP (MIT) is the alternative; note
   that its *training data* licence is unverified — facet's own docs make the
   same disclaimer, and it is inherited by anything using CLIP features.
4. **Keep the licensing gate.** Adopt facet's idea: a registry of components with
   a `commercial` flag and a mode that refuses restricted components at startup,
   so a future contributor cannot silently reintroduce a non-commercial model.
5. MIT requires preserving copyright + permission notices for any copied
   portion. This project re-implements the algorithms in new code and credits
   the design in `NOTICE.md`; if you copy literal code later, keep headers.

---

## 4. Architecture

### 4.1 Shape

Local-first desktop-class service: one Python process owns the pipeline and the
HTTP API; a React SPA is the review surface. Nothing leaves the machine, and the
core path needs no GPU and no network.

```
┌──────────────────────────────────────────────────────────────────┐
│  Review UI (React + Vite)          keyboard-first grid / groups  │
└───────────────▲──────────────────────────────────────────────────┘
                │ REST + SSE (127.0.0.1)
┌───────────────┴──────────────────────────────────────────────────┐
│  FastAPI service                                                 │
│   · job runner (ThreadPool, cancellable, resumable, progress)     │
│   · preview server (decoded JPEG cache, path-safe)               │
└───────────────▲──────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────┐
│  Analysis pipeline                                               │
│                                                                  │
│  1 ingest ─────────► list images, stat, content key              │
│  2 decode ─────────► EXIF (time/ISO/shutter/f), analysis frame    │
│                      (long edge 1600), thumbnail (512)            │
│  3 technical ──────► Laplacian/Tenengrad sharpness, exposure,     │
│                      clipping, noise, contrast                    │
│  4 faces ──────────► MediaPipe landmarks + blendshapes:           │
│                      eyes-open, smile, yaw/pitch, per-face sharp  │
│  5 similarity ─────► pHash + dHash, EXIF burst windows,           │
│                      optional embedding cosine                    │
│  6 group ──────────► union-find → burst / similar groups          │
│  7 score ──────────► per-axis 0..1 signals → weighted composite   │
│                      (missing axes dropped, not zeroed)           │
│  8 decide ─────────► reject rules (blur/exposure/blink/dupe)      │
│  9 lead ───────────► per-group keeper + bounded tie-breaks        │
└───────────────▲──────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────┐
│  SQLite catalog + on-disk preview/thumb cache                    │
└───────────────▲──────────────────────────────────────────────────┘
                │
┌───────────────┴──────────────────────────────────────────────────┐
│  Export: XMP sidecars (rating/label) · copy keepers · CSV        │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Why these choices

| Decision | Rationale (from the references) |
| --- | --- |
| Analyse a fixed-size preview (1600 px) | Laplacian variance is resolution-dependent. A fixed analysis size makes thresholds comparable across a 12 MP and a 61 MP body. photo-pipeline analyses previews; retoucan computes features from ~256 px. |
| Threads, not processes, for metrics | OpenCV/numpy release the GIL for these ops; photo-pipeline's `technical_screen` uses a ThreadPool for exactly this reason. MediaPipe's landmarker is stateful → guarded by a lock. |
| Absolute **and** group-relative blur gates | Absolute catches genuinely soft frames; relative-to-group catches "this is the soft one in the burst". Aftershoot exposes strictness rather than a threshold; facet's low-light/shallow-DOF problem is solved by comparing within a group. |
| pHash first, embeddings optional | pHash needs no model and no network; it is the recall stage. facet's two-stage gate (loose Hamming → cosine confirmation) is the pattern, with the second stage only active when an embedder is installed. |
| Union-find over groups | Both photo-pipeline and facet use it; it makes similarity transitive without O(n²) group bookkeeping. Distance matrix is still O(n²) but vectorised over uint64 with popcount. |
| Burst windows from EXIF | `captured_at` deltas catch bursts whose content differs too much for pHash (movement, expression). facet: 0.8 min window / 0.4 s rapid burst; photo-pipeline: 3 s. Configurable; default 2 s for rapid bursts, 4 s for "similar scene". |
| Weighted composite with `None`-dropping | Copied conceptually from photo-pipeline's `axis_composite_score`: a landscape must not be punished for having no eyes. |
| Bounded tie-breaks when picking the lead | facet's `composite_lead_score` — a real quality gap must never be overturned by a blink bonus, but two near-identical frames should be separated by eyes/sharpness. |
| SQLite catalog | Single-file, zero-ops, instantaneous filtering over 10k rows; facet's schema is the model. |
| XMP sidecars | Non-destructive, and what Aftershoot does for RAW. Ratings + labels are what Lightroom/Capture One read. |

### 4.3 Data model (SQLite)

```sql
libraries(id, root_path, name, created_at, settings_json, export_mode, export_dest)
jobs(id, library_id, status, stage, total, done, error, started_at, finished_at)
photos(id, library_id, path UNIQUE, filename, size_bytes, mtime, content_key,
       width, height, captured_at, camera, iso, shutter, aperture, focal_len,
       sharpness, sharpness_norm, exposure, clip_hi, clip_lo, noise, contrast,
       face_count, eyes_open, blink, smile, face_sharpness, frontal,
       phash, dhash, quality, verdict, reason, user_decision, star,
       group_id, is_lead, analysis_version)
groups(id, library_id, kind, size, lead_photo_id, lead_score)
persons(id, library_id, label, face_count, photo_count, keep_count, covered,
        representative_photo_id, representative_face_id, max_area_ratio)
faces(id, photo_id, bbox_x, bbox_y, bbox_w, bbox_h, eyes_open, smile,
      yaw, pitch, sharpness, embedding BLOB, person_id)
```

`photos.covered_person` records the person a frame was promoted for, so the UI
can explain a keeper that would otherwise look like a duplicate.

`verdict` is the machine decision (`keep` / `reject`), `user_decision` is the
human override (`pick` / `reject` / null) and always wins. `reason` is the
explainable reject cause (`blurry`, `overexposed`, `underexposed`, `blink`,
`closed_eyes`, `duplicate`), which the UI shows as a badge — this is the
"why", which is what makes auto-culling trustworthy.

### 4.4 Scoring model

Each axis is normalised to `0..1` (or `None` when inapplicable):

| Axis | Source | Notes |
| --- | --- | --- |
| `sharpness` | Laplacian variance (log-squashed) after a 0.8 px Gaussian pre-blur, so sensor noise does not inflate it | Absolute floor, face-crop gate, then compared within its group |
| `exposure` | clipping percentages, mean luminance, histogram spread | Silhouette/low-key exemption at the extremes |
| `noise` | Immerkaer estimator | Inverted |
| `contrast` | RMS contrast | |
| `eyes_open` | MediaPipe `1 − max(blinkL, blinkR)` over the primary face (and worst face) | `None` without faces |
| `face_sharpness` | Laplacian on face crops | `None` without faces |
| `subject` | face area fraction + frontal pose | `None` without faces |

Composite = weighted mean over **populated** axes (weights renormalised). Genre
profiles reweight and adjust gates; `strictness` (0–100) scales the blur and
eyes gates:

```
frame_floor       = 5.0 Laplacian variance       # mush, content-independent
face_blur_gate    = preset 45..85, scaled by strictness   # on the 192px face crop
relative_gate     = 0.30 .. 0.65 of group median  # "soft for this group"
eyes_closed_max   = 5.0 .. 2.0 (0–10 scale)       # facet: balanced 4.0
keep_min_per_group = 1 (user-raisable)
```

The face-crop gate carries the absolute decision because measurement showed the
frame-wide Laplacian variance of a real studio portrait swings from 62 to 10 with
a single pixel of Gaussian blur, while its face crop falls 510 → 136 → 33 for
1 → 2 px blur. The face crop is also content-normalised (a 192 px resample), so
one threshold holds across cameras. Photos without faces rely on the group
comparison plus the frame floor.

### 4.5 Best-of-group

```
lead_score = quality
           + 0.03 · (eyes_open − 5)        # ±0.15 on the 0–10 scale
           + 0.01 · (face_sharpness − 5)   # ±0.05
           + 0.02 · (smile − 5)            # ±0.10
```

A mid-blink frame (eyes ≈ 1) therefore loses ≈ 0.12 to an open-eyes frame
(eyes ≈ 9) of equal quality: enough to decide a tie, never enough to rescue a
bad frame. The lead is always the best *acceptable* frame, and ties go to the
earliest capture. Groups always keep ≥ `keep_min_per_group`
frames, so "auto-cull" can never empty a moment.

### 4.5b Cull to target

A second question sits next to the gates: not "is this frame acceptable" but
"how many frames do I want". Aftershoot exposes exactly this as a target
percentage, and it is the one place where a per-group guarantee and a global
count are in direct tension.

```
limit   = round(target_ratio · analysed_frames)
keepers = rank(all keepers by composite quality) [ : limit ]  ∪  coverage_keepers
```

Ranking is by measured quality (ties by photo id, so a rescan is deterministic),
and losers are rejected with reason `target` instead of vanishing — the sheet
still explains itself. Two deliberate exceptions:

* **Coverage is never cut.** A frame that is someone's only keeper survives the
  cap, so the achieved count can exceed the target by a few. Silently losing a
guest to hit a number is the one outcome the people pass exists to prevent.
* **A group can end up with no keeper.** The per-group minimum is a property of
  the gates, not of the target; below the target count the cap wins, which is
  what a tighter edit means. `is_lead` stays on the group's best frame so the
  comparison is still visible.

### 4.6 People: identity clustering and the coverage guarantee

For weddings and events the flagship behaviour is not "best frame" but "nobody
disappears": a guest, a parent or the second half of the couple must not be
silently culled because a near-identical frame without them won the group. Both
mature references have a version of this (facet's key-face handling,
photo-pipeline's facility-location pass over detected people); Aftershoot exposes
it as part of its key-face selection. This build implements it directly:

1. **Descriptor** (`metrics/faceid.py`). A similarity transform fitted to the
   five-point ArcFace template normalises scale, rotation and eye position. On the
   aligned 112 px crop, an 8×8 DCT block captures structure and a 3×3 grid of
   gradient-orientation histograms captures texture; the two are concatenated and
   L2-normalised (117 floats, ~470 B per face). No training, no download — the
   landmarks already come from MediaPipe.
2. **Clustering** (`persons.py`). Greedy centroid matching at cosine ≥ 0.90, a
   merge pass at centroid cosine ≥ 0.92 (removes most order dependence), then
   stable 1-based numbering by first appearance. Cost is O(faces × people).
   Order independence is not the only reason for the pass: the same-photo
   constraint below is enforced in both stages.
3. **Cannot-link within a frame.** Two faces detected in one photo are provably
   two people, so no frame-mate may ever join the same person. This is checked
   both when a face joins an existing centroid and when two finished clusters
   would merge. It matters in group shots, where two similar-looking guests at
   the same table would otherwise collapse into one identity and the guarantee
   would count one of them as covered by a frame that only contains the other.
4. **Confirmable merges.** Because the auto-join gate is deliberately strict, the
   interesting cases are the ones it refused: two people at centroid cosine
   ≥ `person_propose` (0.84) whose photo sets do not intersect. `persons.py`
   `propose_merges` lists them; the People strip offers *Same person* / *Two
   people*; `store.merge_persons` folds a confirmed group and `mark_persons_distinct`
   records a dismissal, both as **face ids**, and the next scan re-applies them
   (`apply_merge_groups`, and a proposal filter respectively). The face ids are
   what makes this survive a rescan: they are stable across analysis and even a
   partial match still links the clusters. The floor sits below the auto-join
   gate simply because a wrong proposal costs the photographer one click, while a
   person they never see again is unrecoverable.
5. **Guarantee.** A person is *required* when they appear in ≥ 2 distinct photos
   and at least one of their faces covers ≥ 1.5 % of the frame. If no keeper
   contains them, the best frame that does (same bounded tie-break ranking as
   best-of-group) is promoted to keeper. Promotion only ever *adds* a keeper. If
   every frame of theirs is technically flawed, the best one is still promoted
   and marked `forced`, so the sheet says "Kept for Person 2 · soft focus"
rather than pretending the frame is clean.

Measured separation (labelled studio portraits plus a 12-portrait gallery):

| Pair type | Cosine similarity |
| --- | --- |
| Same person, same light | 0.94 – 1.00 |
| Same person, 1.5 stops underexposed | 0.94 |
| Same person, different crop / pose | 0.94 – 0.98 |
| Different people (typical) | 0.65 – 0.81 |
| Different people (closest pair) | 0.88 |

The tails nearly touch, so the threshold is deliberately above the overlap. The
costs are asymmetric: a false merge can silently defeat the guarantee (one merged
cluster counts as covered by a frame that contains only one of the two people),
while a false split costs exactly one extra keeper. That is why the gate sits at
0.90 rather than at the midpoint of the two distributions.

Behaviour on the 9-frame real shoot (three people, 7 described faces): clustering
puts the groom in 4 frames, the newborn in 2, a second man in 1, and proposes
nothing — the closest different-people pair measures 0.798, i.e. 0.042 under the
proposal floor. Forcing the failure the feature exists to repair, by raising the
auto-join gate to 0.97, splits the subject's fourth frame (A7001004) into its own
person at cosine 0.952: the split person then has one frame, so the guarantee no
longer applies to them and they silently lose their keeper. The strip offers that
pair at 0.95 and confirming it restores one person with 4 frames and a keeper; a
rescan at the same 0.97 gate keeps it merged (link stored as the four face ids),
while marking the pair *two people* keeps them split and suppresses the
suggestion on every later scan. Verified through the HTTP API and by clicking
both buttons in the browser UI.

Two alternatives were measured and rejected. A **geometry** descriptor built from
the same mesh (inter-ocular-normalised distances and face proportions) resisted
exposure changes but did not discriminate identities at all: intra-person 1.000
vs inter-person 0.982 on the labelled set. A learned embedder — OpenCV Zoo
**SFace** (Apache-2.0) — would separate better in principle, but loading it
through `cv2.dnn` in plain `opencv-python-headless` produced near-constant
embeddings (mean cosine 0.85 between *different* people), so it needs
`opencv-contrib`'s `FaceRecognizerSF`. That is a heavier dependency than a
within-shoot question justifies; it is recorded as the upgrade path.

---

## 5. Stack recommendation

| Layer | Choice | Why |
| --- | --- | --- |
| Language | **Python 3.11** | Every reference and every permissive CV/ML package is Python. The pipeline is the product; the API is thin. |
| Image I/O | OpenCV (headless) + Pillow + `pillow-heif`; `rawpy` optional | Same stack as both mature references; HEIC coverage without extra binaries. |
| Faces / eyes | **MediaPipe Face Landmarker** (Apache-2.0) | Best permissive blink/blendshape option; facet's measured improvement over landmark geometry; no InsightFace licence problem. |
| Similarity | pHash + dHash (own implementation, ~30 lines) + EXIF burst windows; **optional** DINOv2-small ONNX | Heartbeat needs zero model downloads; semantics are additive. |
| Scoring | Deterministic weighted composite (no learned model in v1) | Explainable, testable, and it is what makes reject reasons possible. A learned keeper ranker is a later, optional layer (facet's `keeper_head` is the pattern, and it must beat the deterministic baseline to be enabled). |
| Storage | SQLite (+ JPEG preview cache) | Zero-ops local catalogue; 10k rows is trivial for it. |
| API | FastAPI + Uvicorn | Same as both mature references; SSE for progress. |
| Frontend | React 18 + TypeScript + Vite | Keyboard-driven review UI; static build served by the API. |
| Packaging (later) | Tauri shell or PyInstaller + `pywebview` | retoucan proves Tauri is a good fit for this class of app; not needed for the MVP. |

Explicitly **rejected**: a Docker Compose/Redis/MinIO architecture (photo-pipeline)
as the default — it is the right call for a GPU fleet processing 10k-photo
weddings server-side, and the wrong call for a photographer's laptop. The MVP is
single-process and CPU-only; the interfaces (`Analyzer`, `Embedder`) leave room
for a worker split later.

---

## 6. MVP scope (implemented in this repository)

**In scope**

* Folder ingestion (recursive, common RAW/JPEG/HEIC extensions), resumable scans.
* Technical analysis: sharpness (Laplacian + ISO compensation), exposure/clipping,
  noise, contrast.
* Face analysis: detection, eyes-open/blink via blendshapes, smile, head pose,
  per-face sharpness.
* Similarity: pHash/dHash, EXIF burst grouping, union-find groups with a
  best-frame lead and guaranteed minimum keepers.
* People: face descriptors, identity clustering, per-person frames, and the
  keeper-coverage guarantee (with forced promotion for flawed frames).
* Scoring, EXPLAINABLE reject verdicts, genre + strictness presets.
* Review UI: group-first grid, filters (keepers / rejects / blur / blink /
  duplicates), a people strip, loupe with person badges, keyboard decisions,
  per-group compare, stats.
* Export: XMP sidecars, copy keepers to a folder, CSV report.
* CLI for headless runs (batch/server use, CI).

**Deliberately out of scope for v1**

* RAW decoding via `rawpy` (optional install; JPEG previews of RAWs work if the
  camera wrote an embedded JPEG — otherwise install `rawpy`).
* Semantic duplicate confirmation (DINOv2-small ONNX) — implemented and
  **opt-in**: the `semantic` extra plus a 42 MB checksum-pinned download, a
  per-library flag, 768 stored bytes per photo and about 40 ms per photo
  (measured: 81 → 117 ms/photo on the 72-frame Teasers shoot). Measured on that
  shoot, it added no merges at facet's 12-bit prefilter — the pairs that looked
  like duplicates sat 16–30 bits apart — so it stays off by default until a
  labelled set justifies widening the prefilter. CLIP is not used.
* Learned (as opposed to classical) face embeddings; see §4.6 for why the
  permissive options were deferred.
* Naming people, per-person export albums, "bride & groom priority" weights.
* Learned per-photographer ranking, editing/tethering/Lightroom catalogue
  round-trip.

**Roadmap after v1**: labelled duplicate evaluation (and the wider prefilter it
would justify) for the opt-in DINOv2 dedup → optional SFace embeddings
(`opencv-contrib`) behind the same clustering interface → named people and
per-person export → learned keeper ranker on user decisions → Lightroom
catalogue integration (retoucan's `catalog.rs` approach) → Tauri packaging.

---

## 7. Attribution

The algorithms in this repository were independently implemented from the
designs above. Where a threshold or formula was taken from a reference, the
source is named in the code comment and in `NOTICE.md`. No third-party source
files were copied verbatim; the reference repositories are MIT (retoucan,
facet, photo-pipeline) and are credited accordingly.
