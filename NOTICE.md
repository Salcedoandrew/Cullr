# Third-party notices

Cullr implements its algorithms as new code. No source files were copied
verbatim from the projects below; their designs, thresholds and data structures
informed this implementation and are credited here.

## Reference projects (design only)

| Project | Licence | What informed this implementation |
| --- | --- | --- |
| [viktordrukker/photo-pipeline](https://github.com/viktordrukker/photo-pipeline) (Viktor Drukker) | MIT | Weighted composite that drops inapplicable axes; thread-pooled technical screen; EXIF burst pre-clustering; multi-hash voting with union-find; genre presets. |
| [facet](https://github.com/ncoevoet/facet) (Nicolas Coevoet) | MIT | Eyes-open score scale and 4.0/10 closed-eye cutoff; pose gate for unreliable eye geometry; blendshape-based blink scoring; two-stage pHash + embedding duplicate gate; bounded best-of-group tie-breaks; per-genre cull profiles; subject-region exposure reasoning; subject-region sharpness from a centre crop (facet's `sqrt(variance)/7` normalisation and background-separation ratio, minus the BiRefNet mask). |
| [retoucan](https://github.com/joemalott/retoucan) (Joe Malott) | MIT | Desktop app shape (local-first, no uploads); XMP/Lightroom round-trip expectations; compact interpretable per-photo feature vectors. |
| [AI-POWERED-PHOTO-CULLING](https://github.com/sundhars2823/AI-POWERED-PHOTO-CULLING) | **none** | Read for context only. No code, thresholds or structure were reused: the repository has no licence, so it is all rights reserved. |

## Models and data

| Component | Licence | Notes |
| --- | --- | --- |
| MediaPipe Face Landmarker bundle (`face_landmarker.task`) | Apache-2.0 (Google) | Downloaded from Google's model storage, pinned by SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. Provides 478 landmarks, `eyeBlink*`/`mouthSmile*` blendshapes and head pose. |
| Face identity descriptor (`backend/culler/metrics/faceid.py`) | Original code | Built from the landmarker's own mesh: five-point alignment plus a DCT/gradient-histogram descriptor. No extra model, weights or training data are involved. |
| OpenCV | Apache-2.0 | Decoding, Laplacian/Sobel metrics, perceptual hashing primitives. |
| Pillow, pillow-heif | HPND / BSD-3 | Decoding, EXIF, HEIC support. |
| rawpy / LibRaw (optional) | MIT / LGPL-2.1 (LibRaw: CDDL-1.0 or LGPL-2.1) | Optional RAW decode. Not required for JPEG/HEIC workflows; review LibRaw's terms before redistributing binaries. |
| FastAPI, Starlette, Uvicorn, Pydantic | MIT / BSD-3 | HTTP API. |
| React, Vite | MIT | Review interface. |
| Test fixtures in `.research/fixtures/` | CC BY-SA 3.0/4.0 (Wikimedia Commons) | Downloaded only for local verification and **not** redistributed: the directory is git-ignored. |

## Deliberately not used

* **InsightFace / `buffalo_l`** and **PyIQA** — weights are non-commercial. Both
  photo-pipeline and facet ship them by default; facet also ships a licensing
  gate for exactly this reason. Cullr uses MediaPipe instead.
* **SAMP-Net, U2-Net-P, LAION aesthetic head** — licence unclear per facet's own
  registry.
* **CLIP / DINOv2** — permissive code (MIT / Apache-2.0) but unverified training
  data. Not enabled in v1; the embedder interface exists for a later opt-in.
* **OpenCV Zoo SFace** (`face_recognition_sface_2021dec.onnx`, Apache-2.0) —
  evaluated for identity clustering. It needs `opencv-contrib`'s
  `FaceRecognizerSF`; loaded through plain `cv2.dnn` it returned near-constant
  embeddings (mean cosine 0.85 between different people), so it is not shipped.
  See `docs/RESEARCH.md` §4.6 for the measurement.
