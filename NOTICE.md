# Third-party notices

Cullr implements its algorithms as new code. No source files were copied
verbatim from the projects below; their designs, thresholds and data structures
informed this implementation and are credited here.

## Reference projects (design only)

| Project | Licence | What informed this implementation |
| --- | --- | --- |
| [viktordrukker/photo-pipeline](https://github.com/viktordrukker/photo-pipeline) (Viktor Drukker) | MIT | Weighted composite that drops inapplicable axes; thread-pooled technical screen; EXIF burst pre-clustering; multi-hash voting with union-find; genre presets. |
| [facet](https://github.com/ncoevoet/facet) (Nicolas Coevoet) | MIT | Eyes-open score scale and 4.0/10 closed-eye cutoff; pose gate for unreliable eye geometry; blendshape-based blink scoring; two-stage pHash + embedding duplicate gate; bounded best-of-group tie-breaks and the inert learned-term slot; one shared ranking function for leads and the target cap; per-genre cull profiles; subject-region exposure reasoning; subject-region sharpness from a centre crop (facet's `sqrt(variance)/7` normalisation and background-separation ratio, minus the BiRefNet mask); histogram bimodality (facet's negated excess kurtosis) and the shadow/highlight mass flags; colour harmony as HSV-histogram entropy on facet's 0..10 scale. |
| [retoucan](https://github.com/joemalott/retoucan) (Joe Malott) | MIT | Desktop app shape (local-first, no uploads); XMP/Lightroom round-trip expectations; compact interpretable per-photo feature vectors. |
| [AI-POWERED-PHOTO-CULLING](https://github.com/sundhars2823/AI-POWERED-PHOTO-CULLING) | **none** | Read for context only. No code, thresholds or structure were reused: the repository has no licence, so it is all rights reserved. |

## Models and data

| Component | Licence | Notes |
| --- | --- | --- |
| MediaPipe Face Landmarker bundle (`face_landmarker.task`) | Apache-2.0 (Google) | Downloaded from Google's model storage, pinned by SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. Provides 478 landmarks, `eyeBlink*`/`mouthSmile*` blendshapes and head pose. |
| Face identity descriptor (`backend/culler/metrics/faceid.py`) | Original code | Built from the landmarker's own mesh: five-point alignment plus a DCT/gradient-histogram descriptor. No extra model, weights or training data are involved. |
| DINOv2 ViT-S/14 scene model (`dinov2_vits14_fp16.onnx`, 42 MB) | Apache-2.0 (Meta AI) — code and weights | **Optional, not bundled.** Downloaded only by `cull fetch-scene-model`, pinned by SHA-256 `16845e153bbaf3fd1ef8a2154c454940f901f6fe8a80dd4c6c319eaecdb4d2ee`. The ONNX fp16 conversion is `onnx-community/dinov2-small` on Hugging Face; measured 0.9999 cosine against the fp32 graph on the same frames. DINOv2's training data is not fully documented, which is why it is opt-in and off by default. |
| onnxruntime (optional `semantic` extra) | MIT | CPU inference for the scene model. Not shipped unless the extra is installed at build time. |
| OpenCV | Apache-2.0 | Decoding, Laplacian/Sobel metrics, perceptual hashing primitives. |
| Pillow, pillow-heif | HPND / BSD-3 | Decoding, EXIF, HEIC support. |
| rawpy / LibRaw | MIT / LGPL-2.1 (LibRaw: CDDL-1.0 or LGPL-2.1) | RAW decoding. rawpy calls a LibRaw build linked into its own extension module, so the released `.app` **redistributes LibRaw** — it is part of the bundle, not a user-installed extra, and this notice travels inside the app. Redistributing it means meeting LibRaw's terms (either CDDL-1.0 or LGPL-2.1; take the LGPL route and the app must allow the library to be replaced). |
| FastAPI, Starlette, Uvicorn, Pydantic | MIT / BSD-3 | HTTP API. |
| React, Vite | MIT | Review interface. |
| Test fixtures in `.research/fixtures/` | CC BY-SA 3.0/4.0 (Wikimedia Commons) | Downloaded only for local verification and **not** redistributed: the directory is git-ignored. |
| `backend/tests/fixtures/canon_eos_rp.cr3` | CC0 (public domain) | RAW test fixture from [raw.pixls.us](https://raw.pixls.us/) (Canon EOS RP, C-RAW, crop mode, SHA-256 `4a859ab137f792aedaf3416b0aad1497ebaab02052a49453902e226699154aa7`), committed because 0.8.0 shipped unable to read exactly this container: Canon's CRX is decoded by a separate part of libraw from the TIFF-family formats, so the ORF fixture alone cannot catch a build that lost CR3 support. Used by the backend tests and by the release build's smoke test, which culls one frame of it. CC0 needs no attribution; recorded here so the file's origin is not a mystery later. |
| `backend/tests/fixtures/olympus_c5050z.orf` | CC0 (public domain) | RAW test fixture from [raw.pixls.us](https://raw.pixls.us/) (Olympus C-5050Z, SHA-256 `f4b013e78204ebaadd740d93416080856b3c51ec154760e2e2de0e552873044a`), committed so the RAW decode path — and the release build's own smoke test — can be verified without a camera. CC0 needs no attribution; recorded here so the file's origin is not a mystery later. |

## Deliberately not used

* **InsightFace / `buffalo_l`** and **PyIQA** — weights are non-commercial. Both
  photo-pipeline and facet ship them by default; facet also ships a licensing
  gate for exactly this reason. Cullr uses MediaPipe instead.
* **SAMP-Net, U2-Net-P, LAION aesthetic head** — licence unclear per facet's own
  registry.
* **CLIP** — MIT code, but its training-data licence is unverified and it offers
  no measured advantage for duplicate detection. DINOv2-small is the opt-in
  scene embedder instead (see above), and one embedder is enough.
* **OpenCV Zoo SFace** (`face_recognition_sface_2021dec.onnx`, Apache-2.0) —
  evaluated for identity clustering. It needs `opencv-contrib`'s
  `FaceRecognizerSF`; loaded through plain `cv2.dnn` it returned near-constant
  embeddings (mean cosine 0.85 between different people), so it is not shipped.
  See `docs/RESEARCH.md` §4.6 for the measurement.
