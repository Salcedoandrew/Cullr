"""Scene embeddings: preprocessing, storage, and the opt-in duplicate gate.

The numbers pinned here were measured on this Mac, not guessed:

* A genuine near-duplicate pair (the same frame exported twice) scored cosine
  0.908; the nearest unrelated pair in the same 12-frame sample scored 0.63.
  That gap is why the threshold is 0.88 and not photo-pipeline's 0.92.
* fp16 ONNX against the fp32 reference graph on the same frames: cosine
  0.9999, so the half-size weights are numerically faithful.
* 31 ms/image at 224 px on 8 threads, 41 ms on one thread; the analysis pool
  already runs 8 photos in flight, which is why the session uses one thread.

Anything that needs the real model or `onnxruntime` is skip-gated: the extra
is optional, and the default build must pass without it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from culler.config import ANALYSIS_VERSION, CullingSettings
from culler.db import Store
from culler.imaging import ImageMeta
from culler.metrics.faces import FaceAnalyzer, FaceMetrics
from culler.metrics.scene import (
    EMBEDDING_DIM, SceneEmbedder, embedding_bytes, embedding_from_bytes,
    preprocess, semantic_available,
)
from culler.metrics.technical import TechnicalMetrics
from culler.pipeline import AnalysedPhoto, _from_row, analyse_file, plan


def unit(*values: float) -> tuple[float, ...]:
    """Pad with zeros to EMBEDDING_DIM and normalise."""
    padded = list(values) + [0.0] * (EMBEDDING_DIM - len(values))
    norm = math.sqrt(sum(v * v for v in padded))
    return tuple(v / norm for v in padded)


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Preprocessing and storage
# --------------------------------------------------------------------------- #

def test_preprocess_makes_an_imagenet_tensor(scene):
    tensor = preprocess(scene)
    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))


def test_preprocess_rejects_a_frame_with_no_content():
    assert preprocess(np.zeros((4, 4, 3), dtype=np.uint8)) is None
    assert preprocess(None) is None


def test_embedding_storage_round_trips_as_fp16():
    vector = unit(*[1.0, 2.0, 3.0])
    blob = embedding_bytes(vector)
    assert isinstance(blob, bytes)
    # 384 fp16 values: 768 bytes per photo, ~4.4 MB for a 5,745-frame wedding.
    assert len(blob) == EMBEDDING_DIM * 2

    restored = embedding_from_bytes(blob)
    assert restored is not None
    assert len(restored) == EMBEDDING_DIM
    assert abs(math.sqrt(sum(v * v for v in restored)) - 1.0) < 1e-3
    assert cosine(restored, vector) > 0.9999


def test_corrupt_embeddings_are_treated_as_absent():
    assert embedding_from_bytes(None) is None
    assert embedding_from_bytes(b"") is None
    assert embedding_from_bytes(b"\x00\x01\x02") is None


# --------------------------------------------------------------------------- #
# The embedder itself
# --------------------------------------------------------------------------- #

class FakeSession:
    """Stands in for an ONNX session; keeps the tests runtime-free."""

    def __init__(self, size: int = EMBEDDING_DIM):
        self.size = size

    def run(self, _outputs, _feeds):
        return [np.ones((1, 1, self.size), dtype=np.float32)]


def test_missing_model_is_unavailable_not_an_error(tmp_path):
    embedder = SceneEmbedder(tmp_path / "absent.onnx")
    assert embedder.available is False
    assert embedder.embed(np.zeros((64, 64, 3), dtype=np.uint8)) is None


def test_embed_returns_a_unit_vector_from_the_session(tmp_path):
    embedder = SceneEmbedder(tmp_path / "stub.onnx")
    embedder._session = FakeSession()          # skip the file/runtime checks
    vector = embedder.embed(np.zeros((64, 64, 3), dtype=np.uint8))
    assert vector is not None
    assert len(vector) == EMBEDDING_DIM
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-6
    assert abs(vector[0] - 1.0 / math.sqrt(EMBEDDING_DIM)) < 1e-6


def test_a_wrong_dimension_graph_is_rejected(tmp_path):
    embedder = SceneEmbedder(tmp_path / "stub.onnx")
    embedder._session = FakeSession(size=10)
    assert embedder.embed(np.zeros((64, 64, 3), dtype=np.uint8)) is None


def test_semantic_available_needs_the_model_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CULLR_DATA_DIR", str(tmp_path))
    assert semantic_available() is False


# --------------------------------------------------------------------------- #
# Pipeline wiring: analysis stores the vector, grouping only reads it when asked
# --------------------------------------------------------------------------- #

def sample_photo(photo_id: int, phash: str, dhash: str,
                 scene: tuple[float, ...] | None = None) -> AnalysedPhoto:
    return AnalysedPhoto(
        photo_id=photo_id,
        path=f"/shoot/{photo_id}.jpg",
        filename=f"{photo_id}.jpg",
        meta=ImageMeta(width=4000, height=3000),
        tech=TechnicalMetrics(sharpness=200.0, tenengrad=12.0, clip_hi=0.0,
                              clip_lo=0.01, mean_luma=0.5, p95_luma=0.9,
                              noise=3.0, contrast=40.0),
        faces=FaceMetrics(faces=[], available=True),
        phash=phash,
        dhash=dhash,
        scene_embedding=scene,
    )


def test_analysis_payload_persists_the_vector(tmp_path):
    store = Store(tmp_path / "catalog.db")
    library = store.create_or_get_library(str(tmp_path / "shoot"))
    photo_id = store.upsert_photo(library["id"], tmp_path / "a.jpg", "key", 10, 1.0)
    vector = unit(1.0, 2.0, 3.0)
    photo = sample_photo(photo_id, "0" * 16, "f" * 16, scene=vector)

    store.update_analysis(photo_id, photo.analysis_payload(), ANALYSIS_VERSION)

    row = store.stored_analysis_for(photo_id)
    assert isinstance(row["scene_embedding"], bytes)
    assert len(row["scene_embedding"]) == EMBEDDING_DIM * 2
    restored = _from_row(row, [])
    assert restored.scene_embedding is not None
    assert cosine(restored.scene_embedding, vector) > 0.999


def test_analysis_payload_without_a_vector_stores_null(tmp_path):
    store = Store(tmp_path / "catalog.db")
    library = store.create_or_get_library(str(tmp_path / "shoot"))
    photo_id = store.upsert_photo(library["id"], tmp_path / "a.jpg", "key", 10, 1.0)
    photo = sample_photo(photo_id, "0" * 16, "f" * 16)

    store.update_analysis(photo_id, photo.analysis_payload(), ANALYSIS_VERSION)

    assert store.stored_analysis_for(photo_id)["scene_embedding"] is None


# 9 bits, deliberately split across both 32-bit halves of the pHash: the exact
# bit pattern the hash-only gate misses. Blocking shares a whole half, so a pair
# that differs in both is never a candidate anywhere in the shoot, and with the
# dHash 64 bits apart the time-window rules fail too.
PHASH_B = ((1 << 0) | (1 << 1) | (1 << 2)
           | (1 << 33) | (1 << 40) | (1 << 45) | (1 << 50) | (1 << 55) | (1 << 60))


def test_grouping_uses_embeddings_only_when_the_library_opts_in():
    """The loose pair a hash-only gate misses: 9 pHash bits, 64 dHash bits.

    A hand or expression moved, so the second hash disagrees completely -- the
    case the embedding exists to catch, and the case it must not catch when the
    library has not opted in (stored vectors are inert then).
    """
    vector_a = unit(1.0)
    # cosine 0.95: above the 0.88 confirmation floor.
    vector_b = unit(0.95, math.sqrt(1 - 0.95 ** 2))
    photos = [
        sample_photo(1, f"{0:016x}", f"{0:016x}", scene=vector_a),
        sample_photo(2, f"{PHASH_B:016x}", f"{2**64 - 1:016x}", scene=vector_b),
    ]

    off = plan(photos, CullingSettings(semantic_dedup=False))
    assert off.groups == [], "vectors must be ignored unless the library asks"

    on = plan(photos, CullingSettings(semantic_dedup=True))
    assert len(on.groups) == 1
    assert sorted(on.membership[1]) == [1, 2]


def test_low_cosine_does_not_widen_recall():
    photos = [
        sample_photo(1, f"{0:016x}", f"{0:016x}", scene=unit(1.0)),
        sample_photo(2, f"{PHASH_B:016x}", f"{2**64 - 1:016x}",
                     scene=unit(0.5, math.sqrt(1 - 0.25))),
    ]
    assert plan(photos, CullingSettings(semantic_dedup=True)).groups == []


def test_analyse_file_attaches_a_vector_only_when_an_embedder_is_passed(photo_dir):
    path = sorted(photo_dir.glob("*.jpg"))[0]
    analyzer = FaceAnalyzer()
    expected = unit(1.0, 2.0)

    class FakeEmbedder:
        def embed(self, _bgr):
            return expected

    without = analyse_file(path, analyzer)
    assert without[6] is None
    with_embedder = analyse_file(path, analyzer, embedder=FakeEmbedder())
    assert with_embedder[6] == expected


# --------------------------------------------------------------------------- #
# The real model, when it is installed (the optional `semantic` extra)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not semantic_available(), reason="scene model/runtime not installed")
def test_real_model_produces_normalised_vectors(scene):
    embedder = SceneEmbedder()
    assert embedder.available is True
    first = embedder.embed(scene)
    assert first is not None and len(first) == EMBEDDING_DIM
    assert abs(math.sqrt(sum(v * v for v in first)) - 1.0) < 1e-3

    # Deterministic for the same pixels.
    again = embedder.embed(scene)
    assert again is not None and cosine(first, again) > 0.9999
