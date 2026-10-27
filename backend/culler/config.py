"""Configuration: analysis sizes, shoot presets, scoring weights and gates.

The numbers here are calibrated for a fixed-size analysis frame (long edge
1600 px). Sharpness metrics are resolution dependent, so analysing a normalised
preview is what makes one threshold valid for a 12 MP and a 61 MP body.

Threshold provenance (see docs/RESEARCH.md):
  * Laplacian-variance blur screens: photo-pipeline (default 80, 30..250 range)
    and facet (min(10, var/50)).
  * Eyes-closed cutoff 4.0/10 and pose gate: facet.
  * Bounded per-group lead tie-breaks: facet `utils/selection.py`.
  * Per-genre profiles: facet `cull_profiles`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def bundle_root() -> Path:
    """Where read-only shipped files live.

    Inside a frozen app (PyInstaller) the resources are unpacked next to the
    executable, so ``__file__`` no longer points into the repository. Everything
    that ships with the app — the face model, the built frontend — resolves
    through here.
    """
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return bundle_root().joinpath(*parts)

ANALYSIS_LONG_EDGE = 1600
THUMB_LONG_EDGE = 512
# 1: technical + face metrics. 2: per-face identity descriptors (people
# clustering) are part of analysis. 3: the metrics the new ranking axes read
# back from the stored row — subject-region sharpness, background separation,
# histogram bimodality, the shadow/highlight masses and the colour-histogram
# entropy — plus the optional per-photo scene embedding. A row analysed before
# they existed would score differently from a fresh analysis, so version 2 rows
# are re-analysed once. Bumped as one step because all of it ships together;
# there are no version-3 rows in the wild.
ANALYSIS_VERSION = 3

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp",
    ".heic", ".heif", ".avif",
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef", ".srw",
}
RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".orf", ".pef", ".srw",
}

# Import screen: the four photo types the panel offers, in the order it shows
# them (Aftershoot's import panel lists the same four). "jpeg" is the
# everyday-raster group — JPEG plus the other lossless/raster formats that are
# neither RAW, TIFF nor HEIC — so unchecking it removes PNG/WebP/BMP too.
# The panel shows the extensions under each label, so the grouping is visible.
IMPORT_TYPE_EXTENSIONS: dict[str, frozenset[str]] = {
    "raw": frozenset(RAW_EXTENSIONS),
    "jpeg": frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"}),
    "tiff": frozenset({".tif", ".tiff"}),
    "heic": frozenset({".heic", ".heif", ".avif"}),
}
IMPORT_TYPE_KEYS = ("raw", "jpeg", "tiff", "heic")
IMPORT_TYPE_LABELS = {
    "raw": "RAW",
    "jpeg": "JPEG",
    "tiff": "TIFF",
    "heic": "HEIC",
}


def extensions_for_types(types: Any) -> set[str]:
    """The file suffixes a set of import-type keys admits (empty = none)."""
    extensions: set[str] = set()
    for key in types or ():
        extensions |= set(IMPORT_TYPE_EXTENSIONS.get(str(key), frozenset()))
    return extensions

# Axes that can contribute to the composite score. Missing axes are dropped
# from numerator AND denominator, so a landscape is never punished for having
# no eyes (the pattern used by photo-pipeline's axis_composite_score).
AXES = (
    "sharpness", "subject_sharpness", "exposure", "noise", "contrast", "color",
    "eyes_open", "face_sharpness", "subject",
)


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    weights: dict[str, float]
    strictness: int
    keep_min_per_group: int = 1
    eyes_closed_max: float = 4.0
    # Absolute focus gate applied to the *face crop* (192 px normalised).
    # Calibrated on studio portraits: a sharp face crop measures 450-550, a
    # 1 px Gaussian blur ~135, a 2 px blur ~35. Frame-wide Laplacian variance
    # is not used as an absolute gate because it swings 6x with content.
    face_blur_gate: float = 55.0


PRESETS: dict[str, Preset] = {
    "portrait": Preset(
        key="portrait",
        label="Portrait",
        description="Single subject, shallow depth of field. Eyes and face sharpness weighted heavily; soft backgrounds are not penalised.",
        weights={"sharpness": 0.26, "subject_sharpness": 0.04, "exposure": 0.18,
                 "noise": 0.06, "contrast": 0.06, "color": 0.04,
                 "eyes_open": 0.22,
                 "face_sharpness": 0.14, "subject": 0.08},
        strictness=50,
        keep_min_per_group=1,
        eyes_closed_max=4.0,
        face_blur_gate=55.0,
    ),
    "wedding": Preset(
        key="wedding",
        label="Wedding / event",
        description="Many faces per frame, emotion matters. Lenient on softness because shallow depth of field is normal, strict on closed eyes.",
        weights={"sharpness": 0.24, "subject_sharpness": 0.05, "exposure": 0.16,
                 "noise": 0.05, "contrast": 0.05, "color": 0.04,
                 "eyes_open": 0.26,
                 "face_sharpness": 0.14, "subject": 0.10},
        strictness=35,
        keep_min_per_group=1,
        eyes_closed_max=5.0,
        face_blur_gate=45.0,
    ),
    "event": Preset(
        key="event",
        label="Event / documentary",
        description="Mixed subjects and venues. Balanced gates; tolerates some motion as long as a frame in the group is sharp.",
        weights={"sharpness": 0.30, "subject_sharpness": 0.05, "exposure": 0.20,
                 "noise": 0.08, "contrast": 0.08, "color": 0.05,
                 "eyes_open": 0.16,
                 "face_sharpness": 0.10, "subject": 0.08},
        strictness=50,
        keep_min_per_group=1,
        eyes_closed_max=4.0,
        face_blur_gate=55.0,
    ),
    "sports": Preset(
        key="sports",
        label="Sports / action",
        description="Fast bursts, motion blur is often intentional. Non-face blur allowed; the sharpest frame in each burst wins.",
        weights={"sharpness": 0.34, "subject_sharpness": 0.04, "exposure": 0.18,
                 "noise": 0.05, "contrast": 0.05, "color": 0.04,
                 "eyes_open": 0.10,
                 "face_sharpness": 0.12, "subject": 0.16},
        strictness=85,
        keep_min_per_group=1,
        eyes_closed_max=2.0,
        face_blur_gate=85.0,
    ),
    "landscape": Preset(
        key="landscape",
        label="Landscape / travel",
        description="No faces: eyes axes are absent and therefore ignored. Sharpness and dynamic range dominate.",
        weights={"sharpness": 0.40, "subject_sharpness": 0.08, "exposure": 0.24,
                 "noise": 0.12, "contrast": 0.16, "color": 0.08,
                 "subject": 0.08},
        strictness=60,
        keep_min_per_group=1,
        eyes_closed_max=0.0,
        face_blur_gate=0.0,
    ),
}

DEFAULT_PRESET = "wedding"

# The Auto Target tab's "Set selection strictness" stops, in Aftershoot's order:
# EXTREME keeps the fewest frames, MORE keeps everything the gates allow. The
# values anchor to the preset strictness range (Sports 85 ... Wedding 35) rather
# than being invented: EXTREME is the Sports gate, FEW sits between it and the
# Portrait/Event middle, STANDARD is the Wedding default, MORE is lenient enough
# that only clearly bad frames are dropped. The UI shows the name, never a
# number, and `strictness_stops()` is the one definition both sides read.
#
# Each stop carries a keep fraction as well, because the gate scaling alone
# cannot move a shoot's keeper count: measured on a real 5,745-frame wedding
# (Uslea + Ernesto, wedding preset), gates-only kept 19.9% at EXTREME and 21.5%
# at MORE -- and sweeping the whole 0..100 range moved it only 18.4%..21.6%.
# Two thirds of the rejections are best-of-group demotions, which no gate
# touches, so a looser stop's extra survivors just become duplicates inside
# groups that already keep a frame. The keep fraction is applied after the
# gates (Auto Target mode only; Cull to Target sets its own count) and is what
# makes the scale mean what the dialog says: keep the strongest 25 / 45 / 70%
# of the frames that passed, or, at MORE, everything that passed. Measured end
# to end on that same wedding: 288 kept (5.0%) / 532 (9.3%) / 847 (14.7%) /
# 1,233 (21.5%). STANDARD lands on the 14% (804 frames) the photographer kept by
# hand. The cost of a tighter stop is real and measured: at STANDARD 680 of his
# 804 picks still survive (94 kept exactly, 586 demoted inside a group that kept
# another frame) and 124 lose their moment; at MORE only 24 do. The floor is
# calibrated on the count, not on his taste -- see docs/RESEARCH.md 6 for why
# the learned ranker, not a tighter threshold, is the answer to that gap.
SELECTION_STOPS: tuple[tuple[str, str, int, float | None], ...] = (
    ("extreme", "EXTREME", 85, 0.25),
    ("few", "FEW", 60, 0.45),
    ("standard", "STANDARD", 35, 0.70),
    ("more", "MORE", 15, None),
)


def strictness_stops() -> list[dict[str, Any]]:
    return [{"key": key, "label": label, "value": value}
            for key, label, value, _share in SELECTION_STOPS]


def stop_keep_fraction(strictness: int | None) -> float | None:
    """How much of the gate-passing shoot a named stop keeps.

    The stop is defined by its gate-strictness value, so a hand-written value
    (the CLI takes any 0..100) matches no stop and gets gates only, exactly the
    behaviour libraries had before the stops existed.
    """
    if strictness is None:
        return None
    for _key, _label, value, share in SELECTION_STOPS:
        if value == int(strictness):
            return share
    return None


# ---- The XMP rating scheme ------------------------------------------------ #
# One editable row per outcome, as in Aftershoot's "Set Star/Color Ratings"
# panel. The defaults reproduce Cullr's old hardcoded pair exactly -- keepers
# ★1 Green, everything rejected -1 Red -- so an untouched shoot exports the
# same sidecar it did before the scheme existed.
RATING_CATEGORIES = ("selected", "highlights", "blurry", "closed_eyes",
                     "duplicate", "unrated")
# Lightroom's five colour labels, plus "" for no label at all. Star -1 is XMP's
# own "rejected" value; 0 is unrated.
XMP_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple", "")
RATING_REASONS = {"overexposed": "highlights", "blurry": "blurry",
                  "closed_eyes": "closed_eyes", "duplicate": "duplicate"}


def default_rating_scheme() -> dict[str, dict[str, Any]]:
    return {
        "selected": {"stars": 1, "color": "Green"},
        "highlights": {"stars": -1, "color": "Red"},
        "blurry": {"stars": -1, "color": "Red"},
        "closed_eyes": {"stars": -1, "color": "Red"},
        "duplicate": {"stars": -1, "color": "Red"},
        "unrated": {"stars": -1, "color": "Red"},
    }


def normalise_rating_scheme(raw: Any) -> dict[str, dict[str, Any]]:
    """Clamp a stored scheme back to what XMP can express.

    A hand-edited settings_json (or an older library) must never make export
    write a star outside -1..5 or a colour Lightroom would not show, so each
    row is validated and anything unreadable falls back to the default.
    """
    scheme = default_rating_scheme()
    if not isinstance(raw, dict):
        return scheme
    for key in RATING_CATEGORIES:
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        try:
            stars = int(entry.get("stars", scheme[key]["stars"]))
        except (TypeError, ValueError):
            continue
        color = entry.get("color", scheme[key]["color"])
        scheme[key] = {
            "stars": stars if stars in (-1, 0, 1, 2, 3, 4, 5) else scheme[key]["stars"],
            "color": color if color in XMP_LABELS else scheme[key]["color"],
        }
    return scheme


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class CullingSettings:
    """User-tunable settings for one library scan."""

    preset: str = DEFAULT_PRESET
    # Blur-only strictness (the slider under Detect Blurry Photos). 0..100,
    # None = follow the overall strictness below. It moves the two blur gates
    # and nothing else: eyes and exposure are the overall knob's business.
    strictness: int | None = None
    # The Auto Target tab's overall selection strictness, 0..100 or one of the
    # named stops in SELECTION_STOPS. None = the preset's own value. It scales
    # every gate, so it is the "how picky is this cull" control.
    selection_strictness: int | None = None
    keep_min_per_group: int | None = None
    detect_blinks: bool = True
    detect_blur: bool = True               # reject soft frames (all three blur gates)
    detect_highlights: bool = True         # reject blown highlights (overexposure)
    group_duplicates: bool = True
    # Similarity gates (pHash 64-bit Hamming distance).
    duplicate_distance: int = 6            # facet: 90% similarity => 6
    similar_distance: int = 14             # looser candidate gate within a burst
    burst_window_seconds: float = 2.0      # EXIF capture-time window
    similar_window_seconds: float = 8.0    # same-scene window
    min_group_size: int = 2
    # Semantic duplicate confirmation: one DINOv2-small scene vector per photo,
    # stored with the analysis and used by `similarity.find_groups` to confirm
    # loose hash matches. Off by default, because it needs the optional
    # `semantic` extra plus a one-time model download; see
    # `metrics/scene.py`. With it off, grouping is hash-only and unchanged.
    semantic_dedup: bool = False
    # Cosine floor for that confirmation. Measured on this Mac: a genuine
    # near-duplicate pair (the same frame exported twice) scored 0.908, the
    # nearest unrelated pair in the sample 0.63. photo-pipeline's 0.92 would
    # have missed the real pair, hence 0.88 rather than a copied constant.
    embedding_threshold: float = 0.88
    # Relative blur gate: reject when a frame is this fraction of its group median.
    relative_blur_ratio: float | None = None
    # People: cluster faces across the shoot and guarantee every subject appears
    # in at least one keeper (Aftershoot-style "key face" coverage).
    person_coverage: bool = True
    # Cosine similarity for joining an existing person, and for merging two
    # clusters. Deliberately high: measured separation on real portraits is
    # intra-person 0.94-1.00 vs inter-person up to 0.88, and a false merge can
    # silently drop the coverage guarantee (one merged cluster counted as
    # covered by a frame that only contains one of the two people). Splitting a
    # person into two clusters only costs an extra keeper.
    person_match: float = 0.90
    person_merge: float = 0.95
    person_min_photos: int = 2          # frames before a person must be covered
    person_min_face_ratio: float = 0.015  # face size that counts as a subject
    # "Cull to target": keep roughly this fraction of the shoot, ranked by
    # measured quality. None means the gates decide (the "auto target" mode).
    # Coverage still wins, so the achieved count can sit slightly above it.
    target_ratio: float | None = None
    # Export: rewrite an existing <file>.xmp instead of leaving it alone.
    overwrite_xmp: bool = True
    # Export: the star/colour every outcome gets in the sidecar. Editable from
    # the dialog's Stars & Colors panel; the defaults are the old fixed pair.
    rating_scheme: dict[str, dict[str, Any]] = field(default_factory=default_rating_scheme)
    # Import (the import screen): which folders to walk and which files count.
    # These travel with the library in settings_json, so a rescan reuses them.
    recursive: bool = True               # include sub-folders
    import_types: tuple[str, ...] = ("raw", "jpeg", "tiff", "heic")
    # Ingest and backup: copy the imported photos into a working directory
    # under the app data dir (and, when set, a backup directory), then scan
    # the working copy. The source folder is remembered as the library root;
    # originals are never modified.
    ingest: bool = False
    backup_dir: str | None = None

    def __post_init__(self) -> None:
        self.rating_scheme = normalise_rating_scheme(self.rating_scheme)
        # settings_json round-trips through JSON, so a tuple arrives as a list.
        # Unknown keys are dropped rather than failing a stored library.
        cleaned: list[str] = []
        for key in self.import_types or ():
            key = str(key)
            if key in IMPORT_TYPE_EXTENSIONS and key not in cleaned:
                cleaned.append(key)
        self.import_types = tuple(cleaned)

    def import_extensions(self) -> set[str]:
        """The file suffixes this library's import admits."""
        return extensions_for_types(self.import_types)
    # Similarity floor for *suggesting* a merge between two people. Proposals are
    # only ever confirmed by the photographer, so this can sit below the automatic
    # match threshold (0.90) and still be safe.
    person_propose: float = 0.84

    def resolved_preset(self) -> Preset:
        return PRESETS.get(self.preset, PRESETS[DEFAULT_PRESET])

    def effective_strictness(self) -> int:
        """The overall gate strictness: the Auto Target stop, else the preset."""
        if self.selection_strictness is not None:
            return int(self.selection_strictness)
        return self.resolved_preset().strictness

    def selection_keep_fraction(self) -> float | None:
        """The share of gate-passing frames the Auto Target stop keeps.

        None when no stop is selected (the preset alone decides, so a library
        that never opened the dialog scores exactly as before) and None in Cull
        to Target mode, where the target ratio is already the count control.
        """
        if self.target_ratio:
            return None
        return stop_keep_fraction(self.selection_strictness)

    def blur_strictness(self) -> int:
        """The blur-only strictness: the Customize slider, else the overall."""
        if self.strictness is not None:
            return int(self.strictness)
        return self.effective_strictness()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CullingSettings":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def gates(self) -> "Gates":
        preset = self.resolved_preset()
        t = max(0, min(100, self.effective_strictness())) / 100.0
        # Blur is its own slider on purpose (Aftershoot's dialog has two
        # independent ones); unset it follows the overall value, which is what
        # keeps every existing library scoring exactly as before.
        b = max(0, min(100, self.blur_strictness())) / 100.0
        eyes = (preset.eyes_closed_max if t == 0.5
                else lerp(preset.eyes_closed_max + 1.5, preset.eyes_closed_max - 1.5, t))
        return Gates(
            blur_abs=ABS_BLUR_FLOOR,
            blur_rel=lerp(0.30, 0.65, b),
            face_blur=lerp(preset.face_blur_gate * 0.65,
                           preset.face_blur_gate * 1.60, b),
            eyes_closed_max=max(0.0, min(10.0, eyes)),
            clip_hi_max=lerp(0.12, 0.03, t),
            clip_lo_max=lerp(0.35, 0.08, t),
            mean_luma_min=lerp(0.05, 0.10, t),
            mean_luma_max=lerp(0.95, 0.88, t),
            keep_min_per_group=(preset.keep_min_per_group
                                if self.keep_min_per_group is None
                                else max(0, self.keep_min_per_group)),
        )

    def weights(self) -> dict[str, float]:
        return dict(self.resolved_preset().weights)


# Frame-wide Laplacian variance below this is mush regardless of content; it is
# only a safety net. Real blur decisions are relative to the group, or absolute
# on the face crop (see Preset.face_blur_gate).
ABS_BLUR_FLOOR = 5.0


@dataclass(frozen=True)
class Gates:
    blur_abs: float
    blur_rel: float
    face_blur: float
    eyes_closed_max: float
    clip_hi_max: float
    clip_lo_max: float
    mean_luma_min: float
    mean_luma_max: float
    keep_min_per_group: int
