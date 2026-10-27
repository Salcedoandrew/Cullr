export type Reason =
  | "blurry"
  | "overexposed"
  | "underexposed"
  | "closed_eyes"
  | "duplicate"
  | "unreadable"
  | null;

export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
  eyes_open: number | null;
  smile: number | null;
  yaw: number | null;
  pitch: number | null;
  sharpness: number;
  area_ratio: number;
  clip_hi: number | null;
  clip_lo: number | null;
  mean_luma: number | null;
  person_id: number | null;
}

export interface Photo {
  id: number;
  filename: string;
  path: string;
  width: number | null;
  height: number | null;
  captured_at: number | null;
  camera: string | null;
  lens: string | null;
  iso: number | null;
  shutter: number | null;
  aperture: number | null;
  focal_len: number | null;
  quality: number | null;
  verdict: "keep" | "reject";
  ai_verdict: "keep" | "reject" | null;
  reason: Reason;
  user_decision: "pick" | "reject" | null;
  star: number;
  group_id: number | null;
  is_lead: boolean;
  covered_person: number | null;
  sharpness: number | null;
  exposure: number | null;
  clip_hi: number | null;
  clip_lo: number | null;
  noise: number | null;
  face_count: number;
  eyes_open: number | null;
  blink: boolean;
  error: string | null;
  faces: FaceBox[];
}

export interface Stats {
  total: number;
  keep: number;
  reject: number;
  blurry: number;
  blink: number;
  duplicates: number;
  exposure: number;
  groups: number;
  persons: number;
  uncovered_persons: number;
}

export interface Person {
  id: number;
  label: string;
  face_count: number;
  photo_count: number;
  keep_count: number;
  covered: number;
  required: number;
  representative_photo_id: number | null;
  representative_face_id: number | null;
  max_area_ratio: number | null;
}

export interface MergeSuggestion {
  a: number;
  b: number;
  a_label: string | null;
  b_label: string | null;
  similarity: number;
}

export interface FolderEntry {
  name: string;
  path: string;
  kind: "folder" | "image" | "file";
  /** Folders only: how many visible children, when it could be counted. */
  items?: number | null;
  writable?: boolean;
  size?: number | null;
  width?: number;
  height?: number;
}

export interface FolderShortcut {
  name: string;
  path: string;
}

export interface FolderListing {
  path: string;
  parent: string | null;
  home: string;
  writable: boolean;
  /** Directories only — kept for callers that just want sub-folders. */
  folders: { name: string; path: string }[];
  entries: FolderEntry[];
  sidebar: {
    favorites: FolderShortcut[];
    shared: FolderShortcut[];
    locations: FolderShortcut[];
  };
  truncated: boolean;
}

export interface ExportMemory {
  mode: string;
  dest: string;
}

export interface Library {
  id: number;
  root_path: string;
  name: string;
  photo_count?: number;
  settings?: Partial<Settings>;
  export?: ExportMemory | null;
  /** Working copy, when the library was ingested; null otherwise. */
  ingest_root?: string | null;
}

/** The import screen's options. They travel with the library, so a rescan
 *  reuses them — including the working copy an ingest created. */
export interface ImportOptions {
  recursive: boolean;
  import_types: string[];
  ingest: boolean;
  backup_dir: string | null;
}

export interface ImportType {
  key: string;
  label: string;
  extensions: string[];
}

export interface Settings {
  preset: string;
  /** Blur-only strictness (the slider under Detect Blurry Photos). */
  strictness: number | null;
  /** Auto Target's overall selection strictness; null = the preset's value. */
  selection_strictness: number | null;
  keep_min_per_group: number | null;
  detect_blinks: boolean;
  detect_blur: boolean;
  detect_highlights: boolean;
  group_duplicates: boolean;
  /** Confirm loose hash matches with a scene embedding; needs a 42 MB model. */
  semantic_dedup: boolean;
  person_coverage: boolean;
  /** "Cull to target" fraction (0..1); null means the quality gates decide. */
  target_ratio: number | null;
  /** Export: replace an existing <file>.xmp rather than leaving it alone. */
  overwrite_xmp: boolean;
  /** Import: walk sub-folders, and which photo types count. */
  recursive: boolean;
  import_types: string[];
  /** Import: copy into a working directory (and an optional backup) first. */
  ingest: boolean;
  backup_dir: string | null;
}

export interface Preset {
  key: string;
  label: string;
  description: string;
  strictness: number;
}

/** One stop on the Auto Target "Set selection strictness" scale. */
export interface StrictnessStop {
  key: string;
  label: string;
  value: number;
}

export interface Capabilities {
  faces: boolean;
  faces_error: string | null;
  /** Scene runtime installed and model downloaded: the toggle is usable. */
  semantic: boolean;
  /** Scene runtime is in this build (the weights may still be missing). */
  semantic_runtime: boolean;
  presets: Preset[];
  default_settings: Settings;
  strictness_stops: StrictnessStop[];
  analysis_long_edge: number;
  import_types: ImportType[];
}

/** State of the first-use scene-model download. */
export interface SemanticStatus {
  runtime: boolean;
  model: boolean;
  available: boolean;
  downloading: boolean;
  done: number;
  total: number;
  error: string | null;
}

export interface PhotoResponse {
  total: number;
  photos: Photo[];
  stats: Stats;
}

export interface Job {
  id: number;
  status: "running" | "done" | "error" | "cancelled";
  stage: string;
  message: string | null;
  total: number;
  done: number;
  error: string | null;
}

export type FilterKey =
  | "all"
  | "keep"
  | "reject"
  | "pick"
  | "duplicates"
  | "blur"
  | "blink"
  | "exposure";
