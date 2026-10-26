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

export interface FolderListing {
  path: string;
  parent: string | null;
  home: string;
  writable: boolean;
  folders: { name: string; path: string }[];
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
}

export interface Settings {
  preset: string;
  strictness: number | null;
  keep_min_per_group: number | null;
  detect_blinks: boolean;
  group_duplicates: boolean;
  person_coverage: boolean;
}

export interface Preset {
  key: string;
  label: string;
  description: string;
  strictness: number;
}

export interface Capabilities {
  faces: boolean;
  faces_error: string | null;
  presets: Preset[];
  default_settings: Settings;
  analysis_long_edge: number;
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
