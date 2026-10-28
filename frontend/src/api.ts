import type {
  Capabilities, FilterKey, FolderListing, ImportOptions, Job, Library, MergeSuggestion,
  Person, Photo, PhotoResponse, SemanticStatus, Settings, Stats, UpdateStatus,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  capabilities: () => request<Capabilities>("/api/capabilities"),

  /**
   * In-app update. The check is manual and happens on the server, which is the
   * only side allowed to reach GitHub and the filesystem; the panel polls
   * `/api/update` while a download runs.
   */
  updateStatus: () => request<UpdateStatus>("/api/update"),
  checkForUpdate: () => request<UpdateStatus>("/api/update/check", { method: "POST" }),
  downloadUpdate: () => request<UpdateStatus>("/api/update/download", { method: "POST" }),
  cancelUpdate: () => request<UpdateStatus>("/api/update/cancel", { method: "POST" }),
  installUpdate: () => request<UpdateStatus>("/api/update/install", { method: "POST" }),

  /** Scene model state; polled by the cull dialog while a download runs. */
  semanticStatus: () => request<SemanticStatus>("/api/semantic"),
  downloadSceneModel: () =>
    request<SemanticStatus>("/api/semantic/model", { method: "POST" }),

  libraries: () => request<Library[]>("/api/libraries"),
  createLibrary: (path: string, settings?: ImportOptions) =>
    request<Library>("/api/libraries", {
      method: "POST",
      body: JSON.stringify(settings === undefined ? { path } : { path, settings }),
    }),
  library: (id: number) => request<Library>(`/api/libraries/${id}`),

  /** One listing. `query` searches the whole folder on the server -- the
   *  entries that come back are capped, but the search is not. */
  folders: (path?: string, query?: string) => {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (query?.trim()) params.set("q", query.trim());
    const suffix = params.toString();
    return request<FolderListing>(suffix ? `/api/folders?${suffix}` : "/api/folders");
  },
  createFolder: (parent: string, name: string) =>
    request<{ path: string; name: string }>("/api/folders", {
      method: "POST",
      body: JSON.stringify({ parent, name }),
    }),

  scan: (id: number, settings: Settings, force = false) =>
    request<Job>(`/api/libraries/${id}/scan`, {
      method: "POST",
      body: JSON.stringify({ settings, force }),
    }),
  /** Ask a running cull to stop at the next frame boundary. */
  pauseScan: (jobId: number) =>
    request<{ pausing: number }>(`/api/jobs/${jobId}/pause`, { method: "POST" }),
  /** Stop a running cull for good. What it analysed stays stored. */
  cancelScan: (jobId: number) =>
    request<{ cancelling: number }>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  /** Continue a paused cull: the frames already analysed are reused. */
  resumeScan: (id: number) =>
    request<Job>(`/api/libraries/${id}/resume`, { method: "POST" }),
  latestJob: (id: number) => request<Job | Record<string, never>>(`/api/libraries/${id}/job`),
  job: (jobId: number) => request<Job>(`/api/jobs/${jobId}`),

  discover: (id: number) =>
    request<{ total: number }>(`/api/libraries/${id}/discover`, { method: "POST" }),

  photos: (
    id: number,
    filter: FilterKey,
    sort: "group" | "quality" | "time" = "group",
    personId: number | null = null,
  ) =>
    request<PhotoResponse>(
      `/api/libraries/${id}/photos?filter=${filter}&sort=${sort}` +
        (personId !== null ? `&person_id=${personId}` : ""),
    ),
  stats: (id: number) => request<Stats>(`/api/libraries/${id}/stats`),
  persons: (id: number) => request<Person[]>(`/api/libraries/${id}/persons`),
  mergeSuggestions: (id: number) =>
    request<MergeSuggestion[]>(`/api/libraries/${id}/persons/suggestions`),
  mergePersons: (id: number, ids: number[]) =>
    request<{ merged_into: number; persons: Person[] }>(
      `/api/libraries/${id}/persons/merge`,
      { method: "POST", body: JSON.stringify({ ids }) },
    ),
  markDistinct: (id: number, ids: number[]) =>
    request<{ dismissed: boolean }>(`/api/libraries/${id}/persons/distinct`, {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),

  decide: (
    photoId: number,
    decision?: "pick" | "reject" | "clear",
    star?: number,
  ) => {
    const body: Record<string, unknown> = {};
    if (decision !== undefined) body.decision = decision === "clear" ? "" : decision;
    if (star !== undefined) body.star = star;
    return request<Photo>(`/api/photos/${photoId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
  },

  export: (id: number, mode: string, dest?: string) =>
    request<{ written: number; copied: number; skipped: number; dest: string | null; errors: string[] }>(
      `/api/libraries/${id}/export`,
      { method: "POST", body: JSON.stringify({ mode, dest }) },
    ),
};

export const thumbUrl = (photo: Photo): string =>
  `/api/photos/${photo.id}/image?size=thumb`;
export const previewUrl = (photo: Photo): string =>
  `/api/photos/${photo.id}/image?size=preview`;
export const faceUrl = (photoId: number, faceId: number): string =>
  `/api/photos/${photoId}/face/${faceId}`;

/** Tile for one image in the folder picker (decoded and cached on demand). */
export const folderPreviewUrl = (path: string, size = 256): string =>
  `/api/folders/preview?size=${size}&path=${encodeURIComponent(path)}`;

export function personLabel(people: Person[], personId: number | null): string | null {
  if (personId === null) return null;
  return people.find((person) => person.id === personId)?.label ?? null;
}

export function reasonLabel(reason: string | null): string {
  switch (reason) {
    case "blurry":
      return "Soft focus";
    case "overexposed":
      return "Overexposed";
    case "underexposed":
      return "Underexposed";
    case "closed_eyes":
      return "Eyes closed";
    case "duplicate":
      return "Duplicate";
    case "target":
      return "Over target";
    case "unreadable":
      return "Unreadable file";
    default:
      return "";
  }
}

export function formatTime(epoch: number | null): string {
  if (!epoch) return "no capture time";
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** "168.9 MB" — what a download progress line reads in. */
export function formatBytes(bytes: number): string {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return mb >= 1000 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

export function formatShutter(shutter: number | null): string {
  if (!shutter) return "";
  if (shutter >= 1) return `${shutter}s`;
  return `1/${Math.round(1 / shutter)}s`;
}
