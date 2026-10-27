import type {
  Capabilities, FilterKey, FolderListing, Job, Library, MergeSuggestion, Person, Photo,
  PhotoResponse, Settings, Stats,
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

  libraries: () => request<Library[]>("/api/libraries"),
  createLibrary: (path: string) =>
    request<Library>("/api/libraries", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
  library: (id: number) => request<Library>(`/api/libraries/${id}`),

  folders: (path?: string) =>
    request<FolderListing>(path ? `/api/folders?path=${encodeURIComponent(path)}` : "/api/folders"),
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
  latestJob: (id: number) => request<Job | Record<string, never>>(`/api/libraries/${id}/job`),
  job: (jobId: number) => request<Job>(`/api/jobs/${jobId}`),

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

export function formatShutter(shutter: number | null): string {
  if (!shutter) return "";
  if (shutter >= 1) return `${shutter}s`;
  return `1/${Math.round(1 / shutter)}s`;
}
