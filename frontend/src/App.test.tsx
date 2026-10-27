import {
  cleanup, fireEvent, render, screen, waitFor, within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Capabilities, Library, PhotoResponse, Settings, Stats } from "./types";

/**
 * The count the cull options show before any scan — "~20% of 72 photos", the
 * field reading 14 — is frontend behaviour, so it is asserted on the rendered
 * dialog rather than on the state behind it.
 *
 * The trap it guards is the photo-list refresh: the server reports
 * `stats.total === 0` for a library that has no scan yet, and 0 is not
 * nullish. A plain `??` chain treats that as a known total, swallows the count
 * taken at import, and leaves the dialog reading 0.
 */

const api = vi.hoisted(() => ({
  capabilities: vi.fn(),
  libraries: vi.fn(),
  createLibrary: vi.fn(),
  library: vi.fn(),
  folders: vi.fn(),
  createFolder: vi.fn(),
  scan: vi.fn(),
  latestJob: vi.fn(),
  job: vi.fn(),
  discover: vi.fn(),
  photos: vi.fn(),
  stats: vi.fn(),
  persons: vi.fn(),
  mergeSuggestions: vi.fn(),
  mergePersons: vi.fn(),
  markDistinct: vi.fn(),
  decide: vi.fn(),
  export: vi.fn(),
}));

vi.mock("./api", () => ({
  api,
  thumbUrl: (photo: { id: number }) => `/api/photos/${photo.id}/image?size=thumb`,
  previewUrl: (photo: { id: number }) => `/api/photos/${photo.id}/image?size=preview`,
  faceUrl: (photoId: number, faceId: number) => `/api/photos/${photoId}/face/${faceId}`,
  folderPreviewUrl: (path: string) => `/api/folders/preview?path=${encodeURIComponent(path)}`,
  personLabel: () => null,
  reasonLabel: () => "",
  formatTime: () => "",
  formatShutter: () => "",
}));

const TOTAL = 72;

const settings: Settings = {
  preset: "wedding",
  strictness: null,
  keep_min_per_group: 1,
  detect_blinks: true,
  detect_blur: true,
  detect_highlights: true,
  group_duplicates: true,
  person_coverage: true,
  target_ratio: null,
  overwrite_xmp: true,
  recursive: true,
  import_types: ["raw", "jpeg", "tiff", "heic"],
  ingest: false,
  backup_dir: null,
};

const capabilities: Capabilities = {
  faces: true,
  faces_error: null,
  presets: [
    { key: "wedding", label: "Wedding / event", description: "", strictness: 35 },
  ],
  default_settings: settings,
  analysis_long_edge: 1600,
  import_types: [
    { key: "raw", label: "RAW", extensions: [".arw"] },
    { key: "jpeg", label: "JPEG", extensions: [".jpg"] },
    { key: "tiff", label: "TIFF", extensions: [".tif"] },
    { key: "heic", label: "HEIC", extensions: [".heic"] },
  ],
};

const statsOf = (total: number): Stats => ({
  total,
  keep: 0,
  reject: 0,
  blurry: 0,
  blink: 0,
  duplicates: 0,
  exposure: 0,
  groups: 0,
  persons: 0,
  uncovered_persons: 0,
});

const library = (over: Partial<Library> = {}): Library => ({
  id: 1,
  root_path: "/shoots/Teasers",
  name: "Teasers",
  photo_count: 0,
  settings: {},
  export: null,
  ...over,
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

async function renderApp() {
  render(<App />);
  await screen.findByLabelText("Folder of photos");
}

const keepField = (dialog: HTMLElement) =>
  within(dialog).getByLabelText("Photos to keep") as HTMLInputElement;

const cullSettings = () =>
  within(screen.getByRole("banner")).getByRole("button", { name: "Cull settings" });

beforeEach(() => {
  vi.clearAllMocks();
  api.capabilities.mockResolvedValue(capabilities);
  api.libraries.mockResolvedValue([]);
  // A library with no scan yet: the photo list reports total 0.
  api.photos.mockResolvedValue({ photos: [], total: 0, stats: statsOf(0) });
  api.persons.mockResolvedValue([]);
  api.mergeSuggestions.mockResolvedValue([]);
  api.latestJob.mockResolvedValue({});
  api.discover.mockResolvedValue({ total: 0 });
});

afterEach(cleanup);

describe("the cull options count", () => {
  it("shows the imported count before any scan, though the photo list reports 0", async () => {
    api.createLibrary.mockResolvedValue(library({ id: 1 }));
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.change(screen.getByLabelText("Folder of photos"),
                    { target: { value: "/shoots/Teasers" } });
    fireEvent.click(screen.getByRole("button", { name: "Import" }));

    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    // The refresh has landed by now — that is the moment the count used to vanish.
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    expect(within(dialog).getByText(`~20% of ${TOTAL} photos`)).toBeTruthy();
    expect(keepField(dialog).value).toBe("14");
  });

  it("counts a library opened from Recent that has never been scanned", async () => {
    // No import on this path: the library is reopened from the list.
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));

    fireEvent.click(cullSettings());

    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    expect(within(dialog).getByText(`~20% of ${TOTAL} photos`)).toBeTruthy();
    expect(keepField(dialog).value).toBe("14");
  });

  it("keeps the previous shoot's total out of the next one", async () => {
    const first = library({ id: 1, name: "Teasers", root_path: "/shoots/Teasers",
                            photo_count: TOTAL });
    const second = library({ id: 2, name: "Five", root_path: "/shoots/Five", photo_count: 0 });
    api.libraries.mockResolvedValue([first, second]);
    api.photos.mockResolvedValueOnce({ photos: [], total: TOTAL, stats: statsOf(TOTAL) });
    // The second shoot's photo list never answers, so the dialog has to open
    // while only the previous shoot's stats are in hand.
    const pending = deferred<PhotoResponse>();
    api.photos.mockReturnValueOnce(pending.promise);
    api.discover.mockResolvedValue({ total: 5 });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: "Change folder" }));
    fireEvent.click(screen.getByRole("button", { name: /Five/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(2));

    fireEvent.click(cullSettings());

    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    expect(within(dialog).getByText("~20% of 5 photos")).toBeTruthy();
    expect(keepField(dialog).value).toBe("1");
  });

  it("uses the scanned total once a cull has run", async () => {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.photos.mockResolvedValue({ photos: [], total: TOTAL, stats: statsOf(TOTAL) });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));

    fireEvent.click(cullSettings());

    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    expect(within(dialog).getByText(`~20% of ${TOTAL} photos`)).toBeTruthy();
    expect(keepField(dialog).value).toBe("14");
    // A scan already knows the number, so the folder is not walked again.
    expect(api.discover).not.toHaveBeenCalled();
  });

  it("still opens the cull options when the count cannot be taken", async () => {
    api.createLibrary.mockResolvedValue(library({ id: 1 }));
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockRejectedValue(new Error("not a directory: /shoots/Teasers"));

    await renderApp();
    fireEvent.change(screen.getByLabelText("Folder of photos"),
                    { target: { value: "/shoots/Teasers" } });
    fireEvent.click(screen.getByRole("button", { name: "Import" }));

    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    // Percent only: no total is claimed, the field stays at 0, nothing errors.
    expect(within(dialog).getByText("~20%")).toBeTruthy();
    expect(within(dialog).queryByText(/of [\d,]+ photos/)).toBeNull();
    expect(keepField(dialog).value).toBe("0");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
