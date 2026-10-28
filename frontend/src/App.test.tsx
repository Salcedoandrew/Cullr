import {
  cleanup, fireEvent, render, screen, waitFor, within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type {
  Capabilities, FolderEntry, FolderListing, Job, Library, Photo, PhotoResponse,
  Settings, Stats,
} from "./types";

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
  semanticStatus: vi.fn(),
  downloadSceneModel: vi.fn(),
  updateStatus: vi.fn(),
  checkForUpdate: vi.fn(),
  downloadUpdate: vi.fn(),
  cancelUpdate: vi.fn(),
  installUpdate: vi.fn(),
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

// jsdom has no layout, so the picker's scroll-the-selection-into-view has
// nothing to call. A no-op keeps the click paths that reach it testable.
Element.prototype.scrollIntoView = vi.fn();

const TOTAL = 72;

const settings: Settings = {
  preset: "wedding",
  strictness: null,
  selection_strictness: null,
  keep_min_per_group: 1,
  detect_blinks: true,
  detect_blur: true,
  detect_highlights: true,
  group_duplicates: true,
  semantic_dedup: false,
  person_coverage: true,
  target_ratio: null,
  overwrite_xmp: true,
  rating_scheme: {
    selected: { stars: 1, color: "Green" },
    highlights: { stars: -1, color: "Red" },
    blurry: { stars: -1, color: "Red" },
    closed_eyes: { stars: -1, color: "Red" },
    duplicate: { stars: -1, color: "Red" },
    unrated: { stars: -1, color: "Red" },
  },
  recursive: true,
  import_types: ["raw", "jpeg", "tiff", "heic"],
  ingest: false,
  backup_dir: null,
};

const capabilities: Capabilities = {
  version: "0.8.0",
  faces: true,
  faces_error: null,
  semantic: false,
  semantic_runtime: true,
  presets: [
    { key: "wedding", label: "Wedding / event", description: "", strictness: 35 },
  ],
  strictness_stops: [
    { key: "extreme", label: "EXTREME", value: 85 },
    { key: "few", label: "FEW", value: 60 },
    { key: "standard", label: "STANDARD", value: 35 },
    { key: "more", label: "MORE", value: 15 },
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

const statsOf = (total: number, over: Partial<Stats> = {}): Stats => ({
  total,
  keep: 0,
  reject: 0,
  blurry: 0,
  blink: 0,
  duplicates: 0,
  exposure: 0,
  errors: 0,
  groups: 0,
  persons: 0,
  uncovered_persons: 0,
  ...over,
});

const jobOf = (over: Partial<Job> = {}): Job => ({
  id: 1,
  status: "done",
  stage: "done",
  message: "4 keepers · 2 groups · 72 photos in 8.0s",
  total: TOTAL,
  done: TOTAL,
  error: null,
  error_count: 0,
  error_sample: null,
  ...over,
});

const photoOf = (over: Partial<Photo> = {}): Photo => ({
  id: 1,
  filename: "IMG_0001.jpg",
  path: "/shoots/Teasers/IMG_0001.jpg",
  width: 1200,
  height: 800,
  captured_at: null,
  camera: null,
  lens: null,
  iso: null,
  shutter: null,
  aperture: null,
  focal_len: null,
  quality: null,
  verdict: "keep",
  ai_verdict: null,
  reason: null,
  user_decision: null,
  star: 0,
  group_id: null,
  is_lead: false,
  covered_person: null,
  sharpness: null,
  exposure: null,
  clip_hi: null,
  clip_lo: null,
  noise: null,
  face_count: 0,
  eyes_open: null,
  blink: false,
  error: null,
  faces: [],
  ...over,
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
  // The picker's recents and the last export destination live in localStorage,
  // which jsdom keeps for the whole file: without this, one test's destination
  // becomes the next test's default.
  window.localStorage.clear();
  api.capabilities.mockResolvedValue(capabilities);
  // The update panel's own states are covered in UpdatePanel.test.tsx; here it
  // only has to open.
  api.updateStatus.mockResolvedValue({
    current: "0.8.0",
    phase: "idle",
    error: null,
    latest: null,
    notes_url: null,
    filename: null,
    download_url: null,
    size: 0,
    done: 0,
    total: 0,
    percent: null,
    checked_at: null,
    installed_at: "/Applications/Cullr.app",
    update_log: "/tmp/update.log",
  });
  api.libraries.mockResolvedValue([]);
  // A library with no scan yet: the photo list reports total 0.
  api.photos.mockResolvedValue({ photos: [], total: 0, stats: statsOf(0) });
  api.persons.mockResolvedValue([]);
  api.mergeSuggestions.mockResolvedValue([]);
  api.latestJob.mockResolvedValue({});
  api.discover.mockResolvedValue({ total: 0 });
  api.semanticStatus.mockResolvedValue({ available: false, downloading: false, error: null });
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

describe("the cull amount control", () => {
  it("is the only mode now: no mode tabs, no selection-strictness slider", async () => {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));

    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });

    // Auto Target was removed on purpose; nothing should offer it back.
    expect(within(dialog).queryByRole("tablist")).toBeNull();
    expect(within(dialog).queryByRole("tab")).toBeNull();
    expect(within(dialog).queryByLabelText("Selection strictness: extreme to more"))
      .toBeNull();

    expect(within(dialog).getByLabelText("Target percentage")).toBeTruthy();
    expect(within(dialog).getByText(`~20% of ${TOTAL} photos`)).toBeTruthy();
  });

  it("sends a target even when the stored settings had none", async () => {
    // target_ratio null used to mean "let the gates decide"; the scan must now
    // run the ratio the dialog is showing, not silently fall back to gates-only.
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });
    api.scan.mockResolvedValue({ id: 1, status: "running", stage: "scan" });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });

    fireEvent.click(within(dialog).getByRole("button", { name: "Start Cull" }));

    await waitFor(() => expect(api.scan).toHaveBeenCalled());
    expect(api.scan.mock.calls[0][1].target_ratio).toBeCloseTo(0.2);
  });
});

describe("the dialog's guard rails", () => {
  async function openDialog() {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    fireEvent.click(cullSettings());
    return screen.findByRole("dialog", { name: "AI Automated Cull" });
  }

  it("makes re-analysing the whole shoot a two-tap action in the footer", async () => {
    const dialog = await openDialog();
    api.scan.mockResolvedValue({ id: 1, status: "running", stage: "scan" });

    fireEvent.click(within(dialog).getByRole("button", { name: "Re-analyse every frame" }));
    expect(api.scan).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole(
      "button", { name: "Confirm: re-analyse every frame" }));
    await waitFor(() => expect(api.scan).toHaveBeenCalledWith(1, expect.anything(), true));
  });

  it("bounds keep-at-least with a stepper instead of a free number", async () => {
    const dialog = await openDialog();
    const group = within(dialog).getByRole("group", { name: "Keep at least per group" });
    const fewer = within(group).getByRole("button", { name: "Fewer keepers per group" });
    const more = within(group).getByRole("button", { name: "More keepers per group" });

    expect((fewer as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(more);
    fireEvent.click(more);
    expect(group.textContent).toContain("3");
  });

  it("turns semantic dedup off with its parent, so nothing stuck on is hidden", async () => {
    const dialog = await openDialog();
    api.downloadSceneModel.mockResolvedValue(
      { available: false, downloading: true, error: null });

    fireEvent.click(within(dialog).getByLabelText("Detect Duplicates (AI)"));
    await waitFor(() => expect(api.downloadSceneModel).toHaveBeenCalledTimes(1));

    // The parent goes off and back on: the AI row must come back unchecked.
    fireEvent.click(within(dialog).getByLabelText("Detect Duplicates"));
    fireEvent.click(within(dialog).getByLabelText("Detect Duplicates"));
    const ai = within(dialog).getByLabelText("Detect Duplicates (AI)") as HTMLInputElement;
    expect(ai.checked).toBe(false);
  });

  it("shows the download in the footer while Customize is collapsed", async () => {
    const dialog = await openDialog();
    api.downloadSceneModel.mockResolvedValue(
      { available: false, downloading: true, done: 5 << 20, total: 42 << 20,
        error: null });

    fireEvent.click(within(dialog).getByLabelText("Detect Duplicates (AI)"));
    const progress = () => dialog.querySelector(".cull__foot-progress")?.textContent ?? "";
    await waitFor(() => expect(progress()).toContain("Downloading model"));
    expect(progress()).toContain("42 MB");
  });
});

describe("the target slider callout", () => {
  it("shows the helper as a callout while held, and returns it afterwards", async () => {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });

    const slider = within(dialog).getByLabelText("Target percentage");
    const staticHelp = dialog.querySelector(".cull__help")?.textContent ?? "";
    expect(staticHelp).toMatch(/target/i);
    expect(dialog.querySelector(".cull__callout")).toBeNull();

    fireEvent.pointerDown(slider);
    const callout = dialog.querySelector(".cull__callout");
    expect(callout).toBeTruthy();
    // Same copy, only the presentation changes.
    expect(callout?.textContent).toBe(staticHelp);

    fireEvent.pointerUp(slider);
    expect(dialog.querySelector(".cull__callout")).toBeNull();
    expect(dialog.querySelector(".cull__help")).toBeTruthy();
  });
});

describe("the Stars & Colors panel", () => {
  it("edits the scheme in place and the main row reads it back", async () => {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });

    fireEvent.click(within(dialog).getByLabelText("Edit star and colour ratings"));
    expect(within(dialog).getByText("Set Star/Color Ratings")).toBeTruthy();

    fireEvent.change(within(dialog).getByLabelText("Selected stars"),
                     { target: { value: "2" } });
    fireEvent.change(within(dialog).getByLabelText("Selected colour"),
                     { target: { value: "Blue" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    // Back on the cull panel, the row summarises what was saved -- it opens
    // export, not the editor, so a stale sentence would be misleading.
    expect(within(dialog).getByText(/Keepers get ★2 · Blue/)).toBeTruthy();

    // The back chevron walks out of the editor and only closes from the panel.
    fireEvent.click(within(dialog).getByLabelText("Edit star and colour ratings"));
    fireEvent.click(within(dialog).getByRole("button", { name: "Back" }));
    expect(within(dialog).getByText("AI Automated Cull")).toBeTruthy();
  });

  it("previews what each sidecar will say, per row, as the scheme is edited", async () => {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    fireEvent.click(within(dialog).getByLabelText("Edit star and colour ratings"));

    // One chip per row, in row order, reading the row's star and colour.
    const chips = () => [...dialog.querySelectorAll(".cull__chip")]
      .map((chip) => ({
        text: chip.textContent,
        bg: (chip as HTMLElement).style.background,
      }));
    expect(chips().map((chip) => chip.text)).toEqual(
      ["★1", "✗", "✗", "✗", "✗", "✗"]);
    expect(chips()[0].bg).toBe("rgb(63, 174, 106)");          // Green
    expect(chips()[1].bg).toBe("rgb(226, 86, 77)");           // Red

    // Editing a row moves its own chip and leaves the other five alone.
    fireEvent.change(within(dialog).getByLabelText("Closed Eyes stars"),
                     { target: { value: "0" } });
    fireEvent.change(within(dialog).getByLabelText("Closed Eyes colour"),
                     { target: { value: "Purple" } });
    const after = chips();
    expect(after[3]).toEqual({ text: "0 ★", bg: "rgb(155, 116, 248)" });
    expect(after.length).toBe(6);
    expect(after.filter((chip) => chip.text === "★1").length).toBe(1);

    // "No label" still previews the star; the pill just loses its fill.
    fireEvent.change(within(dialog).getByLabelText("Blurred colour"),
                     { target: { value: "" } });
    const bare = dialog.querySelectorAll(".cull__chip")[2] as HTMLElement;
    expect(bare.className).toContain("cull__chip--bare");
    expect(bare.style.background).toBe("");
    expect(bare.textContent).toBe("✗");

    // The preview is decoration: the selects already announce both values.
    expect(bare.getAttribute("aria-hidden")).toBe("true");
  });
});

describe("the semantic dedup toggle", () => {
  it("downloads the scene model on first use and holds the start until it lands", async () => {
    api.libraries.mockResolvedValue([library({ id: 1 })]);
    api.discover.mockResolvedValue({ total: TOTAL });
    api.downloadSceneModel.mockResolvedValue(
      { available: false, downloading: true, error: null });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.discover).toHaveBeenCalledWith(1));

    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    fireEvent.click(within(dialog).getByLabelText("Detect Duplicates (AI)"));

    // The toggle is the whole opt-in: one click downloads the weights, and the
    // cull cannot start until they are on disk.
    await waitFor(() => expect(api.downloadSceneModel).toHaveBeenCalledTimes(1));
    const start = await within(dialog).findByRole(
      "button", { name: "Downloading model…" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
  });
});

describe("a scan that could not read files", () => {
  const banner = () => (screen.queryByRole("button", { name: "View unreadable files" })
    ?.closest(".sheet-note") ?? null) as HTMLElement | null;

  async function openWithJob(job: Job, stats: Partial<Stats>) {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.photos.mockResolvedValue({ photos: [], total: TOTAL,
                                   stats: statsOf(TOTAL, stats) });
    api.latestJob.mockResolvedValue(job);
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.latestJob).toHaveBeenCalledWith(1));
  }

  it("says so above the grid, with the reason, and points at the files", async () => {
    await openWithJob(
      jobOf({ status: "error", error_count: TOTAL, error_sample:
        "UnidentifiedImageError: cannot identify image file 'SALR3900.CR3'" }),
      { reject: TOTAL, errors: TOTAL },
    );

    await waitFor(() => expect(banner()).not.toBeNull());
    const note = banner() as HTMLElement;
    expect(note.textContent).toContain(`None of the ${TOTAL} frames could be read.`);
    expect(note.textContent).toContain(
      "UnidentifiedImageError: cannot identify image file");
    expect(note.className).toContain("sheet-note--warning");

    fireEvent.click(within(note).getByRole("button", { name: "View unreadable files" }));
    await waitFor(() =>
      expect(api.photos).toHaveBeenLastCalledWith(1, "errors", "group", null));

    fireEvent.click(within(note).getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(banner()).toBeNull());
  });

  it("counts the failures when the rest of the shoot read fine", async () => {
    await openWithJob(jobOf({ error_count: 3, message: "40 keepers · 12 groups · 72 photos in 9.1s · 3 could not be read",
                              error_sample: "UnidentifiedImageError: cannot identify image file 'IMG_0007.CR3'" }),
                      { reject: 32, errors: 3 });

    await waitFor(() => expect(banner()).not.toBeNull());
    expect((banner() as HTMLElement).textContent).toContain(
      `3 of ${TOTAL} frames could not be read.`);
  });

  it("stays quiet on a shoot where every frame was read", async () => {
    await openWithJob(jobOf(), { keep: TOTAL });

    await waitFor(() => expect(api.latestJob).toHaveBeenCalledWith(1));
    expect(banner()).toBeNull();
  });
});

describe("exporting a shoot that kept nothing", () => {
  async function openExport(stats: Partial<Stats>) {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.photos.mockResolvedValue({ photos: [], total: TOTAL,
                                   stats: statsOf(TOTAL, stats) });
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    return screen.findByRole("dialog", { name: "Export decisions" });
  }

  const copyRow = (dialog: HTMLElement) =>
    within(dialog).getByRole("radio", { name: /Copy keepers to a folder/ }) as HTMLInputElement;

  it("says how many keepers there are and refuses to copy none of them", async () => {
    const dialog = await openExport({ reject: TOTAL, errors: TOTAL });

    expect(within(dialog).getByText(
      new RegExp(`^0 keepers, ${TOTAL} rejected, ${TOTAL} unreadable$`))).toBeTruthy();
    expect(copyRow(dialog).disabled).toBe(true);
    expect(within(dialog).getByText(/No keepers to copy/)).toBeTruthy();
    // And the panel is not sitting on a mode it cannot run.
    expect(within(dialog).getByRole("button", { name: "Write sidecars" })).toBeTruthy();
  });

  it("warns that sidecars would record decisions nothing made", async () => {
    const dialog = await openExport({ reject: TOTAL, errors: TOTAL });

    // Both sidecar modes say it: they are the two that write a verdict.
    expect(within(dialog).getAllByText(/were never evaluated/)).toHaveLength(2);
    // The report is still offered: a list of what could not be read is useful.
    const report = within(dialog).getByRole("radio", { name: /Culling report/ });
    expect((report as HTMLInputElement).disabled).toBe(false);
  });

  it("keeps the sidecars quiet on a shoot that read fine, keepers or not", async () => {
    const dialog = await openExport({ keep: 0, reject: TOTAL });

    expect(within(dialog).queryByText(/were never evaluated/)).toBeNull();
    expect(within(dialog).getByText(
      new RegExp(`^0 keepers, ${TOTAL} rejected$`))).toBeTruthy();
  });

  it("offers the copy flow when there is something to copy", async () => {
    api.export.mockResolvedValue({ written: 0, copied: 40, skipped: 0,
                                   dest: "/Users/you/Deliveries/Katie", errors: [] });
    const dialog = await openExport({ keep: 40, reject: 32 });

    expect(within(dialog).getByText(/^40 keepers, 32 rejected$/)).toBeTruthy();
    expect(window.localStorage.getItem("cullr.export.last-destination")).toBeNull();
    const copy = copyRow(dialog);
    expect(copy.disabled).toBe(false);
    fireEvent.click(copy);

    // The destination field appears with the copy mode; the export runs and the
    // panel reports what happened instead of a bare "Nothing to export".
    fireEvent.change(within(dialog).getByLabelText("Destination folder"),
                     { target: { value: "/Users/you/Deliveries/Katie" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Copy to folder" }));

    await waitFor(() => expect(api.export).toHaveBeenCalledWith(
      1, "copy", "/Users/you/Deliveries/Katie"));
    expect(await within(dialog).findByText("40 copied")).toBeTruthy();
    // A destination that was actually exported to becomes the next shoot's
    // default; that is the memory the picker falls back on.
    expect(window.localStorage.getItem("cullr.export.last-destination"))
      .toBe("/Users/you/Deliveries/Katie");
  });

  it("hands the folder the picker chose back to the panel", async () => {
    api.folders.mockResolvedValue({
      path: "/tmp/cullr-e2e", parent: "/tmp", home: "/Users/you", writable: true,
      folders: [{ name: "delivery", path: "/tmp/cullr-e2e/delivery" }],
      entries: [{ name: "delivery", path: "/tmp/cullr-e2e/delivery", kind: "folder" }],
      truncated: false,
      sidebar: { favorites: [], shared: [], locations: [] },
    });
    api.export.mockResolvedValue({ written: 0, copied: 3, skipped: 0,
                                   dest: "/tmp/cullr-e2e/delivery", errors: [] });
    const dialog = await openExport({ keep: 40, reject: 32 });

    fireEvent.click(copyRow(dialog));
    fireEvent.click(within(dialog).getByRole("button", { name: "Select a folder" }));

    const picker = await screen.findByRole("dialog", { name: "Copy keepers to" });
    fireEvent.click(within(picker).getByRole("button", { name: /delivery/ }));
    fireEvent.click(within(picker).getByRole("button", { name: "Use this folder" }));

    // The picker closes and the panel keeps the choice -- the step the video
    // never reached, so it could neither confirm nor deny this.
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Copy keepers to" })).toBeNull());
    expect((within(dialog).getByLabelText("Destination folder") as HTMLInputElement).value)
      .toBe("/tmp/cullr-e2e/delivery");

    fireEvent.click(within(dialog).getByRole("button", { name: "Copy to folder" }));
    await waitFor(() => expect(api.export).toHaveBeenCalledWith(
      1, "copy", "/tmp/cullr-e2e/delivery"));
    expect(await within(dialog).findByText("3 copied")).toBeTruthy();
  });

  it("starts the picker away from the shoot folder", async () => {
    api.folders.mockResolvedValue({
      path: "/Users/you", parent: "/Users", home: "/Users/you", writable: true,
      folders: [], entries: [], truncated: false,
      sidebar: { favorites: [], shared: [], locations: [] },
    });
    const dialog = await openExport({ keep: 40, reject: 32 });

    fireEvent.click(copyRow(dialog));
    fireEvent.click(within(dialog).getByRole("button", { name: "Select a folder" }));

    // No destination remembered for this shoot, so the picker opens at Home --
    // never inside /shoots/Teasers, which is where the video's copy would have
    // put the delivery.
    await waitFor(() => expect(api.folders).toHaveBeenCalledWith(undefined));
  });
});

describe("the unreadable filter", () => {
  const sidebar = () => screen.getByRole("navigation", { name: "Review filters" });

  async function openScannedLibrary(stats: Partial<Stats>) {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.photos.mockResolvedValue({ photos: [], total: TOTAL,
                                   stats: statsOf(TOTAL, stats) });
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
  }

  it("offers a row for files the scan could not read, and lists exactly those", async () => {
    await openScannedLibrary({ reject: TOTAL, errors: TOTAL });

    const row = await within(sidebar()).findByRole(
      "button", { name: /Unreadable/ }) as HTMLButtonElement;
    // The row sits under Rejected: that is where a user looks for why frames
    // were dropped, and "unreadable" is a different answer from "soft".
    const labels = [...sidebar().querySelectorAll(".filter span:first-child")]
      .map((span) => span.textContent);
    expect(labels.indexOf("Unreadable")).toBe(labels.indexOf("Rejected") + 1);
    expect(within(row).getByText(String(TOTAL))).toBeTruthy();

    fireEvent.click(row);
    await waitFor(() =>
      expect(api.photos).toHaveBeenLastCalledWith(1, "errors", "group", null));
  });

  it("stays out of the way on a shoot where every frame was readable", async () => {
    await openScannedLibrary({ keep: TOTAL });

    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
    expect(within(sidebar()).queryByRole("button", { name: /Unreadable/ })).toBeNull();
    expect(within(sidebar()).getByRole("button", { name: /Rejected/ })).toBeTruthy();
  });
});

describe("the grid while a scan runs", () => {
  async function startScan(job: Job, listPhotos: () => Photo[]) {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.discover.mockResolvedValue({ total: TOTAL });
    api.photos.mockImplementation(() => {
      const photos = listPhotos();
      return Promise.resolve({ photos, total: photos.length, stats: statsOf(TOTAL) });
    });
    api.scan.mockResolvedValue(job);
    api.job.mockResolvedValue(job);

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Start Cull" }));
  }

  it("says what the scan is doing instead of claiming nothing matches", async () => {
    await startScan(
      jobOf({ status: "running", stage: "analyzing", done: 12,
              message: "Analysed 12 of 72 photos" }),
      () => [],
    );

    expect(await screen.findByText(
      /Analysed 12 of 72 photos — frames appear here as they are checked\./,
    )).toBeTruthy();
    // The message this replaces: the filter is "All frames", the shoot has 72
    // photos, and nothing has been filtered out — the rows are not there yet.
    expect(screen.queryByText("No frames match this filter.")).toBeNull();
  });

  it("fills the grid in mid-scan instead of waiting for the last frame", async () => {
    const shown: Photo[] = [];
    await startScan(
      jobOf({ status: "running", stage: "analyzing", done: 3,
              message: "Analysed 3 of 72 photos" }),
      () => shown,
    );

    expect(await screen.findByText(/frames appear here/)).toBeTruthy();
    // The job never leaves "running" here, so tiles can only come from the
    // slower list refresh — the difference between a grid that fills in and one
    // that appears all at once when the scan ends.
    shown.push(photoOf({ id: 1, filename: "IMG_0001.jpg" }));
    await waitFor(
      () => expect(screen.getByRole("button", { name: /IMG_0001\.jpg/ })).toBeTruthy(),
      { timeout: 4000 },
    );
  });

  it("goes back to no-frames-match once the scan has finished", async () => {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.discover.mockResolvedValue({ total: TOTAL });
    api.photos.mockResolvedValue({ photos: [], total: 0, stats: statsOf(TOTAL) });
    const running = jobOf({ status: "running", stage: "analyzing", done: 70,
                            message: "Analysed 70 of 72 photos" });
    api.scan.mockResolvedValue(running);
    api.job.mockResolvedValueOnce(running)
      .mockResolvedValue(jobOf({ message: "0 keepers · 0 groups · 72 photos in 9.0s" }));

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
    fireEvent.click(cullSettings());
    const dialog = await screen.findByRole("dialog", { name: "AI Automated Cull" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Start Cull" }));

    expect(await screen.findByText(/frames appear here/)).toBeTruthy();
    // Two 650 ms job polls away, so this needs more than the default timeout.
    expect(await screen.findByText("No frames match this filter.",
                                   {}, { timeout: 4000 })).toBeTruthy();
  });
});

describe("the folder picker's search", () => {
  const entry = (name: string, kind: "folder" | "file" = "folder"): FolderEntry => ({
    name,
    path: `/shoots/Katie/${name}`,
    kind,
  });
  const pickerListing = (over: Partial<FolderListing> = {}): FolderListing => ({
    path: "/shoots/Katie",
    parent: "/shoots",
    home: "/Users/you",
    writable: true,
    folders: [],
    entries: [],
    truncated: false,
    limit: 400,
    sidebar: { favorites: [], shared: [], locations: [] },
    ...over,
  });

  async function openPicker() {
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: TOTAL })]);
    api.photos.mockResolvedValue({
      photos: [], total: TOTAL, stats: { ...statsOf(TOTAL), keep: 40, reject: 32 },
    });
    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));
    await waitFor(() => expect(api.photos).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    const panel = await screen.findByRole("dialog", { name: "Export decisions" });
    fireEvent.click(within(panel).getByRole("radio", { name: /Copy keepers/ }));
    fireEvent.click(within(panel).getByRole("button", { name: "Select a folder" }));
    return screen.findByRole("dialog", { name: "Copy keepers to" });
  }

  it("asks the server, so a name past the entry window is still findable", async () => {
    // The window holds the first 400 names and "zzz-delivery" is not among
    // them: filtering what was sent could never find it.
    api.folders
      .mockResolvedValueOnce(pickerListing({
        truncated: true,
        entries: [entry("shoot_0001"), entry("shoot_0002")],
      }))
      .mockResolvedValueOnce(pickerListing({ entries: [entry("zzz-delivery")] }));

    const picker = await openPicker();
    fireEvent.change(within(picker).getByLabelText("Search this folder"),
                     { target: { value: "delivery" } });

    await waitFor(() => expect(api.folders).toHaveBeenLastCalledWith(
      "/shoots/Katie", "delivery"));
    expect(await within(picker).findByRole("button", { name: /zzz-delivery/ }))
      .toBeTruthy();
  });

  it("says the search reaches the whole folder, not just the shown names", async () => {
    api.folders.mockResolvedValue(pickerListing({
      truncated: true,
      entries: [entry("shoot_0001")],
    }));

    const picker = await openPicker();
    expect(await within(picker).findByText(
      "Showing the first 400 names — search covers the whole folder.")).toBeTruthy();
  });
});

describe("a frame the scan could not read", () => {
  // Block-scoped so the suite keeps one module-level helper per shape.
  const frame = (filename: string, over: Partial<Photo> = {}): Photo => ({
    id: 1,
    filename,
    path: `/shoots/Katie/${filename}`,
    width: null,
    height: null,
    captured_at: null,
    camera: null,
    lens: null,
    iso: null,
    shutter: null,
    aperture: null,
    focal_len: null,
    quality: null,
    verdict: "keep",
    ai_verdict: null,
    reason: null,
    user_decision: null,
    star: 0,
    group_id: null,
    is_lead: false,
    covered_person: null,
    sharpness: null,
    exposure: null,
    clip_hi: null,
    clip_lo: null,
    noise: null,
    face_count: 0,
    eyes_open: null,
    blink: false,
    error: null,
    faces: [],
    ...over,
  });

  it("names the reason in its tooltip instead of only in the loupe", async () => {
    const reason = "UnidentifiedImageError: cannot identify image file 'SALR3900.CR3'";
    const broken = frame("SALR3900.CR3", { id: 1, error: reason, reason: "unreadable",
                                           verdict: "reject" });
    const fine = frame("IMG_0002.jpg", { id: 2 });
    api.libraries.mockResolvedValue([library({ id: 1, photo_count: 2 })]);
    api.photos.mockResolvedValue({
      photos: [broken, fine], total: 2,
      stats: { ...statsOf(2), keep: 1, reject: 1 },
    });

    await renderApp();
    fireEvent.click(screen.getByRole("button", { name: /Teasers/ }));

    const tile = await screen.findByRole("button", { name: /SALR3900\.CR3/ });
    expect(tile.getAttribute("title")).toBe(`SALR3900.CR3 — ${reason}`);
    // A readable frame still just says where it lives.
    const good = screen.getByRole("button", { name: /IMG_0002\.jpg/ });
    expect(good.getAttribute("title")).toBe(fine.path);
  });
});

describe("the in-app update entry", () => {
  it("checks nothing until the user asks, then opens the manual panel", async () => {
    await renderApp();

    // No auto-check: the app has not touched the update endpoints on its own.
    expect(api.checkForUpdate).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Check for updates/ }));

    const dialog = await screen.findByRole("dialog", { name: "Check for updates" });
    expect(within(dialog).getByText(/0\.8\.0/)).toBeTruthy();
    expect(api.checkForUpdate).not.toHaveBeenCalled();

    // The check is the button in the panel, and nothing else.
    fireEvent.click(within(dialog).getByRole("button", { name: "Check for updates" }));
    await waitFor(() => expect(api.checkForUpdate).toHaveBeenCalledTimes(1));
  });
});
