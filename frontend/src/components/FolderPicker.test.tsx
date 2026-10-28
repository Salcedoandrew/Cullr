import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FolderPicker } from "./FolderPicker";
import type { FolderListing } from "../types";

/**
 * New Folder used to be an editable row inside the browser grid, with no
 * Cancel and no statement of where the folder would land. It is now the
 * system's prompt: "Name of new folder inside <parent>", Cancel / Create.
 * These tests drive it through the picker the export panel opens.
 */

const api = vi.hoisted(() => ({
  folders: vi.fn(),
  createFolder: vi.fn(),
  folderPreviewUrl: (path: string) => `/api/folders/preview?path=${encodeURIComponent(path)}`,
}));

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, api, folderPreviewUrl: api.folderPreviewUrl };
});

const listing = (over: Partial<FolderListing> = {}): FolderListing => ({
  path: "/Users/you/Documents",
  parent: "/Users/you",
  home: "/Users/you",
  writable: true,
  folders: [],
  entries: [],
  sidebar: { favorites: [], shared: [], locations: [] },
  truncated: false,
  ...over,
});

// jsdom does not implement scrollIntoView, which the picker calls when a
// selection lands; the app never sees that gap in a real browser.
Object.defineProperty(Element.prototype, "scrollIntoView", {
  value: vi.fn(),
  configurable: true,
});

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

const openPicker = async () => {
  api.folders.mockResolvedValue(listing());
  render(
    <FolderPicker
      title="Copy keepers to"
      actionLabel="Use this folder"
      onClose={vi.fn()}
      onPick={vi.fn()}
    />,
  );
  return screen.findByRole("dialog", { name: "Copy keepers to" });
};

const openPrompt = async () => {
  const picker = await openPicker();
  fireEvent.click(within(picker).getByRole("button", { name: "New Folder" }));
  return screen.findByRole("dialog", { name: "New Folder" });
};

describe("the New Folder prompt", () => {
  it("names the folder the new folder will live inside", async () => {
    const prompt = await openPrompt();
    // "Documents" is a <span>-free text node inside the sentence, so match
    // the sentence's shape rather than a single text node.
    expect(within(prompt).getByText(/Name of new folder inside/)).toBeTruthy();
    expect(within(prompt).getByText(/Documents/)).toBeTruthy();
    expect(
      (within(prompt).getByLabelText("New folder name") as HTMLInputElement).value,
    ).toBe("untitled folder");
  });

  it("Cancel closes the prompt and creates nothing", async () => {
    const prompt = await openPrompt();
    fireEvent.click(within(prompt).getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New Folder" })).toBeNull());
    expect(api.createFolder).not.toHaveBeenCalled();
  });

  it("Create makes the folder, then selects it in the picker", async () => {
    api.createFolder.mockResolvedValue({
      path: "/Users/you/Documents/delivery",
      name: "delivery",
    });
    api.folders.mockResolvedValueOnce(listing());
    // Second listing, after creation, contains the new folder.
    api.folders.mockResolvedValueOnce(
      listing({
        entries: [
          { name: "delivery", path: "/Users/you/Documents/delivery", kind: "folder", items: 0 },
        ],
      }),
    );
    const prompt = await openPrompt();
    const field = within(prompt).getByLabelText("New folder name") as HTMLInputElement;
    fireEvent.change(field, { target: { value: "delivery" } });
    fireEvent.click(within(prompt).getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(api.createFolder).toHaveBeenCalledWith("/Users/you/Documents", "delivery"));
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New Folder" })).toBeNull());
    const picker = await screen.findByRole("dialog", { name: "Copy keepers to" });
    const created = within(picker).getByRole("button", { name: /delivery/ });
    expect(created.className).toContain("is-selected");
  });

  it("a failed create keeps the prompt open with the error", async () => {
    api.createFolder.mockRejectedValue(new Error("a folder named that already exists"));
    const prompt = await openPrompt();
    fireEvent.change(within(prompt).getByLabelText("New folder name"), {
      target: { value: "delivery" },
    });
    fireEvent.click(within(prompt).getByRole("button", { name: "Create" }));

    expect(await within(prompt).findByText(/already exists/)).toBeTruthy();
    expect(screen.getByRole("dialog", { name: "New Folder" })).toBeTruthy();
  });
});
