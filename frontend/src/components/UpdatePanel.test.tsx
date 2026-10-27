import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UpdatePanel } from "./UpdatePanel";
import type { UpdateStatus } from "../types";

/**
 * The five states the update flow has to keep apart — nothing to do, something
 * available, downloading, downloaded and verified, failed — because they mean
 * different things to the person reading them. The panel is driven here through
 * the same api module the app uses.
 */

const api = vi.hoisted(() => ({
  updateStatus: vi.fn(),
  checkForUpdate: vi.fn(),
  downloadUpdate: vi.fn(),
  cancelUpdate: vi.fn(),
  installUpdate: vi.fn(),
}));

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, api };
});

const state = (over: Partial<UpdateStatus> = {}): UpdateStatus => ({
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
  update_log: "/Users/you/Library/Application Support/Cullr/logs/update.log",
  ...over,
});

const open = (over: Partial<UpdateStatus> = {}, onClose = vi.fn()) => {
  api.updateStatus.mockResolvedValue(state(over));
  render(<UpdatePanel current="0.8.0" onClose={onClose} />);
  return onClose;
};

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

describe("the update panel", () => {
  it("shows the running version and asks before it checks anything", async () => {
    open();
    const dialog = await screen.findByRole("dialog", { name: "Check for updates" });
    expect(within(dialog).getByText(/0\.8\.0/)).toBeTruthy();
    expect(api.checkForUpdate).not.toHaveBeenCalled();
  });

  it("reports no update without offering anything to install", async () => {
    api.checkForUpdate.mockResolvedValue(state({ phase: "up_to_date", latest: "0.8.0" }));
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText(/No updates available/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Install/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Check again" })).toBeTruthy();
  });

  it("reports how a check failed, and where the installer logs", async () => {
    api.checkForUpdate.mockResolvedValue(state({
      phase: "error",
      error: "could not check for updates: nothing published yet (HTTP 404)",
    }));
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText(/HTTP 404/)).toBeTruthy();
    expect(screen.getByText(/Nothing was installed/)).toBeTruthy();
    expect(screen.getByText(/update\.log/)).toBeTruthy();
  });

  it("asks for confirmation, then counts the bytes it is downloading", async () => {
    api.checkForUpdate.mockResolvedValue(state({
      phase: "available", latest: "0.9.0", size: 168_900_000,
      notes_url: "https://github.com/Salcedoandrew/cullr/releases/tag/v0.9.0",
    }));
    api.downloadUpdate.mockResolvedValue(state({
      phase: "downloading", latest: "0.9.0", done: 84_450_000, total: 168_900_000,
      percent: 50,
    }));
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    // The version is a <b> inside the sentence, so match the sentence's text.
    expect(await screen.findByText(/is available/)).toBeTruthy();
    expect(screen.getByText("0.9.0")).toBeTruthy();
    const notes = screen.getByRole("link", { name: "Release notes" }) as HTMLAnchorElement;
    expect(notes.href).toContain("v0.9.0");

    fireEvent.click(screen.getByRole("button", { name: "Download update" }));
    await waitFor(() => expect(api.downloadUpdate).toHaveBeenCalledTimes(1));
    // 16.1 MB downloaded of a 168.9 MB download, at the byte counts the server reports.
    expect(await screen.findByText(/80\.5 MB of 161\.1 MB · 50%/)).toBeTruthy();
  });

  it("cancelling a download leaves the installed version alone", async () => {
    api.checkForUpdate.mockResolvedValue(state({ phase: "available", latest: "0.9.0" }));
    api.downloadUpdate.mockResolvedValue(state({ phase: "downloading", latest: "0.9.0" }));
    api.cancelUpdate.mockResolvedValue(state({ phase: "cancelled", latest: "0.9.0" }));
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));
    fireEvent.click(await screen.findByRole("button", { name: "Download update" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel download" }));

    expect(await screen.findByText(/cancelled and deleted/)).toBeTruthy();
    expect(screen.getByText(/0\.8\.0 is still/)).toBeTruthy();
    expect(api.installUpdate).not.toHaveBeenCalled();
  });

  it("installs only after the verified download, naming the app it replaces", async () => {
    api.checkForUpdate.mockResolvedValue(state({ phase: "available", latest: "0.9.0" }));
    api.downloadUpdate.mockResolvedValue(state({ phase: "ready", latest: "0.9.0" }));
    api.installUpdate.mockResolvedValue(state({ phase: "installing", latest: "0.9.0" }));
    const onClose = open();
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));
    fireEvent.click(await screen.findByRole("button", { name: "Download update" }));

    const dialog = screen.getByRole("dialog");
    expect(await within(dialog).findByText(/SHA-256 matches/)).toBeTruthy();
    expect(within(dialog).getAllByText(/\/Applications\/Cullr\.app/).length).toBeGreaterThan(0);

    fireEvent.click(within(dialog).getByRole("button", { name: "Install and restart" }));
    await waitFor(() => expect(api.installUpdate).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/Restarting on Cullr 0\.9\.0/)).toBeTruthy();

    // Once it is installing, Escape must not pretend to cancel the install.
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes on Escape before anything is installed", async () => {
    const onClose = open();
    await screen.findByRole("dialog");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
