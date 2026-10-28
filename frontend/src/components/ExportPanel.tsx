import { useState } from "react";
import { api } from "../api";
import type { ExportMemory, Stats } from "../types";
import { FolderPicker } from "./FolderPicker";

interface Props {
  libraryId: number;
  /** The scan's counts. The panel needs them: "copy keepers" is meaningless on
   *  a shoot with none, and sidecars are worse than meaningless when nothing
   *  could be read. */
  stats: Stats | null;
  remembered?: ExportMemory | null;
  onRemembered?: (value: ExportMemory) => void;
  onClose: () => void;
}

/* The destination a photographer last exported to, kept across shoots. Without
   it the picker falls back to Home, which is at least never the shoot folder
   itself -- copying keepers back into the folder they came from is almost never
   what was wanted. Same localStorage pattern the picker uses for its recents. */
const LAST_DEST_KEY = "cullr.export.last-destination";

function readLastDestination(): string {
  try {
    return window.localStorage.getItem(LAST_DEST_KEY) ?? "";
  } catch {
    return "";
  }
}

function rememberLastDestination(path: string): void {
  try {
    window.localStorage.setItem(LAST_DEST_KEY, path);
  } catch {
    /* storage may be unavailable; the default destination is a convenience */
  }
}

type Mode = "sidecar" | "sidecar_all" | "copy" | "csv";

const MODES: { key: Mode; label: string; help: string; needsDest: boolean }[] = [
  {
    key: "sidecar",
    label: "XMP sidecars for RAW files",
    help: "Writes <file>.xmp next to each RAW. Ratings travel with the file into Lightroom or Capture One.",
    needsDest: false,
  },
  {
    key: "sidecar_all",
    label: "XMP sidecars for every file",
    help: "Same as above, including JPEG and TIFF. Only useful if your editor reads external sidecars for those formats.",
    needsDest: false,
  },
  {
    key: "copy",
    label: "Copy keepers to a folder",
    help: "Copies every kept frame, leaving the originals untouched.",
    needsDest: true,
  },
  {
    key: "csv",
    label: "Culling report (CSV)",
    help: "One row per frame with the decision, reason and measured quality.",
    needsDest: true,
  },
];

export function ExportPanel({
  libraryId, stats, remembered, onRemembered, onClose,
}: Props) {
  const keepers = stats?.keep ?? 0;
  const rejected = stats?.reject ?? 0;
  const total = stats?.total ?? 0;
  const hasKeepers = keepers > 0;
  /** Every frame failed to read: not one was judged, so a sidecar for them
   *  records a decision the app never made. */
  const nothingRead = total > 0 && (stats?.errors ?? 0) >= total;
  const [lastDestination] = useState(readLastDestination);

  // The catalog carries the last export for this shoot, so a different browser
  // (or a restart) opens on the same destination.
  const rememberedMode = MODES.find((item) => item.key === remembered?.mode)?.key;
  const [mode, setMode] = useState<Mode>(
    // A remembered "copy" on a shoot that kept nothing would open on a mode it
    // cannot run, so the panel opens on the sidecars instead.
    rememberedMode === "copy" && keepers === 0 ? "sidecar" : rememberedMode ?? "sidecar",
  );
  const [dest, setDest] = useState(remembered?.dest ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);
  const selected = MODES.find((item) => item.key === mode)!;
  const isCopy = mode === "copy";
  const actionLabel = isCopy ? "Copy to folder" : mode === "csv" ? "Write report" : "Write sidecars";

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await api.export(libraryId, mode, dest || undefined);
      const parts = [
        response.written ? `${response.written} written` : null,
        response.copied ? `${response.copied} copied` : null,
        response.skipped ? `${response.skipped} skipped` : null,
      ].filter(Boolean);
      setResult(parts.join(", ") || "Nothing to export");
      if (response.errors.length) setError(response.errors[0]);
      // The server stores this too; mirror it so the hint and a reopened panel
      // agree inside this session.
      onRemembered?.({ mode, dest: dest.trim() });
      // Only a destination that was actually exported to becomes the default
      // for the next shoot: an abandoned pick should not move it.
      if (dest.trim() && !response.errors.length) rememberLastDestination(dest.trim());
    } catch (event) {
      setError(event instanceof Error ? event.message : String(event));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Export decisions">
      <div className="modal__body">
        <header className="modal__head">
          <h2>Export decisions</h2>
          <button type="button" className="loupe__close loupe__close--inline" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {stats && (
          <p className="modal__hint export-counts">
            {keepers} keeper{keepers === 1 ? "" : "s"}, {rejected} rejected
            {stats.errors > 0 ? `, ${stats.errors} unreadable` : ""}
          </p>
        )}

        <fieldset className="export-modes">
          {MODES.map((item) => {
            // Copying nothing used to look exactly like copying 3,000 keepers:
            // the mode ran, reported "Nothing to export", and left the user to
            // work out why.
            const empty = item.key === "copy" && !hasKeepers;
            const unjudged = item.key !== "copy" && item.key !== "csv" && nothingRead;
            return (
              <label key={item.key}
                     className={mode === item.key ? "export-mode is-on" : "export-mode"}>
                <input
                  type="radio"
                  name="mode"
                  value={item.key}
                  checked={mode === item.key}
                  onChange={() => setMode(item.key)}
                  disabled={empty}
                />
                <span>
                  <strong>{item.label}</strong>
                  <em>{item.help}</em>
                  {empty && (
                    <em className="export-mode__warn">
                      No keepers to copy — this cull kept 0 frames.
                    </em>
                  )}
                  {unjudged && (
                    <em className="export-mode__warn">
                      Nothing in this shoot could be read, so these sidecars would
                      mark frames that were never evaluated. Fix the unreadable
                      files first.
                    </em>
                  )}
                </span>
              </label>
            );
          })}
        </fieldset>

        {selected.needsDest && (
          <div className="modal__field">
            <label htmlFor="export-dest">
              {mode === "copy" ? "Destination folder" : "Destination"}
            </label>
            <div className="field-with-button">
              <input
                id="export-dest"
                value={dest}
                onChange={(event) => setDest(event.target.value)}
                placeholder={mode === "copy" ? "/Users/you/Pictures/delivery" : "/Users/you/Desktop"}
                spellCheck={false}
              />
              <button
                type="button"
                className="button field-with-button__action"
                onClick={() => setPicking(true)}
              >
                Select a folder
              </button>
            </div>
            <p className="modal__hint">
              {mode === "copy"
                ? "Every keeper is copied here; originals stay where they are."
                : "The report is written into this folder as culling-report.csv."}
              {remembered?.dest === dest.trim() && dest.trim() !== ""
                ? " Prefilled from this shoot's last export."
                : null}
            </p>
          </div>
        )}

        <div className="modal__actions">
          <button
            type="button"
            className="button button--primary"
            onClick={() => void run()}
            disabled={busy || (selected.needsDest && !dest.trim())
              || (mode === "copy" && !hasKeepers)}
          >
            {busy ? "Working…" : actionLabel}
          </button>
        </div>

        {picking && (
          <FolderPicker
            title={isCopy ? "Copy keepers to" : "Write the report in"}
            // Not the shoot folder: the destination is a delivery folder, and
            // the last one used beats guessing. Home is the fallback, so the
            // picker never opens inside the shoot by accident.
            startPath={dest.trim() || lastDestination || undefined}
            actionLabel="Use this folder"
            onClose={() => setPicking(false)}
            onPick={(path) => {
              setDest(path);
              setPicking(false);
              setResult(null);
            }}
          />
        )}

        {result && <p className="modal__result">{result}</p>}
        {error && <p className="modal__error">{error}</p>}
      </div>
    </div>
  );
}
