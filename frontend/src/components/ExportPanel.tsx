import { useState } from "react";
import { api } from "../api";
import type { ExportMemory, Stats } from "../types";
import { FolderPicker } from "./FolderPicker";

interface Props {
  libraryId: number;
  /** The scan's counts. The panel needs them: "export to a folder" is
   *  meaningless on a shoot with no keepers, and sidecars are worse than
   *  meaningless when nothing could be read. */
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

/* The panel offers two methods only: a folder copy, and Lightroom Classic.
   The backend and the headless CLI still offer XMP sidecars and a CSV report
   directly, but they are no longer rows in this panel. And "Lightroom Classic"
   here is deliberately not a .lrcat writer: Lightroom's catalog format is
   undocumented and versioned, and writing it by hand risks a catalog
   Lightroom refuses or, worse, quietly repairs. The interchange Aftershoot
   itself documents is XMP sidecars -- Lightroom reads them on import, or on
   Metadata -> Read Metadata from Files for photos already imported -- so
   this action writes sidecars for every frame (the sidecar_all mode) and
   says exactly that, rather than pretending a catalog was written. */
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
  const [dest, setDest] = useState(remembered?.dest ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);

  const run = async (mode: "copy" | "sidecar_all", destArg?: string) => {
    setBusy(true);
    setError(null);
    try {
      const response = await api.export(libraryId, mode, destArg || undefined);
      const parts = [
        response.written ? `${response.written} written` : null,
        response.copied ? `${response.copied} copied` : null,
        response.skipped ? `${response.skipped} skipped` : null,
      ].filter(Boolean);
      setResult(parts.join(", ") || "Nothing to export");
      if (response.errors.length) setError(response.errors[0]);
      if (mode === "copy") {
        // The server stores this too; mirror it so the hint and a reopened
        // panel agree inside this session.
        onRemembered?.({ mode, dest: (destArg ?? "").trim() });
        // Only a destination that was actually exported to becomes the
        // default for the next shoot: an abandoned pick should not move it.
        if (destArg?.trim() && !response.errors.length) {
          rememberLastDestination(destArg.trim());
        }
      }
      // A Lightroom run must not overwrite the remembered folder destination:
      // it has none, and the next folder export should still open where the
      // last folder export went.
    } catch (event) {
      setError(event instanceof Error ? event.message : String(event));
    } finally {
      setBusy(false);
    }
  };

  const title = total > 0 ? `Export ${total} Photos` : "Export Photos";

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
      <div className="modal__body export-panel">
        <header className="export-panel__head">
          <div>
            <h2>{title}</h2>
            {stats && (
              <p className="export-panel__counts">
                {keepers} keeper{keepers === 1 ? "" : "s"}, {rejected} rejected
                {stats.errors > 0 ? `, ${stats.errors} unreadable` : ""}
              </p>
            )}
          </div>
          <button type="button" className="export-panel__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <p className="export-panel__intro">Choose a method to export</p>

        <section className="export-panel__section">
          <h3>Export to a folder</h3>
          <p>Choose where you want to move / copy your photos</p>
          <div className="export-panel__field">
            <label htmlFor="export-dest">Destination folder</label>
            <div className="export-panel__fieldrow">
              <input
                id="export-dest"
                value={dest}
                onChange={(event) => setDest(event.target.value)}
                placeholder="/Users/you/Pictures/delivery"
                spellCheck={false}
              />
              <button
                type="button"
                className="export-panel__select"
                onClick={() => setPicking(true)}
              >
                Select a folder
              </button>
            </div>
            <p className="export-panel__hint">
              Every keeper is copied here; originals stay where they are.
              {remembered?.dest === dest.trim() && dest.trim() !== ""
                ? " Prefilled from this shoot's last export."
                : null}
            </p>
            {!hasKeepers && (
              <p className="export-panel__warn">
                No keepers to copy — this cull kept 0 frames.
              </p>
            )}
          </div>
          <button
            type="button"
            className="export-panel__primary"
            onClick={() => void run("copy", dest)}
            disabled={busy || !dest.trim() || !hasKeepers}
          >
            {busy ? "Working…" : "Export to a folder"}
          </button>
        </section>

        <section className="export-panel__section">
          <h3>One click export</h3>
          <p>Already have these photos in Lightroom Classic?</p>
          <button
            type="button"
            className="export-panel__tile"
            onClick={() => void run("sidecar_all")}
            disabled={busy || nothingRead}
          >
            <strong>Lightroom Classic</strong>
          </button>
          <p className="export-panel__hint">
            Writes XMP sidecars next to every frame with Cullr&apos;s ratings and
            labels. In Lightroom Classic, import the shoot — or, if it is already
            imported, choose Metadata → Read Metadata from Files. This does not
            write Lightroom&apos;s catalog file itself.
          </p>
          {nothingRead && (
            <p className="export-panel__warn">
              Nothing in this shoot could be read, so these sidecars would
              mark frames that were never evaluated. Fix the unreadable
              files first.
            </p>
          )}
        </section>

        {picking && (
          <FolderPicker
            title="Copy keepers to"
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

        {result && <p className="export-panel__result">{result}</p>}
        {error && <p className="export-panel__error">{error}</p>}
      </div>
    </div>
  );
}
