import { useState } from "react";
import { api } from "../api";
import type { ExportMemory } from "../types";
import { FolderPicker } from "./FolderPicker";

interface Props {
  libraryId: number;
  libraryPath?: string;
  remembered?: ExportMemory | null;
  onRemembered?: (value: ExportMemory) => void;
  onClose: () => void;
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
  libraryId, libraryPath, remembered, onRemembered, onClose,
}: Props) {
  // The catalog carries the last export for this shoot, so a different browser
  // (or a restart) opens on the same destination.
  const [mode, setMode] = useState<Mode>(
    MODES.find((item) => item.key === remembered?.mode)?.key ?? "sidecar",
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

        <fieldset className="export-modes">
          {MODES.map((item) => (
            <label key={item.key} className={mode === item.key ? "export-mode is-on" : "export-mode"}>
              <input
                type="radio"
                name="mode"
                value={item.key}
                checked={mode === item.key}
                onChange={() => setMode(item.key)}
              />
              <span>
                <strong>{item.label}</strong>
                <em>{item.help}</em>
              </span>
            </label>
          ))}
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
            disabled={busy || (selected.needsDest && !dest.trim())}
          >
            {busy ? "Working…" : actionLabel}
          </button>
        </div>

        {picking && (
          <FolderPicker
            title={isCopy ? "Copy keepers to" : "Write the report in"}
            startPath={dest.trim() || libraryPath}
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
