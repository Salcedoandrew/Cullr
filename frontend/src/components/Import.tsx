import { useEffect, useRef, useState } from "react";
import type { Capabilities, ImportOptions, Library } from "../types";
import { FolderPicker } from "./FolderPicker";

declare global {
  interface Window {
    /** Set by the desktop wrapper once its native drop listener is attached. */
    __cullrNativeDrop?: boolean;
    /** Called by the desktop wrapper with the absolute path of a dropped folder. */
    __cullrDrop?: (path: string) => void;
    /** Called by the desktop wrapper when a drop could not be resolved. */
    __cullrDropError?: (reason: string) => void;
  }
}

interface Props {
  libraries: Library[];
  capabilities: Capabilities;
  error: string | null;
  /** Parent of the most recently opened shoot — where a dropped folder most
   *  likely came from. */
  lastParent?: string;
  onOpen: (path: string, options: ImportOptions) => Promise<void>;
  onSelect: (library: Library) => void;
}

const ALL_TYPES = ["raw", "jpeg", "tiff", "heic"];

/** Top-level names of what was dropped: the folder name for a dropped folder
 *  (via webkitGetAsEntry), or the first path segment for files inside one. */
function droppedNames(event: React.DragEvent): string[] {
  const names: string[] = [];
  const items = event.dataTransfer?.items;
  if (items) {
    for (const item of Array.from(items)) {
      if (item.kind !== "file") continue;
      const entry = item.webkitGetAsEntry?.();
      if (entry) {
        names.push(entry.name);
        continue;
      }
      const file = item.getAsFile();
      if (!file) continue;
      names.push(
        file.webkitRelativePath
          ? file.webkitRelativePath.split("/")[0]
          : file.name,
      );
    }
  }
  if (!names.length) {
    for (const file of Array.from(event.dataTransfer?.files ?? [])) {
      names.push(
        file.webkitRelativePath
          ? file.webkitRelativePath.split("/")[0]
          : file.name,
      );
    }
  }
  return [...new Set(names)];
}

export function Import({ libraries, capabilities, error, lastParent, onOpen, onSelect }: Props) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState<"shoot" | "backup" | null>(null);
  const [droppedName, setDroppedName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [options, setOptions] = useState<ImportOptions>(() => {
    const defaults = capabilities.default_settings;
    const offered = capabilities.import_types?.map((type) => type.key) ?? ALL_TYPES;
    const types = defaults.import_types?.length ? defaults.import_types : offered;
    return {
      recursive: defaults.recursive ?? true,
      import_types: types.length ? types : ALL_TYPES,
      ingest: defaults.ingest ?? false,
      backup_dir: defaults.backup_dir ?? null,
    };
  });

  const optionsRef = useRef(options);
  optionsRef.current = options;
  const resolveTimer = useRef<number | null>(null);
  const droppedRef = useRef<string | null>(null);

  const submit = async (folder?: string) => {
    const target = (folder ?? path).trim();
    if (!target || busy) return;
    setBusy(true);
    try {
      await onOpen(target, optionsRef.current);
    } finally {
      setBusy(false);
    }
  };
  const submitRef = useRef(submit);
  submitRef.current = submit;

  const openPickerFallback = (name: string | null) => {
    setResolving(false);
    setDroppedName(name);
    setPicking("shoot");
  };

  // The desktop wrapper delivers a dropped folder's real path here (see
  // desktop/cullr_desktop.py); in a plain browser there is no such bridge and
  // the drop falls back to the folder picker below.
  useEffect(() => {
    window.__cullrDrop = (droppedPath: string) => {
      if (resolveTimer.current !== null) {
        window.clearTimeout(resolveTimer.current);
        resolveTimer.current = null;
      }
      setResolving(false);
      setPath(droppedPath);
      void submitRef.current(droppedPath);
    };
    window.__cullrDropError = () => {
      if (resolveTimer.current !== null) {
        window.clearTimeout(resolveTimer.current);
        resolveTimer.current = null;
      }
      openPickerFallback(droppedRef.current);
    };
    return () => {
      delete window.__cullrDrop;
      delete window.__cullrDropError;
      if (resolveTimer.current !== null) {
        window.clearTimeout(resolveTimer.current);
        resolveTimer.current = null;
      }
    };
  }, []);

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    if (busy) return;
    const names = droppedNames(event);
    const first = names[0] ?? null;
    droppedRef.current = first;
    if (window.__cullrNativeDrop) {
      // The desktop wrapper resolves the drop to a real path and calls
      // window.__cullrDrop. If it never answers, fall back to the picker —
      // a drop must never leave the user staring at a spinner.
      setResolving(true);
      resolveTimer.current = window.setTimeout(() => {
        resolveTimer.current = null;
        openPickerFallback(first);
      }, 2500);
      return;
    }
    openPickerFallback(first);
  };

  const toggleType = (key: string, checked: boolean) => {
    setOptions((current) => {
      const next = checked
        ? [...current.import_types, key]
        : current.import_types.filter((item) => item !== key);
      // At least one type must stay checked: an import with no types admits
      // nothing, and that is never what unchecking the last box means.
      if (!next.length) return current;
      return { ...current, import_types: next };
    });
  };

  const pickerStart = picking === "backup"
    ? options.backup_dir ?? lastParent
    : path.trim() || lastParent;

  return (
    <div className="import">
      <div className="import__pitch">
        <h1>
          Cull a shoot
          <span>without opening every frame.</span>
        </h1>
        <p>
          Cullr scores each photo for focus, exposure and open eyes, groups the
          near-identical frames from a burst, and picks the one worth keeping. You
          review the result, override anything, and export ratings as sidecars.
        </p>

        <form
          className="import__form"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <div
            className={`import__drop${dragging ? " is-dragging" : ""}${
              resolving ? " is-resolving" : ""
            }`}
            role="button"
            tabIndex={0}
            aria-label="Import a folder: click to choose, or drag and drop a folder here"
            onClick={() => {
              if (!busy && !resolving) {
                setDroppedName(null);
                setPicking("shoot");
              }
            }}
            onKeyDown={(event) => {
              if ((event.key === "Enter" || event.key === " ") && !busy && !resolving) {
                event.preventDefault();
                setDroppedName(null);
                setPicking("shoot");
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              if (!dragging) setDragging(true);
            }}
            onDragLeave={(event) => {
              // Moving between children of the zone is not leaving it.
              if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
              setDragging(false);
            }}
            onDrop={handleDrop}
          >
            <span className="import__drop-icon" aria-hidden="true">
              <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 16V4m0 0 4.5 4.5M12 4 7.5 8.5" />
                <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" />
              </svg>
            </span>
            <span className="import__drop-title">
              {resolving
                ? "Resolving the dropped folder…"
                : "Click to import folder(s) or drag and drop them here"}
            </span>
            <span className="import__drop-sub">
              A shoot folder from Finder, a card, or an external drive
            </span>
          </div>

          <label htmlFor="folder">Folder of photos</label>
          <div className="import__row">
            <div className="field-with-button">
              <input
                id="folder"
                value={path}
                onChange={(event) => setPath(event.target.value)}
                placeholder="/Users/you/Pictures/2026-06-15 Wedding"
                spellCheck={false}
                autoFocus
              />
              <button
                type="button"
                className="button field-with-button__action"
                onClick={() => {
                  setDroppedName(null);
                  setPicking("shoot");
                }}
                disabled={busy}
              >
                Select a folder
              </button>
            </div>
          </div>

          <fieldset className="import__settings">
            <legend>Import settings</legend>

            <div className="import__setting">
              <span className="import__setting-label">Type of photos to import</span>
              <div className="import__types">
                {(capabilities.import_types ?? []).map((type) => (
                  <label key={type.key} className="import__check">
                    <input
                      type="checkbox"
                      checked={options.import_types.includes(type.key)}
                      onChange={(event) => toggleType(type.key, event.target.checked)}
                      disabled={busy}
                    />
                    <span>
                      {type.label}
                      <small>
                        {type.extensions
                          .map((ext) => ext.replace(/^\./, "").toUpperCase())
                          .join(" · ")}
                      </small>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <label className="import__check">
              <input
                type="checkbox"
                checked={options.recursive}
                onChange={(event) =>
                  setOptions({ ...options, recursive: event.target.checked })
                }
                disabled={busy}
              />
              <span>Include sub-folders</span>
            </label>

            <label className="import__check">
              <input
                type="checkbox"
                checked={options.ingest}
                onChange={(event) =>
                  setOptions({ ...options, ingest: event.target.checked })
                }
                disabled={busy}
              />
              <span>Ingest and backup photos</span>
            </label>
            {options.ingest && (
              <div className="import__ingest">
                <p>
                  Photos are copied into the app&rsquo;s working folder before
                  culling, so the card can be reformatted while you review.
                  Originals are never modified, and a rescan reuses the same
                  working copy.
                </p>
                <div className="import__backup">
                  <span className="import__backup-path" title={options.backup_dir ?? ""}>
                    {options.backup_dir ?? "No backup folder chosen"}
                  </span>
                  <button
                    type="button"
                    className="button"
                    onClick={() => setPicking("backup")}
                    disabled={busy}
                  >
                    Choose backup folder…
                  </button>
                  {options.backup_dir && (
                    <button
                      type="button"
                      className="button"
                      onClick={() => setOptions({ ...options, backup_dir: null })}
                      disabled={busy}
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
            )}
          </fieldset>

          <div className="import__actions">
            <button type="submit" className="button button--primary" disabled={busy}>
              {busy ? "Importing…" : "Import"}
            </button>
            <p className="import__note">
              Analysis runs on this machine. Nothing is uploaded, and originals are
              never modified.
            </p>
          </div>
          {error && <p className="import__error">{error}</p>}
          {capabilities && !capabilities.faces && (
            <p className="import__note import__note--warn">
              Face detection is off ({capabilities.faces_error}). Blur, exposure and
              duplicate grouping still work; closed-eye rejection needs the face
              model.
            </p>
          )}
        </form>
      </div>

      {picking && (
        <FolderPicker
          title={picking === "backup" ? "Choose a backup folder" : "Open a shoot"}
          startPath={pickerStart}
          actionLabel={picking === "backup" ? "Use this folder" : "Import this folder"}
          droppedName={picking === "shoot" ? droppedName : null}
          onClose={() => {
            setPicking(null);
            setDroppedName(null);
          }}
          onPick={(chosen) => {
            if (picking === "backup") {
              setOptions({ ...options, backup_dir: chosen });
            } else {
              setPath(chosen);
            }
            setPicking(null);
            setDroppedName(null);
          }}
        />
      )}

      {libraries.length > 0 && (
        <aside className="import__recent">
          <h2>Recent folders</h2>
          <ul>
            {libraries.map((library) => (
              <li key={library.id}>
                <button type="button" onClick={() => onSelect(library)}>
                  <span>{library.name}</span>
                  <span className="import__recent-path">{library.root_path}</span>
                  {library.photo_count ? (
                    <span className="import__recent-count">{library.photo_count} frames</span>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}
