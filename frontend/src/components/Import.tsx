import { useState } from "react";
import type { Capabilities, Library } from "../types";
import { FolderPicker } from "./FolderPicker";

interface Props {
  libraries: Library[];
  capabilities: Capabilities | null;
  error: string | null;
  onOpen: (path: string) => Promise<void>;
  onSelect: (library: Library) => void;
}

export function Import({ libraries, capabilities, error, onOpen, onSelect }: Props) {
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [picking, setPicking] = useState(false);

  const submit = async () => {
    if (!path.trim()) return;
    setBusy(true);
    try {
      await onOpen(path.trim());
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="import">
      <div className="import__pitch">
        <h1>
          Cull a shoot
          <span>without opening every frame.</span>
        </h1>
        <p>
          Sharpen scores each photo for focus, exposure and open eyes, groups the
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
                onClick={() => setPicking(true)}
                disabled={busy}
              >
                Select a folder
              </button>
            </div>
            <button type="submit" className="button button--primary" disabled={busy}>
              {busy ? "Opening…" : "Open folder"}
            </button>
          </div>
          <p className="import__note">
            Analysis runs on this machine. Nothing is uploaded, and originals are
            never modified.
          </p>
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
          title="Open a shoot"
          startPath={path.trim() || undefined}
          actionLabel="Open this folder"
          onClose={() => setPicking(false)}
          onPick={(chosen) => {
            setPath(chosen);
            setPicking(false);
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
