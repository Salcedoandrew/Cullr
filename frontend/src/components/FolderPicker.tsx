import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { FolderListing } from "../types";

interface Props {
  title: string;
  startPath?: string;
  actionLabel: string;
  onPick: (path: string) => void;
  onClose: () => void;
}

const errorText = (value: unknown): string =>
  value instanceof Error ? value.message : String(value);

export function FolderPicker({ title, startPath, actionLabel, onPick, onClose }: Props) {
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const dialog = useRef<HTMLDivElement>(null);
  const draftField = useRef<HTMLInputElement>(null);

  // Take focus off whatever was focused behind the dialog, so typing cannot
  // reach the path field underneath (or submit its form on Enter).
  useEffect(() => {
    dialog.current?.focus();
  }, []);

  const go = useCallback((path?: string) => {
    setBusy(true);
    setError(null);
    api
      .folders(path)
      .then((next) => {
        setListing(next);
        setBusy(false);
      })
      .catch((event) => {
        setError(errorText(event));
        setBusy(false);
      });
  }, []);

  useEffect(() => {
    go(startPath || undefined);
  }, [go, startPath]);

  useEffect(() => {
    if (creating) draftField.current?.focus();
  }, [creating]);

  useEffect(() => {
    // Capture phase, so Escape never reaches the app's own handler (which would
    // close the export dialog too). It backs out one step at a time.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      if (creating) {
        setCreating(false);
        setDraft("");
        setError(null);
      } else {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [creating, onClose]);

  const create = useCallback(async () => {
    const name = draft.trim();
    if (!listing || !name || saving) return;
    setSaving(true);
    setError(null);
    try {
      const made = await api.createFolder(listing.path, name);
      setDraft("");
      setCreating(false);
      // Land inside the new folder, so "Use this folder" is one click away.
      go(made.path);
    } catch (event) {
      setError(errorText(event));
    } finally {
      setSaving(false);
    }
  }, [draft, go, listing, saving]);

  const usable = listing !== null && listing.writable && !busy;

  return (
    <div
      className="modal"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal__body modal__body--picker" ref={dialog} tabIndex={-1}>
        <header className="modal__head">
          <h2>{title}</h2>
          <button
            type="button"
            className="loupe__close loupe__close--inline"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        <div className="picker__bar">
          <button
            type="button"
            className="picker__jump"
            onClick={() => listing && go(listing.parent ?? undefined)}
            disabled={!listing?.parent || busy}
            title="Parent folder"
          >
            ↑
          </button>
          <button
            type="button"
            className="picker__jump"
            onClick={() => go(undefined)}
            disabled={busy}
            title="Home folder"
          >
            Home
          </button>
          <p className="picker__path" title={listing?.path}>
            {listing?.path ?? "Opening…"}
          </p>
          <button
            type="button"
            className="picker__jump"
            onClick={() => setCreating(true)}
            disabled={busy || saving || !listing?.writable}
            title="Create a folder here"
          >
            New folder
          </button>
        </div>

        {creating && (
          <form
            className="picker__create"
            onSubmit={(event) => {
              event.preventDefault();
              void create();
            }}
          >
            <input
              ref={draftField}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="New folder name"
              aria-label="New folder name"
              spellCheck={false}
            />
            <button
              type="submit"
              className="button button--primary picker__create-action"
              disabled={!draft.trim() || saving}
            >
              {saving ? "Creating…" : "Create"}
            </button>
          </form>
        )}

        <ul className="picker__list" aria-label="Sub-folders">
          {listing?.parent && (
            <li>
              <button
                type="button"
                className="picker__row picker__row--up"
                onClick={() => go(listing.parent ?? undefined)}
                disabled={busy}
              >
                <span className="picker__chev">↑</span>
                <span className="picker__name">Parent folder</span>
              </button>
            </li>
          )}
          {listing?.folders.map((folder) => (
            <li key={folder.path}>
              <button
                type="button"
                className="picker__row"
                onClick={() => go(folder.path)}
                disabled={busy}
              >
                <span className="picker__chev">›</span>
                <span className="picker__name">{folder.name}</span>
              </button>
            </li>
          ))}
          {listing && listing.folders.length === 0 && (
            <li className="picker__empty">No sub-folders.</li>
          )}
        </ul>

        {listing?.truncated && (
          <p className="picker__hint">First 500 folders shown.</p>
        )}
        {listing && !listing.writable && (
          <p className="picker__hint picker__hint--warn">
            Not writable — pick a folder you own, or a sub-folder of this one.
          </p>
        )}
        {error && <p className="modal__error">{error}</p>}

        <div className="modal__actions">
          <button type="button" className="linkbutton" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="button button--primary"
            onClick={() => listing && onPick(listing.path)}
            disabled={!usable}
          >
            {actionLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
