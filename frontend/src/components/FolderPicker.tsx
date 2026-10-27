import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { api, folderPreviewUrl } from "../api";
import type { FolderEntry, FolderListing, FolderShortcut } from "../types";

/**
 * The folder picker, shaped like the panel macOS puts in front of you: places on
 * the left, a browser in the middle, New Folder / Cancel / Select along the
 * bottom. It is styled with the system's dark materials rather than the app's
 * warm palette on purpose — choosing a folder is the one moment the user wants
 * Finder, not a bespoke dialog.
 *
 * Semantics follow NSOpenPanel too: the current folder is the answer, unless a
 * sub-folder is selected, in which case that one is.
 */

interface Props {
  title: string;
  startPath?: string;
  actionLabel: string;
  onPick: (path: string) => void;
  onClose: () => void;
}

const RECENTS_KEY = "cullr.recentFolders";
const MAX_RECENTS = 5;

const errorText = (value: unknown): string =>
  value instanceof Error ? value.message : String(value);

const basename = (path: string): string => path.split("/").filter(Boolean).pop() ?? "/";

function readRecents(): string[] {
  try {
    const raw: unknown = JSON.parse(window.localStorage.getItem(RECENTS_KEY) ?? "[]");
    if (!Array.isArray(raw)) return [];
    return raw.filter((item): item is string => typeof item === "string").slice(0, MAX_RECENTS);
  } catch {
    return [];
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 ** 3) {
    const mb = bytes / 1024 ** 2;
    return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
  }
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function metaOf(entry: FolderEntry): string {
  if (entry.kind === "folder") {
    if (entry.items == null) return "";
    return `${entry.items.toLocaleString()} ${entry.items === 1 ? "item" : "items"}`;
  }
  if (entry.kind === "image" && entry.width && entry.height) {
    return `${entry.width.toLocaleString()} × ${entry.height.toLocaleString()}`;
  }
  return entry.size != null ? formatSize(entry.size) : "";
}

/* ------------------------------------------------------------------ icons -- */

type GlyphName =
  | "recents" | "shared" | "home" | "desktop" | "file" | "download"
  | "picture" | "film" | "apps" | "disk";

function placeGlyph(name: string): GlyphName {
  switch (name) {
    case "Home": return "home";
    case "Desktop": return "desktop";
    case "Documents": return "file";
    case "Downloads": return "download";
    case "Pictures": return "picture";
    case "Movies": return "film";
    case "Applications": return "apps";
    default: return "disk";
  }
}

function Glyph({ name, size = 15 }: { name: GlyphName; size?: number }) {
  return (
    <svg
      className="finder__ink"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {name === "recents" && (<><circle cx="12" cy="12" r="8.4" /><path d="M12 7.6V12l3.1 1.9" /></>)}
      {name === "shared" && (<><circle cx="9.2" cy="9" r="2.9" /><path d="M3.8 19c.6-2.8 2.9-4.5 5.4-4.5s4.8 1.7 5.4 4.5" /><circle cx="17" cy="10.2" r="2.2" /><path d="M15.6 14.8c2.1.2 3.7 1.7 4.2 4" /></>)}
      {name === "home" && (<path d="M4 10.6 12 4l8 6.6V20a1 1 0 0 1-1 1h-4.2v-6H9.2v6H5a1 1 0 0 1-1-1z" />)}
      {name === "desktop" && (<><rect x="3" y="4.6" width="18" height="11.8" rx="1.7" /><path d="M9 20h6M12 16.4V20" /></>)}
      {name === "file" && (<><path d="M6.2 3.6h6.9l4.7 4.7V20a1 1 0 0 1-1 1H7.2a1 1 0 0 1-1-1V4.6a1 1 0 0 1 1-1z" /><path d="M13 3.6v4.7h4.7" /></>)}
      {name === "download" && (<><circle cx="12" cy="12" r="8.4" /><path d="M12 7.8v7.4m0 0 3-3m-3 3-3-3" /></>)}
      {name === "picture" && (<><rect x="3.4" y="5" width="17.2" height="14" rx="1.8" /><circle cx="9" cy="10" r="1.5" /><path d="m4.6 17.2 4.1-4.1 3.2 3.2 2.9-2.5 4.6 4" /></>)}
      {name === "film" && (<><rect x="3.4" y="5" width="17.2" height="14" rx="1.8" /><path d="M3.4 9.6h17.2M3.4 14.4h17.2M8 5v14M16 5v14" /></>)}
      {name === "apps" && (<><rect x="4" y="4" width="6.4" height="6.4" rx="1.6" /><rect x="13.6" y="4" width="6.4" height="6.4" rx="1.6" /><rect x="4" y="13.6" width="6.4" height="6.4" rx="1.6" /><rect x="13.6" y="13.6" width="6.4" height="6.4" rx="1.6" /></>)}
      {name === "disk" && (<><rect x="3" y="6" width="18" height="12" rx="2.4" /><circle cx="12" cy="12" r="2.1" /><path d="M6.6 9.8h.01M6.6 14.2h.01" /></>)}
    </svg>
  );
}

/** macOS folder: a tab, a lighter front face, both in system blue. */
function FolderGlyph({ edge = 62 }: { edge?: number }) {
  // React's ids contain colons, which are awkward inside url(#…): strip them.
  const id = useId().replace(/:/g, "");
  return (
    <svg width={edge} height={edge * 0.82} viewBox="0 0 64 52" aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-back`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7cc0f4" />
          <stop offset="100%" stopColor="#3d8ad4" />
        </linearGradient>
        <linearGradient id={`${id}-front`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#b3dbfa" />
          <stop offset="100%" stopColor="#5aa6e9" />
        </linearGradient>
      </defs>
      <path
        d="M2 11.4c0-2.4 2-4.4 4.4-4.4h14.4l4.6 4.2h34.2c2.4 0 4.4 2 4.4 4.4V44c0 2.4-2 4.4-4.4 4.4H6.4C4 48.4 2 46.4 2 44z"
        fill={`url(#${id}-back)`}
      />
      <path
        d="M2 17.8h60V44c0 2.4-2 4.4-4.4 4.4H6.4C4 48.4 2 46.4 2 44z"
        fill={`url(#${id}-front)`}
      />
      <path d="M2 17.8h60" stroke="rgba(255,255,255,0.5)" strokeWidth="0.9" />
    </svg>
  );
}

/** Anything the picker cannot preview: a page with a turned corner. */
function DocGlyph({ edge = 50 }: { edge?: number }) {
  return (
    <svg width={edge * 0.78} height={edge} viewBox="0 0 44 56" aria-hidden="true">
      <path
        d="M5 4.2A2.2 2.2 0 0 1 7.2 2h20.4L40 14.4V51.8A2.2 2.2 0 0 1 37.8 54H7.2A2.2 2.2 0 0 1 5 51.8z"
        fill="#f4f4f6"
      />
      <path d="M27.6 2 40 14.4H29.8a2.2 2.2 0 0 1-2.2-2.2z" fill="#c8c8cd" />
      <g stroke="#d6d6db" strokeWidth="2" strokeLinecap="round">
        <path d="M12 26h20M12 33h20M12 40h12" />
      </g>
    </svg>
  );
}

function ImageTile({ entry }: { entry: FolderEntry }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <DocGlyph />;
  return (
    <span className="finder__thumb">
      <img
        src={folderPreviewUrl(entry.path)}
        alt=""
        loading="lazy"
        decoding="async"
        draggable={false}
        onError={() => setFailed(true)}
      />
    </span>
  );
}

function itemGlyph(entry: FolderEntry) {
  if (entry.kind === "folder") return <FolderGlyph />;
  if (entry.kind === "image") return <ImageTile entry={entry} />;
  return <DocGlyph />;
}

/* ------------------------------------------------------------- component -- */

export function FolderPicker({ title, startPath, actionLabel, onPick, onClose }: Props) {
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const [history, setHistory] = useState<{ stack: string[]; index: number }>({ stack: [], index: -1 });
  const [view, setView] = useState<"grid" | "list">("grid");
  const [query, setQuery] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [recents, setRecents] = useState<string[]>(readRecents);
  const [recentsOpen, setRecentsOpen] = useState(true);
  const [sharedOpen, setSharedOpen] = useState(true);

  const dialog = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const draftField = useRef<HTMLInputElement>(null);

  // Take focus off whatever was focused behind the dialog, so typing cannot
  // reach the path field underneath (or submit its form on Enter).
  useEffect(() => {
    dialog.current?.focus();
  }, []);

  const remember = useCallback((path: string) => {
    setRecents((prev) => {
      const next = [path, ...prev.filter((item) => item !== path)].slice(0, MAX_RECENTS);
      try {
        window.localStorage.setItem(RECENTS_KEY, JSON.stringify(next));
      } catch {
        /* storage may be unavailable; recents are a convenience */
      }
      return next;
    });
  }, []);

  const open = useCallback(
    (path: string | undefined, mode: "push" | "reset" = "push") => {
      setBusy(true);
      setError(null);
      api
        .folders(path)
        .then((next) => {
          setListing(next);
          setBusy(false);
          setSelected(null);
          setQuery("");
          setHistory((prev) => {
            const stack =
              mode === "reset"
                ? [next.path]
                : [...prev.stack.slice(0, prev.index + 1), next.path];
            return { stack, index: stack.length - 1 };
          });
          remember(next.path);
        })
        .catch((event) => {
          setError(errorText(event));
          setBusy(false);
        });
    },
    [remember],
  );

  useEffect(() => {
    open(startPath || undefined, "reset");
  }, [open, startPath]);

  const step = useCallback(
    (delta: number) => {
      const index = history.index + delta;
      const path = history.stack[index];
      if (!path) return;
      setBusy(true);
      setError(null);
      api
        .folders(path)
        .then((next) => {
          setListing(next);
          setBusy(false);
          setSelected(null);
          setQuery("");
          setHistory((prev) => ({ ...prev, index }));
          remember(next.path);
        })
        .catch((event) => {
          setError(errorText(event));
          setBusy(false);
        });
    },
    [history, remember],
  );

  const entries = useMemo(() => {
    const all = listing?.entries ?? [];
    const needle = query.trim().toLowerCase();
    return needle ? all.filter((entry) => entry.name.toLowerCase().includes(needle)) : all;
  }, [listing, query]);

  const selectedFolder = entries.find(
    (entry) => entry.path === selected && entry.kind === "folder",
  );
  const target = selectedFolder?.path ?? listing?.path ?? null;

  const ancestors = useMemo(() => {
    if (!listing) return [];
    const bootName =
      listing.sidebar.locations.find((place) => place.path === "/")?.name ?? "Macintosh HD";
    const items = [{ name: bootName, path: "/", depth: 0 }];
    let walked = "";
    for (const part of listing.path.split("/").filter(Boolean)) {
      walked += `/${part}`;
      items.push({ name: part, path: walked, depth: items.length });
    }
    return items;
  }, [listing]);

  const confirm = useCallback(() => {
    if (target) onPick(target);
  }, [onPick, target]);

  const move = useCallback(
    (delta: number) => {
      if (!entries.length) return;
      const current = entries.findIndex((entry) => entry.path === selected);
      const next =
        current === -1
          ? delta > 0
            ? 0
            : entries.length - 1
          : Math.min(entries.length - 1, Math.max(0, current + delta));
      setSelected(entries[next].path);
    },
    [entries, selected],
  );

  const startCreate = useCallback(() => {
    setDraft("untitled folder");
    setCreating(true);
    setError(null);
    setMenuOpen(false);
  }, []);

  const create = useCallback(async () => {
    const name = draft.trim();
    if (!listing || !name || saving) return;
    setSaving(true);
    setError(null);
    try {
      const made = await api.createFolder(listing.path, name);
      const next = await api.folders(listing.path);
      setCreating(false);
      setDraft("");
      setListing(next);
      setSelected(made.path);
    } catch (event) {
      setError(errorText(event));
    } finally {
      setSaving(false);
    }
  }, [draft, listing, saving]);

  useEffect(() => {
    if (!creating) return;
    draftField.current?.focus();
    draftField.current?.select();
  }, [creating]);

  useEffect(() => {
    if (!selected) return;
    content.current
      ?.querySelector(".finder__item.is-selected")
      ?.scrollIntoView({ block: "nearest" });
  }, [selected, view]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [menuOpen]);

  useEffect(() => {
    // Capture phase: Escape must back out of the menu or the draft first, and
    // must never reach the app handler (which would close the export dialog).
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        if (menuOpen) setMenuOpen(false);
        else if (creating) {
          setCreating(false);
          setDraft("");
          setError(null);
        } else onClose();
        return;
      }
      const element = event.target as HTMLElement | null;
      if (element && ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        event.stopPropagation();
        move(1);
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        event.stopPropagation();
        move(-1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        const entry = entries.find((item) => item.path === selected);
        if (entry?.kind === "folder") open(entry.path, "push");
        else confirm();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [confirm, creating, entries, menuOpen, move, onClose, open, selected]);

  // Exactly one row is highlighted, the one the browser is sitting in — a
  // prefix match would light up Home and Documents at the same time.
  const isHere = (path: string) => listing?.path === path;

  const renderPlace = (place: FolderShortcut, nest = false) => (
    <li key={place.path}>
      <button
        type="button"
        className={`finder__place${nest ? " finder__place--nested" : ""}${
          !nest && isHere(place.path) ? " is-current" : ""
        }`}
        onClick={() => open(place.path, "push")}
        title={place.path}
      >
        {!nest && <Glyph name={placeGlyph(place.name)} />}
        <span>{place.name}</span>
      </button>
    </li>
  );

  const sidebar = listing?.sidebar;

  return (
    <div
      className="finder-scrim"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="finder" ref={dialog} tabIndex={-1}>
        <aside className="finder__sidebar" aria-label="Places">
          <ul className="finder__places">
            <li>
              <button
                type="button"
                className={`finder__place${recentsOpen ? " is-open" : ""}`}
                onClick={() => setRecentsOpen((value) => !value)}
                aria-expanded={recentsOpen}
                disabled={recents.length === 0}
              >
                <Glyph name="recents" />
                <span>Recents</span>
                <span className={`finder__twist${recentsOpen ? " is-open" : ""}`}>›</span>
              </button>
            </li>
            {recentsOpen && recents.map((path) => renderPlace({ name: basename(path), path }, true))}
            <li>
              <button
                type="button"
                className={`finder__place${sharedOpen ? " is-open" : ""}`}
                onClick={() => setSharedOpen((value) => !value)}
                aria-expanded={sharedOpen}
                disabled={!sidebar?.shared.length}
              >
                <Glyph name="shared" />
                <span>Shared</span>
                <span className={`finder__twist${sharedOpen ? " is-open" : ""}`}>›</span>
              </button>
            </li>
            {sharedOpen && sidebar?.shared.map((place) => renderPlace(place, true))}
          </ul>

          <p className="finder__section">Favorites</p>
          <ul className="finder__places">
            {sidebar?.favorites.map((place) => renderPlace(place))}
          </ul>

          <p className="finder__section">Locations</p>
          <ul className="finder__places">
            {sidebar?.locations.map((place) => renderPlace(place))}
            {sidebar && sidebar.locations.length === 0 && (
              <li className="finder__empty-place">No volumes</li>
            )}
          </ul>
        </aside>

        <div className="finder__head">
          <p className="finder__message">{title}</p>
          <div className="finder__toolbar">
            <div className="finder__group">
              <button
                type="button"
                className="finder__icon-button"
                onClick={() => step(-1)}
                disabled={history.index <= 0 || busy}
                aria-label="Back"
                title="Back"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M14.5 5.5 8 12l6.5 6.5" />
                </svg>
              </button>
              <button
                type="button"
                className="finder__icon-button"
                onClick={() => step(1)}
                disabled={history.index < 0 || history.index >= history.stack.length - 1 || busy}
                aria-label="Forward"
                title="Forward"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M9.5 5.5 16 12l-6.5 6.5" />
                </svg>
              </button>
            </div>

            <div className="finder__segmented" role="group" aria-label="View">
              <button
                type="button"
                className={`finder__segment${view === "grid" ? " is-on" : ""}`}
                onClick={() => setView("grid")}
                aria-pressed={view === "grid"}
                aria-label="Icon view"
                title="Icon view"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                  <rect x="3.5" y="3.5" width="7" height="7" rx="1.4" />
                  <rect x="13.5" y="3.5" width="7" height="7" rx="1.4" />
                  <rect x="3.5" y="13.5" width="7" height="7" rx="1.4" />
                  <rect x="13.5" y="13.5" width="7" height="7" rx="1.4" />
                </svg>
              </button>
              <button
                type="button"
                className={`finder__segment${view === "list" ? " is-on" : ""}`}
                onClick={() => setView("list")}
                aria-pressed={view === "list"}
                aria-label="List view"
                title="List view"
              >
                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
                  <path d="M4 6.5h16M4 12h16M4 17.5h16" />
                </svg>
              </button>
            </div>

            <div className="finder__pathwrap" onMouseDown={(event) => event.stopPropagation()}>
              <button
                type="button"
                className={`finder__path${menuOpen ? " is-open" : ""}`}
                onClick={() => setMenuOpen((value) => !value)}
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                title={listing?.path ?? "Opening…"}
              >
                <span className="finder__path-name">
                  {listing ? basename(listing.path) : "Opening…"}
                </span>
                <svg viewBox="0 0 12 12" width="9" height="9" aria-hidden="true">
                  <path d="M2.6 4.3 6 7.7l3.4-3.4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {menuOpen && (
                <div className="finder__menu" role="menu">
                  {ancestors.map((item, index) => (
                    <button
                      key={item.path}
                      type="button"
                      role="menuitem"
                      className={`finder__menu-row${
                        index === ancestors.length - 1 ? " is-current" : ""
                      }`}
                      style={{ paddingLeft: 12 + item.depth * 13 }}
                      onClick={() => {
                        setMenuOpen(false);
                        open(item.path, "push");
                      }}
                    >
                      {item.name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <label className="finder__search">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <circle cx="11" cy="11" r="6.4" />
                <path d="m16 16 4 4" />
              </svg>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search"
                aria-label="Search this folder"
                spellCheck={false}
              />
            </label>
          </div>
        </div>

        <div className={`finder__content${view === "list" ? " finder__content--list" : ""}`} ref={content}>
          {creating && (
            <div className="finder__item finder__item--draft">
              <FolderGlyph />
              <input
                ref={draftField}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    event.stopPropagation();
                    void create();
                  }
                }}
                aria-label="New folder name"
                spellCheck={false}
              />
            </div>
          )}

          {view === "grid"
            ? entries.map((entry) => (
                <button
                  key={entry.path}
                  type="button"
                  className={`finder__item${selected === entry.path ? " is-selected" : ""}${
                    entry.kind === "folder" ? "" : " finder__item--file"
                  }`}
                  onClick={() => setSelected(entry.path)}
                  onDoubleClick={() => {
                    if (entry.kind === "folder") open(entry.path, "push");
                  }}
                  title={entry.path}
                >
                  {itemGlyph(entry)}
                  <span className="finder__label">{entry.name}</span>
                  <span className="finder__meta">{metaOf(entry)}</span>
                </button>
              ))
            : entries.map((entry) => (
                <button
                  key={entry.path}
                  type="button"
                  className={`finder__row${selected === entry.path ? " is-selected" : ""}`}
                  onClick={() => setSelected(entry.path)}
                  onDoubleClick={() => {
                    if (entry.kind === "folder") open(entry.path, "push");
                  }}
                  title={entry.path}
                >
                  <span className="finder__row-icon">{itemGlyph(entry)}</span>
                  <span className="finder__row-name">{entry.name}</span>
                  <span className="finder__row-meta">{metaOf(entry)}</span>
                </button>
              ))}

          {!busy && listing && entries.length === 0 && !creating && (
            <p className="finder__blank">
              {query.trim() ? `Nothing matches “${query.trim()}”.` : "This folder is empty."}
            </p>
          )}
        </div>

        <div className="finder__foot">
          <button
            type="button"
            className="finder__new-folder"
            onClick={startCreate}
            disabled={busy || saving || !listing?.writable}
            title={listing?.writable ? "New folder here" : "This folder is read-only"}
          >
            New Folder
          </button>
          <div className="finder__foot-end">
            {error && <p className="finder__error">{error}</p>}
            {!error && listing && !listing.writable && (
              <p className="finder__error">Read-only — choose another folder.</p>
            )}
            {!error && listing?.truncated && (
              <p className="finder__note">Showing the first 400 names.</p>
            )}
            {!error && target && (
              <p className="finder__target" title={target}>
                {target}
              </p>
            )}
            <button type="button" className="finder__button" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="finder__button finder__button--primary"
              onClick={confirm}
              disabled={!target}
            >
              {actionLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
