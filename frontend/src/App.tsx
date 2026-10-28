import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { CullDialog } from "./components/CullDialog";
import { ExportPanel } from "./components/ExportPanel";
import { Import } from "./components/Import";
import { Loupe } from "./components/Loupe";
import { People } from "./components/People";
import { Sheet } from "./components/Sheet";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { UpdatePanel } from "./components/UpdatePanel";
import type {
  Capabilities, ExportMemory, FilterKey, ImportOptions, Job, Library, MergeSuggestion,
  Person, Photo, Settings, Stats,
} from "./types";

declare global {
  interface Window {
    /** Called by the desktop wrapper's app-menu item (desktop/cullr_desktop.py).
     *  Opens the update panel; there is still no check until the user clicks. */
    __cullrCheckUpdates?: () => void;
  }
}

const FALLBACK_SETTINGS: Settings = {
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

/** Parent of a path, for the drop fallback: the folder a dropped shoot most
 *  likely came from is beside the last shoot that was opened. */
const parentOf = (path: string): string => {
  const parts = path.split("/").filter(Boolean);
  parts.pop();
  return parts.length ? `/${parts.join("/")}` : "/";
};

function mergeSettings(
  base: Partial<Settings> | null | undefined,
  override?: Partial<Settings>,
): Settings {
  return { ...FALLBACK_SETTINGS, ...(base ?? {}), ...(override ?? {}) };
}

const errorText = (value: unknown): string =>
  value instanceof Error ? value.message : String(value);

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [libraries, setLibraries] = useState<Library[]>([]);
  const [library, setLibrary] = useState<Library | null>(null);
  const [settings, setSettings] = useState<Settings>(FALLBACK_SETTINGS);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [listed, setListed] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  const [suggestions, setSuggestions] = useState<MergeSuggestion[]>([]);
  const [pendingMerge, setPendingMerge] = useState<string | null>(null);
  const [exportMemory, setExportMemory] = useState<ExportMemory | null>(null);
  const [filter, setFilter] = useState<FilterKey>("all");
  const [personId, setPersonId] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loupeId, setLoupeId] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [updateOpen, setUpdateOpen] = useState(false);
  const [cullOpen, setCullOpen] = useState(false);
  const [discoveredTotal, setDiscoveredTotal] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Which finished job's read-failure banner has been dismissed. The banner
  // itself is derived from the job below, so it survives a photo-list refresh
  // and a page reload; only an explicit dismissal (or a new scan) changes it.
  const [dismissedJob, setDismissedJob] = useState<number | null>(null);

  const busy = job?.status === "running";
  const paused = job?.status === "paused";
  const scanProgress = busy ? job?.message ?? job?.stage ?? "Scanning" : null;
  /** How far the cull has got, for the bar on the window's bottom edge. It stays
   *  on screen while paused so the fill shows exactly where it stopped.
   *  Null until the job knows its total, which is the one case a bar cannot
   *  honestly claim to be at 0 %. */
  const cullPercent = (busy || paused) && job && job.total > 0
    ? Math.min(100, Math.round((job.done / job.total) * 100))
    : null;
  /** A finished scan that could not read files: the one case where "0 keepers"
   *  is not an answer. Derived from the job row (re-read with the library), not
   *  from state the refresh would drop. */
  const unreadable = job && job.status !== "running" && (job.error_count ?? 0) > 0
    ? job
    : null;
  const selected = useMemo(
    () => photos.find((photo) => photo.id === selectedId) ?? null,
    [photos, selectedId],
  );
  const loupePhoto = useMemo(
    () => photos.find((photo) => photo.id === loupeId) ?? null,
    [photos, loupeId],
  );

  const refresh = useCallback(async (
    libraryId: number,
    key: FilterKey,
    person: number | null = null,
  ) => {
    const [response, roster, proposals] = await Promise.all([
      api.photos(libraryId, key, "group", person),
      api.persons(libraryId),
      api.mergeSuggestions(libraryId),
    ]);
    setPhotos(response.photos);
    setListed(response.total);
    setStats(response.stats);
    setPeople(roster);
    setSuggestions(proposals);
    setSelectedId((current) =>
      current !== null && response.photos.some((photo) => photo.id === current)
        ? current
        : null,
    );
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const [caps, libs] = await Promise.all([api.capabilities(), api.libraries()]);
        setCapabilities(caps);
        setLibraries(libs);
        setSettings(mergeSettings(caps.default_settings));
      } catch (event) {
        setError(errorText(event));
      }
    })();
  }, []);

  // The macOS app menu's "Check for Updates…" lands here. The panel is the only
  // update UI, so the menu item cannot drift from the buttons on either screen.
  useEffect(() => {
    window.__cullrCheckUpdates = () => setUpdateOpen(true);
    return () => {
      delete window.__cullrCheckUpdates;
    };
  }, []);

  // Load the catalog and the stored settings when a library is opened.
  useEffect(() => {
    if (!library) return;
    setSettings((current) => mergeSettings(capabilities?.default_settings ?? current,
                                           library.settings));
    setExportMemory(library.export ?? null);
    void api
      .latestJob(library.id)
      .then((latest) => {
        if (latest && "id" in latest) setJob(latest as Job);
      })
      .catch(() => undefined);
  }, [library?.id, capabilities]);

  useEffect(() => {
    if (!library) return;
    void refresh(library.id, filter, personId).catch((event) => setError(errorText(event)));
  }, [library?.id, filter, personId, refresh]);

  // Poll the running scan.
  useEffect(() => {
    if (!job || job.status !== "running" || !library) return;
    let stopped = false;
    let ticks = 0;
    const timer = window.setInterval(() => {
      void api
        .job(job.id)
        .then((next) => {
          if (stopped) return;
          setJob(next);
          ticks += 1;
          // The list is re-read every third tick (~2 s) while the scan runs, so
          // the grid fills in as frames are analysed instead of appearing all
          // at once at the end. The 650 ms job poll stays as it is; the photo
          // list is the expensive one to re-read, and the scan is still writing
          // rows between the two.
          if (next.status !== "running" || ticks % 3 === 0) {
            return refresh(library.id, filter, personId);
          }
          return undefined;
        })
        .catch((event) => setError(errorText(event)));
    }, 650);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [job?.id, job?.status, library?.id, filter, personId, refresh]);

  /** Re-read capabilities after the scene model finishes downloading, so the
   *  cull dialog can stop offering the download. */
  const refreshCapabilities = useCallback(() => {
    void api
      .capabilities()
      .then((next) => setCapabilities(next))
      .catch(() => undefined);
  }, []);

  const startScan = useCallback(
    async (libraryId: number, active: Settings, force: boolean) => {
      try {
        const started = await api.scan(libraryId, active, force);
        setJob(started);
      } catch (event) {
        setError(errorText(event));
      }
    },
    [],
  );

  /** Stop the running cull at the next frame boundary. The job answers as
   *  "paused" on the next poll, and the frames already analysed stay stored. */
  const pauseScan = useCallback(() => {
    if (!job || job.status !== "running") return;
    void api.pauseScan(job.id).catch((event) => setError(errorText(event)));
  }, [job]);

  /** Continue a paused cull. The scan is resumable by construction — analysed
   *  frames are reused — so the server starts it again and hands back the job
   *  the poll below tracks. */
  const resumeScan = useCallback(() => {
    if (!library) return;
    void api
      .resumeScan(library.id)
      .then((started) => setJob(started))
      .catch((event) => setError(errorText(event)));
  }, [library]);

  /** Count the photos an import set holds, for the cull options to show before
   *  any scan has run. A names-only walk on the server; null when the count
   *  cannot be taken, so a failed count never blocks the dialog — it falls
   *  back to the percent-only display. */
  const countImportSet = useCallback(
    async (libraryId: number): Promise<number | null> => {
      try {
        const found = await api.discover(libraryId);
        return found.total;
      } catch {
        return null;
      }
    },
    [],
  );

  const openFolder = useCallback(
    async (path: string, options?: ImportOptions) => {
      try {
        setError(null);
        const created = await api.createLibrary(path, options);
        setLibraries(await api.libraries());
        setLibrary(created);
        // The import options are part of the scan's settings: they decide
        // what gets discovered and whether the photos are ingested first.
        // They persist with the library, so a rescan reuses them.
        const active = mergeSettings(capabilities?.default_settings ?? null,
                                     { ...created.settings, ...(options ?? {}) });
        setSettings(active);
        setFilter("all");
        setPersonId(null);
        // Drop the previous shoot's stats: the photo list refresh reports
        // total 0 for a library with no scan yet, and a stale total would
        // leak into the cull options before it lands.
        setStats(null);
        // Count the import set first, so the cull options show the real
        // total instead of 0.
        setDiscoveredTotal(await countImportSet(created.id));
        // No auto-cull: the cull options open first, and the job starts only
        // when the user presses Start Cull in that window.
        setCullOpen(true);
      } catch (event) {
        setError(errorText(event));
        throw event;
      }
    },
    [capabilities, startScan, countImportSet],
  );

  /** Open a library from Recent folders. Its cull options need a total before
   *  the first scan, so a library that has never been scanned is counted here
   *  the way an import counts one — the same names-only walk, which honours
   *  the stored import options and falls back to the working copy when the
   *  source folder is gone. A library that already has a scan takes its total
   *  from stats, which the photo list refresh brings in. */
  const openRecent = useCallback(
    async (chosen: Library) => {
      setLibrary(chosen);
      setFilter("all");
      setPersonId(null);
      // The shoot before this one must not leak its total into the cull
      // options while this library's photo list loads.
      setStats(null);
      if (chosen.photo_count) {
        setDiscoveredTotal(null);
        return;
      }
      setDiscoveredTotal(await countImportSet(chosen.id));
    },
    [countImportSet],
  );

  const decide = useCallback(
    async (photo: Photo, decision?: "pick" | "reject" | "clear", star?: number) => {
      if (!library) return;
      try {
        const updated = await api.decide(photo.id, decision, star);
        setPhotos((list) =>
          list.map((item) => (item.id === updated.id ? updated : item)),
        );
        setStats(await api.stats(library.id));
      } catch (event) {
        setError(errorText(event));
      }
    },
    [library],
  );

  // Confirmed merges and dismissals both change the roster, so re-read it all.
  const linkPeople = useCallback(
    async (suggestion: MergeSuggestion, samePerson: boolean) => {
      if (!library || pendingMerge) return;
      const key = `${suggestion.a}-${suggestion.b}`;
      const ids = [suggestion.a, suggestion.b];
      setPendingMerge(key);
      try {
        let nextPersonId = personId;
        if (samePerson) {
          const merged = await api.mergePersons(library.id, ids);
          setPeople(merged.persons);
          if (personId !== null && !merged.persons.some((p) => p.id === personId)) {
            nextPersonId = null;
            setPersonId(null);
          }
        } else {
          await api.markDistinct(library.id, ids);
        }
        await refresh(library.id, filter, nextPersonId);
      } catch (event) {
        setError(errorText(event));
      } finally {
        setPendingMerge(null);
      }
    },
    [library, personId, filter, pendingMerge, refresh],
  );

  const move = useCallback(
    (delta: number) => {
      if (!photos.length) return;
      const index = photos.findIndex((photo) => photo.id === selectedId);
      const next = index === -1 ? 0 : Math.min(photos.length - 1, Math.max(0, index + delta));
      setSelectedId(photos[next].id);
    },
    [photos, selectedId],
  );

  const nextGroup = useCallback(() => {
    if (!photos.length) return;
    const index = photos.findIndex((photo) => photo.id === selectedId);
    const currentGroup = index >= 0 ? photos[index].group_id : null;
    for (let i = Math.max(0, index + 1); i < photos.length; i += 1) {
      if (photos[i].group_id !== currentGroup || currentGroup === null) {
        setSelectedId(photos[i].id);
        return;
      }
    }
  }, [photos, selectedId]);

  // Keyboard: the whole review loop without the mouse.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Escape closes a dialog even from inside one of its inputs.
      if (event.key === "Escape") {
        if (exportOpen) setExportOpen(false);
        else if (cullOpen) setCullOpen(false);
        else if (loupeId !== null) setLoupeId(null);
        return;
      }

      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      // A dialog owns the keyboard while it is open: no reviewing through it.
      if (exportOpen || cullOpen) return;
      if (!selected) {
        if (event.key === " " && photos.length) {
          event.preventDefault();
          setSelectedId(photos[0].id);
        }
        return;
      }

      switch (event.key) {
        case "ArrowRight":
        case "ArrowDown":
        case "j":
          event.preventDefault();
          move(1);
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "k":
          event.preventDefault();
          move(-1);
          break;
        case "g":
          event.preventDefault();
          nextGroup();
          break;
        case " ":
          event.preventDefault();
          setLoupeId(loupeId === null ? selected.id : null);
          break;
        case "x":
          event.preventDefault();
          void decide(selected, "reject");
          break;
        case "p":
        case "Enter":
          event.preventDefault();
          void decide(selected, "pick");
          break;
        case "u":
          event.preventDefault();
          void decide(selected, "clear");
          break;
        case "1":
        case "2":
        case "3":
        case "4":
        case "5":
          event.preventDefault();
          void decide(selected, undefined, Number(event.key));
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, photos, loupeId, exportOpen, cullOpen, move, nextGroup, decide]);

  if (!capabilities) {
    return <div className="boot">Starting Cullr…</div>;
  }

  // The manual update flow is reachable from both screens — the import screen
  // before a shoot is open and the review screen after — so it is built once.
  const updatePanel = updateOpen && (
    <UpdatePanel current={capabilities.version} onClose={() => setUpdateOpen(false)} />
  );

  if (!library) {
    return (
      <>
        <Import
          libraries={libraries}
          capabilities={capabilities}
          error={error}
          lastParent={libraries[0] ? parentOf(libraries[0].root_path) : undefined}
          onOpen={openFolder}
          onSelect={(chosen) => void openRecent(chosen)}
          onCheckUpdates={() => setUpdateOpen(true)}
        />
        {updatePanel}
      </>
    );
  }

  return (
    <div className="app">
      <TopBar
        library={library}
        stats={stats}
        job={job}
        busy={busy}
        onOpenCull={() => setCullOpen(true)}
        onExport={() => setExportOpen(true)}
        onPause={pauseScan}
        onResume={resumeScan}
      />

      <div className="app__body">
        <Sidebar filter={filter} stats={stats} onChange={setFilter} />
        <main className="app__main">
          {unreadable && unreadable.id !== dismissedJob && (
            <div className="sheet-note sheet-note--warning" role="status">
              <span>
                {(unreadable.error_count ?? 0) >= (unreadable.total || 0)
                  ? `None of the ${unreadable.total} frames could be read.`
                  : `${unreadable.error_count} of ${unreadable.total} frames could not be read.`}
                {unreadable.error_sample ? <code>{unreadable.error_sample}</code> : null}
              </span>
              <button type="button" className="button" onClick={() => setFilter("errors")}>
                View unreadable files
              </button>
              <button
                type="button"
                className="button"
                onClick={() => setDismissedJob(unreadable.id)}
              >
                Dismiss
              </button>
            </div>
          )}
          {listed === 0 && !busy && !job && (
            <p className="sheet-note">
              No cull yet.{" "}
              <button type="button" className="button" onClick={() => setCullOpen(true)}>
                Cull settings
              </button>{" "}
              to set the parameters and start the cull.
            </p>
          )}
          <People
            people={people}
            suggestions={suggestions}
            pending={pendingMerge}
            activeId={personId}
            onSelect={setPersonId}
            onMerge={(suggestion) => void linkPeople(suggestion, true)}
            onDistinct={(suggestion) => void linkPeople(suggestion, false)}
          />
          {listed > photos.length && (
            <p className="sheet-note">
              Showing {photos.length} of {listed} frames. Narrow the filter to
              review the rest.
            </p>
          )}
          {busy && photos.length === 0 ? (
            // "No frames match this filter." is wrong while a scan is running:
            // nothing has been filtered out, the rows simply are not there yet.
            // The job's own line matches the top bar, so the two cannot
            // disagree about what is happening.
            <div className="sheet-empty" role="status">
              <p>{scanProgress} — frames appear here as they are checked.</p>
            </div>
          ) : (
            <Sheet
              photos={photos}
              people={people}
              selectedId={selectedId}
              onSelect={(photo) => setSelectedId(photo.id)}
              onOpen={(photo) => {
                setSelectedId(photo.id);
                setLoupeId(photo.id);
              }}
            />
          )}
        </main>
      </div>

      <footer className="shortcuts">
        <span><kbd>←</kbd><kbd>→</kbd> move</span>
        <span><kbd>space</kbd> inspect</span>
        <span><kbd>p</kbd> keep</span>
        <span><kbd>x</kbd> reject</span>
        <span><kbd>1</kbd>–<kbd>5</kbd> stars</span>
        <span><kbd>g</kbd> next group</span>
        <button type="button" onClick={() => setUpdateOpen(true)}>
          Cullr {capabilities.version} · Check for updates
        </button>
        <button
          type="button"
          onClick={() => {
            setLibrary(null);
            setDiscoveredTotal(null);
          }}
        >
          Change folder
        </button>
      </footer>

      {/* How far the cull has got, at the window's own edge. The top bar names
          the stage ("Analysed 1,400 of 3,016"); this is the part that reads at
          a glance while you are looking at the grid, not at the header. */}
      {(busy || paused) && (
        <div
          className={`cullbar${cullPercent === null ? " is-indeterminate" : ""}`}
          role="progressbar"
          aria-label="Cull progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={cullPercent ?? undefined}
          aria-valuetext={
            cullPercent === null
              ? "Starting…"
              : `${cullPercent}% — ${job?.message ?? job?.stage ?? (paused ? "paused" : "scanning")}`
          }
        >
          <span style={cullPercent === null ? undefined : { width: `${cullPercent}%` }} />
        </div>
      )}

      {updatePanel}

      {loupePhoto && (
        <Loupe
          photo={loupePhoto}
          people={people}
          analysisEdge={capabilities.analysis_long_edge}
          onClose={() => setLoupeId(null)}
          onDecide={(decision) => void decide(loupePhoto, decision ?? "clear")}
          onStar={(star) => void decide(loupePhoto, undefined, star)}
          onPrev={() => move(-1)}
          onNext={() => move(1)}
        />
      )}

      {cullOpen && (
        <CullDialog
          stats={stats}
          discoveredTotal={discoveredTotal}
          settings={settings}
          capabilities={capabilities}
          busy={busy}
          onChange={setSettings}
          onStart={(force) => {
            setCullOpen(false);
            void startScan(library.id, settings, force);
          }}
          onOpenExport={() => {
            setCullOpen(false);
            setExportOpen(true);
          }}
          onClose={() => setCullOpen(false)}
          onSemanticReady={refreshCapabilities}
        />
      )}

      {exportOpen && (
        <ExportPanel
          libraryId={library.id}
          stats={stats}
          remembered={exportMemory}
          onRemembered={setExportMemory}
          onClose={() => setExportOpen(false)}
        />
      )}

      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}
    </div>
  );
}
