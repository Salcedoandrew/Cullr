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
import type {
  Capabilities, ExportMemory, FilterKey, ImportOptions, Job, Library, MergeSuggestion,
  Person, Photo, Settings, Stats,
} from "./types";

const FALLBACK_SETTINGS: Settings = {
  preset: "wedding",
  strictness: null,
  keep_min_per_group: 1,
  detect_blinks: true,
  detect_blur: true,
  detect_highlights: true,
  group_duplicates: true,
  person_coverage: true,
  target_ratio: null,
  overwrite_xmp: true,
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
  const [cullOpen, setCullOpen] = useState(false);
  const [discoveredTotal, setDiscoveredTotal] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const busy = job?.status === "running";
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
    const timer = window.setInterval(() => {
      void api
        .job(job.id)
        .then((next) => {
          if (stopped) return;
          setJob(next);
          if (next.status !== "running") {
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
        // Count the import set first, so the cull options show the real
        // total instead of 0. Discovery is a names-only walk; if it fails
        // the dialog still opens and falls back to the percent-only display.
        try {
          const found = await api.discover(created.id);
          setDiscoveredTotal(found.total);
        } catch {
          setDiscoveredTotal(null);
        }
        // No auto-cull: the cull options open first, and the job starts only
        // when the user presses Start Cull in that window.
        setCullOpen(true);
      } catch (event) {
        setError(errorText(event));
        throw event;
      }
    },
    [capabilities, startScan],
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

  if (!library) {
    return (
      <Import
        libraries={libraries}
        capabilities={capabilities}
        error={error}
        lastParent={libraries[0] ? parentOf(libraries[0].root_path) : undefined}
        onOpen={openFolder}
        onSelect={(chosen) => {
          setLibrary(chosen);
          setFilter("all");
          setPersonId(null);
          setDiscoveredTotal(null);
        }}
      />
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
      />

      <div className="app__body">
        <Sidebar filter={filter} stats={stats} onChange={setFilter} />
        <main className="app__main">
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
        </main>
      </div>

      <footer className="shortcuts">
        <span><kbd>←</kbd><kbd>→</kbd> move</span>
        <span><kbd>space</kbd> inspect</span>
        <span><kbd>p</kbd> keep</span>
        <span><kbd>x</kbd> reject</span>
        <span><kbd>1</kbd>–<kbd>5</kbd> stars</span>
        <span><kbd>g</kbd> next group</span>
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
        />
      )}

      {exportOpen && (
        <ExportPanel
          libraryId={library.id}
          libraryPath={library.root_path}
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
