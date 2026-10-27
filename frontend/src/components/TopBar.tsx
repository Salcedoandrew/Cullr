import type { Job, Library, Stats } from "../types";

interface Props {
  library: Library;
  stats: Stats | null;
  job: Job | null;
  busy: boolean;
  onOpenCull: () => void;
  onExport: () => void;
}

export function TopBar({ library, stats, job, busy, onOpenCull, onExport }: Props) {
  const progress =
    job && job.total > 0 ? Math.min(100, Math.round((job.done / job.total) * 100)) : null;

  return (
    <header className="topbar">
      <div className="topbar__identity">
        <h1>Cullr</h1>
        <div>
          <p className="topbar__library">{library.name}</p>
          <p className="topbar__path">{library.root_path}</p>
        </div>
        {stats && (
          <p className="topbar__summary">
            <span><b>{stats.total}</b>frames</span>
            <span><b>{stats.keep}</b>kept</span>
            <span><b>{stats.groups}</b>groups</span>
            {stats.persons > 0 && <span><b>{stats.persons}</b>people</span>}
          </p>
        )}
      </div>

      <div className="topbar__actions">
        <button type="button" className="button" onClick={onOpenCull}>
          Cull settings
        </button>
        <button type="button" className="button" onClick={onExport} disabled={busy || !stats?.total}>
          Export
        </button>
      </div>

      {job && job.status === "running" && (
        <div className="progress" role="status" aria-live="polite">
          <div className="progress__bar">
            <span style={{ width: `${progress ?? 4}%` }} />
          </div>
          <p>
            {job.message ?? job.stage}
            {job.total > 0 && ` · ${job.done}/${job.total}`}
          </p>
        </div>
      )}
      {job?.status === "error" && <p className="topbar__error">{job.error}</p>}
    </header>
  );
}
