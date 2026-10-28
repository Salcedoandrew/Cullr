import type { Job, Library, Stats } from "../types";

interface Props {
  library: Library;
  stats: Stats | null;
  job: Job | null;
  busy: boolean;
  onOpenCull: () => void;
  onExport: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}

export function TopBar({
  library, stats, job, busy, onOpenCull, onExport, onPause, onResume, onCancel,
}: Props) {
  const paused = job?.status === "paused";

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
        {busy && (
          <>
            <button type="button" className="button" onClick={onPause}>
              Pause
            </button>
            <button type="button" className="button" onClick={onCancel}>
              Cancel
            </button>
          </>
        )}
        {paused && (
          <button type="button" className="button button--primary" onClick={onResume}>
            Resume
          </button>
        )}
        <button type="button" className="button" onClick={onOpenCull}>
          Cull settings
        </button>
        <button type="button" className="button" onClick={onExport} disabled={busy || !stats?.total}>
          Export
        </button>
      </div>

      {job && (job.status === "running" || paused) && (
        // The bar for this lives on the window's bottom edge; here it is only
        // the sentence, so the same number is not drawn twice.
        <div className="progress" role="status" aria-live="polite">
          <p>
            {paused
              ? job.message ?? "Paused"
              : `${job.message ?? job.stage}${job.total > 0 ? ` · ${job.done}/${job.total}` : ""}`}
          </p>
        </div>
      )}
      {job?.status === "cancelled" && (
        // Say that the cancel landed: a scan that stops on its own without a
        // word reads like a crash.
        <div className="progress" role="status" aria-live="polite">
          <p>{job.message ?? "Cancelled"}</p>
        </div>
      )}
      {job?.status === "error" && <p className="topbar__error">{job.error}</p>}
    </header>
  );
}
