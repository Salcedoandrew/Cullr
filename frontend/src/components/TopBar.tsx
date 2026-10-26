import type { Capabilities, Job, Library, Settings, Stats } from "../types";

interface Props {
  library: Library;
  stats: Stats | null;
  settings: Settings;
  capabilities: Capabilities;
  job: Job | null;
  busy: boolean;
  onChange: (settings: Settings) => void;
  onScan: (force: boolean) => void;
  onExport: () => void;
}

export function TopBar({
  library, stats, settings, capabilities, job, busy, onChange, onScan, onExport,
}: Props) {
  const preset = capabilities.presets.find((p) => p.key === settings.preset);
  const strictness = settings.strictness ?? preset?.strictness ?? 50;
  const progress =
    job && job.total > 0 ? Math.min(100, Math.round((job.done / job.total) * 100)) : null;

  return (
    <header className="topbar">
      <div className="topbar__identity">
        <h1>Sharpen</h1>
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

      <details className="scan-settings">
        <summary>Culling settings</summary>
        <div className="scan-settings__body">
          <label>
            <span>Shoot type</span>
            <select
              value={settings.preset}
              onChange={(event) => onChange({ ...settings, preset: event.target.value })}
            >
              {capabilities.presets.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <p className="scan-settings__hint">{preset?.description}</p>

          <label>
            <span>Strictness {strictness}</span>
            <input
              type="range"
              min={0}
              max={100}
              step={5}
              value={strictness}
              onChange={(event) =>
                onChange({ ...settings, strictness: Number(event.target.value) })
              }
            />
          </label>

          <label>
            <span>Keep at least, per group</span>
            <input
              type="number"
              min={1}
              max={10}
              value={settings.keep_min_per_group ?? 1}
              onChange={(event) =>
                onChange({
                  ...settings,
                  keep_min_per_group: Math.max(1, Number(event.target.value) || 1),
                })
              }
            />
          </label>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={settings.detect_blinks}
              onChange={(event) =>
                onChange({ ...settings, detect_blinks: event.target.checked })
              }
            />
            <span>Reject closed eyes</span>
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={settings.group_duplicates}
              onChange={(event) =>
                onChange({ ...settings, group_duplicates: event.target.checked })
              }
            />
            <span>Group similar shots</span>
          </label>
          <label className="checkbox" title="Promotes one extra frame when a person would otherwise be culled">
            <input
              type="checkbox"
              checked={settings.person_coverage}
              onChange={(event) =>
                onChange({ ...settings, person_coverage: event.target.checked })
              }
            />
            <span>Keep a frame of everyone</span>
          </label>

          {!capabilities.faces && (
            <p className="scan-settings__warning">
              Face model unavailable, so blur and duplicate detection still work but
              eyes-closed is off. {capabilities.faces_error}
            </p>
          )}

          <div className="scan-settings__actions">
            <button type="button" className="button button--primary" onClick={() => onScan(false)}>
              {busy ? "Scanning…" : "Scan folder"}
            </button>
            <button type="button" className="button" onClick={() => onScan(true)} disabled={busy}>
              Re-analyse all
            </button>
          </div>
        </div>
      </details>

      <div className="topbar__actions">
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
