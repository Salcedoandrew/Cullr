import { useEffect, useMemo, useRef, useState } from "react";
import type { Capabilities, Settings, Stats } from "../types";

/**
 * The pre-cull window: what kind of shoot this is, how many frames to keep, and
 * the advanced gates behind a disclosure.
 *
 * Two modes, and they answer different questions:
 *
 *  · "Auto target" asks *is this frame acceptable* — the quality gates decide.
 *  · "Cull to target" asks *how many frames do I want* — keepers are
 *    ranked by measured quality and cut at the target. Coverage still wins, so
 *    someone in the shoot is never dropped to hit a number.
 */

interface Props {
  stats: Stats | null;
  settings: Settings;
  capabilities: Capabilities;
  busy: boolean;
  onChange: (settings: Settings) => void;
  onStart: (force: boolean) => void;
  onOpenExport: () => void;
  onClose: () => void;
}

const MIN_PERCENT = 10;
const MAX_PERCENT = 60;
const DEFAULT_RATIO = 0.2;

function Chevron({ direction = "right" }: { direction?: "left" | "right" | "down" }) {
  const rotation = direction === "left" ? 180 : direction === "down" ? 90 : 0;
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: `rotate(${rotation}deg)` }}
      aria-hidden="true"
    >
      <path d="M9 5.5 15.5 12 9 18.5" />
    </svg>
  );
}

function InfoDot() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="8.6" />
      <path d="M12 11v5.4M12 7.9v.6" />
    </svg>
  );
}

function helperText(ratio: number): string {
  if (ratio < 0.2) {
    return "A lower target gives a tighter selection, focused on your best images. "
      + "Increase it to include more moments.";
  }
  if (ratio <= 0.35) {
    return "A balanced target: the strongest frame from each part of the shoot, with "
      + "room for the quieter moments.";
  }
  return "A higher target keeps more moments — and more near-duplicates. Lower it "
    + "for a tighter edit.";
}

export function CullDialog({
  stats, settings, capabilities, busy, onChange, onStart, onOpenExport, onClose,
}: Props) {
  const preset = capabilities.presets.find((option) => option.key === settings.preset);
  const total = stats?.total ?? 0;

  const [targetMode, setTargetMode] = useState(settings.target_ratio !== null);
  const [ratio, setRatio] = useState(settings.target_ratio ?? DEFAULT_RATIO);
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    dialog.current?.focus();
  }, []);

  useEffect(() => {
    // Capture phase: Escape must not reach the review loop behind the sheet.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const percent = Math.round(ratio * 100);
  const count = total > 0 ? Math.max(1, Math.round(ratio * total)) : 0;
  const strictness = settings.strictness ?? preset?.strictness ?? 50;

  const clampRatio = (value: number) =>
    Math.min(MAX_PERCENT / 100, Math.max(MIN_PERCENT / 100, value));

  const setTarget = (value: number) => {
    const next = clampRatio(value);
    setRatio(next);
    if (targetMode) onChange({ ...settings, target_ratio: next });
  };

  const chooseMode = (target: boolean) => {
    setTargetMode(target);
    onChange({ ...settings, target_ratio: target ? clampRatio(ratio) : null });
  };

  const helper = useMemo(() => helperText(ratio), [ratio]);
  const handleOffset = ((percent - MIN_PERCENT) / (MAX_PERCENT - MIN_PERCENT)) * 100;

  return (
    <div
      className="cull-scrim"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="cull"
        role="dialog"
        aria-modal="true"
        aria-label="AI Automated Cull"
        ref={dialog}
        tabIndex={-1}
      >
        <header className="cull__head">
          <button type="button" className="cull__close" onClick={onClose} aria-label="Back">
            <Chevron direction="left" />
          </button>
          <h2>AI Automated Cull</h2>
          <button type="button" className="cull__close" onClick={onClose} aria-label="Close">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </header>

        <div className="cull__scroll">
        {/* The shoot type is the profile the gates are tuned for. */}
        <div className="cull__shoot">
          <select
            className="cull__shoot-select"
            value={settings.preset}
            onChange={(event) => onChange({ ...settings, preset: event.target.value })}
            aria-label="Shoot type"
          >
            {capabilities.presets.map((option) => (
              <option key={option.key} value={option.key}>{option.label}</option>
            ))}
          </select>
          <span className="cull__shoot-chevron"><Chevron /></span>
        </div>
        {preset?.description && <p className="cull__shoot-hint">{preset.description}</p>}

        <section className={`cull__card${targetMode ? "" : " is-auto"}`}>
          <div className="cull__modes" role="tablist" aria-label="Culling mode">
            <button
              type="button"
              role="tab"
              aria-selected={!targetMode}
              className={`cull__mode${targetMode ? "" : " is-on"}`}
              onClick={() => chooseMode(false)}
            >
              Auto Target
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={targetMode}
              className={`cull__mode${targetMode ? " is-on" : ""}`}
              onClick={() => chooseMode(true)}
            >
              Cull to Target
            </button>
          </div>

          <div className={`cull__target${targetMode ? "" : " is-off"}`}>
            <div className="cull__target-head">
              <span className="cull__target-label">
                Set amount of photos
                <span className="cull__info" title="How many frames the cull should keep.">
                  <InfoDot />
                </span>
              </span>
              <span className="cull__target-right">
                <span className="cull__percent">
                  ~{percent}%{total > 0 ? ` of ${total.toLocaleString()} photos` : ""}
                </span>
                <input
                  className="cull__count"
                  type="number"
                  inputMode="numeric"
                  aria-label="Photos to keep"
                  min={1}
                  value={count}
                  disabled={!targetMode}
                  onChange={(event) => {
                    const typed = Number(event.target.value);
                    if (!Number.isFinite(typed) || total === 0) return;
                    setTarget(typed / total);
                  }}
                />
              </span>
            </div>

            <div className="cull__slider">
              <input
                type="range"
                min={MIN_PERCENT}
                max={MAX_PERCENT}
                step={1}
                value={percent}
                disabled={!targetMode}
                aria-label="Target percentage"
                aria-valuetext={total > 0 ? `${percent}% of ${total} photos` : `${percent}%`}
                style={{ ["--fill" as string]: `${handleOffset}%` }}
                onChange={(event) => setTarget(Number(event.target.value) / 100)}
              />
              <output
                className="cull__bubble"
                style={{ left: `${Math.min(92, Math.max(8, handleOffset))}%` }}
              >
                {percent}%
              </output>
            </div>
            <div className="cull__scale">
              <span>{MIN_PERCENT}%</span>
              <span>{MAX_PERCENT}%</span>
            </div>

            <p className="cull__help">
              <span className="cull__help-icon"><InfoDot /></span>
              {targetMode
                ? helper
                : "Auto Target keeps every frame that passes the quality gates — sharp, "
                  + "open eyes, one per burst. Pick Cull to Target to also cap the count."}
            </p>
          </div>
        </section>

        <details className="cull__customize">
          <summary>
            <span className="cull__twist"><Chevron /></span>
            Customize
          </summary>
          <div className="cull__advanced">
            <label className="cull__field">
              <span>Strictness <b>{strictness}</b></span>
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

            <label className="cull__field cull__field--inline">
              <span>Keep at least per group</span>
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

            <label className="cull__check">
              <input
                type="checkbox"
                checked={settings.detect_blinks}
                onChange={(event) =>
                  onChange({ ...settings, detect_blinks: event.target.checked })
                }
              />
              <span>Reject closed eyes</span>
            </label>
            <label className="cull__check">
              <input
                type="checkbox"
                checked={settings.group_duplicates}
                onChange={(event) =>
                  onChange({ ...settings, group_duplicates: event.target.checked })
                }
              />
              <span>Group similar shots</span>
            </label>
            <label className="cull__check">
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
              <p className="cull__warn">
                Face model unavailable, so blur and duplicate detection still work but
                eyes-closed is off. {capabilities.faces_error}
              </p>
            )}

            <button
              type="button"
              className="cull__relink"
              onClick={() => onStart(true)}
              disabled={busy}
            >
              Re-analyse every frame (ignore cached results)
            </button>
          </div>
        </details>

        <div className="cull__row">
          <div>
            <p className="cull__row-label">Stars &amp; Colors Rating</p>
            <p className="cull__row-sub">
              Keepers get ★1 · Green, rejects ✗ · Red, written as XMP sidecars.
            </p>
          </div>
          <button type="button" className="cull__row-action" onClick={onOpenExport}>
            Update
            <Chevron />
          </button>
        </div>
        </div>

        <div className="cull__foot">
          <label className="cull__check">
            <input
              type="checkbox"
              checked={settings.overwrite_xmp}
              onChange={(event) =>
                onChange({ ...settings, overwrite_xmp: event.target.checked })
              }
            />
            <span>Overwrite ratings in XMP files</span>
          </label>
          <button
            type="button"
            className="cull__start"
            onClick={() => onStart(false)}
            disabled={busy}
          >
            {busy ? "Culling…" : "Start Cull"}
          </button>
        </div>
      </div>
    </div>
  );
}
