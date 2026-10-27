import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type {
  Capabilities, RatingCategory, RatingScheme, SemanticStatus, Settings, Stats,
} from "../types";

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
  /** Photos counted at import, before any scan. Used only while stats is null. */
  discoveredTotal?: number | null;
  settings: Settings;
  capabilities: Capabilities;
  busy: boolean;
  onChange: (settings: Settings) => void;
  onStart: (force: boolean) => void;
  onOpenExport: () => void;
  onClose: () => void;
  /** The scene model finished downloading; capabilities can be refreshed. */
  onSemanticReady: () => void;
}

const MIN_PERCENT = 10;
const MAX_PERCENT = 60;
const DEFAULT_RATIO = 0.2;

/** The Stars & Colors rows, in the order the panel shows them. */
const RATING_ROWS: { key: RatingCategory; label: string }[] = [
  { key: "selected", label: "Selected" },
  { key: "highlights", label: "Highlights" },
  { key: "blurry", label: "Blurred" },
  { key: "closed_eyes", label: "Closed Eyes" },
  { key: "duplicate", label: "Duplicates" },
  { key: "unrated", label: "Unrated" },
];
const STAR_CHOICES = [-1, 0, 1, 2, 3, 4, 5];
// Lightroom's five colour labels (what other tools actually display), plus the
// option to write no label at all. "Orange", for instance, is not one of them.
const LABEL_CHOICES = ["Red", "Yellow", "Green", "Blue", "Purple", ""];

const starLabel = (stars: number): string =>
  stars < 0 ? "✗" : stars === 0 ? "unrated" : `★${stars}`;

/** One line under the main row, derived from whatever scheme is saved. */
function schemeSummary(scheme: RatingScheme): string {
  const selected = scheme.selected;
  const rejects = RATING_ROWS.filter((row) => row.key !== "selected")
    .map((row) => scheme[row.key]);
  const sameStars = new Set(rejects.map((entry) => entry.stars)).size === 1;
  const sameColor = new Set(rejects.map((entry) => entry.color)).size === 1;
  const reject = sameStars && sameColor
    ? `${starLabel(rejects[0].stars)} · ${rejects[0].color || "No label"}`
    : "their own rows";
  return `Keepers get ${starLabel(selected.stars)} · ${selected.color || "No label"}, `
    + `rejects ${reject}, written as XMP sidecars.`;
}

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

/**
 * One Customize row: a label with an info dot on the left, a switch on the
 * right. The switch is brass, the app's accent — not the dialog's blue.
 */
function DetectToggle({
  label, info, checked, onChange,
}: {
  label: string;
  info: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="cull__toggle">
      <span className="cull__toggle-text">
        {label}
        <span className="cull__info" title={info}><InfoDot /></span>
      </span>
      <input
        type="checkbox"
        className="cull__toggle-input"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span className="cull__toggle-track" aria-hidden="true">
        <span className="cull__toggle-thumb" />
      </span>
    </label>
  );
}

const sizeLabel = (bytes: number): string =>
  `${Math.max(1, Math.round(bytes / (1 << 20)))} MB`;

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
  stats, discoveredTotal, settings, capabilities, busy, onChange, onStart, onOpenExport,
  onClose, onSemanticReady,
}: Props) {
  const preset = capabilities.presets.find((option) => option.key === settings.preset);
  // A scan that already ran knows the real total; before the first scan the
  // count from import stands in, so the target never reads 0. A plain `??`
  // does not work here: the photo list refresh reports total 0 for a library
  // with no scan yet, and 0 is not nullish, so it would swallow the
  // discovered count. Only a scanned total above 0 wins.
  const total = stats?.total ? stats.total : (discoveredTotal ?? 0);

  const [targetMode, setTargetMode] = useState(settings.target_ratio !== null);
  const [ratio, setRatio] = useState(settings.target_ratio ?? DEFAULT_RATIO);
  // The scene model is downloaded from here on first use; the dialog polls
  // its progress and unlocks Start Cull when the file is installed.
  const [semantic, setSemantic] = useState<SemanticStatus | null>(null);
  const [semanticError, setSemanticError] = useState<string | null>(null);
  const downloading = semantic?.downloading ?? false;
  // The dialog has two panels: the cull itself, and the drill-in ratings
  // editor. The back chevron walks out of the second, closes from the first.
  const [view, setView] = useState<"main" | "ratings">("main");
  const [scheme, setScheme] = useState<RatingScheme>(
    settings.rating_scheme ?? capabilities.default_settings.rating_scheme,
  );
  const [reExport, setReExport] = useState(false);
  // The target helper doubles as a callout while the slider is held, the way
  // Aftershoot shows it; releasing returns it to the always-on block so the
  // copy is never hidden.
  const [draggingTarget, setDraggingTarget] = useState(false);
  const goBack = () => (view === "ratings" ? setView("main") : onClose());
  const dialog = useRef<HTMLDivElement>(null);

  useEffect(() => {
    dialog.current?.focus();
  }, []);

  useEffect(() => {
    // Capture phase: Escape must not reach the review loop behind the sheet.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      // Escape walks back out of the ratings panel before closing the dialog.
      if (view === "ratings") setView("main");
      else onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, view]);

  const percent = Math.round(ratio * 100);
  const count = total > 0 ? Math.max(1, Math.round(ratio * total)) : 0;
  const strictness = settings.strictness ?? preset?.strictness ?? 50;

  // Auto Target's own control: the overall selection strictness as four named
  // stops, never a raw number. The nearest stop to the effective value is the
  // one shown, so an untouched library reads the preset's own strictness.
  const stops = capabilities.strictness_stops ?? [];
  const effectiveSelection = settings.selection_strictness ?? preset?.strictness ?? 50;
  const stopIndex = stops.reduce(
    (best, stop, index) =>
      Math.abs(stop.value - effectiveSelection) < Math.abs(stops[best].value - effectiveSelection)
        ? index
        : best,
    0,
  );
  const stop = stops[stopIndex];
  const chooseStop = (index: number) => {
    const next = stops[index];
    if (next) onChange({ ...settings, selection_strictness: next.value });
  };

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

  // Poll only while a download is running. A finished download refreshes the
  // capabilities, which is what makes the toggle's info text honest.
  useEffect(() => {
    if (!semantic?.downloading) return;
    let stopped = false;
    const timer = window.setInterval(() => {
      void api
        .semanticStatus()
        .then((next) => {
          if (stopped) return;
          setSemantic(next);
          if (next.error) setSemanticError(next.error);
          if (next.available) onSemanticReady();
        })
        .catch(() => undefined);
    }, 800);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [semantic?.downloading, onSemanticReady]);

  const toggleSemantic = async (value: boolean) => {
    setSemanticError(null);
    onChange({ ...settings, semantic_dedup: value });
    if (!value) {
      // The download carries on in the background, but this cull no longer
      // waits for it, so Start Cull unlocks right away.
      setSemantic(null);
      return;
    }
    if (capabilities.semantic) return;
    try {
      const started = await api.downloadSceneModel();
      setSemantic(started);
      if (started.error) setSemanticError(started.error);
      else if (started.available) onSemanticReady();
    } catch (event) {
      setSemanticError(event instanceof Error ? event.message : String(event));
    }
  };

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
          <button type="button" className="cull__close" onClick={goBack} aria-label="Back">
            <Chevron direction="left" />
          </button>
          <h2>{view === "ratings" ? "Set Star/Color Ratings" : "AI Automated Cull"}</h2>
          <button type="button" className="cull__close" onClick={onClose} aria-label="Close">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </header>

        <div className="cull__scroll">
        {view === "ratings" ? (
          <>
            <p className="cull__ratings-intro">
              Every outcome writes one XMP rating and colour label. A star you
              pick by hand still wins over the row below.
            </p>
            <div className="cull__ratings">
              {RATING_ROWS.map(({ key, label }) => (
                <div className="cull__rating-row" key={key}>
                  <span className="cull__rating-name">{label}</span>
                  <select
                    className="cull__rating-select cull__rating-select--stars"
                    aria-label={`${label} stars`}
                    value={scheme[key].stars}
                    onChange={(event) => setScheme({
                      ...scheme,
                      [key]: { ...scheme[key], stars: Number(event.target.value) },
                    })}
                  >
                    {STAR_CHOICES.map((stars) => (
                      <option key={stars} value={stars}>
                        {stars < 0 ? "✗ Rejected" : stars === 0 ? "0 stars" : `★${stars}`}
                      </option>
                    ))}
                  </select>
                  <select
                    className="cull__rating-select"
                    aria-label={`${label} colour`}
                    value={scheme[key].color}
                    onChange={(event) => setScheme({
                      ...scheme,
                      [key]: { ...scheme[key], color: event.target.value },
                    })}
                  >
                    {LABEL_CHOICES.map((color) => (
                      <option key={color || "none"} value={color}>
                        {color || "No label"}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
            <label className="cull__check">
              <input
                type="checkbox"
                checked={reExport}
                onChange={(event) => setReExport(event.target.checked)}
              />
              <span>Re-export the shoot to apply rating preferences</span>
            </label>
            <div className="cull__foot cull__foot--panel">
              <button
                type="button"
                className="cull__quiet"
                onClick={() => setScheme(capabilities.default_settings.rating_scheme)}
              >
                Restore defaults
              </button>
              <button
                type="button"
                className="cull__start"
                onClick={() => {
                  onChange({ ...settings, rating_scheme: scheme });
                  setView("main");
                  if (reExport) onOpenExport();
                }}
              >
                Save
              </button>
            </div>
          </>
        ) : (
        <>
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

          {!targetMode && stop && (
            <div className="cull__selection">
              <div className="cull__target-head">
                <span className="cull__target-label">
                  Set selection strictness
                  <span
                    className="cull__info"
                    title="How picky the cull is: EXTREME keeps only the strongest frames, MORE keeps anything that passes the quality gates."
                  >
                    <InfoDot />
                  </span>
                </span>
                <span className="cull__stop-name">{stop.label}</span>
              </div>
              <div className="cull__slider">
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, stops.length - 1)}
                  step={1}
                  value={stopIndex}
                  aria-label="Selection strictness: extreme to more"
                  aria-valuetext={stop.label}
                  onChange={(event) => chooseStop(Number(event.target.value))}
                />
              </div>
              <div className="cull__scale">
                <span>{stops[0].label}</span>
                <span>{stops[stops.length - 1].label}</span>
              </div>
            </div>
          )}

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
                onPointerDown={() => setDraggingTarget(true)}
                onPointerUp={() => setDraggingTarget(false)}
                onPointerCancel={() => setDraggingTarget(false)}
                onKeyDown={() => setDraggingTarget(true)}
                onKeyUp={() => setDraggingTarget(false)}
                onBlur={() => setDraggingTarget(false)}
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

            {targetMode && draggingTarget ? (
              <output className="cull__callout">{helper}</output>
            ) : (
              <p className="cull__help">
                <span className="cull__help-icon"><InfoDot /></span>
                {targetMode
                  ? helper
                  : "Auto Target keeps every frame that passes the quality gates — sharp, "
                    + "open eyes, one per burst. Pick Cull to Target to also cap the count."}
              </p>
            )}
          </div>
        </section>

        <details className="cull__customize">
          <summary>
            <span className="cull__twist"><Chevron /></span>
            Customize
          </summary>
          <div className="cull__advanced">
            {/* The four Detect toggles first, in Aftershoot's order; Cullr's
                own extras (keep-min, coverage, re-analyse) follow them. */}
            <DetectToggle
              label="Detect Duplicates"
              info="Groups near-identical frames so only the best of each burst is kept."
              checked={settings.group_duplicates}
              onChange={(value) => onChange({ ...settings, group_duplicates: value })}
            />
            {settings.group_duplicates && capabilities.semantic_runtime && (
              <DetectToggle
                label="Detect Duplicates (AI)"
                info="Also catches the same shot when a hand or expression moved. The first use downloads a 42 MB scene model, once."
                checked={settings.semantic_dedup}
                onChange={(value) => void toggleSemantic(value)}
              />
            )}
            {settings.group_duplicates && !capabilities.semantic_runtime && (
              <p className="cull__warn">
                Scene-model duplicate detection is not installed in this build.
              </p>
            )}
            {downloading && (
              <p className="cull__help">
                <span className="cull__help-icon"><InfoDot /></span>
                Downloading the scene model — {sizeLabel(semantic?.done ?? 0)}
                {semantic && semantic.total > 0 ? ` of ${sizeLabel(semantic.total)}` : ""}.
                Start Cull unlocks when it finishes.
              </p>
            )}
            {semanticError && (
              <p className="cull__warn">
                Scene model download failed: {semanticError}. Turn the option off
                and on to try again.
              </p>
            )}
            <DetectToggle
              label="Detect Highlights"
              info="Rejects frames with blown highlights."
              checked={settings.detect_highlights}
              onChange={(value) => onChange({ ...settings, detect_highlights: value })}
            />
            <DetectToggle
              label="Detect Blurry Photos"
              info="Rejects soft frames — frame-wide, face-crop, and soft-for-its-burst."
              checked={settings.detect_blur}
              onChange={(value) => onChange({ ...settings, detect_blur: value })}
            />

            <div className="cull__strict">
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={strictness}
                aria-label="Blur strictness: lenient to strict"
                onChange={(event) =>
                  onChange({ ...settings, strictness: Number(event.target.value) })
                }
              />
              <div className="cull__strict-labels" aria-hidden="true">
                <span>Lenient</span>
                <span>Moderate</span>
                <span>Strict</span>
              </div>
            </div>

            <DetectToggle
              label="Detect Closed Eyes"
              info="Rejects frames where the subject's eyes are closed."
              checked={settings.detect_blinks}
              onChange={(value) => onChange({ ...settings, detect_blinks: value })}
            />

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
              disabled={busy || downloading}
            >
              Re-analyse every frame (ignore cached results)
            </button>
          </div>
        </details>

        <div className="cull__row">
          <div>
            <p className="cull__row-label">Stars &amp; Colors Rating</p>
            <p className="cull__row-sub">
              {schemeSummary(settings.rating_scheme ?? capabilities.default_settings.rating_scheme)}
            </p>
          </div>
          <button
            type="button"
            className="cull__row-action"
            onClick={() => setView("ratings")}
            aria-label="Edit star and colour ratings"
          >
            Update
            <Chevron />
          </button>
        </div>
        </>
        )}
        </div>

        {view === "main" && (
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
            disabled={busy || downloading}
          >
            {busy ? "Culling…" : downloading ? "Downloading model…" : "Start Cull"}
          </button>
        </div>
        )}
      </div>
    </div>
  );
}
