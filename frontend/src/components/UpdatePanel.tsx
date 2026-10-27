import { useCallback, useEffect, useRef, useState } from "react";
import { api, formatBytes } from "../api";
import type { UpdateStatus } from "../types";

/**
 * "Check for Updates" — a manual, on-demand update.
 *
 * Nothing here runs on its own: the panel fetches the server's current state
 * when it opens (so a download that already finished is still there), and the
 * only network call the app ever makes to GitHub starts with a click.
 *
 * The states are kept visibly distinct because they mean different things: a
 * check that found nothing, a download in flight, a verified file waiting, and
 * an install that is about to close the window are not the same message.
 */

interface Props {
  /** The running version, from /api/capabilities. */
  current: string;
  onClose: () => void;
}

/** Progress polls: the same cadence the cull dialog uses for a model download. */
const POLL_MS = 800;

export function UpdatePanel({ current, onClose }: Props) {
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialog = useRef<HTMLDivElement>(null);

  const phase = status?.phase ?? "idle";
  const version = status?.current ?? current;
  const installing = phase === "installing";

  useEffect(() => {
    dialog.current?.focus();
  }, []);

  // Escape closes, as in every other dialog here — except once the installer
  // is running, when the window is on its way out anyway.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !installing) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, installing]);

  // Pick up whatever the server already knows about.
  useEffect(() => {
    let live = true;
    void api
      .updateStatus()
      .then((state) => {
        if (live) setStatus(state);
      })
      .catch(() => {
        /* the panel still works from a cold start */
      });
    return () => {
      live = false;
    };
  }, []);

  // While bytes are moving, poll. The check itself is synchronous, so it needs
  // no polling — the button says "Checking…" until the request answers.
  useEffect(() => {
    if (phase !== "downloading") return;
    const timer = window.setInterval(() => {
      void api
        .updateStatus()
        .then(setStatus)
        .catch(() => {
          /* keep the last known progress rather than blanking the bar */
        });
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [phase]);

  const run = useCallback(
    async (call: () => Promise<UpdateStatus>, markChecking = false) => {
      setError(null);
      if (markChecking) setChecking(true);
      try {
        setStatus(await call());
      } catch (event) {
        setError(event instanceof Error ? event.message : String(event));
      } finally {
        if (markChecking) setChecking(false);
      }
    },
    [],
  );

  const check = () => void run(api.checkForUpdate, true);
  const download = () => void run(api.downloadUpdate);
  const cancel = () => void run(api.cancelUpdate);
  const install = () => void run(api.installUpdate);

  const progress = status?.percent ?? null;
  const total = status?.total ?? 0;
  const done = status?.done ?? 0;
  const latest = status?.latest ?? null;

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label="Check for updates">
      <div className="modal__body update" ref={dialog} tabIndex={-1}>
        <header className="modal__head">
          <h2>Check for updates</h2>
          <button
            type="button"
            className="loupe__close loupe__close--inline"
            onClick={onClose}
            aria-label="Close"
            disabled={installing}
          >
            ×
          </button>
        </header>

        <p className="update__version" role="status" aria-live="polite">
          Cullr <b>{version}</b> is installed
          {status?.installed_at ? <> · {status.installed_at}</> : null}
        </p>

        {checking && <p className="modal__hint">Checking for updates…</p>}

        {!checking && phase === "idle" && (
          <p className="modal__hint">
            Updates are manual and checked only when you ask. Cullr downloads the
            new version, verifies it against the release&rsquo;s SHA-256, replaces
            the installed app and reopens it.
          </p>
        )}

        {phase === "up_to_date" && (
          <p className="update__state update__state--ok">
            No updates available — Cullr {version} is the latest version.
          </p>
        )}

        {phase === "available" && (
          <>
            <p className="update__state">
              Cullr <b>{latest}</b> is available
              {status?.size ? ` (${formatBytes(status.size)})` : null}.
            </p>
            <p className="modal__hint">
              The release notes are on GitHub; nothing has been downloaded yet.
            </p>
          </>
        )}

        {phase === "downloading" && (
          <>
            <div className="progress__bar update__bar">
              <span style={{ width: `${progress ?? 4}%` }} />
            </div>
            <p className="update__state">
              Downloading Cullr {latest} — {formatBytes(done)}
              {total ? ` of ${formatBytes(total)}` : ""}
              {progress !== null ? ` · ${progress}%` : ""}
            </p>
          </>
        )}

        {phase === "ready" && (
          <>
            <p className="update__state update__state--ok">
              Cullr {latest} is downloaded and its SHA-256 matches the release.
            </p>
            <p className="modal__hint">
              Installing quits Cullr, replaces{" "}
              {status?.installed_at ?? "the installed app"} and reopens the new
              version. Your catalog, settings and photos are not touched.
            </p>
          </>
        )}

        {phase === "cancelled" && (
          <p className="update__state">
            The download was cancelled and deleted. Cullr {version} is still
            installed and unchanged.
          </p>
        )}

        {phase === "installing" && (
          <p className="update__state update__state--ok">
            Restarting on Cullr {latest}. The window closes while the new version
            is put in place — Cullr reopens by itself.
          </p>
        )}

        {phase === "error" && (
          <>
            <p className="update__state update__state--bad">
              {status?.error ?? "The update could not be completed."}
            </p>
            <p className="modal__hint">
              Nothing was installed; Cullr {version} is untouched. The installer
              logs to {status?.update_log ?? "the update log"} when one has run.
            </p>
          </>
        )}

        {error && <p className="modal__error">{error}</p>}

        <div className="modal__actions update__actions">
          {phase === "idle" && (
            <button type="button" className="button button--primary" onClick={check}>
              {checking ? "Checking…" : "Check for updates"}
            </button>
          )}
          {(phase === "up_to_date" || phase === "cancelled" || phase === "error") && (
            <>
              {(phase === "cancelled" || phase === "error") && latest && (
                <button type="button" className="button button--primary" onClick={download}>
                  Download Cullr {latest}
                </button>
              )}
              <button type="button" className="button" onClick={check}>
                {phase === "up_to_date" ? "Check again" : "Check for updates"}
              </button>
            </>
          )}
          {phase === "available" && (
            <>
              <button type="button" className="button button--primary" onClick={download}>
                Download update
              </button>
              {status?.notes_url && (
                <a
                  className="button update__notes"
                  href={status.notes_url}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  Release notes
                </a>
              )}
              <button type="button" className="button" onClick={onClose}>
                Not now
              </button>
            </>
          )}
          {phase === "downloading" && (
            <button type="button" className="button" onClick={cancel}>
              Cancel download
            </button>
          )}
          {phase === "ready" && (
            <>
              <button type="button" className="button button--primary" onClick={install}>
                Install and restart
              </button>
              <button type="button" className="button" onClick={cancel}>
                Cancel
              </button>
            </>
          )}
          {installing && <span className="modal__hint">Installing…</span>}
        </div>
      </div>
    </div>
  );
}
