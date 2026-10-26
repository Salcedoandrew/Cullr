import { reasonLabel, thumbUrl } from "../api";
import type { Photo } from "../types";

interface Props {
  photo: Photo;
  active: boolean;
  groupSize?: number;
  coveredLabel?: string | null;
  onSelect: () => void;
  onOpen: () => void;
}

export function Frame({
  photo, active, groupSize, coveredLabel, onSelect, onOpen,
}: Props) {
  const rejected = photo.verdict === "reject";
  const score = photo.quality === null ? null : Math.round(photo.quality * 100);
  // A frame kept only to cover a person says so, and keeps its technical
  // caveat visible: "Kept for Person 2 · soft focus".
  const coverage = photo.covered_person === null
    ? null
    : [
        `Kept for ${coveredLabel ?? "a person"}`,
        photo.reason ? reasonLabel(photo.reason).toLowerCase() : null,
      ]
        .filter(Boolean)
        .join(" · ");
  const note = coverage
    ?? (photo.reason
      ? reasonLabel(photo.reason)
      : rejected && photo.user_decision === "reject"
        ? "Rejected by you"
        : photo.is_lead
          ? `Best of ${groupSize ?? 2}`
          : null);
  const classes = [
    "frame",
    active ? "is-active" : "",
    rejected ? "is-rejected" : "",
    photo.is_lead ? "is-lead" : "",
    photo.covered_person !== null ? "is-coverage" : "",
    photo.error ? "is-broken" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type="button"
      className={classes}
      onClick={onSelect}
      onDoubleClick={onOpen}
      aria-label={`${photo.filename}, ${rejected ? "rejected" : "kept"}${
        photo.reason ? `, ${reasonLabel(photo.reason)}` : ""
      }`}
      aria-pressed={active}
      title={photo.path}
    >
      <span className="frame__image">
        {photo.error ? (
          <span className="frame__broken">No preview</span>
        ) : (
          <img src={thumbUrl(photo)} alt="" loading="lazy" draggable={false} />
        )}
        {rejected && <span className="frame__mark" aria-hidden="true">×</span>}
        {photo.star > 0 && (
          <span className="frame__stars" aria-label={`${photo.star} stars`}>
            {"•".repeat(photo.star)}
          </span>
        )}
      </span>
      <span className="frame__meta">
        <span className="frame__name">{photo.filename}</span>
        {score !== null && <span className="frame__score">{score}</span>}
      </span>
      {note && <span className="frame__note">{note}</span>}
    </button>
  );
}
