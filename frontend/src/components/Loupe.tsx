import { formatShutter, formatTime, personLabel, previewUrl, reasonLabel } from "../api";
import type { Person, Photo } from "../types";

interface Props {
  photo: Photo;
  people: Person[];
  analysisEdge: number;
  onClose: () => void;
  onDecide: (decision: "pick" | "reject" | null) => void;
  onStar: (star: number) => void;
  onPrev: () => void;
  onNext: () => void;
}

function explain(photo: Photo): string {
  switch (photo.reason) {
    case "blurry":
      return "Softer than the sharpness gate. Either the frame is out of focus or it is the softest of its burst.";
    case "duplicate":
      return "Another frame in this group scored higher and is kept instead.";
    case "closed_eyes":
      return "Eyes closed on the subject. Lower strictness or disable the check for newborn and editorial work.";
    case "overexposed":
      return "Highlights on the subject are clipped.";
    case "underexposed":
      return "The subject is too dark to recover.";
    case "target":
      // Both count controls land here: the cull-to-target cap and the Auto
      // Target stop's trim. The frame passed the gates and ranked below the cut.
      return "It passed the gates and ranked below the cut: more frames survived than the count or selection strictness you asked for.";
    case "unreadable":
      return photo.error ?? "The file could not be decoded.";
    default:
      return "Kept by the technical gates.";
  }
}

export function Loupe({
  photo, people, analysisEdge, onClose, onDecide, onStar, onPrev, onNext,
}: Props) {
  const personNumber = new Map(people.map((person, index) => [person.id, index + 1]));
  const facePeople = Array.from(new Set(
    photo.faces
      .map((face) => face.person_id)
      .filter((id): id is number => id !== null),
  ));
  const width = photo.width || 1;
  const height = photo.height || 1;
  const scale = width >= height
    ? { w: analysisEdge, h: (analysisEdge * height) / width }
    : { w: (analysisEdge * width) / height, h: analysisEdge };

  const score = photo.quality === null ? "–" : (photo.quality * 100).toFixed(0);
  const eyes = photo.eyes_open === null ? "–" : photo.eyes_open.toFixed(1);

  return (
    <div className="loupe" role="dialog" aria-modal="true" aria-label={photo.filename}>
      <button type="button" className="loupe__close" onClick={onClose} aria-label="Close">
        ×
      </button>

      <div className="loupe__stage">
        <button type="button" className="loupe__nav loupe__nav--prev" onClick={onPrev} aria-label="Previous frame">
          ‹
        </button>
        <figure className="loupe__figure">
          <div className="loupe__image">
            <img src={previewUrl(photo)} alt={photo.filename} />
            {photo.faces.map((face, index) => (
              <span
                key={index}
                className={
                  face.eyes_open !== null && face.eyes_open <= 4
                    ? "facebox facebox--closed"
                    : "facebox"
                }
                style={{
                  left: `${(face.x / scale.w) * 100}%`,
                  top: `${(face.y / scale.h) * 100}%`,
                  width: `${(face.w / scale.w) * 100}%`,
                  height: `${(face.h / scale.h) * 100}%`,
                }}
                title={[
                  face.person_id === null
                    ? null
                    : personLabel(people, face.person_id),
                  face.eyes_open === null
                    ? "eyes not measured"
                    : `eyes open ${face.eyes_open.toFixed(1)} / 10`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              >
                {face.person_id !== null && personNumber.has(face.person_id) && (
                  <span className="facebox__index">{personNumber.get(face.person_id)}</span>
                )}
              </span>
            ))}
          </div>
          <figcaption>
            <button type="button" className="linkbutton" onClick={onPrev} aria-label="Previous">
              Previous
            </button>
            <span>{photo.filename}</span>
            <button type="button" className="linkbutton" onClick={onNext} aria-label="Next">
              Next
            </button>
          </figcaption>
        </figure>
        <button type="button" className="loupe__nav loupe__nav--next" onClick={onNext} aria-label="Next frame">
          ›
        </button>
      </div>

      <aside className="loupe__panel">
        <div className="loupe__verdict">
          <p className={photo.verdict === "keep" ? "verdict verdict--keep" : "verdict verdict--reject"}>
            {photo.verdict === "keep" ? "Keep" : reasonLabel(photo.reason) || "Reject"}
          </p>
          <p className="loupe__explain">{explain(photo)}</p>
          {photo.covered_person !== null && (
            <p className="loupe__coverage">
              Kept so {personLabel(people, photo.covered_person) ?? "someone in this frame"}
              {" "}is not lost to duplicate culling.
            </p>
          )}
        </div>

        <div className="loupe__decisions">
          <button type="button" className="button button--primary" onClick={() => onDecide("pick")}>
            Keep <kbd>P</kbd>
          </button>
          <button type="button" className="button button--danger" onClick={() => onDecide("reject")}>
            Reject <kbd>X</kbd>
          </button>
          {photo.user_decision && (
            <button type="button" className="button" onClick={() => onDecide(null)}>
              Use AI verdict
            </button>
          )}
        </div>

        <div className="stars" role="group" aria-label="Star rating">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              className={photo.star >= star ? "star is-on" : "star"}
              onClick={() => onStar(photo.star === star ? 0 : star)}
              aria-label={`${star} star${star > 1 ? "s" : ""}`}
            >
              ★
            </button>
          ))}
        </div>

        <dl className="metrics">
          <div>
            <dt>Score</dt>
            <dd>{score}</dd>
          </div>
          <div>
            <dt>Sharpness</dt>
            <dd>{photo.sharpness === null ? "–" : photo.sharpness.toFixed(0)}</dd>
          </div>
          <div>
            <dt>Mean luma</dt>
            <dd>{photo.exposure === null ? "–" : photo.exposure.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Clipped hi / lo</dt>
            <dd>
              {photo.clip_hi === null ? "–" : (photo.clip_hi * 100).toFixed(1)}% /{" "}
              {photo.clip_lo === null ? "–" : (photo.clip_lo * 100).toFixed(1)}%
            </dd>
          </div>
          <div>
            <dt>Faces</dt>
            <dd>{photo.face_count}</dd>
          </div>
          <div>
            <dt>Eyes open</dt>
            <dd>{eyes} <span className="metrics__unit">/ 10</span></dd>
          </div>
          <div>
            <dt>People</dt>
            <dd>
              {facePeople.length === 0
                ? "–"
                : facePeople
                    .map((id) => personLabel(people, id) ?? "unclustered")
                    .join(", ")}
            </dd>
          </div>
          <div>
            <dt>Noise σ</dt>
            <dd>{photo.noise === null ? "–" : photo.noise.toFixed(1)}</dd>
          </div>
          <div>
            <dt>Group</dt>
            <dd>{photo.group_id === null ? "–" : photo.is_lead ? `Group ${photo.group_id}, best` : `Group ${photo.group_id}`}</dd>
          </div>
        </dl>

        <dl className="exif">
          <div><dt>Captured</dt><dd>{formatTime(photo.captured_at)}</dd></div>
          {photo.camera && <div><dt>Camera</dt><dd>{photo.camera}</dd></div>}
          {photo.lens && <div><dt>Lens</dt><dd>{photo.lens}</dd></div>}
          <div>
            <dt>Settings</dt>
            <dd>
              {[
                photo.iso ? `ISO ${photo.iso}` : null,
                formatShutter(photo.shutter),
                photo.aperture ? `f/${photo.aperture}` : null,
                photo.focal_len ? `${Math.round(photo.focal_len)}mm` : null,
              ]
                .filter(Boolean)
                .join(", ") || "–"}
            </dd>
          </div>
          <div><dt>Path</dt><dd className="exif__path">{photo.path}</dd></div>
        </dl>
      </aside>
    </div>
  );
}
