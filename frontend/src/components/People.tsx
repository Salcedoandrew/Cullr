import { faceUrl } from "../api";
import type { MergeSuggestion, Person } from "../types";

interface Props {
  people: Person[];
  suggestions: MergeSuggestion[];
  pending: string | null;
  activeId: number | null;
  onSelect: (personId: number | null) => void;
  onMerge: (suggestion: MergeSuggestion) => void;
  onDistinct: (suggestion: MergeSuggestion) => void;
}

function Thumb({ person }: { person: Person | undefined }) {
  if (!person || person.representative_photo_id === null ||
      person.representative_face_id === null) {
    return <span className="merge__face" />;
  }
  return (
    <span className="merge__face">
      <img
        src={faceUrl(person.representative_photo_id, person.representative_face_id)}
        alt=""
        loading="lazy"
        draggable={false}
      />
    </span>
  );
}

export function People({
  people, suggestions, pending, activeId, onSelect, onMerge, onDistinct,
}: Props) {
  if (people.length === 0) return null;

  return (
    <section className="people" aria-label="People in this folder">
      <p className="people__intro">
        Faces grouped across the folder. Culling keeps at least one frame of
        everyone who appears twice or more.
      </p>
      <ul>
        {people.map((person) => {
          const active = person.id === activeId;
          return (
            <li key={person.id}>
              <button
                type="button"
                className={active ? "person is-active" : "person"}
                onClick={() => onSelect(active ? null : person.id)}
                aria-pressed={active}
                title={`${person.photo_count} frames, ${person.face_count} faces`}
              >
                <span className="person__face">
                  {person.representative_photo_id !== null &&
                  person.representative_face_id !== null ? (
                    <img
                      src={faceUrl(person.representative_photo_id,
                                   person.representative_face_id)}
                      alt=""
                      loading="lazy"
                      draggable={false}
                    />
                  ) : null}
                </span>
                <span className="person__name">{person.label}</span>
                <span className="person__count">{person.photo_count}</span>
                {person.required === 1 && !person.covered && (
                  <span className="person__miss" title="No keeper for this person">
                    no keeper
                  </span>
                )}
              </button>
            </li>
          );
        })}
        {activeId !== null && (
          <li>
            <button type="button" className="person person--all" onClick={() => onSelect(null)}>
              Show everyone
            </button>
          </li>
        )}
      </ul>

      {suggestions.length > 0 && (
        <div className="merges">
          <p className="merges__intro">
            Grouping stays strict on purpose, so a few people stay split.
            Confirm the ones that are the same — two faces in one frame are
            never suggested, and the scan remembers your answer.
          </p>
          <ul>
            {suggestions.map((suggestion) => {
              const key = `${suggestion.a}-${suggestion.b}`;
              const busyHere = pending === key;
              return (
                <li className="merge" key={key}>
                  <span className="merge__pair">
                    <Thumb person={people.find((p) => p.id === suggestion.a)} />
                    <Thumb person={people.find((p) => p.id === suggestion.b)} />
                  </span>
                  <span className="merge__names">
                    {suggestion.a_label} · {suggestion.b_label}
                  </span>
                  <span
                    className="merge__similarity"
                    title="Face-descriptor similarity. The culler joins people automatically above 0.90"
                  >
                    {suggestion.similarity.toFixed(2)}
                  </span>
                  <span className="merge__actions">
                    <button
                      type="button"
                      className="merge__button merge__button--yes"
                      disabled={busyHere}
                      onClick={() => onMerge(suggestion)}
                    >
                      Same person
                    </button>
                    <button
                      type="button"
                      className="merge__button"
                      disabled={busyHere}
                      onClick={() => onDistinct(suggestion)}
                    >
                      Two people
                    </button>
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
