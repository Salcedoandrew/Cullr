import { useMemo } from "react";
import { Frame } from "./Frame";
import type { Person, Photo } from "../types";

interface Props {
  photos: Photo[];
  people: Person[];
  selectedId: number | null;
  onSelect: (photo: Photo) => void;
  onOpen: (photo: Photo) => void;
}

interface Section {
  key: string;
  groupId: number | null;
  photos: Photo[];
}

function sectionize(photos: Photo[]): Section[] {
  const sections: Section[] = [];
  for (const photo of photos) {
    const last = sections[sections.length - 1];
    if (last && photo.group_id !== null && last.groupId === photo.group_id) {
      last.photos.push(photo);
    } else {
      sections.push({
        key: photo.group_id !== null ? `g${photo.group_id}` : `p${photo.id}`,
        groupId: photo.group_id,
        photos: [photo],
      });
    }
  }
  return sections;
}

export function Sheet({ photos, people, selectedId, onSelect, onOpen }: Props) {
  const sections = useMemo(() => sectionize(photos), [photos]);
  const labels = useMemo(
    () => new Map(people.map((person) => [person.id, person.label])),
    [people],
  );

  if (photos.length === 0) {
    return (
      <div className="sheet-empty">
        <p>No frames match this filter.</p>
      </div>
    );
  }

  return (
    <div className="sheet">
      {sections.map((section) => (
        <section
          key={section.key}
          className={section.groupId === null ? "group group--single" : "group"}
        >
          {section.groupId !== null && (
            <header className="group__head">
              <span className="group__kind">
                {section.photos.length > 1
                  ? `${section.photos.length} similar frames`
                  : "1 frame"}
              </span>
              <span className="group__rule" aria-hidden="true" />
            </header>
          )}
          <div className="group__frames">
            {section.photos.map((photo) => (
              <Frame
                key={photo.id}
                photo={photo}
                coveredLabel={
                  photo.covered_person === null
                    ? null
                    : labels.get(photo.covered_person) ?? null
                }
                active={photo.id === selectedId}
                groupSize={section.photos.length}
                onSelect={() => onSelect(photo)}
                onOpen={() => onOpen(photo)}
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
