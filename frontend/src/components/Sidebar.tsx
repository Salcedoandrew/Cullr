import type { FilterKey, Stats } from "../types";

interface Props {
  filter: FilterKey;
  stats: Stats | null;
  onChange: (filter: FilterKey) => void;
}

const FILTERS: { key: FilterKey; label: string; stat?: keyof Stats }[] = [
  { key: "all", label: "All frames", stat: "total" },
  { key: "keep", label: "Keepers", stat: "keep" },
  { key: "reject", label: "Rejected", stat: "reject" },
  { key: "duplicates", label: "Duplicates", stat: "duplicates" },
  { key: "blur", label: "Soft focus", stat: "blurry" },
  { key: "blink", label: "Eyes closed", stat: "blink" },
  { key: "exposure", label: "Exposure", stat: "exposure" },
  { key: "pick", label: "My picks" },
];

export function Sidebar({ filter, stats, onChange }: Props) {
  return (
    <nav className="sidebar" aria-label="Review filters">
      <ul>
        {FILTERS.map((item) => {
          const count = item.stat && stats ? stats[item.stat] : null;
          return (
            <li key={item.key}>
              <button
                type="button"
                className={filter === item.key ? "filter is-active" : "filter"}
                onClick={() => onChange(item.key)}
                aria-current={filter === item.key ? "true" : undefined}
              >
                <span>{item.label}</span>
                {count !== null && <span className="filter__count">{count}</span>}
              </button>
            </li>
          );
        })}
      </ul>
      <div className="sidebar__foot">
        <p>
          Every frame gets a reason when it is rejected. Override anything with
          <kbd>P</kbd> or <kbd>X</kbd>.
        </p>
      </div>
    </nav>
  );
}
