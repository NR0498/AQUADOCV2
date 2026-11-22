"use client";

import { useEffect, useRef, useState } from "react";
import { getJSON, type SearchResult } from "@/lib/api";
import featuredList from "@/lib/featured.json";

type Featured = { name: string; country: string; latitude: number; longitude: number; osm_ref: string | null };
const FEATURED = featuredList as Featured[];
const asResult = (f: Featured): SearchResult => ({
  name: f.name, display_name: `${f.name}, ${f.country}`, latitude: f.latitude, longitude: f.longitude,
  type: "featured lake", osm: f.osm_ref ?? "", bbox: [],
});

type Props = { onSelect: (result: SearchResult) => void };

export default function SearchBox({ onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchable = query.trim().length >= 3;
  const needle = query.trim().toLowerCase();
  // Featured lakes match instantly (by name or state) and come first; OpenStreetMap results follow.
  const local = needle.length >= 2
    ? FEATURED.filter((f) => f.name.toLowerCase().includes(needle) || f.country.toLowerCase().includes(needle)).slice(0, 5).map(asResult)
    : [];
  const remote = searchable ? results.filter((r) => !local.some((l) => l.osm === r.osm)) : [];
  const shown = [...local, ...remote];

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (query.trim().length < 3) return; // shorter queries show nothing (derived below)
    // Debounce: Nominatim allows about one request per second.
    timer.current = setTimeout(async () => {
      setStatus("loading");
      try {
        const data = await getJSON<{ results: SearchResult[] }>(`/api/waterbodies/search?q=${encodeURIComponent(query.trim())}`);
        setResults(data.results);
        setStatus("idle");
        setOpen(true);
      } catch {
        setStatus("error");
      }
    }, 700);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [query]);

  return (
    <div className="relative">
      <label htmlFor="search" className="mb-1 block text-xs font-medium text-ink-2">
        Find a lake, reservoir or river
      </label>
      <input
        id="search"
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        placeholder="e.g. Powai, Naini, Lake Erie, Kerala"
        autoComplete="off"
        className="w-full border border-hairline bg-page px-3 py-2.5 text-sm outline-none focus:border-accent"
        role="combobox"
        aria-expanded={open}
        aria-controls="search-results"
      />
      {status === "loading" && <p className="mt-1 text-xs text-muted">Searching OpenStreetMap…</p>}
      {status === "error" && <p className="mt-1 text-xs text-muted">Search is unavailable right now; click the map instead.</p>}
      {open && shown.length > 0 && (
        <ul id="search-results" role="listbox" className="panel absolute z-20 mt-1 max-h-72 w-full overflow-auto py-1 shadow-lg">
          {shown.map((result) => (
            <li key={`${result.type}-${result.osm}-${result.name}`}>
              <button
                type="button"
                className="w-full px-3 py-2 text-left hover:bg-surface-2"
                onClick={() => {
                  onSelect(result);
                  setQuery(result.name);
                  setOpen(false);
                }}
              >
                <span className="block text-sm font-medium">{result.name}</span>
                <span className="block truncate text-xs text-muted">
                  {result.type} · {result.display_name}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {status === "idle" && searchable && shown.length === 0 && results.length === 0 && (
        <p className="mt-1 text-xs text-muted">No water bodies found with that name.</p>
      )}
    </div>
  );
}
