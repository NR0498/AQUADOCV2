"use client";

import { useEffect, useRef, useState } from "react";
import MapView from "@/components/MapView";
import SearchBox from "@/components/SearchBox";
import type { SearchResult } from "@/lib/api";
import featuredList from "@/lib/featured.json";
import { shortDate } from "@/lib/format";

export type Target = { latitude: number; longitude: number; osmRef: string | null; name: string | null; detail?: string };
export type Recent = { id: string; created_at: string; water_body: string | null; latitude: number; longitude: number; scene_datetime: string };
export type DateWindow = { start: string; end: string };

type Featured = { group: string; name: string; country: string; latitude: number; longitude: number; osm_ref: string | null; why: string };
// Indian lakes are listed by state so the long list is easy to scan; world lakes keep their order.
const FEATURED = (featuredList as Featured[]).map((f, i) => ({ ...f, i })).sort((a, b) =>
  a.group !== b.group ? a.i - b.i : a.group === "India" ? a.country.localeCompare(b.country) || a.name.localeCompare(b.name) : a.i - b.i);

type Tab = "featured" | "search" | "coords" | "pin";
const TABS: { key: Tab; label: string }[] = [
  { key: "featured", label: "Featured lakes" },
  { key: "search", label: "Search by name" },
  { key: "coords", label: "Coordinates" },
  { key: "pin", label: "Pin on map" },
];

const WINDOWS = [
  { days: 30, label: "30 days" },
  { days: 60, label: "60 days" },
  { days: 120, label: "4 months" },
  { days: 365, label: "1 year" },
];

const iso = (d: Date) => d.toISOString().slice(0, 10);
export const windowFor = (days: number): DateWindow => {
  const end = new Date();
  return { start: iso(new Date(end.getTime() - days * 86_400_000)), end: iso(end) };
};

type Props = {
  target: Target | null;
  onTarget: (target: Target, view?: { bbox?: [number, number, number, number]; center?: [number, number] }) => void;
  onWindow: (window: DateWindow | null) => void; // null = invalid custom range
  recent: Recent[];
  onOpenRecent: (id: string) => void;
  onRun: () => void; // called without arguments: runs the chosen target
  running: boolean;
};

/** Parses "17.42, 78.47" (or "17.42 78.47") pasted into a coordinate field. */
function parsePair(text: string): [number, number] | null {
  const parts = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  return parts.length === 2 && parts.every(Number.isFinite) ? [parts[0], parts[1]] : null;
}

function FeaturedMenu({ target, onPick, recent, onOpenRecent }: {
  target: Target | null; onPick: (f: Featured) => void; recent: Recent[]; onOpenRecent: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const needle = filter.trim().toLowerCase();
  const visible = FEATURED.filter((f) => !needle || f.name.toLowerCase().includes(needle) || f.country.toLowerCase().includes(needle));
  const visibleRecent = recent.filter((r) => !needle || (r.water_body ?? "").toLowerCase().includes(needle));
  const chosen = FEATURED.find((f) => f.name === target?.name);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);

  const groups = [...new Set(visible.map((f) => f.group))];
  return (
    <div ref={box} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 border border-hairline bg-page px-3.5 py-3 text-left hover:border-ink-2">
        <span className="min-w-0">
          <span className="block text-[15px]">{chosen?.name ?? "Choose a lake…"}</span>
          <span className="block truncate text-xs text-muted">
            {chosen ? `${chosen.country} · ${chosen.why}` : `${FEATURED.length} lakes chosen for interesting water`}
          </span>
        </span>
        <span className="font-mono text-muted" aria-hidden>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="anim-drop absolute inset-x-0 top-[calc(100%+6px)] z-30 border border-ink bg-page shadow-[0_18px_40px_-18px_oklch(0.2_0.02_80/0.35)]">
          <div className="border-b border-hairline p-2">
            <input autoFocus value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter lakes by name or state"
              placeholder={`Filter ${FEATURED.length} lakes by name or state…`}
              className="w-full border border-hairline bg-page px-3 py-2 text-sm outline-none focus:border-accent" />
          </div>
          <div role="listbox" className="max-h-[340px] overflow-auto">
          {groups.length === 0 && visibleRecent.length === 0 && (
            <p className="px-3.5 py-3 text-sm text-muted">No featured lake matches. Try the “Search by name” tab.</p>
          )}
          {groups.map((g) => (
            <div key={g}>
              <div className="border-t border-hairline bg-surface-2 px-3.5 pb-1 pt-2.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-accent first:border-t-0">{g}</div>
              {visible.filter((f) => f.group === g).map((f) => (
                <button key={`${f.name}-${f.i}`} type="button" role="option" aria-selected={f.name === chosen?.name}
                  onClick={() => { onPick(f); setOpen(false); }}
                  className="grid w-full grid-cols-[minmax(0,1fr)_auto] gap-x-3 px-3.5 py-2 text-left hover:bg-accent-wash">
                  <b className="text-sm font-medium">{f.name}</b>
                  <em className="font-mono text-[11px] not-italic text-muted">{f.country}</em>
                  <i className="col-span-2 text-xs not-italic text-muted">{f.why}</i>
                </button>
              ))}
            </div>
          ))}
          {visibleRecent.length > 0 && (
            <div>
              <div className="border-t border-hairline bg-surface-2 px-3.5 pb-1 pt-2.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-accent">Recent analyses</div>
              {visibleRecent.map((r) => (
                <button key={r.id} type="button" role="option" aria-selected={false}
                  onClick={() => { onOpenRecent(r.id); setOpen(false); }}
                  className="grid w-full grid-cols-[minmax(0,1fr)_auto] gap-x-3 px-3.5 py-2 text-left hover:bg-accent-wash">
                  <b className="truncate text-sm font-medium">{r.water_body ?? `${r.latitude.toFixed(3)}, ${r.longitude.toFixed(3)}`}</b>
                  <em className="font-mono text-[11px] not-italic text-muted">{shortDate(r.scene_datetime)}</em>
                  <i className="col-span-2 text-xs not-italic text-muted">Saved result · opens instantly</i>
                </button>
              ))}
            </div>
          )}
          </div>
        </div>
      )}
    </div>
  );
}

function CoordinateFields({ target, onTarget }: { target: Target | null; onTarget: Props["onTarget"] }) {
  const [lat, setLat] = useState(target ? target.latitude.toFixed(4) : "");
  const [lon, setLon] = useState(target ? target.longitude.toFixed(4) : "");
  const latN = Number(lat);
  const lonN = Number(lon);
  const latOk = lat.trim() !== "" && Number.isFinite(latN) && Math.abs(latN) <= 90;
  const lonOk = lon.trim() !== "" && Number.isFinite(lonN) && Math.abs(lonN) <= 180;

  const change = (which: "lat" | "lon", text: string) => {
    const pair = parsePair(text);
    let nextLat = which === "lat" ? text : lat;
    let nextLon = which === "lon" ? text : lon;
    if (pair) [nextLat, nextLon] = [String(pair[0]), String(pair[1])];
    setLat(nextLat);
    setLon(nextLon);
    const a = Number(nextLat);
    const b = Number(nextLon);
    if (nextLat.trim() && nextLon.trim() && Math.abs(a) <= 90 && Math.abs(b) <= 180 && Number.isFinite(a) && Number.isFinite(b)) {
      onTarget({ latitude: a, longitude: b, osmRef: null, name: null, detail: "Coordinates entered" }, { center: [b, a] });
    }
  };

  const field = "w-full border bg-page px-3 py-2.5 font-mono text-sm outline-none focus:border-accent";
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-3">
        <label className="block">
          <span className="mb-1 block text-xs text-ink-2">Latitude</span>
          <input value={lat} onChange={(e) => change("lat", e.target.value)} inputMode="decimal" placeholder="17.4239"
            aria-invalid={lat !== "" && !latOk} className={`${field} ${lat !== "" && !latOk ? "border-[var(--critical)]" : "border-hairline"}`} />
        </label>
        <label className="block">
          <span className="mb-1 block text-xs text-ink-2">Longitude</span>
          <input value={lon} onChange={(e) => change("lon", e.target.value)} inputMode="decimal" placeholder="78.4738"
            aria-invalid={lon !== "" && !lonOk} className={`${field} ${lon !== "" && !lonOk ? "border-[var(--critical)]" : "border-hairline"}`} />
        </label>
      </div>
      <p className="text-xs text-muted">Decimal degrees. Paste “17.4239, 78.4738” into either field. The lake outline is looked up automatically.</p>
    </div>
  );
}

export default function LocationChooser({ target, onTarget, onWindow, recent, onOpenRecent, onRun, running }: Props) {
  const [tab, setTab] = useState<Tab>("featured");
  const [days, setDays] = useState<number | "custom">(60);
  const [custom, setCustom] = useState<DateWindow>({ start: "", end: "" });
  // Dates are only computed after interaction, never during prerender (the static shell must not bake in "today").
  const today = days === "custom" ? iso(new Date()) : "";
  const customOk = Boolean(custom.start && custom.end) && custom.start <= custom.end && custom.end <= today && custom.start >= "2017-01-01";

  useEffect(() => {
    onWindow(days === "custom" ? (customOk ? custom : null) : windowFor(days));
  }, [days, custom, customOk, onWindow]);

  const pickFeatured = (f: Featured) =>
    onTarget({ latitude: f.latitude, longitude: f.longitude, osmRef: f.osm_ref, name: f.name, detail: f.country }, { center: [f.longitude, f.latitude] });
  const pickSearch = (place: SearchResult) => {
    const [south, north, west, east] = place.bbox;
    onTarget({ latitude: place.latitude, longitude: place.longitude, osmRef: place.osm || null, name: place.name, detail: place.type },
      place.bbox.length === 4 ? { bbox: [west, south, east, north] } : { center: [place.longitude, place.latitude] });
  };

  const canRun = Boolean(target) && !running && (days !== "custom" || customOk);
  return (
    <div className="flex flex-col gap-[18px] border border-ink bg-page p-5 sm:p-[26px]">
      <div>
        <div className="secnum">01 — Location</div>
        <h2 className="mt-0.5 font-serif text-[26px] font-medium">Choose a water body</h2>
      </div>

      <div className="flex gap-x-5 gap-y-1 overflow-x-auto border-b border-hairline" role="tablist" aria-label="How to choose a location">
        {TABS.map((t) => (
          <button key={t.key} type="button" role="tab" aria-selected={tab === t.key} onClick={() => setTab(t.key)}
            className={`-mb-px shrink-0 border-b-2 py-2 text-sm ${tab === t.key ? "border-accent font-semibold text-ink" : "border-transparent text-ink-2 hover:text-ink"}`}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="min-h-[86px]">
        {tab === "featured" && <FeaturedMenu target={target} onPick={pickFeatured} recent={recent} onOpenRecent={onOpenRecent} />}
        {tab === "search" && <SearchBox onSelect={pickSearch} />}
        {tab === "coords" && <CoordinateFields target={target} onTarget={onTarget} />}
        {tab === "pin" && (
          <div className="space-y-2">
            <div className="relative aspect-[16/10] border border-hairline">
              <MapView point={target} waterBody={null} overlay={null} fitTo={null}
                flyTo={target ? { center: [target.longitude, target.latitude] } : null}
                onPick={(latitude, longitude) => onTarget({ latitude, longitude, osmRef: null, name: null, detail: "Pin dropped on the map" })} />
            </div>
            <p className="text-xs text-muted">Click on open water to drop a pin. Zoom in for small lakes.</p>
          </div>
        )}
      </div>

      <div className="flex items-start justify-between gap-3 bg-surface-2 px-3.5 py-2.5 text-sm" aria-live="polite">
        {target ? (
          <span className="min-w-0">
            <span className="block font-medium">{target.name ?? "Point on the map"}</span>
            <span className="block truncate font-mono text-[11px] text-ink-2">
              {target.latitude.toFixed(4)}, {target.longitude.toFixed(4)}
              {target.osmRef ? ` · OSM ${target.osmRef}` : " · outline found automatically"}
            </span>
          </span>
        ) : (
          <span className="text-ink-2">No location selected yet.</span>
        )}
      </div>

      <div>
        <span className="mb-1.5 block text-xs text-ink-2">Look for clear images in the last</span>
        <div className="flex flex-wrap gap-2">
          {WINDOWS.map((w) => (
            <button key={w.days} type="button" aria-pressed={days === w.days} onClick={() => setDays(w.days)}
              className={`border px-3 py-1.5 text-[13px] ${days === w.days ? "border-ink bg-surface-2 text-ink" : "border-hairline text-ink-2 hover:text-ink"}`}>
              {w.label}
            </button>
          ))}
          <button type="button" aria-pressed={days === "custom"}
            onClick={() => { if (!custom.start) setCustom(windowFor(60)); setDays("custom"); }}
            className={`border px-3 py-1.5 text-[13px] ${days === "custom" ? "border-ink bg-surface-2 text-ink" : "border-hairline text-ink-2 hover:text-ink"}`}>
            Custom dates
          </button>
        </div>
        {days === "custom" && (
          <div className="anim-drop mt-3 grid grid-cols-2 gap-3">
            {(["start", "end"] as const).map((k) => (
              <label key={k} className="block">
                <span className="mb-1 block text-xs text-ink-2">{k === "start" ? "From" : "To"}</span>
                <input type="date" value={custom[k]} min="2017-01-01" max={today}
                  onChange={(e) => setCustom((c) => ({ ...c, [k]: e.target.value }))}
                  className="w-full border border-hairline bg-page px-3 py-2 font-mono text-sm" />
              </label>
            ))}
            {!customOk && <p className="col-span-2 text-xs text-[var(--critical)]">Choose a start before the end, between 2017 and today.</p>}
          </div>
        )}
      </div>

      <button type="button" onClick={() => onRun()} disabled={!canRun}
        className="flex items-center justify-between gap-3 bg-ink px-[18px] py-3.5 text-[15px] font-medium text-page transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-ink">
        {running ? "Analysing…" : "Run analysis"}
        <span className="font-mono text-xs opacity-70">{running ? "live below ↓" : "→ about 10–30 seconds"}</span>
      </button>
    </div>
  );
}
