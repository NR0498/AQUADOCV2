"use client";

import { useCallback, useState } from "react";
import type { Target } from "@/components/LocationChooser";
import { getJSON, type NearbyLake, type NearbyResponse } from "@/lib/api";
import featuredList from "@/lib/featured.json";
import { fmt, shortDate } from "@/lib/format";

type Featured = { name: string; latitude: number; longitude: number; osm_ref: string | null };

function distanceBearing(lat1: number, lon1: number, lat2: number, lon2: number): [number, number] {
  const rad = Math.PI / 180;
  const dp = (lat2 - lat1) * rad;
  const dl = (lon2 - lon1) * rad;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dl / 2) ** 2;
  const y = Math.sin(dl) * Math.cos(lat2 * rad);
  const x = Math.cos(lat1 * rad) * Math.sin(lat2 * rad) - Math.sin(lat1 * rad) * Math.cos(lat2 * rad) * Math.cos(dl);
  return [2 * 6371 * Math.asin(Math.sqrt(a)), ((Math.atan2(y, x) / rad) + 360) % 360];
}

/** Featured lakes inside the radius that the directory search missed (e.g. unnamed on OpenStreetMap). */
function withFeatured(data: NearbyResponse): NearbyLake[] {
  const { latitude, longitude } = data.origin;
  const known = new Set(data.lakes.map((l) => l.osm));
  const extra = (featuredList as Featured[]).flatMap((f): NearbyLake[] => {
    if (f.osm_ref && known.has(f.osm_ref)) return [];
    const [distance, bearing] = distanceBearing(latitude, longitude, f.latitude, f.longitude);
    if (distance > data.radius_km) return [];
    return [{ name: f.name, osm: f.osm_ref ?? `featured/${f.name}`, type: "featured", latitude: f.latitude, longitude: f.longitude,
      bbox: [f.latitude, f.latitude, f.longitude, f.longitude], distance_km: Math.round(distance * 100) / 100,
      bearing_deg: Math.round(bearing), extent_km: 0.5, latest_analysis: null }];
  });
  return [...data.lakes, ...extra].sort((a, b) => a.distance_km - b.distance_km);
}

type Props = {
  onAnalyse: (target: Target) => void;
  onOpen: (analysisId: string) => void;
  running: boolean;
};

type State =
  | { kind: "idle" }
  | { kind: "locating" }
  | { kind: "searching" }
  | { kind: "ready"; data: NearbyResponse }
  | { kind: "error"; message: string };

const RADII = [10, 25, 50];
const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const direction = (deg: number) => COMPASS[Math.round(deg / 45) % 8];

function geolocationMessage(error: GeolocationPositionError): string {
  if (error.code === error.PERMISSION_DENIED)
    return "Location permission was denied. Allow it in your browser's site settings, or use the Coordinates or Pin on map tab above.";
  if (error.code === error.TIMEOUT) return "Finding your location took too long. Try again, or use the Pin on map tab above.";
  return "Your location is unavailable on this device. Use the Coordinates or Pin on map tab above.";
}

/** You at the centre, lakes placed by their real bearing and distance; filled dots have a saved analysis. */
function Radar({ lakes, radius, selected, onSelect }: {
  lakes: NearbyLake[]; radius: number; selected: string | null; onSelect: (osm: string) => void;
}) {
  const R = 46;
  const maxExtent = Math.max(...lakes.map((l) => l.extent_km), 0.1);
  return (
    <svg viewBox="-50 -50 100 100" className="h-full w-full" role="img"
      aria-label={`${lakes.length} lakes within ${radius} km, placed by direction and distance from you`}>
      {[0.5, 1].map((f) => (
        <circle key={f} r={R * f} fill="none" stroke="var(--hairline)" strokeWidth="0.4" />
      ))}
      <line x1={-R} x2={R} y1="0" y2="0" stroke="var(--hairline)" strokeWidth="0.3" />
      <line y1={-R} y2={R} x1="0" x2="0" stroke="var(--hairline)" strokeWidth="0.3" />
      <text x="0" y={-R - 1.2} textAnchor="middle" fontSize="3" fill="var(--ink-muted)" fontFamily="var(--font-mono)">N</text>
      <text x={R * 0.5 + 0.8} y="-0.8" fontSize="2.6" fill="var(--ink-muted)" fontFamily="var(--font-mono)">{radius / 2} km</text>
      <text x={R + 0.8} y="-0.8" fontSize="2.6" fill="var(--ink-muted)" fontFamily="var(--font-mono)">{radius} km</text>
      {lakes.map((lake, i) => {
        const r = (lake.distance_km / radius) * R;
        const a = (lake.bearing_deg * Math.PI) / 180;
        const x = r * Math.sin(a);
        const y = -r * Math.cos(a);
        const size = 1.2 + 3.2 * Math.sqrt(lake.extent_km / maxExtent);
        const on = lake.osm === selected;
        const analysed = Boolean(lake.latest_analysis);
        return (
          <g key={lake.osm} className="cursor-pointer" onClick={() => onSelect(lake.osm)} style={{ animationDelay: `${i * 40}ms` }}>
            <title>{`${lake.name}: ${fmt(lake.distance_km, 2)} km ${direction(lake.bearing_deg)}`}</title>
            {on && <circle cx={x} cy={y} r={size + 1.6} fill="none" stroke="var(--ink)" strokeWidth="0.5" />}
            <circle cx={x} cy={y} r={size} fill={analysed ? "var(--accent)" : "var(--page)"}
              stroke="var(--accent)" strokeWidth="0.6" fillOpacity={analysed ? 0.85 : 1} />
          </g>
        );
      })}
      <circle r="1.6" fill="var(--ink)" />
      <circle r="3.2" fill="none" stroke="var(--ink)" strokeWidth="0.3" className="anim-pulse" />
    </svg>
  );
}

export default function NearbyLakes({ onAnalyse, onOpen, running }: Props) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [radius, setRadius] = useState(25);
  const [sort, setSort] = useState<"near" | "large">("near");
  const [selected, setSelected] = useState<string | null>(null);
  const [origin, setOrigin] = useState<{ latitude: number; longitude: number } | null>(null);

  const search = useCallback(async (lat: number, lon: number, km: number) => {
    setState({ kind: "searching" });
    try {
      const data = await getJSON<NearbyResponse>(`/api/waterbodies/nearby?latitude=${lat}&longitude=${lon}&radius_km=${km}`);
      setState({ kind: "ready", data });
      setSelected(withFeatured(data)[0]?.osm ?? null);
    } catch (e) {
      setState({ kind: "error", message: (e as Error).message });
    }
  }, []);

  const locate = useCallback(() => {
    if (!("geolocation" in navigator)) {
      setState({ kind: "error", message: "This browser cannot share a location. Use the Coordinates or Pin on map tab above." });
      return;
    }
    setState({ kind: "locating" });
    navigator.geolocation.getCurrentPosition(
      (position) => {
        // Rounded to ~1 km in the browser: enough to find lakes, without sending a precise location anywhere.
        const point = { latitude: Number(position.coords.latitude.toFixed(2)), longitude: Number(position.coords.longitude.toFixed(2)) };
        setOrigin(point);
        search(point.latitude, point.longitude, radius);
      },
      (error) => setState({ kind: "error", message: geolocationMessage(error) }),
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 600000 },
    );
  }, [radius, search]);

  const changeRadius = (km: number) => {
    setRadius(km);
    if (origin) search(origin.latitude, origin.longitude, km);
  };

  const lakes = state.kind === "ready" ? withFeatured(state.data) : [];
  const shown = sort === "near" ? lakes : [...lakes].sort((a, b) => b.extent_km - a.extent_km);
  const analysedCount = lakes.filter((l) => l.latest_analysis).length;
  const analyse = (lake: NearbyLake) =>
    onAnalyse({ latitude: lake.latitude, longitude: lake.longitude, osmRef: lake.osm.startsWith("featured/") ? null : lake.osm,
      name: lake.name, detail: "Near you" });

  return (
    <section className="border-b border-hairline py-12" aria-labelledby="nearby-title">
      <div className="mb-7 grid gap-2 sm:grid-cols-[200px_minmax(0,1fr)_auto] sm:items-baseline sm:gap-6">
        <div className="secnum">Near you</div>
        <div>
          <h2 id="nearby-title" className="font-serif text-[28px] font-medium leading-tight tracking-tight sm:text-[32px]">
            Lakes near you
          </h2>
          <p className="mt-1.5 max-w-[46em] text-[15px] text-ink-2 text-pretty">
            Find the lakes and reservoirs around you, see which have already been analysed, and analyse any of them with one click.
          </p>
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Search radius">
          {RADII.map((km) => (
            <button key={km} type="button" aria-pressed={radius === km} onClick={() => changeRadius(km)}
              className={`border px-2.5 py-1 font-mono text-xs ${radius === km ? "border-ink bg-surface-2 text-ink" : "border-hairline text-ink-2 hover:text-ink"}`}>
              {km} km
            </button>
          ))}
        </div>
      </div>

      {(state.kind === "idle" || state.kind === "error") && (
        <div className="grid gap-6 border border-hairline bg-surface-2 p-6 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
          <div>
            {state.kind === "error" ? (
              <p role="alert" className="text-sm"><b className="font-semibold">Could not find lakes near you. </b>{state.message}</p>
            ) : (
              <p className="text-sm text-ink-2">
                Your browser will ask to share your location. It is rounded to about 1 km and used only to look up nearby water in
                OpenStreetMap; it is not stored.
              </p>
            )}
          </div>
          <button type="button" onClick={locate}
            className="flex items-center justify-between gap-6 bg-ink px-[18px] py-3 text-[15px] font-medium text-page transition-colors hover:bg-accent">
            {state.kind === "error" ? "Try again" : "Find lakes near me"}
            <span className="font-mono text-xs opacity-70" aria-hidden>◎</span>
          </button>
        </div>
      )}

      {(state.kind === "locating" || state.kind === "searching") && (
        <p className="anim-pulse border border-hairline p-6 font-mono text-xs text-ink-2" aria-live="polite">
          {state.kind === "locating" ? "Waiting for your location…" : `Looking up lakes within ${radius} km in OpenStreetMap…`}
        </p>
      )}

      {state.kind === "ready" && lakes.length === 0 && (
        <p className="border border-hairline p-6 text-sm text-ink-2">
          No named lakes or reservoirs larger than about 150 m were found within {radius} km.
          {radius < 50 && <> Try <button type="button" className="underline" onClick={() => changeRadius(50)}>50 km</button>.</>}
        </p>
      )}

      {state.kind === "ready" && lakes.length > 0 && (
        <div className="anim-rise grid gap-10 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <figure>
            <div className="mx-auto aspect-square max-w-[420px]">
              <Radar lakes={lakes} radius={radius} selected={selected} onSelect={setSelected} />
            </div>
            <figcaption className="mt-2 text-xs text-muted">
              <b className="font-semibold text-ink">Fig. 0</b> {lakes.length} lakes within {radius} km of you
              ({state.data.origin.latitude.toFixed(2)}°, {state.data.origin.longitude.toFixed(2)}°). Dot size shows lake size;
              filled dots ({analysedCount}) already have an analysis.
            </figcaption>
          </figure>

          <div className="min-w-0">
            <div className="mb-2 flex items-center justify-between gap-3 text-xs text-ink-2">
              <span>{analysedCount} of {lakes.length} already analysed</span>
              <span className="flex gap-3" role="group" aria-label="Sort lakes">
                {(["near", "large"] as const).map((k) => (
                  <button key={k} type="button" aria-pressed={sort === k} onClick={() => setSort(k)}
                    className={sort === k ? "font-semibold text-ink underline underline-offset-4" : "hover:text-ink"}>
                    {k === "near" ? "Nearest" : "Largest"}
                  </button>
                ))}
              </span>
            </div>
            <ul className="max-h-[520px] overflow-auto border-t border-ink">
              {shown.map((lake) => {
                const saved = lake.latest_analysis;
                const on = lake.osm === selected;
                return (
                  <li key={lake.osm} onMouseEnter={() => setSelected(lake.osm)}
                    className={`grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-b border-hairline px-2 py-3 ${
                      on ? "bg-[linear-gradient(90deg,var(--accent-wash),transparent_70%)]" : ""}`}>
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold">{lake.name}</span>
                      <span className="block font-mono text-[11px] text-ink-2">
                        {fmt(lake.distance_km, 2)} km {direction(lake.bearing_deg)}{lake.type === "featured" ? "" : ` · ${fmt(lake.extent_km, 2)} km across`}
                        {lake.type === "featured" ? " · featured lake" : lake.type && lake.type !== "water" ? ` · ${lake.type}` : ""}
                      </span>
                      {saved && (
                        <span className="mt-0.5 block text-xs text-ink-2">
                          Chlorophyll-a <b className="font-serif text-[15px] font-medium text-ink">{fmt(saved.chlorophyll_a)}</b> µg/L
                          {saved.trophic_state ? ` · ${saved.trophic_state}` : ""} · clarity {fmt(saved.secchi_depth)} m ·
                          image {shortDate(saved.scene_datetime)}
                        </span>
                      )}
                    </span>
                    <span className="flex gap-2">
                      {saved && (
                        <button type="button" onClick={() => onOpen(saved.id)}
                          className="border border-ink px-3 py-1.5 text-xs hover:bg-surface-2">
                          View result
                        </button>
                      )}
                      <button type="button" disabled={running} onClick={() => analyse(lake)}
                        className={`px-3 py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-45 ${
                          saved ? "border border-hairline text-ink-2 hover:text-ink" : "bg-ink text-page hover:bg-accent"}`}>
                        {saved ? "Re-analyse" : "Analyse"}
                      </button>
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-2 text-xs text-muted">Lake directory © OpenStreetMap contributors. Distances are from your rounded location.</p>
          </div>
        </div>
      )}
    </section>
  );
}
