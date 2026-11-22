"use client";

import { useMemo, useState } from "react";
import PixelBreakdown from "@/components/PixelBreakdown";
import type { PixelCounts, ReflectanceBand, SceneSummary, StageEvent } from "@/lib/api";
import { fmt, pct, shortDate } from "@/lib/format";

type StageKey = "locate" | "search" | "scene" | "mask" | "correct" | "estimate" | "explain";

const STAGES: { key: StageKey; title: string; line: string; heading: string; body: string; images: string[] }[] = [
  { key: "locate", title: "Locate", line: "Lake outline from OpenStreetMap", heading: "Find the water body's outline",
    body: "The OpenStreetMap outline of the lake is fetched and shrunk inward (up to 100 m) so that light scattered from bright shorelines — the adjacency effect — cannot leak into the estimate. The analysis window is sized to fit the lake.",
    images: [] },
  { key: "search", title: "Acquire", line: "Search the Sentinel-2 archive", heading: "Search the satellite archive",
    body: "Every Sentinel-2 pass over the lake in the chosen period is listed, newest first. Tile-level cloud cover only rules out hopeless images; the real cloud check happens pixel by pixel in the next stages.",
    images: [] },
  { key: "scene", title: "Read", line: "10 bands + scene classification", heading: "Read only the pixels needed",
    body: "Ten spectral bands and the satellite's own scene classification are read from cloud-optimised files and warped onto one 20 m grid. The classification is the satellite product's first guess at water, cloud and land.",
    images: ["true_color", "scene_classification"] },
  { key: "mask", title: "Mask", line: "Clouds, shadows, shore removed", heading: "Keep only clear open water",
    body: "Cloud shadows over water are usually missed by the satellite's own mask, so AquaDoc projects each cloud along the sun's direction and removes pixels darker than their surroundings. Haze near clouds, sun glint and floating vegetation are separated too. If too little water is clear, an older image is tried.",
    images: ["pixel_classes"] },
  { key: "correct", title: "Correct", line: "Remove atmospheric offset", heading: "Correct reflectance over water",
    body: "Sentinel-2's atmospheric correction is designed for land and leaves extra light over water. A correction learned from 506 satellite–field match-ups subtracts it, using the shortwave-infrared signal that clean water should not have. Each pixel is also averaged with its 3×3 neighbours to suppress sensor noise.",
    images: [] },
  { key: "estimate", title: "Estimate", line: "Model on every clear pixel", heading: "Estimate each parameter, pixel by pixel",
    body: "A small neural-network ensemble trained on 5,761 field spectra estimates five parameters for every clear pixel, with an 80% range calibrated on real satellite data.",
    images: ["chlorophyll_a", "tss", "turbidity", "secchi_depth", "cdom_a440"] },
  { key: "explain", title: "Explain", line: "Which bands drove it", heading: "Explain the result with the bands",
    body: "Each band is nudged ±10% at this lake's median spectrum to measure how much it moves each estimate. Index maps show the raw spectral signals behind the numbers: NDCI (red-edge vs red) for algae and the Floating Debris Index for material on the surface.",
    images: ["ndci", "floating_debris_index"] },
];

const IMAGE_LABELS: Record<string, string> = {
  true_color: "True colour", scene_classification: "Satellite classification", pixel_classes: "AquaDoc mask",
  chlorophyll_a: "Chlorophyll-a", tss: "Suspended solids", turbidity: "Turbidity", secchi_depth: "Clarity",
  cdom_a440: "CDOM", ndci: "NDCI", floating_debris_index: "Floating debris",
};

type StageState = {
  status: "pending" | "active" | "done" | "failed";
  message: string;
  elapsed: number | null;
  data: StageEvent["data"];
  rejected: StageEvent[];
};

type Props = {
  events: StageEvent[];
  running: boolean;
  layerUrl?: (id: string) => string | undefined; // resolves stored preview_layers to image URLs
  replay?: boolean; // events come from a stored timeline rather than a live run
  onReplay?: () => void;
};

function deriveStates(events: StageEvent[], running: boolean): Record<StageKey, StageState> {
  const states = Object.fromEntries(
    STAGES.map((s) => [s.key, { status: "pending", message: "", elapsed: null, data: {}, rejected: [] } as StageState]),
  ) as Record<StageKey, StageState>;
  let lastStarted: StageKey | null = null;
  for (const event of events) {
    const key = event.stage as StageKey;
    const state = states[key];
    if (!state) continue;
    if (event.status === "started") {
      lastStarted = key;
      if (state.status !== "done") Object.assign(state, { status: "active", message: event.message });
    } else if (event.status === "completed") {
      Object.assign(state, { status: "done", message: event.message, elapsed: event.elapsed_ms, data: { ...state.data, ...event.data } });
    } else if (event.status === "rejected") {
      state.rejected.push(event);
      if (key === "mask") states.scene.status = "active"; // trying an older image
    } else if (event.status === "failed") {
      Object.assign(state, { status: "failed", message: event.message, elapsed: event.elapsed_ms });
    }
  }
  if (!running) {
    for (const s of STAGES) if (states[s.key].status === "active") states[s.key].status = lastStarted === s.key ? "failed" : "pending";
  }
  return states;
}

function previewsFor(stage: (typeof STAGES)[number], state: StageState, layerUrl?: Props["layerUrl"]) {
  const live = state.data.previews ?? {};
  const stored = new Set(state.data.preview_layers ?? []);
  return stage.images
    .map((id) => ({ id, url: live[id] ?? (stored.has(id) ? layerUrl?.(id) : undefined) }))
    .filter((p): p is { id: string; url: string } => Boolean(p.url));
}

function stats(key: StageKey, data: StageEvent["data"], state: StageState): { v: string; k: string }[] {
  const d = data as Record<string, unknown>;
  if (key === "locate") {
    const body = d.water_body as { name?: string; water_type?: string } | null | undefined;
    const grid = d.grid as { width?: number; resolution_m?: number } | undefined;
    return [
      { v: body?.name ?? "Spectral", k: body ? body.water_type ?? "water body" : "no mapped outline" },
      { v: grid?.width && grid.resolution_m ? `${((grid.width * grid.resolution_m) / 1000).toFixed(1)} km` : "–", k: "analysis window" },
      { v: state.elapsed !== null ? `${(state.elapsed / 1000).toFixed(1)} s` : "–", k: "elapsed" },
    ];
  }
  if (key === "search") {
    const candidates = (d.candidates as SceneSummary[] | undefined) ?? [];
    return [
      { v: String(candidates.length), k: "candidate images" },
      { v: candidates[0] ? shortDate(candidates[0].datetime) : "–", k: "newest" },
      { v: candidates[0]?.cloud_cover_pct != null ? pct(candidates[0].cloud_cover_pct) : "–", k: "its tile cloud cover" },
    ];
  }
  if (key === "scene") {
    const scene = d.scene as SceneSummary | undefined;
    return [
      { v: scene ? shortDate(scene.datetime) : "–", k: "image date" },
      { v: scene?.platform ?? "–", k: "satellite" },
      { v: String(state.rejected.length), k: "cloudy images skipped" },
    ];
  }
  if (key === "mask") {
    const p = d.pixels as PixelCounts | undefined;
    return [
      { v: p ? p.valid_open_water.toLocaleString() : "–", k: "clear water pixels" },
      { v: p ? pct(p.clear_fraction * 100) : "–", k: "of the water area" },
      { v: p ? (p.cloud + p.cloud_shadow).toLocaleString() : "–", k: "cloud & shadow pixels" },
    ];
  }
  if (key === "correct") {
    const bands = (d.reflectance as ReflectanceBand[] | undefined) ?? [];
    const b4 = bands.find((b) => b.band === "B4");
    const using = (d.targets_using_correction as string[] | undefined) ?? [];
    return [
      { v: b4 ? `${fmt(b4.rrs_l2a, 2)} → ${fmt(b4.rrs_corrected, 2)}` : "–", k: "red (B4) reflectance, sr⁻¹" },
      { v: `${using.length} of 5`, k: "parameters use it" },
      { v: "3×3", k: "noise smoothing" },
    ];
  }
  if (key === "estimate") {
    const e = (d.estimates as Record<string, { value: number | null; unit: string }> | undefined) ?? {};
    const show = (t: string) => (e[t] ? `${fmt(e[t].value)} ${e[t].unit.replace("ug/L", "µg/L")}` : "–");
    return [
      { v: show("chlorophyll_a"), k: "chlorophyll-a" },
      { v: show("tss"), k: "suspended solids" },
      { v: show("secchi_depth"), k: "clarity (Secchi)" },
    ];
  }
  const top = (d.top_bands as Record<string, string> | undefined) ?? {};
  return [
    { v: top.chlorophyll_a ?? "–", k: "top band for algae" },
    { v: top.tss ?? "–", k: "top band for sediment" },
    { v: state.elapsed !== null ? `${(state.elapsed / 1000).toFixed(1)} s` : "–", k: "total so far" },
  ];
}

/** The lake outline drawn inside the analysis window (lon/lat projected linearly; fine at lake scale). */
function LakeOutline({ geometry, corners }: { geometry?: GeoJSON.Geometry | null; corners?: [number, number][] }) {
  if (!geometry || !corners?.length || (geometry.type !== "Polygon" && geometry.type !== "MultiPolygon")) return null;
  const lons = corners.map((c) => c[0]);
  const lats = corners.map((c) => c[1]);
  const [west, east, south, north] = [Math.min(...lons), Math.max(...lons), Math.min(...lats), Math.max(...lats)];
  const x = (lon: number) => ((lon - west) / (east - west || 1)) * 100;
  const y = (lat: number) => ((north - lat) / (north - south || 1)) * 100;
  const polygons = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  const d = polygons
    .flatMap((rings) => rings.map((ring) => ring.map(([lon, lat], i) => `${i ? "L" : "M"}${x(lon).toFixed(2)},${y(lat).toFixed(2)}`).join(" ") + "Z"))
    .join(" ");
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full" role="img" aria-label="Lake outline within the analysis window">
      <rect x="0.5" y="0.5" width="99" height="99" fill="none" stroke="var(--axis)" strokeWidth="0.6" strokeDasharray="2 2" />
      <path d={d} fill="var(--accent-wash)" stroke="var(--accent)" strokeWidth="1" fillRule="evenodd" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function CloudBars({ candidates, chosen }: { candidates: SceneSummary[]; chosen?: string }) {
  const items = candidates.slice(0, 10).reverse();
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full" role="img" aria-label="Tile cloud cover of candidate images, oldest to newest">
      <line x1="4" x2="96" y1="88" y2="88" stroke="var(--axis)" strokeWidth="0.6" />
      {items.map((c, i) => {
        const w = 92 / Math.max(items.length, 1);
        const h = Math.max(2, ((c.cloud_cover_pct ?? 0) / 100) * 72);
        return (
          <rect key={c.id} x={4 + i * w + w * 0.2} y={88 - h} width={w * 0.6} height={h} rx="1"
            fill={c.id === chosen ? "var(--accent)" : "var(--axis)"} />
        );
      })}
      <text x="4" y="98" fontSize="6" fill="var(--ink-muted)" fontFamily="var(--font-mono)">tile cloud %, oldest → newest</text>
    </svg>
  );
}

function MiniSpectrum({ bands, large = false }: { bands: ReflectanceBand[]; large?: boolean }) {
  const values = bands.flatMap((b) => [b.rrs_l2a ?? 0, b.rrs_corrected ?? 0]);
  const max = Math.max(...values, 0.001) * 1.1;
  const W = large ? 320 : 100;
  const H = large ? 200 : 100;
  const pad = large ? 28 : 8;
  const x = (i: number) => pad + ((W - 2 * pad) * i) / Math.max(bands.length - 1, 1);
  const y = (v: number) => H - pad - (v / max) * (H - 2 * pad);
  const line = (key: "rrs_l2a" | "rrs_corrected") => bands.map((b, i) => `${i ? "L" : "M"}${x(i)},${y(b[key] ?? 0)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img" aria-label="Median reflectance per band before and after correction">
      <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} stroke="var(--axis)" strokeWidth={large ? 1 : 0.6} />
      <path d={line("rrs_l2a")} fill="none" stroke="#eb6834" strokeWidth={large ? 2 : 1.4} strokeLinejoin="round" />
      <path d={line("rrs_corrected")} fill="none" stroke="var(--series-1)" strokeWidth={large ? 2 : 1.4} strokeLinejoin="round" />
      {large && bands.map((b, i) => (
        <text key={b.band} x={x(i)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--ink-muted)">{b.band}</text>
      ))}
      {large && (
        <>
          <text x={x(bands.length - 1)} y={y(bands[bands.length - 1]?.rrs_l2a ?? 0) - 6} textAnchor="end" fontSize="10" fill="var(--ink-2)">as delivered</text>
          <text x={x(bands.length - 1)} y={y(bands[bands.length - 1]?.rrs_corrected ?? 0) + 14} textAnchor="end" fontSize="10" fill="var(--ink-2)">corrected</text>
        </>
      )}
    </svg>
  );
}

export default function Pipeline({ events, running, layerUrl, replay = false, onReplay }: Props) {
  const states = useMemo(() => deriveStates(events, running), [events, running]);
  const [picked, setPicked] = useState<StageKey | null>(null);
  const [imageChoice, setImageChoice] = useState<Record<string, string>>({});

  const doneCount = STAGES.filter((s) => states[s.key].status === "done").length;
  const activeIndex = STAGES.findIndex((s) => states[s.key].status === "active");
  const failed = STAGES.find((s) => states[s.key].status === "failed");
  const lastDone = [...STAGES].reverse().find((s) => states[s.key].status === "done");
  const selectedKey: StageKey = picked ?? (activeIndex >= 0 ? STAGES[activeIndex].key : failed?.key ?? lastDone?.key ?? "locate");
  const selected = STAGES.find((s) => s.key === selectedKey)!;
  const selState = states[selectedKey];
  const selPreviews = previewsFor(selected, selState, layerUrl);
  const shownImage = selPreviews.find((p) => p.id === imageChoice[selectedKey]) ?? selPreviews[0];
  const total = events.length ? events[events.length - 1].elapsed_ms : 0;
  const statusText = failed ? "Stopped" : running ? `${replay ? "Replaying" : "Running"} · stage ${Math.min(doneCount + 1, 7)} of 7`
    : doneCount === 7 ? `${replay ? "Saved run" : "Complete"} · ${(total / 1000).toFixed(1)} s` : "Waiting for a location";

  return (
    <section aria-labelledby="pipeline-title">
      <div className="mb-7 grid gap-2 sm:grid-cols-[200px_minmax(0,1fr)_auto] sm:items-baseline sm:gap-6">
        <div className="secnum">02 — Fig. 1</div>
        <div>
          <h2 id="pipeline-title" className="font-serif text-[32px] font-medium leading-tight tracking-tight">How this image became a result</h2>
          <p className="mt-1.5 max-w-[46em] text-[15px] text-ink-2">Each stage runs live on the server. Select a stage to see what it did to the image and why.</p>
        </div>
        <div className="flex items-center gap-2 whitespace-nowrap font-mono text-xs text-ink-2" aria-live="polite">
          <i className={`inline-block h-2 w-2 rounded-full ${failed ? "bg-[var(--critical)]" : "bg-accent"} ${running ? "anim-pulse" : ""}`} aria-hidden />
          {statusText}
          {onReplay && !running && doneCount === 7 && (
            <button type="button" onClick={() => { setPicked(null); onReplay(); }} className="ml-2 underline hover:text-ink">Replay ↻</button>
          )}
        </div>
      </div>

      <div className="relative grid grid-cols-2 gap-3.5 sm:grid-cols-4 lg:grid-cols-7">
        <div className="absolute left-0 right-0 top-[13px] hidden h-px bg-hairline lg:block" aria-hidden>
          <div className="h-px bg-accent transition-[width] duration-700 ease-out" style={{ width: `${Math.min(100, (doneCount / 6) * 100)}%` }} />
        </div>
        {STAGES.map((stage, i) => {
          const state = states[stage.key];
          const previews = previewsFor(stage, state, layerUrl);
          const active = state.status === "active";
          const done = state.status === "done";
          const isSelected = stage.key === selectedKey;
          const candidates = (states.search.data.candidates as SceneSummary[] | undefined) ?? [];
          const chosenId = (states.scene.data.scene as SceneSummary | undefined)?.id;
          const bands = (state.data.reflectance as ReflectanceBand[] | undefined) ?? [];
          return (
            <button key={stage.key} type="button" onClick={() => setPicked(stage.key)} aria-pressed={isSelected}
              className="group relative flex flex-col gap-2.5 text-left">
              <span className={`relative z-10 grid h-[27px] w-[27px] place-items-center rounded-full border font-mono text-[11px] transition-colors ${
                done ? "border-ink bg-ink text-page" : state.status === "failed" ? "border-[var(--critical)] text-[var(--critical)]" : active ? "anim-ring border-accent bg-page text-accent" : "border-hairline bg-page text-muted"}`}>
                {String(i + 1).padStart(2, "0")}
              </span>
              <span>
                <span className="block text-sm font-semibold">{stage.title}</span>
                <span className="block min-h-[32px] text-xs leading-snug text-muted">
                  {stage.key === "scene" && states.mask.rejected.length > 0
                    ? `${states.mask.rejected.length} cloudy image${states.mask.rejected.length > 1 ? "s" : ""} skipped`
                    : stage.line}
                </span>
              </span>
              <span className={`relative block aspect-square overflow-hidden border border-hairline bg-surface-2 transition-opacity ${
                state.status === "pending" ? "opacity-40" : ""} ${isSelected ? "outline outline-2 outline-offset-[3px] outline-ink" : ""}`}>
                {done && previews[0] ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previews[0].url} alt={`${stage.title}: ${IMAGE_LABELS[previews[0].id] ?? previews[0].id}`} className="anim-reveal pixelated h-full w-full object-cover" />
                ) : done && stage.key === "search" ? (
                  <span className="anim-reveal block h-full w-full p-1.5"><CloudBars candidates={candidates} chosen={chosenId} /></span>
                ) : done && stage.key === "correct" && bands.length ? (
                  <span className="anim-reveal block h-full w-full"><MiniSpectrum bands={bands} /></span>
                ) : done && stage.key === "locate" && (state.data.water_body as { geometry?: GeoJSON.Geometry } | null)?.geometry ? (
                  <span className="anim-reveal block h-full w-full p-1.5">
                    <LakeOutline geometry={(state.data.water_body as { geometry?: GeoJSON.Geometry }).geometry} corners={state.data.grid_corners as [number, number][]} />
                  </span>
                ) : done && stage.key === "locate" ? (
                  <span className="anim-reveal flex h-full w-full flex-col justify-end p-2.5">
                    <span className="font-serif text-lg leading-tight">{(state.data.water_body as { name?: string } | null)?.name ?? "Point on map"}</span>
                    <span className="font-mono text-[10px] text-muted">{stats("locate", state.data, state)[1].v} window</span>
                  </span>
                ) : (
                  <span className="ph absolute inset-0 grid place-items-center p-2 text-center font-mono text-[10px] text-muted">
                    {state.status === "failed" ? "stopped" : active ? "working…" : "waiting"}
                  </span>
                )}
                {active && <span className="anim-scan absolute left-0 right-0 h-[30%] bg-gradient-to-b from-transparent via-accent-wash to-transparent" aria-hidden />}
              </span>
            </button>
          );
        })}
      </div>

      <div key={selectedKey} className="anim-rise mt-8 grid gap-8 border-t border-ink pt-7 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:gap-10">
        <figure>
          <div className="relative aspect-[4/3] overflow-hidden border border-hairline bg-surface-2">
            {shownImage ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={shownImage.id} src={shownImage.url} alt={IMAGE_LABELS[shownImage.id] ?? shownImage.id} className="anim-reveal pixelated h-full w-full object-contain" />
            ) : selectedKey === "locate" && (selState.data.water_body as { geometry?: GeoJSON.Geometry } | null)?.geometry ? (
              <div className="anim-reveal h-full w-full p-4">
                <LakeOutline geometry={(selState.data.water_body as { geometry?: GeoJSON.Geometry }).geometry} corners={selState.data.grid_corners as [number, number][]} />
              </div>
            ) : selectedKey === "correct" && (selState.data.reflectance as ReflectanceBand[] | undefined)?.length ? (
              <div className="h-full w-full p-3"><MiniSpectrum bands={selState.data.reflectance as ReflectanceBand[]} large /></div>
            ) : selectedKey === "search" && (selState.data.candidates as SceneSummary[] | undefined)?.length ? (
              <div className="h-full overflow-auto p-3">
                <table className="w-full text-xs tabular">
                  <thead><tr className="text-left text-muted"><th className="pb-1 font-normal">Acquired</th><th className="font-normal">Satellite</th><th className="text-right font-normal">Tile cloud</th></tr></thead>
                  <tbody>
                    {(selState.data.candidates as SceneSummary[]).map((c) => (
                      <tr key={c.id} className="border-t border-hairline">
                        <td className="py-1">{shortDate(c.datetime)}</td><td>{c.platform}</td><td className="text-right">{pct(c.cloud_cover_pct ?? undefined)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="ph absolute inset-0 grid place-items-center p-4 text-center font-mono text-[11px] text-muted">
                {selState.status === "pending" ? "This stage has not run yet" : selState.status === "active" ? "Working…" : selected.heading}
              </div>
            )}
          </div>
          {selPreviews.length > 1 && (
            <div className="mt-2 flex flex-wrap gap-1.5" role="tablist" aria-label="Images from this stage">
              {selPreviews.map((p) => (
                <button key={p.id} type="button" role="tab" aria-selected={p.id === shownImage?.id}
                  onClick={() => setImageChoice((c) => ({ ...c, [selectedKey]: p.id }))}
                  className={`border px-2.5 py-1 text-xs ${p.id === shownImage?.id ? "border-ink bg-surface-2 text-ink" : "border-hairline text-ink-2 hover:text-ink"}`}>
                  {IMAGE_LABELS[p.id] ?? p.id}
                </button>
              ))}
            </div>
          )}
          <figcaption className="mt-2 text-xs text-muted">
            <b className="font-semibold text-ink">Fig. 1.{STAGES.indexOf(selected) + 1}</b>{" "}
            {selState.message || selected.line}
          </figcaption>
        </figure>
        <div>
          <div className="secnum">Stage {String(STAGES.indexOf(selected) + 1).padStart(2, "0")}</div>
          <h3 className="mb-2.5 mt-0.5 font-serif text-[28px] font-medium leading-tight">{selected.heading}</h3>
          <p className="mb-5 max-w-[38em] text-[15px] leading-relaxed text-ink-2 text-pretty">{selected.body}</p>
          <dl className="grid grid-cols-3 border-t border-hairline">
            {stats(selectedKey, selState.data, selState).map((s) => (
              <div key={s.k} className="border-b border-hairline py-3 pr-3">
                <dd className="font-serif text-2xl font-medium leading-tight">{s.v}</dd>
                <dt className="mt-0.5 text-xs text-muted">{s.k}</dt>
              </div>
            ))}
          </dl>
          {selectedKey === "mask" && (selState.data.pixels as PixelCounts | undefined) && (
            <div className="mt-5"><PixelBreakdown pixels={selState.data.pixels as PixelCounts} /></div>
          )}
          {selectedKey === "scene" && states.mask.rejected.length > 0 && (
            <ul className="mt-4 space-y-1 text-xs text-ink-2">
              {states.mask.rejected.map((r, i) => <li key={i}>↷ {r.message}</li>)}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
