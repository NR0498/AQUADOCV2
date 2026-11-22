"use client";

import Link from "next/link";
import MapPanel from "@/components/MapPanel";
import ParamRows from "@/components/ParamRows";
import SensitivityBars from "@/components/SensitivityBars";
import SpectrumChart from "@/components/SpectrumChart";
import { TARGET_ORDER, type AnalysisResult } from "@/lib/api";
import { fmt, pct, shortDate, unitLabel } from "@/lib/format";
import { useGuide } from "@/lib/useGuide";

type Props = {
  result: AnalysisResult;
  selected: string;
  onSelect: (target: string) => void;
  layerId: string;
  onLayerChange: (id: string) => void;
  shareLink?: boolean;
  placeName?: string | null; // the name the user chose, for water OpenStreetMap leaves unnamed
};

export function SectionHead({ num, title, children, aside, id }: {
  num: string; title: string; children?: React.ReactNode; aside?: React.ReactNode; id?: string;
}) {
  return (
    <div className="mb-7 grid gap-2 sm:grid-cols-[200px_minmax(0,1fr)_auto] sm:items-baseline sm:gap-6">
      <div className="secnum">{num}</div>
      <div>
        <h2 id={id} className="font-serif text-[28px] font-medium leading-tight tracking-tight sm:text-[32px]">{title}</h2>
        {children && <p className="mt-1.5 max-w-[46em] text-[15px] text-ink-2 text-pretty">{children}</p>}
      </div>
      {aside && <div className="font-mono text-xs text-ink-2">{aside}</div>}
    </div>
  );
}

export default function ResultPanels({ result, selected, onSelect, layerId, onLayerChange, shareLink = true, placeName }: Props) {
  const { guide } = useGuide();
  const estimates = TARGET_ORDER.filter((t) => result.estimates[t]).map((t) => [t, result.estimates[t]] as [string, typeof result.estimates[string]]);
  const estimate = result.estimates[selected];
  const descriptions = Object.fromEntries((guide?.bands ?? []).map((b) => [b.band, b.water]));
  // Without an OpenStreetMap outline (lookup timed out or nothing mapped) name the place by its coordinates.
  const name = result.water_body?.name ?? placeName
    ?? `Water at ${result.request.latitude.toFixed(3)}°, ${result.request.longitude.toFixed(3)}°`;
  const floating = result.floating_material;
  const layer = result.layers.find((l) => l.id === layerId);
  const sens = result.band_sensitivity[selected] ?? [];
  const top = [...sens].sort((a, b) => b.share_pct - a.share_pct)[0];

  return (
    <>
      <section className="border-b border-hairline py-12" aria-labelledby="findings-title">
        <SectionHead num="03 — Findings" id="findings-title" title={`${name}, ${shortDate(result.scene.datetime)}`}
          aside={shareLink ? <Link href={`/analyses/${result.id}`} className="no-underline hover:underline">Shareable link →</Link> : undefined}>
          {result.pixels.valid_open_water.toLocaleString()} clear open-water pixels ({pct(result.pixels.clear_fraction * 100)} of the
          water area){result.cached ? ", from an analysis made earlier today" : ""}. Select a parameter to map it.
        </SectionHead>

        <div className="grid gap-10 lg:grid-cols-2">
          <figure className="min-w-0">
            <div className="aspect-square max-h-[640px] w-full">
              <MapPanel result={result} layerId={layerId} onLayerChange={onLayerChange}
                point={{ latitude: result.request.latitude, longitude: result.request.longitude }} interactive={false} />
            </div>
            <figcaption className="mt-2 text-xs text-muted">
              <b className="font-semibold text-ink">Fig. 2</b> {layer?.label ?? "Layer"} across the water at {result.grid.resolution_m} m
              resolution{layer?.scale === "log10" ? ", log colour scale" : ""}. Satellite basemap © Esri; outline © OpenStreetMap.
            </figcaption>
          </figure>

          <div className="min-w-0">
            <ParamRows estimates={estimates} selected={selected} onSelect={onSelect} />
            <div className="mt-6 grid gap-4 border-t border-hairline pt-5 sm:grid-cols-2">
              <div>
                <div className="kicker">{floating.label}</div>
                <p className="mt-1 font-serif text-[30px] font-medium leading-none">{pct(floating.vegetation_or_scum_pct, 1)}</p>
                <p className="mt-1 text-xs text-ink-2">of the surface looks like floating vegetation or algal scum</p>
              </div>
              <div>
                <div className="kicker">Possible debris</div>
                <p className="mt-1 font-serif text-[30px] font-medium leading-none">{pct(floating.debris_like_pct, 1)}</p>
                <p className="mt-1 text-xs text-ink-2">unusually bright in near-infrared (debris, foam or plastics). An index, not a count.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {estimate && (
        <section className="border-b border-hairline py-12" aria-labelledby="bands-title">
          <SectionHead num="04 — Fig. 3" id="bands-title" title={`How the bands reveal ${estimate.label.toLowerCase()}`}
            aside={top ? <>top band here: <b className="text-ink">{top.band}</b> · {pct(top.share_pct)}</> : undefined}>
            {estimate.what}
          </SectionHead>
          <div className="grid gap-10 lg:grid-cols-[3fr_2fr]">
            <div className="min-w-0"><SpectrumChart spectrum={result.spectrum} highlight={estimate.bands} descriptions={descriptions} /></div>
            <div className="min-w-0">
              <h3 className="mb-1 text-sm font-semibold">Which bands moved this estimate here</h3>
              <p className="mb-3 text-xs text-muted">
                Each band is nudged ±10% at this water&apos;s median spectrum; bars show its share of the response.
                <Link href="/bands" className="ml-1">Band guide →</Link>
              </p>
              <SensitivityBars items={sens} label={estimate.label} />
              <dl className="mt-5 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 border-t border-hairline text-xs">
                <dt className="border-b border-hairline py-2 text-muted">Variation across the water (10th–90th %)</dt>
                <dd className="border-b border-hairline py-2 text-right tabular">{fmt(estimate.spatial_p10)}–{fmt(estimate.spatial_p90)} {unitLabel(estimate.unit)}</dd>
                <dt className="border-b border-hairline py-2 text-muted">Range basis</dt>
                <dd className="border-b border-hairline py-2 text-right">{estimate.interval_basis}</dd>
                <dt className="border-b border-hairline py-2 text-muted">Reflectance input</dt>
                <dd className="border-b border-hairline py-2 text-right">{estimate.reflectance_input === "corrected" ? "match-up corrected" : "L2A as delivered"}</dd>
              </dl>
            </div>
          </div>
        </section>
      )}

      <section className="grid gap-10 border-b border-hairline py-12 lg:grid-cols-2">
        <div>
          <div className="secnum">05 — Limits</div>
          <h2 className="mb-3.5 mt-1 font-serif text-[26px] font-medium">What a satellite cannot see</h2>
          <ul>
            {result.not_detectable.map((item) => (
              <li key={item.name} className="border-b border-hairline py-2.5 text-sm text-ink-2">
                <b className="font-semibold text-ink">{item.name}.</b> {item.why}
              </li>
            ))}
          </ul>
        </div>
        <div className="min-w-0">
          <div className="secnum">06 — Provenance</div>
          <h2 className="mb-3.5 mt-1 font-serif text-[26px] font-medium">Reproduce this result</h2>
          <dl className="grid grid-cols-[120px_minmax(0,1fr)] gap-x-4 gap-y-2 font-mono text-xs text-ink-2 sm:grid-cols-[160px_minmax(0,1fr)]">
            <dt className="text-muted">Scene</dt><dd className="break-all">{result.scene.id}</dd>
            <dt className="text-muted">Acquired</dt>
            <dd>{new Date(result.scene.datetime).toUTCString().replace(" GMT", " UTC")} · tile cloud {pct(result.scene.cloud_cover_pct ?? undefined)}</dd>
            <dt className="text-muted">Images tried</dt>
            <dd>{result.attempts.map((a) => `${shortDate(a.datetime)} (${a.outcome.replaceAll("_", " ")})`).join(", ")}</dd>
            <dt className="text-muted">Outline</dt>
            <dd>{result.water_body ? `OpenStreetMap ${result.water_body.osm}` : "none (spectral water detection)"} · {result.grid.shoreline_buffer_m ?? 0} m shore buffer</dd>
            <dt className="text-muted">Model</dt>
            <dd>{result.model.version}{result.model.satellite_correction ? ` + ${result.model.satellite_correction}` : ""}</dd>
            <dt className="text-muted">Published algorithms</dt>
            <dd>
              NDCI chl {fmt(result.published_algorithms.chlorophyll_a_ndci_mishra2012?.median)} µg/L · Nechad TSS{" "}
              {fmt(result.published_algorithms.tss_nechad2010?.median)} mg/L · Dogliotti {fmt(result.published_algorithms.turbidity_dogliotti2015?.median)} FNU
            </dd>
            <dt className="text-muted">Run time</dt><dd>{(result.elapsed_ms / 1000).toFixed(1)} s</dd>
          </dl>
          {result.notes.length > 0 && (
            <ul className="mt-5 space-y-1 border-l-2 border-ochre pl-3 text-sm text-ink-2">
              {result.notes.map((note) => <li key={note}>{note}</li>)}
            </ul>
          )}
          <p className="mt-5 text-xs text-muted">{result.disclaimer}</p>
        </div>
      </section>
    </>
  );
}
