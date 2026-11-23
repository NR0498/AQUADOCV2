"use client";

import type { StageEvent } from "@/lib/api";
import { shortDate } from "@/lib/format";

/** Shape of `data.failure` on a pipeline `failed` event (see water_pipeline._failure_summary). */
type FailedImage = {
  datetime: string | null;
  platform: string | null;
  tile_cloud_pct: number | null;
  outcome: string;
  water_pixels?: number;
  clear_pct?: number;
  cloud_pct?: number;
  vegetation_pct?: number;
  not_water_pct?: number;
  glint_pct?: number;
  main_blocker?: "cloud" | "floating" | "not_water" | "glint";
};

export type Failure = {
  reason: string;
  images_tried?: number;
  images?: FailedImage[];
  needed_clear_pct?: number;
  needed_pixels?: number;
  max_tile_cloud_pct?: number;
  start_date?: string;
  end_date?: string;
};

type Copy = { title: string; what: string; why: string; tips: string[] };

const LONGER = "Choose a longer date range (for example the last 6–12 months), so more satellite passes can be checked.";
const DRY_SEASON = "Pick a dry-season window: monsoon months are often cloudy for weeks at a time.";

const COPY: Record<string, Copy> = {
  cloud: {
    title: "Clouds or haze covered the water on every image",
    what: "AquaDoc checked the most recent Sentinel-2 images of this lake, pixel by pixel. On each one, too little of the water was clearly visible, so no estimate was made.",
    why: "Sentinel-2 measures sunlight reflected from the water. Clouds, their shadows and the haze around them block or distort that light. A number computed through them would describe the cloud rather than the water, so these pixels are removed and the image is rejected.",
    tips: [LONGER, DRY_SEASON, "Try again in a few days. Sentinel-2 passes over every location about every 5 days."],
  },
  floating_vegetation: {
    title: "The water is covered by floating vegetation or algal scum",
    what: "The images were clear, but most of the lake surface looked like plants (water hyacinth, duckweed, lotus beds) or a dense algal scum rather than open water.",
    why: "The water-quality model estimates what is dissolved and suspended in open water, using how that water absorbs and scatters light. Floating plants reflect strongly in the near-infrared like land vegetation, which hides the water underneath. With too little open water left, any estimate would be unreliable. The heavy surface cover is itself a sign of nutrient-rich, eutrophic water.",
    tips: ["Try an image from another season, after the vegetation has been cleared or has died back.", LONGER,
      "If only part of the lake is open, drop a pin on the open-water part to analyse that area."],
  },
  not_open_water: {
    title: "The surface did not look like open water",
    what: "On every clear image, most pixels inside the lake outline did not have the spectral signature of open water.",
    why: "Water absorbs strongly in the shortwave infrared, and the model is only applied where that signal is present. Dense vegetation, scum, a dried-up or very shallow lake bed, construction or an outdated map outline all remove it. Unusually coloured water, such as a salt lake, can do the same.",
    tips: ["Check the true-colour image in stage 03. If the lake has shrunk or is dry, analyse a wetter season.",
      "Drop a pin directly on visible water instead of using the mapped outline.", LONGER],
  },
  glint: {
    title: "Sun glint made the water too bright",
    what: "Most of the water reflected sunlight directly towards the satellite, like a mirror, on the images tried.",
    why: "Glint adds a bright signal at every wavelength that has nothing to do with what is in the water. AquaDoc detects it with the shortwave-infrared band and removes those pixels.",
    tips: [LONGER, "Try again later. Glint depends on the sun angle and the wind on the day the image was taken."],
  },
  too_small: {
    title: "This water body is too small for the satellite",
    what: "After removing the shoreline, too few pure water pixels were left to analyse.",
    why: "Each Sentinel-2 pixel covers 20 × 20 m. Pixels near the shore mix water with land, trees and buildings, so a buffer of up to 100 m is removed from the edge. Small ponds and narrow rivers can end up with no usable pixels.",
    tips: ["Choose a larger lake or reservoir (roughly 10 hectares or more works best).",
      "For rivers, pick a wide stretch and drop the pin in the middle of the channel."],
  },
  unreadable: {
    title: "The satellite images could not be downloaded",
    what: "Images were found in the archive, but none of them could be read.",
    why: "This is almost always a temporary problem with the Microsoft Planetary Computer image servers, not with the lake.",
    tips: ["Try again in a minute.", LONGER],
  },
  no_scenes: {
    title: "No usable satellite images in this period",
    what: "The archive had no Sentinel-2 image of this location with acceptable cloud cover between the chosen dates.",
    why: "Images covered mostly by cloud are skipped before download because they cannot show the water. Short date ranges in cloudy seasons often have none left.",
    tips: [LONGER, DRY_SEASON],
  },
  catalogue_error: {
    title: "The satellite archive could not be reached",
    what: "The search for Sentinel-2 images failed before any image was checked.",
    why: "The image catalogue (Microsoft Planetary Computer) was busy or unreachable. This is not related to the lake.",
    tips: ["Try again in a minute."],
  },
  invalid_request: {
    title: "The request was not valid",
    what: "The location or date range could not be used.",
    why: "Coordinates must be valid latitude and longitude, and Sentinel-2 Level-2A images are only available from 2017.",
    tips: ["Check the coordinates and dates, then run the analysis again."],
  },
};

const BLOCKER_LABEL: Record<string, string> = {
  cloud: "clouds / haze", floating: "floating vegetation", not_water: "not water-like", glint: "sun glint",
};

const SEGMENTS: { key: keyof FailedImage; label: string; color: string }[] = [
  { key: "clear_pct", label: "Clear water", color: "#2a78d6" },
  { key: "vegetation_pct", label: "Vegetation / scum", color: "#1baf7a" },
  { key: "not_water_pct", label: "Not water-like", color: "#6250d6" },
  { key: "glint_pct", label: "Glint", color: "#eb6834" },
  { key: "cloud_pct", label: "Cloud, shadow, haze", color: "#c3c2b7" },
];

function ImageBar({ image }: { image: FailedImage }) {
  const parts = SEGMENTS.map((s) => ({ ...s, value: Math.max(0, Number(image[s.key] ?? 0)) }));
  const total = parts.reduce((sum, p) => sum + p.value, 0) || 1;
  return (
    <div className="flex h-2.5 w-full overflow-hidden rounded-sm bg-surface-2" role="img"
      aria-label={parts.map((p) => `${p.label} ${p.value.toFixed(0)}%`).join(", ")}>
      {parts.map((p) => p.value > 0 && (
        <span key={p.key} style={{ width: `${(p.value / total) * 100}%`, background: p.color }} title={`${p.label}: ${p.value.toFixed(0)}%`} />
      ))}
    </div>
  );
}

/** Plain-language explanation shown under the pipeline when an analysis stops. */
export default function FailureExplainer({ event }: { event: StageEvent }) {
  const failure = (event.data.failure as Failure | undefined) ?? { reason: "unknown" };
  const copy = COPY[failure.reason];
  const images = failure.images ?? [];
  const readable = images.filter((i) => i.clear_pct != null);

  return (
    <section role="alert" aria-labelledby="failure-title"
      className="anim-drop mt-8 border border-[var(--critical)] bg-[color-mix(in_oklch,var(--critical)_5%,var(--page))] p-4 sm:p-6">
      <p className="font-mono text-[11px] uppercase tracking-wider text-[var(--critical)]">Analysis stopped · no estimate made</p>
      <h3 id="failure-title" className="mt-1.5 font-serif text-[22px] font-medium leading-tight text-balance sm:text-[26px]">
        {copy?.title ?? "The analysis could not finish"}
      </h3>
      <p className="mt-2 max-w-[44em] text-sm text-ink-2">{event.message}</p>

      {copy && (
        <div className="mt-5 grid gap-5 sm:grid-cols-2 sm:gap-8">
          <div>
            <h4 className="text-sm font-semibold">What happened</h4>
            <p className="mt-1 text-sm leading-relaxed text-ink-2 text-pretty">{copy.what}</p>
          </div>
          <div>
            <h4 className="text-sm font-semibold">Why it cannot be analysed</h4>
            <p className="mt-1 text-sm leading-relaxed text-ink-2 text-pretty">{copy.why}</p>
          </div>
        </div>
      )}

      {readable.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold">
            Images checked ({images.length}){failure.needed_clear_pct != null && (
              <span className="font-normal text-muted"> · at least {failure.needed_clear_pct.toFixed(0)}% of the water must be clear</span>
            )}
          </h4>
          <ul className="mt-2 divide-y divide-hairline border-y border-hairline">
            {images.map((image, i) => (
              <li key={i} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1.5 py-2.5 sm:grid-cols-[9rem_minmax(0,1fr)_11rem]">
                <span className="text-sm tabular">
                  {image.datetime ? shortDate(image.datetime) : "–"}
                  <span className="block text-[11px] text-muted">{image.platform ?? "Sentinel-2"}</span>
                </span>
                <span className="text-right text-xs text-ink-2 sm:order-last">
                  {image.clear_pct != null ? (
                    <><b className="font-semibold text-ink tabular">{image.clear_pct.toFixed(0)}%</b> clear
                      {image.main_blocker && <span className="block text-muted">mostly {BLOCKER_LABEL[image.main_blocker]}</span>}</>
                  ) : "could not be read"}
                </span>
                <span className="col-span-2 sm:col-span-1">{image.clear_pct != null && <ImageBar image={image} />}</span>
              </li>
            ))}
          </ul>
          <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
            {SEGMENTS.map((s) => (
              <li key={s.key} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} aria-hidden />{s.label}
              </li>
            ))}
          </ul>
        </div>
      )}

      {copy?.tips.length ? (
        <div className="mt-6">
          <h4 className="text-sm font-semibold">What you can try</h4>
          <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-ink-2">
            {copy.tips.map((t) => <li key={t}>{t}</li>)}
          </ul>
        </div>
      ) : null}

      <p className="mt-5 text-xs text-muted">
        AquaDoc stops rather than guessing: an estimate from cloudy, vegetated or mixed pixels would look precise but be wrong.
      </p>
    </section>
  );
}
