"use client";

import { useState } from "react";
import type { PixelCounts } from "@/lib/api";

// Order and colours match the backend's validated pixel-class palette (render.PIXEL_CLASS_COLORS).
const CLASSES: { key: keyof PixelCounts | "cloud_total"; label: string; color: string; ink: string }[] = [
  { key: "valid_open_water", label: "Clear open water (used)", color: "#2a78d6", ink: "#fff" },
  { key: "floating_vegetation_or_scum", label: "Floating vegetation / scum", color: "#1baf7a", ink: "#fff" },
  { key: "glint_or_mixed", label: "Glint or mixed", color: "#eb6834", ink: "#fff" },
  { key: "not_water_like", label: "Not water-like", color: "#6250d6", ink: "#fff" },
  { key: "cloud_adjacent", label: "Near cloud (haze buffer)", color: "#c3c2b7", ink: "#0b0b0b" },
  { key: "cloud_total", label: "Cloud, shadow, snow, no data", color: "#e1e0d9", ink: "#0b0b0b" },
];

/** Tall segmented bar: where every pixel of the water area went. */
export default function PixelBreakdown({ pixels }: { pixels: PixelCounts }) {
  const [hover, setHover] = useState<string | null>(null);
  const value = (key: string) =>
    key === "cloud_total"
      ? pixels.cloud + pixels.cloud_shadow + pixels.snow_ice + pixels.saturated_or_defective + pixels.no_data
      : (pixels[key as keyof PixelCounts] as number);
  const total = CLASSES.reduce((sum, c) => sum + value(c.key), 0) || 1;
  const hectares = (n: number) => ((n * 400) / 10_000).toLocaleString("en-US", { maximumFractionDigits: 0 });

  return (
    <div>
      <div className="flex h-7 w-full gap-[2px] overflow-hidden rounded" role="img"
        aria-label={CLASSES.map((c) => `${c.label} ${((value(c.key) / total) * 100).toFixed(0)}%`).join(", ")}>
        {CLASSES.map((c) => {
          const share = value(c.key) / total;
          if (share <= 0) return null;
          const label = `${(share * 100).toFixed(0)}%`;
          return (
            <div key={c.key} className="flex h-full items-center rounded-[4px] px-1.5 text-[11px] font-medium"
              style={{ width: `${share * 100}%`, background: c.color, color: c.ink, opacity: hover && hover !== c.key ? 0.5 : 1 }}
              onMouseEnter={() => setHover(c.key)} onMouseLeave={() => setHover(null)}
              title={`${c.label}: ${value(c.key).toLocaleString()} pixels (${label})`}>
              {share > 0.07 ? label : ""}
            </div>
          );
        })}
      </div>
      <ul className="mt-3 grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
        {CLASSES.map((c) => (
          <li key={c.key} className="flex items-center gap-2" onMouseEnter={() => setHover(c.key)} onMouseLeave={() => setHover(null)}>
            <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: c.color }} aria-hidden />
            <span className="flex-1 text-ink-2">{c.label}</span>
            <span className="tabular text-ink-2">{value(c.key).toLocaleString()} px · {hectares(value(c.key))} ha</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted">
        Each pixel is 20 × 20 m (0.04 ha). Only clear open water feeds the estimates; floating material is reported separately.
      </p>
    </div>
  );
}
