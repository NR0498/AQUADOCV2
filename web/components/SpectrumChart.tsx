"use client";

import { useState } from "react";
import type { SpectrumPoint } from "@/lib/api";
import { fmt, niceTicks } from "@/lib/format";

type Props = { spectrum: SpectrumPoint[]; highlight: string[]; descriptions?: Record<string, string> };

const W = 640;
const H = 240;
const M = { top: 16, right: 16, bottom: 44, left: 52 };

/** Median water reflectance per band, with the interquartile range as a wash.
 *  Bands are spaced evenly (not by wavelength) so the visible/red-edge detail stays readable. */
export default function SpectrumChart({ spectrum, highlight, descriptions = {} }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const points = spectrum.filter((p) => p.reflectance_median !== null);
  if (!points.length) return null;
  const maxY = Math.max(...points.map((p) => p.reflectance_p75 ?? p.reflectance_median ?? 0)) * 1.1;
  const ticks = niceTicks(0, maxY, 4);
  const top = Math.max(maxY, ticks[ticks.length - 1]);
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;
  const x = (i: number) => M.left + (innerW * (i + 0.5)) / points.length;
  const y = (v: number) => M.top + innerH - (v / top) * innerH;
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.reflectance_median!)}`).join(" ");
  const area =
    points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.reflectance_p75 ?? p.reflectance_median!)}`).join(" ") +
    points.slice().reverse().map((p, j) => `L${x(points.length - 1 - j)},${y(p.reflectance_p25 ?? p.reflectance_median!)}`).join(" ") +
    "Z";
  const active = hover !== null ? points[hover] : null;

  return (
    <figure className="relative min-w-0">
      <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[520px]" role="img"
        aria-label="Median surface reflectance of the analysed water for each Sentinel-2 band">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--hairline)" strokeWidth={1} />
            <text x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize={11} fill="var(--ink-muted)" className="tabular">
              {t.toFixed(t < 0.1 ? 3 : 2)}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--axis)" strokeWidth={1} />
        <text x={14} y={M.top + innerH / 2} transform={`rotate(-90 14 ${M.top + innerH / 2})`} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
          Reflectance
        </text>
        <path d={area} fill="var(--series-1)" opacity={0.1} />
        <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => {
          const on = highlight.includes(p.band);
          return (
            <g key={p.band}>
              <circle cx={x(i)} cy={y(p.reflectance_median!)} r={on ? 6 : 4} fill={on ? "var(--series-1)" : "var(--surface)"}
                stroke={on ? "var(--surface)" : "var(--series-1)"} strokeWidth={2} />
              <text x={x(i)} y={H - M.bottom + 16} textAnchor="middle" fontSize={11}
                fill={on ? "var(--ink)" : "var(--ink-muted)"} fontWeight={on ? 600 : 400}>
                {p.band}
              </text>
              <text x={x(i)} y={H - M.bottom + 30} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" className="tabular">
                {p.center_nm}
              </text>
              <rect x={x(i) - innerW / points.length / 2} y={M.top} width={innerW / points.length} height={innerH}
                fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(i)} onBlur={() => setHover(null)} tabIndex={0}
                aria-label={`${p.band} ${p.name}, ${p.center_nm} nanometres, median reflectance ${fmt(p.reflectance_median)}`} />
            </g>
          );
        })}
        {hover !== null && (
          <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + innerH} stroke="var(--axis)" strokeWidth={1} pointerEvents="none" />
        )}
      </svg>
      </div>
      {active && hover !== null && (
        <div className="panel pointer-events-none absolute z-10 w-60 p-3 text-xs shadow-lg"
          style={{ left: `min(calc(${(x(hover) / W) * 100}% + 12px), calc(100% - 15rem))`, top: 8 }}>
          <p className="font-semibold">{active.band} · {active.name} · {active.center_nm} nm</p>
          <p className="mt-1 tabular text-ink-2">
            Median {fmt(active.reflectance_median)} (middle half {fmt(active.reflectance_p25)}–{fmt(active.reflectance_p75)})
          </p>
          {descriptions[active.band] && <p className="mt-1 leading-relaxed text-ink-2">{descriptions[active.band]}</p>}
        </div>
      )}
      <figcaption className="mt-1 text-xs text-muted">
        Filled points are the bands that matter most for the selected parameter. Band labels show the centre wavelength in nm.
      </figcaption>
      <details className="mt-2 text-xs">
        <summary className="cursor-pointer text-ink-2">Show as table</summary>
        <table className="mt-2 w-full tabular">
          <thead><tr className="text-left text-muted"><th>Band</th><th>Name</th><th>nm</th><th className="text-right">Median</th><th className="text-right">25th</th><th className="text-right">75th</th></tr></thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.band} className="border-t border-hairline">
                <td>{p.band}</td><td>{p.name}</td><td>{p.center_nm}</td>
                <td className="text-right">{fmt(p.reflectance_median)}</td><td className="text-right">{fmt(p.reflectance_p25)}</td><td className="text-right">{fmt(p.reflectance_p75)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
