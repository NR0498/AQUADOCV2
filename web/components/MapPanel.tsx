"use client";

import { useMemo, useState } from "react";
import MapView from "@/components/MapView";
import { apiUrl, type AnalysisResult, type Layer } from "@/lib/api";
import { fmt, unitLabel } from "@/lib/format";

type Props = {
  result: AnalysisResult | null;
  layerId: string;
  onLayerChange: (id: string) => void;
  point: { latitude: number; longitude: number } | null;
  onPick?: (latitude: number, longitude: number) => void;
  flyTo?: { bbox?: [number, number, number, number]; center?: [number, number] } | null;
  interactive?: boolean;
};

function Legend({ layer }: { layer: Layer }) {
  if (layer.legend) {
    return (
      <ul className="space-y-1 text-xs">
        {Object.entries(layer.legend).map(([name, color]) => (
          <li key={name} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: color }} aria-hidden />
            <span className="text-ink-2">{name.replaceAll("_", " ")}</span>
          </li>
        ))}
      </ul>
    );
  }
  if (!layer.colors) return null;
  const mid = layer.scale === "log10" && layer.vmin && layer.vmax ? Math.sqrt(layer.vmin * layer.vmax) : null;
  return (
    <div className="w-48">
      <div className="h-2.5 rounded-sm" style={{ background: `linear-gradient(to right, ${layer.colors.join(",")})` }} />
      <div className="mt-1 flex justify-between text-[11px] text-ink-2 tabular">
        <span>{fmt(layer.vmin)}</span>
        {mid && <span>{fmt(mid)}</span>}
        <span>{fmt(layer.vmax)}</span>
      </div>
      <p className="text-[11px] text-muted">{layer.unit ? unitLabel(layer.unit) : ""}{layer.scale === "log10" ? " · log scale" : ""}</p>
    </div>
  );
}

export default function MapPanel({ result, layerId, onLayerChange, point, onPick, flyTo = null, interactive = true }: Props) {
  const [opacity, setOpacity] = useState(0.85);
  const layer = result?.layers.find((l) => l.id === layerId) ?? null;
  const corners = result?.grid.corners_wgs84 ?? null;
  const overlay = useMemo(
    () => (layer && corners ? { url: apiUrl(layer.url), corners, opacity } : null),
    [layer, corners, opacity],
  );

  return (
    <div className="panel flex h-full flex-col overflow-hidden">
      <div className="relative min-h-[320px] flex-1">
        <MapView
          point={point}
          waterBody={result?.water_body?.geometry ?? null}
          overlay={overlay}
          fitTo={corners}
          flyTo={flyTo}
          onPick={onPick ?? (() => {})}
          interactive={interactive}
        />
        {layer && (
          <div className="panel absolute bottom-8 left-3 p-3 shadow-md">
            <p className="mb-2 text-xs font-medium">{layer.label}</p>
            <Legend layer={layer} />
          </div>
        )}
      </div>
      {result && (
        <div className="flex flex-wrap items-center gap-3 border-t border-hairline px-3 py-2 text-xs">
          <label className="flex items-center gap-2">
            <span className="text-ink-2">Layer</span>
            <select value={layerId} onChange={(e) => onLayerChange(e.target.value)}
              className="rounded-md border border-hairline bg-surface px-2 py-1">
              {result.layers.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="text-ink-2">Opacity</span>
            <input type="range" min={0} max={1} step={0.05} value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Layer opacity" />
          </label>
          <span className="ml-auto text-muted">20 m pixels · {result.grid.extent}</span>
        </div>
      )}
    </div>
  );
}
