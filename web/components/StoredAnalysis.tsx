"use client";

import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";
import Pipeline from "@/components/Pipeline";
import ResultPanels from "@/components/ResultPanels";
import { apiUrl, getJSON, type AnalysisResult } from "@/lib/api";
import { shortDate } from "@/lib/format";
import { useReplay } from "@/lib/useReplay";

export default function StoredAnalysis({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  // Keyed by id so a stale result is never shown after navigating to another analysis.
  const [loaded, setLoaded] = useState<{ id: string; result?: AnalysisResult; error?: string } | null>(null);
  const [selected, setSelected] = useState("chlorophyll_a");
  const [layerId, setLayerId] = useState("chlorophyll_a");
  const { events, replaying, replay } = useReplay();

  useEffect(() => {
    getJSON<AnalysisResult>(`/api/analyses/${encodeURIComponent(id)}`)
      .then((result) => {
        setLoaded({ id, result });
        replay(result.timeline);
      })
      .catch((e: Error) => setLoaded({ id, error: e.message }));
  }, [id, replay]);
  const current = loaded?.id === id ? loaded : null;
  const result = current?.result ?? null;
  const error = current?.error ?? null;

  const layerUrl = useMemo(() => {
    const urls = Object.fromEntries((result?.layers ?? []).map((l) => [l.id, apiUrl(l.url)]));
    return (layer: string) => urls[layer];
  }, [result]);

  if (error) {
    return (
      <div className="mx-auto max-w-xl px-4 py-16">
        <p className="border border-hairline p-4 text-sm">
          This analysis could not be loaded ({error}). <Link href="/">Start a new one</Link>.
        </p>
      </div>
    );
  }
  if (!result) return <p className="anim-pulse mx-auto max-w-[1240px] px-4 py-16 font-mono text-xs text-ink-2 sm:px-10">Loading analysis…</p>;

  const name = result.water_body?.name ?? `${result.request.latitude.toFixed(4)}, ${result.request.longitude.toFixed(4)}`;
  return (
    <div className="mx-auto w-full max-w-[1240px] px-4 sm:px-10">
      <header className="anim-rise border-b border-hairline pb-10 pt-10 sm:pt-14">
        <p className="kicker mb-3">Saved analysis · {result.id}</p>
        <h1 className="font-serif text-[40px] font-medium leading-[1.04] tracking-[-0.02em] text-balance sm:text-[52px]">{name}</h1>
        <p className="mt-3 font-serif text-[19px] text-ink-2">
          Sentinel-2 image of {shortDate(result.scene.datetime)} · analysed {shortDate(result.created_at)} ·{" "}
          <Link href="/">Run a new analysis →</Link>
        </p>
      </header>
      <div className="border-b border-hairline py-12">
        <Pipeline events={events} running={replaying} layerUrl={layerUrl} replay
          onReplay={result.timeline?.length ? () => replay(result.timeline) : undefined} />
        {!result.timeline?.length && (
          <p className="mt-4 text-xs text-muted">This result was saved before processing timelines were recorded, so its stages cannot be replayed.</p>
        )}
      </div>
      <ResultPanels result={result} selected={selected} onSelect={(t) => { setSelected(t); setLayerId(t); }}
        layerId={layerId} onLayerChange={setLayerId} shareLink={false} />
    </div>
  );
}
