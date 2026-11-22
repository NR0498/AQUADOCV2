"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import NearbyLakes from "@/components/NearbyLakes";
import LocationChooser, { type DateWindow, type Recent, type Target } from "@/components/LocationChooser";
import Pipeline from "@/components/Pipeline";
import ResultPanels from "@/components/ResultPanels";
import { apiUrl, getJSON, streamAnalysis, type AnalysisResult, type StageEvent } from "@/lib/api";
import { useReplay } from "@/lib/useReplay";

export default function Analyzer() {
  const [target, setTarget] = useState<Target | null>(null);
  const [dates, setDates] = useState<DateWindow | null>(null); // set by the chooser after mount
  const [live, setLive] = useState<StageEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selected, setSelected] = useState("chlorophyll_a");
  const [layerId, setLayerId] = useState("chlorophyll_a");
  const [recent, setRecent] = useState<Recent[]>([]);
  const [runId, setRunId] = useState(0); // remounts the pipeline so a new run starts from its first stage
  const abort = useRef<AbortController | null>(null);
  const pipelineRef = useRef<HTMLDivElement>(null);
  const { events: replayed, replaying, replay, stop: stopReplay } = useReplay();
  // Live runs show streamed events; cached or reopened results replay their stored timeline.
  const [mode, setMode] = useState<"live" | "replay">("live");
  const events = mode === "live" ? live : replayed;

  const loadRecent = useCallback(() => {
    getJSON<{ analyses: Recent[] }>("/api/analyses?limit=6").then((d) => setRecent(d.analyses)).catch(() => {});
  }, []);
  useEffect(loadRecent, [loadRecent]);

  const showResult = useCallback((res: AnalysisResult, fromStore: boolean) => {
    setResult(res);
    if (fromStore || res.cached) {
      setMode("replay");
      replay(res.timeline);
    }
  }, [replay]);

  const scrollToPipeline = () =>
    requestAnimationFrame(() => pipelineRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));

  const run = useCallback(async (override?: Target) => {
    const chosen = override ?? target;
    if (!chosen || !dates) return;
    if (override) setTarget(override);
    abort.current?.abort();
    abort.current = new AbortController();
    stopReplay();
    setMode("live");
    setLive([]);
    setError(null);
    setResult(null);
    setRunning(true);
    setRunId((n) => n + 1);
    scrollToPipeline();
    try {
      await streamAnalysis(
        { latitude: chosen.latitude, longitude: chosen.longitude, osm_ref: chosen.osmRef, start_date: dates.start, end_date: dates.end },
        {
          onStage: (event) => setLive((list) => [...list, event]),
          onComplete: (res) => {
            showResult(res, false);
            loadRecent();
          },
          onError: setError,
        },
        abort.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError("Could not reach the analysis service.");
    } finally {
      setRunning(false);
    }
  }, [target, dates, stopReplay, showResult, loadRecent]);

  const openRecent = useCallback(async (id: string) => {
    abort.current?.abort();
    setError(null);
    try {
      const res = await getJSON<AnalysisResult>(`/api/analyses/${encodeURIComponent(id)}`);
      setTarget({ latitude: res.request.latitude, longitude: res.request.longitude, osmRef: res.water_body?.osm ?? null,
        name: res.water_body?.name ?? null });
      setRunId((n) => n + 1);
      showResult(res, true);
      scrollToPipeline();
    } catch (e) {
      setError(`Could not open that analysis (${(e as Error).message}).`);
    }
  }, [showResult]);

  const choose = useCallback((t: string) => {
    setSelected(t);
    setLayerId(t);
  }, []);

  const layerUrl = useMemo(() => {
    const urls = Object.fromEntries((result?.layers ?? []).map((l) => [l.id, apiUrl(l.url)]));
    return (id: string) => urls[id];
  }, [result]);

  const started = running || events.length > 0 || result || error;

  return (
    <div className="mx-auto w-full max-w-[1240px] px-4 sm:px-10">
      <section className="grid gap-10 border-b border-hairline pb-12 pt-10 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:gap-14 lg:pt-14">
        <div className="anim-rise">
          <p className="kicker mb-3.5">From orbit to estimate</p>
          <h1 className="font-serif text-[40px] font-medium leading-[1.02] tracking-[-0.02em] text-balance sm:text-[56px]">
            What is in the water? Read it from the light it reflects.
          </h1>
          <p className="mt-5 max-w-[34em] font-serif text-[19px] leading-[1.45] text-ink-2 text-pretty sm:text-[21px]">
            Pick any lake, reservoir or river. AquaDoc fetches the latest clear Sentinel-2 image, removes clouds, shadows and
            shoreline, and estimates algae, sediment, clarity and dissolved organic matter — then shows exactly how it got there.
          </p>
          <div className="mt-8 flex flex-wrap gap-8 border-t border-hairline pt-4">
            {[["10", "spectral bands read"], ["5", "water-quality parameters"], ["7", "processing stages, shown live"]].map(([v, k]) => (
              <div key={k}>
                <b className="block font-serif text-[26px] font-medium">{v}</b>
                <span className="text-xs text-muted">{k}</span>
              </div>
            ))}
          </div>
          <ol className="mt-8 space-y-2 text-sm text-ink-2">
            <li><span className="secnum mr-2">A</span>Choose a featured lake, search by name, type coordinates or drop a pin.</li>
            <li><span className="secnum mr-2">B</span>Watch each stage run on the real image, and select any stage to see why.</li>
            <li><span className="secnum mr-2">C</span>Read the estimates with their ranges, maps and the bands behind them.</li>
          </ol>
        </div>
        <LocationChooser target={target} onTarget={(t) => setTarget(t)} onWindow={setDates}
          recent={recent} onOpenRecent={openRecent} onRun={run} running={running} />
      </section>

      <NearbyLakes onAnalyse={(t) => run(t)} onOpen={openRecent} running={running} />

      <div ref={pipelineRef} className="scroll-mt-4 border-b border-hairline py-12">
        {started ? (
          <Pipeline key={runId} events={events} running={running || replaying} layerUrl={layerUrl} replay={mode === "replay"}
            onReplay={result?.timeline?.length ? () => { setMode("replay"); replay(result.timeline); } : undefined} />
        ) : (
          <Pipeline events={[]} running={false} />
        )}
        {error && (
          <p role="alert" className="anim-drop mt-6 border border-[var(--critical)] bg-[color-mix(in_oklch,var(--critical)_6%,var(--page))] p-4 text-sm">
            <b className="font-semibold">Analysis stopped. </b>{error}
          </p>
        )}
        {running && (
          <button type="button" onClick={() => abort.current?.abort()} className="mt-4 font-mono text-xs text-ink-2 underline">
            Stop this analysis
          </button>
        )}
      </div>

      {result && !replaying && (
        <div className="anim-rise">
          <ResultPanels result={result} selected={selected} onSelect={choose} layerId={layerId} onLayerChange={setLayerId}
            placeName={target?.name} />
        </div>
      )}
    </div>
  );
}
