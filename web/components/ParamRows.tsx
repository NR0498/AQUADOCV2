"use client";

import type { Estimate } from "@/lib/api";
import { fmt, pct, unitLabel } from "@/lib/format";

const CONFIDENCE: Record<Estimate["confidence"], { dots: number; text: string }> = {
  high: { dots: 3, text: "High confidence" },
  medium: { dots: 2, text: "Medium confidence" },
  low: { dots: 1, text: "Low confidence" },
};

/** Log-scale strip: 80% interval (wash between ticks) and median (dot). */
function RangeStrip({ estimate }: { estimate: Estimate }) {
  const { interval, value } = estimate;
  if (!value || !interval.low || !interval.high || interval.low <= 0) return <span className="flex-1" />;
  const lo = Math.log10(interval.low / 2);
  const hi = Math.log10(interval.high * 2);
  const x = (v: number) => ((Math.log10(v) - lo) / (hi - lo)) * 100;
  return (
    <span className="relative h-2.5 min-w-[60px] flex-1" role="img"
      aria-label={`${Math.round(interval.level * 100)}% range ${fmt(interval.low)} to ${fmt(interval.high)}, estimate ${fmt(value)}`}>
      <i className="absolute inset-x-0 top-1 h-0.5 bg-hairline" />
      <u className="absolute top-px h-2 border-x border-accent bg-accent-wash transition-all duration-700"
        style={{ left: `${x(interval.low)}%`, width: `${x(interval.high) - x(interval.low)}%` }} />
      <s className="absolute top-0 -ml-[5px] h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_0_2px_var(--page)] transition-all duration-700"
        style={{ left: `${x(value)}%` }} />
    </span>
  );
}

type Props = { estimates: [string, Estimate][]; selected: string; onSelect: (id: string) => void };

export default function ParamRows({ estimates, selected, onSelect }: Props) {
  return (
    <div className="flex flex-col" role="radiogroup" aria-label="Water-quality parameters">
      {estimates.map(([id, e], i) => {
        const conf = CONFIDENCE[e.confidence];
        const on = id === selected;
        return (
          <button key={id} type="button" role="radio" aria-checked={on} onClick={() => onSelect(id)}
            style={{ animationDelay: `${i * 70}ms` }}
            className={`anim-rise grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 border-b border-hairline px-2 py-4 text-left first:border-t first:border-t-ink ${
              on ? "bg-[linear-gradient(90deg,var(--accent-wash),transparent_70%)]" : "hover:bg-surface-2"}`}>
            <span className="text-sm font-semibold">{e.label}</span>
            <span className="row-span-2 text-right">
              <b className="tabular font-serif text-[34px] font-medium tracking-tight">{fmt(e.value)}</b>
              <small className="ml-1 text-[13px] text-ink-2">{unitLabel(e.unit)}</small>
            </span>
            <span className="text-[12.5px] text-muted">
              {e.trophic_state ? `${e.what.split(".")[0]} — ${e.trophic_state}` : e.what.split(".")[0]}
            </span>
            <span className="col-span-2 mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-ink-2">
              <span className="inline-flex items-center gap-1.5" title={conf.text}>
                <span className="inline-flex gap-0.5" aria-hidden>
                  {[0, 1, 2].map((d) => <i key={d} className={`h-1.5 w-1.5 rounded-full ${d < conf.dots ? "bg-ink" : "bg-hairline"}`} />)}
                </span>
                {conf.text}
              </span>
              <RangeStrip estimate={e} />
              <span className="tabular">{fmt(e.interval.low)}–{fmt(e.interval.high)}</span>
              <span className="w-full text-muted sm:w-auto"
                title="Median symmetric accuracy on real Sentinel-2 match-ups; 100% means a typical factor-of-two error">
                ±{e.satellite_validated_mdsa_pct ? pct(e.satellite_validated_mdsa_pct) : "n/a"} typical error · {pct(e.in_training_domain_pct)} pixels in range
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
