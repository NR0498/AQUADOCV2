"use client";

import { useState } from "react";
import { useGuide } from "@/lib/useGuide";
import { unitLabel } from "@/lib/format";

// Wavelength axis: 400–2300 nm, with the visible range (400–700 nm) shown as a colour strip.
const MIN_NM = 400;
const MAX_NM = 2300;
const pos = (nm: number) => ((Math.log(nm) - Math.log(MIN_NM)) / (Math.log(MAX_NM) - Math.log(MIN_NM))) * 100;
const VISIBLE = "linear-gradient(to right, #7a3cc9, #3a5ee0, #1aa6d6, #2fb24c, #d8d22c, #f0902a, #d9352a)";

export default function BandGuide() {
  const { guide, error } = useGuide();
  const [active, setActive] = useState("B5");
  if (error) return <p className="mx-auto max-w-[1240px] px-4 py-16 text-sm sm:px-10">The band guide could not be loaded ({error}).</p>;
  if (!guide) return <p className="anim-pulse mx-auto max-w-[1240px] px-4 py-16 font-mono text-xs text-ink-2 sm:px-10">Loading band guide…</p>;
  const band = guide.bands.find((b) => b.band === active) ?? guide.bands[0];
  // Labels closer than ~4% of the axis would overlap: alternate them onto a second row.
  const rowOf: Record<string, number> = {};
  guide.bands.forEach((b, i) => {
    const prev = guide.bands[i - 1];
    rowOf[b.band] = prev && pos(b.center_nm) - pos(prev.center_nm) < 4 && rowOf[prev.band] === 0 ? 1 : 0;
  });
  const params = guide.parameters;

  return (
    <div className="mx-auto w-full max-w-[1240px] space-y-12 px-4 pb-6 pt-10 sm:px-10 sm:pt-14">
      <header className="border-b border-hairline pb-10">
        <p className="kicker mb-3">Reference · Sentinel-2 bands</p>
        <h1 className="anim-rise font-serif text-[40px] font-medium leading-[1.04] tracking-[-0.02em] text-balance sm:text-[52px]">How Sentinel-2 sees water</h1>
        <p className="mt-4 max-w-[40em] font-serif text-[19px] leading-[1.5] text-ink-2 text-pretty">
          Sentinel-2 measures reflected sunlight in 13 bands. Water absorbs strongly beyond red light, so each substance
          in water leaves a fingerprint: algae absorb red and reflect red-edge light, sediment brightens red and
          near-infrared, dissolved organic matter darkens blue. AquaDoc uses ten bands at 20 m resolution.
        </p>
      </header>

      <section className="border-t border-ink pt-6">
        <h2 className="font-serif text-[26px] font-medium leading-tight">The bands on the light spectrum</h2>
        <p className="text-sm text-ink-2">Select a band to see what it tells us about water. Log wavelength scale.</p>
        <div className="relative mt-6 h-28">
          <div className="absolute top-14 h-3 rounded-sm" style={{ left: 0, width: `${pos(700)}%`, background: VISIBLE }} />
          <div className="absolute top-14 h-3 rounded-sm bg-hairline" style={{ left: `${pos(700)}%`, right: 0 }} />
          <span className="absolute top-[4.6rem] text-[11px] text-muted" style={{ left: 0 }}>Visible</span>
          <span className="absolute top-[4.6rem] text-[11px] text-muted" style={{ left: `${pos(760)}%` }}>Near-infrared</span>
          <span className="absolute top-[4.6rem] text-[11px] text-muted" style={{ left: `${pos(1500)}%` }}>Shortwave infrared</span>
          {guide.bands.map((b) => (
            <div key={b.band}>
              <span aria-hidden className="absolute w-px bg-axis" style={{ left: `${pos(b.center_nm)}%`, top: rowOf[b.band] ? "2.6rem" : "1.4rem", height: rowOf[b.band] ? "0.9rem" : "2.1rem", background: "var(--axis)" }} />
              <button type="button" onClick={() => setActive(b.band)} aria-pressed={b.band === active}
                className={`absolute -translate-x-1/2 px-1.5 py-0.5 font-mono text-[11px] font-medium transition-colors ${b.band === active ? "bg-ink text-page" : "bg-surface-2 text-ink-2 hover:text-ink"}`}
                style={{ left: `${pos(b.center_nm)}%`, top: rowOf[b.band] ? "1.3rem" : 0 }}>
                {b.band}
              </button>
            </div>
          ))}
          {[400, 500, 700, 1000, 1500, 2200].map((nm) => (
            <span key={nm} className="absolute top-[5.8rem] -translate-x-1/2 text-[10px] text-muted tabular" style={{ left: `${pos(nm)}%` }}>{nm} nm</span>
          ))}
        </div>
        <div className="mt-4 anim-rise border-l-2 border-accent bg-surface-2 p-4">
          <p className="font-serif text-xl font-medium">{band.band} · {band.name} · {band.center_nm} nm · {band.resolution_m} m pixels</p>
          <p className="mt-1 text-sm leading-relaxed text-ink-2">{band.water}</p>
          <p className="mt-2 text-sm">
            <span className="text-muted">Helps estimate: </span>
            {band.indicates.map((p) => params[p]?.label ?? p.replaceAll("_", " ")).join(", ")}
          </p>
        </div>
      </section>

      <section>
        <h2 className="mb-5 font-serif text-[26px] font-medium leading-tight">What AquaDoc estimates, and from which bands</h2>
        <div className="grid gap-x-10 sm:grid-cols-2">
          {Object.entries(params).map(([key, p]) => (
            <article key={key} className="border-t border-hairline py-4">
              <h3 className="font-serif text-xl font-medium">{p.label}</h3>
              <p className="text-xs text-muted">{unitLabel(p.unit)} · {p.kind === "model" ? "machine-learning estimate" : "spectral index"}</p>
              <p className="mt-2 text-sm text-ink-2">{p.what}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {p.bands.map((b) => (
                  <button key={b} type="button" onClick={() => setActive(b)}
                    className="border border-hairline px-2 py-0.5 font-mono text-xs hover:border-ink">{b}</button>
                ))}
              </div>
              {p.scale && (
                <p className="mt-3 text-xs text-muted">
                  Scale: {p.scale.map((s) => `${s.label}${s.max ? ` < ${s.max}` : ""}`).join(" · ")} µg/L ({p.scale_source})
                </p>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="border-t border-ink pt-6">
        <h2 className="font-serif text-[26px] font-medium leading-tight">What no satellite can measure</h2>
        <ul className="mt-3 space-y-2 text-sm">
          {guide.not_detectable.map((item) => (
            <li key={item.name}><span className="font-medium">{item.name}.</span> <span className="text-ink-2">{item.why}</span></li>
          ))}
        </ul>
      </section>
    </div>
  );
}
