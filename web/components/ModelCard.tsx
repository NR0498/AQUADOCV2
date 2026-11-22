"use client";

import { useGuide } from "@/lib/useGuide";
import { fmt, pct, unitLabel } from "@/lib/format";
import { TARGET_ORDER } from "@/lib/api";

export default function ModelCard() {
  const { guide, error } = useGuide();
  if (error) return <p className="mx-auto max-w-[1240px] px-4 py-16 text-sm sm:px-10">Model information could not be loaded ({error}).</p>;
  if (!guide) return <p className="anim-pulse mx-auto max-w-[1240px] px-4 py-16 font-mono text-xs text-ink-2 sm:px-10">Loading…</p>;
  const { model, parameters } = guide;
  const correction = model.satellite_correction;

  return (
    <div className="mx-auto w-full max-w-[1240px] space-y-12 px-4 pb-6 pt-10 sm:px-10 sm:pt-14">
      <header className="border-b border-hairline pb-10">
        <p className="kicker mb-3">Methods &amp; accuracy</p>
        <h1 className="anim-rise font-serif text-[40px] font-medium leading-[1.04] tracking-[-0.02em] text-balance sm:text-[52px]">Model &amp; accuracy</h1>
        <p className="mt-4 max-w-[40em] font-serif text-[19px] leading-[1.5] text-ink-2 text-pretty">
          A small neural-network ensemble per parameter, trained on {model.training_data}. Each in-situ spectrum was
          converted into what Sentinel-2 would see. Accuracy was then measured on real satellite pixels at sites the
          model never saw.
        </p>
      </header>

      <section className="overflow-x-auto border-t border-ink pt-6">
        <h2 className="font-serif text-[26px] font-medium leading-tight">Typical error on real Sentinel-2 images</h2>
        <p className="mt-1 text-sm text-ink-2">
          Median symmetric accuracy (lower is better; 100% means estimates are typically off by a factor of two).
          {correction && ` Measured on ${correction.matchups} satellite match-ups at ${correction.sites} sites.`}
        </p>
        <table className="mt-4 w-full min-w-[640px] text-sm tabular">
          <thead>
            <tr className="text-left text-xs text-muted">
              <th className="py-2">Parameter</th>
              <th className="text-right">Ideal input (field spectra)</th>
              <th className="text-right">Satellite, raw</th>
              <th className="text-right">Satellite, corrected</th>
              <th className="text-right">Published algorithm</th>
              <th className="text-right">80% range coverage</th>
              <th className="text-right">Training range</th>
            </tr>
          </thead>
          <tbody>
            {TARGET_ORDER.filter((t) => model.targets[t]).map((t) => {
              const m = model.targets[t];
              const v = m.satellite_validation;
              const used = correction?.targets_using_correction.includes(t) ? "corrected" : "raw";
              const strong = (which: string) => (which === used ? "font-semibold" : "text-ink-2");
              return (
                <tr key={t} className="border-t border-hairline">
                  <td className="py-2">{parameters[t]?.label ?? t}</td>
                  <td className="text-right text-ink-2">{v ? pct(v.in_situ_rrs.mdsa_pct) : pct(m.test_mdsa_pct)}</td>
                  <td className={`text-right ${strong("raw")}`}>{v ? pct(v.satellite_raw.mdsa_pct) : "–"}</td>
                  <td className={`text-right ${strong("corrected")}`}>{v ? pct(v.satellite_corrected.mdsa_pct) : "–"}</td>
                  <td className="text-right text-ink-2">{v?.published_algorithm_on_corrected_satellite ? pct(v.published_algorithm_on_corrected_satellite.mdsa_pct) : "–"}</td>
                  <td className="text-right text-ink-2">{v ? pct((used === "corrected" ? v.interval_coverage_cv_corrected : v.interval_coverage_cv_raw) * 100) : pct(m.interval_coverage * 100)}</td>
                  <td className="text-right text-ink-2">{fmt(m.training_range[0])}–{fmt(m.training_range[1])} {unitLabel(m.unit)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-muted">Bold marks the input the dashboard uses for each parameter.</p>
      </section>

      <div className="grid gap-10 md:grid-cols-2">
        <section className="border-t border-ink pt-6 text-sm">
          <h2 className="font-serif text-[26px] font-medium leading-tight">How confidence is assigned</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-ink-2">
            <li><span className="font-medium text-ink">High</span>: validated satellite error ≤ 40%, nearly all pixels inside the training range, and at least 100 clear pixels.</li>
            <li><span className="font-medium text-ink">Medium</span>: validated error ≤ 100% and the inputs look like the training data.</li>
            <li><span className="font-medium text-ink">Low</span>: error above 100% (for example CDOM), unusual spectra, or too few clear pixels.</li>
          </ul>
        </section>
        <section className="border-t border-ink pt-6 text-sm">
          <h2 className="font-serif text-[26px] font-medium leading-tight">Known limitations</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-ink-2">
            <li>Validation is dominated by lakes; accuracy for rivers and estuaries is largely unknown.</li>
            <li>Sentinel-2&apos;s atmospheric correction is designed for land. A learned correction removes most of its offset over water, but not all.</li>
            <li>Shoreline pixels, shallow clear water and thin haze can bias results; the pipeline masks what it can and reports it.</li>
            <li>No estimate here replaces laboratory testing or a drinking-water assessment.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
