"use client";

import { useState } from "react";
import type { Sensitivity } from "@/lib/api";

/** Share of the estimate's sensitivity carried by each band at this location.
 *  Bars are one series (no legend); the direction of influence is stated in text. */
export default function SensitivityBars({ items, label }: { items: Sensitivity[]; label: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const max = Math.max(...items.map((i) => i.share_pct), 1);
  return (
    <div>
      <ul className="space-y-2" aria-label={`Band influence on ${label}`}>
        {items.map((item) => (
          <li key={item.band} className="grid grid-cols-[2.5rem_1fr_9.5rem] items-center gap-2 text-xs"
            onMouseEnter={() => setHover(item.band)} onMouseLeave={() => setHover(null)}>
            <span className="font-medium">{item.band}</span>
            <span className="relative h-4">
              <span className="absolute left-0 top-0 h-4 rounded-r-[4px] bg-[var(--series-1)]"
                style={{ width: `${(item.share_pct / max) * 100}%`, opacity: hover && hover !== item.band ? 0.45 : 1 }} />
            </span>
            <span className="tabular text-ink-2">
              {item.share_pct.toFixed(0)}% · {item.elasticity >= 0 ? "brighter → higher" : "brighter → lower"}
            </span>
          </li>
        ))}
      </ul>
      {hover && (() => {
        const item = items.find((i) => i.band === hover)!;
        return (
          <p className="mt-2 text-xs text-ink-2" aria-live="polite">
            A 10% brighter {item.band} changes the {label.toLowerCase()} estimate by about{" "}
            {(item.elasticity * 10).toFixed(0)}% here.
          </p>
        );
      })()}
    </div>
  );
}
