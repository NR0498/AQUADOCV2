"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api";

type Status = "checking" | "online" | "offline";

/** Polls /api/health. When the API is down, explains how to start it instead of failing silently. */
export default function ApiStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const response = await fetch(apiUrl("/api/health"), { cache: "no-store" });
        if (!cancelled) setStatus(response.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };
    check();
    const timer = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const colour = status === "online" ? "bg-[var(--good)]" : status === "offline" ? "bg-[var(--critical)]" : "bg-hairline";
  return (
    <>
      <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink-2" role="status">
        <i className={`inline-block h-[7px] w-[7px] rounded-full ${colour}`} aria-hidden />
        {status === "online" ? "API online" : status === "offline" ? "API offline" : "Checking API…"}
      </span>
      {status === "offline" && (
        <div role="alert" className="anim-drop w-full border border-[var(--critical)] bg-[color-mix(in_oklch,var(--critical)_6%,var(--page))] p-4 text-sm">
          <p className="font-semibold">The analysis service is not running, so analyses, search and reference pages cannot load.</p>
          <p className="mt-1 text-ink-2">
            From the <code className="font-mono">web</code> folder run <code className="font-mono">npm run dev:all</code>, or start
            the API yourself from the repository root:
          </p>
          <pre className="mt-2 overflow-x-auto bg-surface-2 p-2 font-mono text-xs">.venv\Scripts\python -m uvicorn aquadoc_core.api:app --port 8000</pre>
          <p className="mt-1 text-xs text-muted">This message disappears automatically once the API responds.</p>
        </div>
      )}
    </>
  );
}
