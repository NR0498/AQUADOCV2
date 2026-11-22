"use client";

import { useEffect, useState } from "react";
import { getJSON, type Guide } from "@/lib/api";

let cached: Promise<Guide> | null = null;

/** Band/parameter reference and model metrics from /api/guide, fetched once per page load. */
export function useGuide() {
  const [guide, setGuide] = useState<Guide | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    cached ??= getJSON<Guide>("/api/guide");
    cached.then(setGuide).catch((e: Error) => {
      cached = null;
      setError(e.message);
    });
  }, []);
  return { guide, error };
}
