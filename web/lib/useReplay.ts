"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { StageEvent } from "@/lib/api";

/** Replays a stored pipeline timeline so a saved or cached result animates like a live run.
 *  Real timing is compressed to about four seconds, with a minimum gap so each stage is visible. */
export function useReplay() {
  const [events, setEvents] = useState<StageEvent[]>([]);
  const [replaying, setReplaying] = useState(false);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const stop = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setReplaying(false);
  }, []);

  const replay = useCallback((timeline: StageEvent[] | undefined) => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    const items = (timeline ?? []).filter((e) => e.stage !== "complete");
    setEvents([]);
    if (!items.length) {
      setReplaying(false);
      return;
    }
    const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setEvents(items);
      setReplaying(false);
      return;
    }
    setReplaying(true);
    const total = Math.max(items[items.length - 1].elapsed_ms, 1);
    let at = 0;
    items.forEach((event, i) => {
      const scaled = (event.elapsed_ms / total) * 4000;
      at = Math.max(at + 180, scaled);
      timers.current.push(setTimeout(() => {
        setEvents((list) => [...list, event]);
        if (i === items.length - 1) setReplaying(false);
      }, at));
    });
  }, []);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  return { events, replaying, replay, stop, setEvents };
}
