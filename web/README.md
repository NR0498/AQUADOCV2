# AquaDoc dashboard

Next.js 16 (App Router, Cache Components) + TypeScript + Tailwind CSS 4 + MapLibre GL 6.

## Run locally

Start the API from the repository root, then the dashboard:

```powershell
# terminal 1 (repo root)
python -m uvicorn aquadoc_core.api:app --port 8000

# terminal 2
cd web
npm install          # also copies the MapLibre worker into public/
npm run dev          # http://localhost:3000  (use localhost, not 127.0.0.1)
```

`.env.development` points the dashboard at `http://127.0.0.1:8000`. In production the dashboard and API share one origin, so `NEXT_PUBLIC_API_BASE` stays unset and requests go to `/api/...`.

## Structure

| Path | Purpose |
| --- | --- |
| `app/page.tsx` | Analyse: search or click, live processing timeline, map layers, results |
| `app/analyses/[id]/page.tsx` | Shareable stored result (params read inside `<Suspense>`) |
| `app/bands/page.tsx` | Band guide: what each Sentinel-2 band reveals about water |
| `app/model/page.tsx` | Model accuracy on real satellite match-ups |
| `lib/api.ts` | Typed API client and Server-Sent Events reader (fetch + stream; EventSource cannot POST) |
| `components/` | Map, timeline, estimate cards, spectrum chart, band sensitivity, pixel breakdown |

## Notes

- MapLibre 6 loads its tile worker from a separate module. `scripts/copy-maplibre-worker.mjs` (run on `postinstall`) copies it to `public/`, and `MapView` calls `setWorkerUrl`.
- Basemaps: OpenFreeMap vector tiles (no key) and Esri World Imagery (attribution shown).
- Chart and pixel-class colours were checked with a colour-vision-deficiency validator. Every chart has hover details and a table or text alternative.
