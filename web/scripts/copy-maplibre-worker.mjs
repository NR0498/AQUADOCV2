// MapLibre GL v6 loads its tile worker from a separate module file, resolved next to its
// own bundle. Bundlers rename chunks, so serve the worker from /public and point MapLibre
// at it (see components/MapView.tsx). Runs on every install, so it tracks the package version.
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const dist = dirname(require.resolve("maplibre-gl/dist/maplibre-gl.mjs"));
mkdirSync("public", { recursive: true });
copyFileSync(join(dist, "maplibre-gl-worker.mjs"), join("public", "maplibre-gl-worker.mjs"));
console.log("copied maplibre-gl-worker.mjs to public/");
