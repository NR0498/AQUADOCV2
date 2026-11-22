// Typed client for the AquaDoc FastAPI backend.

// Same origin everywhere: in development next.config.ts proxies /api to the Python server;
// in production both are served from one domain. NEXT_PUBLIC_API_BASE overrides for special setups.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export const apiUrl = (path: string) => `${API_BASE}${path}`;

export type Interval = { level: number; low: number | null; high: number | null };

export type Estimate = {
  label: string;
  what: string;
  bands: string[];
  unit: string;
  kind: string;
  value: number | null;
  spatial_p10: number | null;
  spatial_p90: number | null;
  interval: Interval;
  in_training_domain_pct: number;
  confidence: "low" | "medium" | "high";
  model_test_mdsa_pct: number;
  satellite_validated_mdsa_pct: number | null;
  reflectance_input: "raw" | "corrected";
  interval_basis: string;
  trophic_state?: string | null;
};

export type SpectrumPoint = {
  band: string;
  name: string;
  center_nm: number;
  reflectance_median: number | null;
  reflectance_p25: number | null;
  reflectance_p75: number | null;
};

export type Sensitivity = { band: string; elasticity: number; share_pct: number };

export type Layer = {
  id: string;
  label: string;
  url: string;
  unit?: string;
  colors?: string[];
  scale?: "log10" | "linear";
  vmin?: number | null;
  vmax?: number | null;
  legend?: Record<string, string>;
};

export type PixelCounts = {
  grid_pixels: number;
  water_area_pixels: number;
  no_data: number;
  cloud: number;
  cloud_shadow: number;
  snow_ice: number;
  saturated_or_defective: number;
  cloud_adjacent: number;
  floating_vegetation_or_scum: number;
  not_water_like: number;
  glint_or_mixed: number;
  valid_open_water: number;
  clear_fraction: number;
};

export type SceneSummary = {
  id: string;
  datetime: string;
  cloud_cover_pct: number | null;
  platform: string | null;
  aoi_coverage: number;
  processing_baseline?: string;
  mgrs_tile?: string;
};

export type AnalysisResult = {
  id: string;
  created_at: string;
  cached: boolean;
  request: { latitude: number; longitude: number; start_date: string; end_date: string; max_size_m: number };
  water_body: {
    name: string | null;
    osm: string;
    water_type: string | null;
    contains_point: boolean;
    geometry: GeoJSON.Geometry | null;
    source: string;
  } | null;
  grid: { corners_wgs84: [number, number][]; width: number; height: number; resolution_m: number; extent: string; shoreline_buffer_m: number | null };
  scene: SceneSummary;
  attempts: (SceneSummary & { outcome: string; pixels?: PixelCounts })[];
  pixels: PixelCounts;
  estimates: Record<string, Estimate>;
  floating_material: {
    label: string;
    what: string;
    bands: string[];
    vegetation_or_scum_pct: number;
    debris_like_pct: number;
    fdi_water_median: number | null;
    fdi_anomaly_threshold: number | null;
  };
  published_algorithms: Record<string, { median: number | null; p10: number | null; p90: number | null }>;
  indices: Record<string, number | null>;
  spectrum: SpectrumPoint[];
  band_sensitivity: Record<string, Sensitivity[]>;
  layers: Layer[];
  reflectance_correction?: { bands: ReflectanceBand[]; targets_using_correction: string[] };
  timeline?: StageEvent[];
  not_detectable: { name: string; why: string }[];
  model: { version: string; satellite_correction: string | null };
  notes: string[];
  disclaimer: string;
  elapsed_ms: number;
};

export type StageEvent = {
  stage: string;
  status: "started" | "completed" | "rejected" | "skipped" | "failed";
  message: string;
  // Live events carry `previews` (PNG data URLs); stored timelines carry `preview_layers` (layer ids).
  data: Record<string, unknown> & { previews?: Record<string, string>; preview_layers?: string[] };
  elapsed_ms: number;
};

export type ReflectanceBand = { band: string; center_nm: number; rrs_l2a: number | null; rrs_corrected: number | null };

export type SearchResult = {
  name: string;
  display_name: string;
  latitude: number;
  longitude: number;
  type: string;
  osm: string;
  bbox: number[]; // [south, north, west, east] as returned by Nominatim
};

export type AnalyzeBody = {
  latitude: number;
  longitude: number;
  osm_ref?: string | null;
  start_date?: string | null;
  end_date?: string | null;
};

export type Guide = {
  bands: { band: string; name: string; center_nm: number; resolution_m: number; water: string; indicates: string[] }[];
  parameters: Record<string, { label: string; unit: string; kind: string; what: string; bands: string[]; colormap: string;
    scale?: { max: number | null; label: string }[]; scale_source?: string }>;
  not_detectable: { name: string; why: string }[];
  model: {
    version: string;
    training_data: string;
    targets: Record<string, {
      unit: string; deployed: boolean; test_mdsa_pct: number; test_mdsa_with_noise_pct: number;
      published_algorithm?: { name: string; mdsa_pct: number } | null; interval_coverage: number;
      training_range: [number, number];
      satellite_validation?: {
        n: number; sites: number;
        in_situ_rrs: { mdsa_pct: number }; satellite_raw: { mdsa_pct: number }; satellite_corrected: { mdsa_pct: number };
        published_algorithm_on_corrected_satellite?: { mdsa_pct: number };
        interval_coverage_cv_raw: number; interval_coverage_cv_corrected: number;
      } | null;
    }>;
    satellite_correction?: { matchups: number; sites: number; targets_using_correction: string[];
      coefficients: Record<string, { intercept: number; gain: number; swir_b11: number }> } | null;
  };
};

export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

type StreamHandlers = {
  onStage: (event: StageEvent) => void;
  onComplete: (result: AnalysisResult) => void;
  onError: (message: string) => void;
};

/** POST an analysis and consume the Server-Sent Events stream.
 *  (EventSource only supports GET, so the stream is read with fetch + a text decoder.) */
export async function streamAnalysis(body: AnalyzeBody, handlers: StreamHandlers, signal?: AbortSignal) {
  const response = await fetch(apiUrl("/api/analyses/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const json = await response.json();
      if (json?.detail) detail = typeof json.detail === "string" ? json.detail : JSON.stringify(json.detail);
    } catch {}
    handlers.onError(detail);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let kind = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue; // keep-alive comment
        if (line.startsWith("event: ")) kind = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (!data.length) continue;
      const payload = JSON.parse(data.join("\n"));
      if (kind === "stage") handlers.onStage(payload);
      else if (kind === "complete") {
        finished = true;
        handlers.onComplete(payload);
      } else if (kind === "error") {
        finished = true;
        handlers.onError(payload.message ?? "Analysis failed");
      }
    }
  }
  if (!finished) handlers.onError("The connection closed before the analysis finished.");
}

export const TARGET_ORDER = ["chlorophyll_a", "tss", "turbidity", "secchi_depth", "cdom_a440"] as const;

export type NearbyLake = {
  name: string;
  osm: string;
  type: string | null;
  latitude: number;
  longitude: number;
  bbox: [number, number, number, number]; // south, north, west, east
  distance_km: number;
  bearing_deg: number;
  extent_km: number;
  latest_analysis: {
    id: string;
    scene_datetime: string | null;
    created_at: string | null;
    chlorophyll_a: number | null;
    turbidity: number | null;
    secchi_depth: number | null;
    trophic_state: string | null;
    clear_fraction: number | null;
  } | null;
};

export type NearbyResponse = { origin: { latitude: number; longitude: number }; radius_km: number; lakes: NearbyLake[] };
