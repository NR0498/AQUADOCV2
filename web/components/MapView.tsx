"use client";

import { useEffect, useRef, useState } from "react";
import type { GeoJSONSource, Map as MapLibreMap, Marker } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";
const IMAGERY_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

export type Overlay = { url: string; corners: [number, number][]; opacity: number } | null;

type Props = {
  point: { latitude: number; longitude: number } | null;
  waterBody: GeoJSON.Geometry | null;
  overlay: Overlay;
  fitTo: [number, number][] | null; // corners to zoom to (lon, lat)
  flyTo: { bbox?: [number, number, number, number]; center?: [number, number] } | null;
  onPick: (latitude: number, longitude: number) => void;
  interactive?: boolean;
};

export default function MapView({ point, waterBody, overlay, fitTo, flyTo, onPick, interactive = true }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markerRef = useRef<Marker | null>(null);
  const markerClass = useRef<typeof Marker | null>(null);
  const pickRef = useRef(onPick);
  const [ready, setReady] = useState(false);
  const [satellite, setSatellite] = useState(true);

  useEffect(() => {
    pickRef.current = onPick; // latest callback for the map click handler registered once
  }, [onPick]);

  useEffect(() => {
    let cancelled = false;
    let map: MapLibreMap | undefined;
    (async () => {
      const maplibre = await import("maplibre-gl"); // v6: ESM named exports
      maplibre.setWorkerUrl("/maplibre-gl-worker.mjs"); // copied to public/ by scripts/copy-maplibre-worker.mjs
      if (cancelled || !container.current) return;
      markerClass.current = maplibre.Marker;
      map = new maplibre.Map({
        container: container.current,
        style: STYLE_URL,
        center: [78.47, 17.42],
        zoom: 2.2,
        attributionControl: { compact: true },
      });
      map.addControl(new maplibre.NavigationControl({ showCompass: false }), "top-right");
      map.on("load", () => {
        if (!map) return;
        map.addSource("imagery", {
          type: "raster",
          tiles: [IMAGERY_TILES],
          tileSize: 256,
          maxzoom: 18,
          attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
        });
        // Insert imagery under the first symbol layer so place names stay readable.
        const firstSymbol = map.getStyle().layers?.find((layer) => layer.type === "symbol")?.id;
        map.addLayer({ id: "imagery", type: "raster", source: "imagery" }, firstSymbol);
        map.addSource("water-body", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        map.addLayer({
          id: "water-body-line",
          type: "line",
          source: "water-body",
          paint: { "line-color": "#ffffff", "line-width": 2, "line-opacity": 0.9 },
        });
        mapRef.current = map;
        setReady(true);
      });
      map.on("click", (event) => {
        if (interactive) pickRef.current(event.lngLat.lat, event.lngLat.lng);
      });
      map.getCanvas().style.cursor = interactive ? "crosshair" : "";
    })();
    return () => {
      cancelled = true;
      map?.remove();
      mapRef.current = null;
    };
  }, [interactive]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    map.setLayoutProperty("imagery", "visibility", satellite ? "visible" : "none");
  }, [ready, satellite]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !markerClass.current) return;
    markerRef.current?.remove();
    markerRef.current = null;
    if (point) {
      markerRef.current = new markerClass.current({ color: "#2a78d6" })
        .setLngLat([point.longitude, point.latitude])
        .addTo(map);
    }
  }, [ready, point]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const source = map.getSource("water-body") as GeoJSONSource | undefined;
    source?.setData(
      waterBody
        ? { type: "Feature", geometry: waterBody, properties: {} }
        : { type: "FeatureCollection", features: [] },
    );
  }, [ready, waterBody]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    if (map.getLayer("overlay")) map.removeLayer("overlay");
    if (map.getSource("overlay")) map.removeSource("overlay");
    if (!overlay) return;
    map.addSource("overlay", {
      type: "image",
      url: overlay.url,
      coordinates: overlay.corners.slice(0, 4) as [[number, number], [number, number], [number, number], [number, number]],
    });
    map.addLayer(
      { id: "overlay", type: "raster", source: "overlay", paint: { "raster-opacity": overlay.opacity, "raster-resampling": "nearest" } },
      "water-body-line",
    );
  }, [ready, overlay?.url, overlay?.corners]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !overlay || !map.getLayer("overlay")) return;
    map.setPaintProperty("overlay", "raster-opacity", overlay.opacity);
  }, [ready, overlay?.opacity]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !fitTo?.length) return;
    const lons = fitTo.map((c) => c[0]);
    const lats = fitTo.map((c) => c[1]);
    map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: 40, duration: 800 });
  }, [ready, fitTo]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !flyTo) return;
    if (flyTo.bbox) {
      const [west, south, east, north] = flyTo.bbox;
      map.fitBounds([[west, south], [east, north]], { padding: 40, duration: 900, maxZoom: 14 });
    } else if (flyTo.center) {
      map.flyTo({ center: flyTo.center, zoom: 12 });
    }
  }, [ready, flyTo]);

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div ref={container} className="h-full w-full" aria-label="Map. Click on water to choose a location." />
      <button
        type="button"
        onClick={() => setSatellite((s) => !s)}
        className="absolute left-3 top-3 rounded-md bg-surface/95 px-3 py-1.5 text-xs font-medium text-ink shadow-sm ring-1 ring-black/10"
      >
        {satellite ? "Show street map" : "Show satellite"}
      </button>
    </div>
  );
}
