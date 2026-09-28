import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { GeoJsonLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { TripsLayer } from "@deck.gl/geo-layers";
import { Ship, Suspect, Track, trackAt } from "./data";

const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const T_OFFSET = 48; // TripsLayer needs non-negative time: hours since T0-48 h

export type Layers = { backward: boolean; forecast: boolean; ais: boolean; ships: boolean; platforms: boolean; reference: boolean };

type Props = {
  events: any; selected: number | null; onSelectEvent: (id: number) => void;
  t: number; backward: any; forecast: any; forward: any;
  tracks: Track[]; suspects: Suspect[]; focus: Suspect | null; onSelectSuspect: (s: Suspect) => void;
  ships: Ship[]; platforms: any[]; reference: any; layers: Layers;
};

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (x: any) => String(x ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

const flat2 = (a: number[] | undefined) => {
  const out: [number, number][] = [];
  if (a) for (let i = 0; i < a.length; i += 2) out.push([a[i], a[i + 1]]);
  return out;
};

const shipColor = (l: string): [number, number, number] =>
  l === "dark_vessel_candidate" ? [244, 63, 94] : l === "ais_silent_at_t0" ? [245, 158, 11]
    : l === "platform_no_ais" ? [148, 163, 184] : l === "no_ais_coverage" ? [100, 116, 139] : [45, 212, 191];

export default function MapView(p: Props) {
  const div = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const overlay = useRef<MapboxOverlay | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; html: string } | null>(null);

  useEffect(() => {
    const m = new maplibregl.Map({ container: div.current!, style: STYLE, center: [-90.2, 27.4], zoom: 7.2, attributionControl: { compact: true } });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
    const o = new MapboxOverlay({ interleaved: false, layers: [] });
    m.addControl(o as any);
    map.current = m; overlay.current = o;
    return () => m.remove();
  }, []);

  // fly to the selected event
  useEffect(() => {
    if (!map.current || p.selected == null || !p.events) return;
    const f = p.events.features.find((x: any) => x.properties.cluster_id === p.selected);
    if (!f) return;
    const b = new maplibregl.LngLatBounds();
    const walk = (c: any) => (typeof c[0] === "number" ? b.extend(c as [number, number]) : c.forEach(walk));
    walk(f.geometry.coordinates);
    map.current.fitBounds(b, { padding: 180, maxZoom: 10.5, duration: 900 });
  }, [p.selected, p.events]);

  const suspectSet = useMemo(() => new Set(p.suspects.map((s) => s.MMSI)), [p.suspects]);

  const layers = useMemo(() => {
    const L: any[] = [];
    const t = p.t;
    if (p.layers.reference && p.reference)
      L.push(new GeoJsonLayer({ id: "reference", data: p.reference, filled: false, stroked: true, getLineColor: [250, 204, 21, 220],
        lineWidthMinPixels: 1.5, getDashArray: [4, 3] } as any));
    if (p.layers.platforms)
      L.push(new ScatterplotLayer({ id: "platforms", data: p.platforms, getPosition: (d: any) => [d.lon, d.lat],
        getRadius: (d: any) => (d.major ? 350 : 180), radiusMinPixels: 1.5, getFillColor: [120, 134, 160, 160], pickable: true }));
    L.push(new GeoJsonLayer({
      id: "events", data: p.events, pickable: true, stroked: true, filled: true,
      getFillColor: (f: any) => (f.properties.cluster_id === p.selected ? [226, 232, 240, 235] : f.properties.selected ? [148, 163, 184, 180] : [100, 116, 139, 110]),
      getLineColor: (f: any) => (f.properties.cluster_id === p.selected ? [45, 212, 191, 255] : [148, 163, 184, 120]),
      lineWidthMinPixels: 1, onClick: (i: any) => i.object && p.onSelectEvent(i.object.properties.cluster_id),
      updateTriggers: { getFillColor: p.selected, getLineColor: p.selected },
    }));
    // hindcast: backward particle cloud of the selected event at age tau = -t
    if (p.layers.backward && p.backward && t <= 0) {
      const tau = Math.min(48, Math.round(-t));
      const pts = flat2(p.backward[tau]);
      L.push(new ScatterplotLayer({ id: "backward", data: pts, getPosition: (d: any) => d, getRadius: 120, radiusMinPixels: 1.6,
        getFillColor: [96, 165, 250, 130] }));
    }
    // forecast: particles at lead t (3-hourly), surface solid, dispersed faint
    if (p.layers.forecast && p.forecast && t > 0) {
      const lead = Math.min(72, Math.round(t / 3) * 3);
      const fr = p.forecast[lead];
      if (fr) {
        const pts = flat2(fr.xy).map((xy, i) => ({ xy, s: fr.surface[i] }));
        L.push(new ScatterplotLayer({ id: "forecast", data: pts, getPosition: (d: any) => d.xy, getRadius: 130, radiusMinPixels: 1.6,
          getFillColor: (d: any) => (d.s ? [251, 146, 60, 200] : [251, 146, 60, 55]), updateTriggers: { getFillColor: lead } }));
      }
    }
    // forward virtual oil of the focused suspect (their hypothetical discharge, drifted to T0)
    if (p.focus && p.forward && p.forward[p.focus.MMSI] && Math.abs(t) <= 3)
      L.push(new ScatterplotLayer({ id: "virtual", data: flat2(p.forward[p.focus.MMSI]), getPosition: (d: any) => d, getRadius: 110,
        radiusMinPixels: 1.8, getFillColor: [167, 139, 250, 170] }));
    if (p.layers.ais && t <= 6) {
      L.push(new TripsLayer({
        id: "trips", data: p.tracks, getPath: (d: Track) => d.path, getTimestamps: (d: Track) => d.t.map((x) => x + T_OFFSET),
        getColor: (d: Track) => (p.focus && d.mmsi === p.focus.MMSI ? [167, 139, 250] : suspectSet.has(d.mmsi) ? [45, 212, 191] : [96, 116, 150]),
        opacity: 0.85, widthMinPixels: 1.4, getWidth: (d: Track) => (p.focus && d.mmsi === p.focus.MMSI ? 4 : suspectSet.has(d.mmsi) ? 2 : 1),
        currentTime: t + T_OFFSET, trailLength: 8, capRounded: true, jointRounded: true,
        updateTriggers: { getColor: [p.focus?.MMSI, suspectSet], getWidth: [p.focus?.MMSI, suspectSet] },
      } as any));
      const heads = p.tracks.map((d) => ({ d, xy: trackAt(d, t) })).filter((x) => x.xy);
      L.push(new ScatterplotLayer({
        id: "heads", data: heads, pickable: true, getPosition: (x: any) => x.xy, radiusUnits: "pixels",
        getRadius: (x: any) => (p.focus && x.d.mmsi === p.focus.MMSI ? 7 : suspectSet.has(x.d.mmsi) ? 4.5 : 2.5),
        getFillColor: (x: any) => (p.focus && x.d.mmsi === p.focus.MMSI ? [167, 139, 250] : suspectSet.has(x.d.mmsi) ? [45, 212, 191] : [120, 140, 170]),
        stroked: true, getLineColor: [7, 13, 24], lineWidthMinPixels: 1,
        onClick: (i: any) => { const s = p.suspects.find((q) => q.MMSI === i.object.d.mmsi); if (s) p.onSelectSuspect(s); },
        updateTriggers: { getRadius: [p.focus?.MMSI, suspectSet], getFillColor: [p.focus?.MMSI, suspectSet] },
      }));
      if (p.focus) {
        const h = heads.find((x) => x.d.mmsi === p.focus!.MMSI);
        if (h) L.push(new TextLayer({ id: "focus-label", data: [h], getPosition: (x: any) => x.xy, getText: () => p.focus!.VesselName || p.focus!.MMSI,
          getColor: [237, 233, 254], getSize: 13, getPixelOffset: [10, -12], fontFamily: "Inter", fontWeight: 600 } as any));
      }
    }
    // SAR ship detections: only meaningful at the acquisition time
    if (p.layers.ships && Math.abs(t) <= 1.5)
      L.push(new ScatterplotLayer({ id: "ships", data: p.ships, pickable: true, getPosition: (d: Ship) => [d.lon, d.lat], radiusUnits: "pixels",
        getRadius: 5, stroked: true, filled: false, lineWidthMinPixels: 2, getLineColor: (d: Ship) => shipColor(d.label) }));
    return L;
  }, [p, suspectSet]);

  useEffect(() => {
    overlay.current?.setProps({
      layers,
      onHover: (i: any) => {
        if (!i.object) return setHover(null);
        const o = i.object;
        let html = "";
        if (i.layer.id === "events") html = `<b>Slick event ${o.properties.cluster_id}</b><br/>${(o.properties.area_m2 / 1e6).toFixed(2)} km² · wind ${o.properties.wind_speed_m_s?.toFixed(1)} m/s`;
        else if (i.layer.id === "ships") html = `<b>SAR ship</b> ${esc(o.label).replaceAll("_", " ")}<br/>${esc(o.match_name || o.silent_name)} · ${o.peak_db?.toFixed(1)} dB`;
        else if (i.layer.id === "heads") html = `<b>${esc(o.d.name || o.d.mmsi)}</b><br/>MMSI ${esc(o.d.mmsi)}`;
        else if (i.layer.id === "platforms") html = `<b>Platform</b><br/>${esc(o.name)}`;
        setHover(html ? { x: i.x, y: i.y, html } : null);
      },
    });
  }, [layers]);

  return (
    <div className="map-wrap">
      <div ref={div} style={{ position: "absolute", inset: 0 }} />
      {hover && <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }} dangerouslySetInnerHTML={{ __html: hover.html }} />}
    </div>
  );
}
