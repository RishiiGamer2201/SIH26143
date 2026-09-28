// Loaders for the JSON exported by portal/build_data.py (static files under /data/<incident>/).
export type Suspect = Record<string, any> & {
  suspect_rank: number; tier: string; v31_decision?: string; MMSI: string; VesselName: string;
  fwd_F: number; bwd_point: number; fwd_window_start_h: number; fwd_window_end_h: number;
};
export type EventProps = Record<string, any> & { cluster_id: number; selected: boolean };
export type Track = { mmsi: string; name: string; type: number | null; path: [number, number][]; t: number[]; reports_t: number[] };
export type Ship = { lon: number; lat: number; label: string; match_name?: string; silent_name?: string; peak_db: number; size_px: number };

const base = (inc: string) => `/data/${inc}`;
const cache = new Map<string, Promise<any>>();

export function get<T = any>(path: string): Promise<T> {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => (r.ok ? r.json() : null)));
  return cache.get(path)!;
}

export const loadCore = (inc: string) =>
  Promise.all([
    get(`${base(inc)}/meta.json`), get(`${base(inc)}/events.geojson`), get(`${base(inc)}/suspects.json`),
    get(`${base(inc)}/ais_tracks.json`), get(`${base(inc)}/ships.json`), get(`${base(inc)}/platforms.json`),
    get(`${base(inc)}/reference.geojson`), get(`${base(inc)}/twin.json`),
  ]).then(([meta, events, suspects, tracks, ships, platforms, reference, twin]) =>
    ({ meta, events, suspects, tracks, ships, platforms, reference, twin }));

export const loadEvent = (inc: string, id: number) =>
  Promise.all([get(`${base(inc)}/backward/${id}.json`), get(`${base(inc)}/forward/${id}.json`), get(`${base(inc)}/forecast/${id}.json`)])
    .then(([backward, forward, forecast]) => ({ backward, forward, forecast }));

/** position of a track at time t (hours rel. T0), linear between 15-min samples; null outside coverage */
export function trackAt(tr: Track, t: number): [number, number] | null {
  const ts = tr.t;
  if (!ts.length || t < ts[0] || t > ts[ts.length - 1]) return null;
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m; }
  if (ts[hi] - ts[lo] > 0.3) return null; // AIS gap (> 15 min sampling) -> unknown position
  const f = ts[hi] === ts[lo] ? 0 : (t - ts[lo]) / (ts[hi] - ts[lo]);
  const a = tr.path[lo], b = tr.path[hi];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
}

export const fmtH = (h: number) => (h === 0 ? "T0" : `${h > 0 ? "+" : "−"}${Math.abs(h).toFixed(Math.abs(h) < 10 && h % 1 ? 1 : 0)} h`);
export const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
