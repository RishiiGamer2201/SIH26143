import { useEffect, useMemo, useState } from "react";
import MapView, { Layers } from "./MapView";
import { EventList, Legend, MethodPanel, SuspectBoard, Timeline } from "./panels";
import { EventProps, Suspect, loadCore, loadEvent } from "./data";

const Q = new URLSearchParams(location.search);
const INCIDENT = Q.get("incident") || "incident_001";

export default function App() {
  const [core, setCore] = useState<any>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [evData, setEvData] = useState<any>({});
  const [focus, setFocus] = useState<Suspect | null>(null);
  const [t, setT] = useState(Q.has("t") ? Number(Q.get("t")) : 0);
  const [playing, setPlaying] = useState(false);
  const [method, setMethod] = useState(Q.get("method") === "1");
  const [layers, setLayers] = useState<Layers>({ backward: true, forecast: true, ais: true, ships: true, platforms: true, reference: false });

  useEffect(() => { loadCore(INCIDENT).then((c) => {
    setCore(c);
    // open on the event carrying the strongest evidence (high-confidence flag first, else best lead)
    const s = c.suspects as Record<string, Suspect[]>;
    const flagged = Object.entries(s).find(([, l]) => l.some((x) => x.v31_decision?.startsWith("high-confidence")));
    setSelected(Q.has("event") ? Number(Q.get("event")) : Number(flagged ? flagged[0] : Object.keys(s)[0]));
  }); }, []);
  useEffect(() => { if (selected != null) { setFocus(null); loadEvent(INCIDENT, selected).then(setEvData); } }, [selected]);
  useEffect(() => { const f = Q.get("focus"); if (f && core && selected != null) { const s = (core.suspects[selected] || []).find((x: Suspect) => x.MMSI === f); if (s) setFocus(s); } }, [core, selected]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setT((x) => (x >= 72 ? -48 : Math.round((x + 0.5) * 4) / 4)), 90);
    return () => clearInterval(id);
  }, [playing]);

  const ev: EventProps | null = useMemo(() => core?.events.features.find((f: any) => f.properties.cluster_id === selected)?.properties ?? null, [core, selected]);
  const suspects: Suspect[] = (selected != null && core?.suspects?.[selected]) || [];
  const onFocus = (s: Suspect) => { setFocus(s); if (s.fwd_F > 0) { setPlaying(false); setT(-s.fwd_window_start_h); } };

  const m = core?.meta;
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand"><span className="dot" />OILWATCH <small>spill attribution console · SIH 26143</small></div>
        {m && (
          <div className="chips">
            <span className="chip">Sentinel-1 · <b>{m.t0.replace("+00:00", " UTC")}</b></span>
            <span className="chip"><b>{m.n_events}</b> slick events · <b>{m.n_selected}</b> attributed</span>
            <span className="chip"><b>{m.n_ais_vessels.toLocaleString()}</b> AIS vessels</span>
            <span className="chip"><b>{m.n_ship_detections}</b> SAR ships · <b>{m.ship_labels?.dark_vessel_candidate ?? 0}</b> without AIS</span>
          </div>
        )}
        <span className="spacer" />
        <button className="btn" onClick={() => setMethod(true)}>How accurate is this?</button>
      </div>
      <div className="side"><EventList events={core?.events} suspects={core?.suspects} selected={selected} onSelect={setSelected} />
        <div className="disclaimer">{m?.disclaimer}</div></div>
      <MapView events={core?.events} selected={selected} onSelectEvent={setSelected} t={t} backward={evData.backward} forecast={evData.forecast}
               forward={evData.forward} tracks={core?.tracks ?? []} suspects={suspects} focus={focus} onSelectSuspect={onFocus}
               ships={core?.ships ?? []} platforms={core?.platforms ?? []} reference={core?.reference} layers={layers} />
      <div className="side right"><SuspectBoard ev={ev} suspects={suspects} focus={focus} onFocus={onFocus} /></div>
      <Timeline t={t} setT={setT} playing={playing} setPlaying={setPlaying} focus={focus} />
      <div style={{ position: "fixed", left: 342, bottom: 108, zIndex: 3 }}>
        <Legend layers={layers} set={(k) => setLayers({ ...layers, [k]: !(layers as any)[k] })} />
      </div>
      {method && <MethodPanel meta={m} twin={core?.twin} onClose={() => setMethod(false)} />}
    </div>
  );
}
