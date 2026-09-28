import { EventProps, Suspect, fmtH, pct } from "./data";

const tierTag = (t: string) =>
  t === "strongly consistent" ? "tag ok" : t === "consistent" ? "tag ok" : t === "weak" ? "tag warn" : "tag muted";

export function EventList({ events, suspects, selected, onSelect }: { events: any; suspects: any; selected: number | null; onSelect: (id: number) => void }) {
  if (!events) return null;
  const fs: EventProps[] = events.features.map((f: any) => f.properties)
    .sort((a: EventProps, b: EventProps) => Number(b.selected) - Number(a.selected) || a.potential_rank - b.potential_rank);
  return (
    <>
      <h3>Detected slick events · {fs.length}</h3>
      {fs.map((e) => {
        const top: Suspect | undefined = suspects?.[e.cluster_id]?.[0];
        const flag = suspects?.[e.cluster_id]?.find((s: Suspect) => s.v31_decision?.startsWith("high-confidence"));
        return (
          <div key={e.cluster_id} className={`event ${selected === e.cluster_id ? "active" : ""} ${e.selected ? "" : "dim"}`}
               onClick={() => e.selected && onSelect(e.cluster_id)} title={e.selected ? "" : e.ineligible_reason || "not selected (below top-12 potential)"}>
            <div className="row">
              <span className="title">Event {e.cluster_id}</span>
              <span className="tag">{(e.area_m2 / 1e6).toFixed(2)} km²</span>
            </div>
            <div className="meta">
              wind {e.wind_speed_m_s?.toFixed(1)} m/s · {e.darkness_class ?? "—"} · {e.selected ? `age ${e.age_lo_h?.toFixed(0)}–${e.age_hi_h?.toFixed(0)} h` : "not attributed"}
            </div>
            {top && (
              <div className="lead" style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ color: "var(--ink-3)" }}>lead</span> <b>{top.VesselName || top.MMSI}</b>
                {flag ? <span className="tag flag">high-confidence flag</span> : <span className={tierTag(top.tier)}>{top.tier}</span>}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

function Bar({ label, v, color, fmt }: { label: string; v: number | null | undefined; color: string; fmt?: (x: number) => string }) {
  const x = Math.max(0, Math.min(1, v ?? 0));
  return (
    <>
      <span className="l">{label}</span>
      <span className="t"><i style={{ width: `${x * 100}%`, background: color }} /></span>
      <span className="n">{v == null ? "–" : fmt ? fmt(v) : v.toFixed(2)}</span>
    </>
  );
}

export function SuspectBoard({ ev, suspects, focus, onFocus }: { ev: EventProps | null; suspects: Suspect[]; focus: Suspect | null; onFocus: (s: Suspect) => void }) {
  if (!ev) return <div className="empty">Select a slick event on the map or in the list.</div>;
  const f = ev.funnel || {};
  const max = f.ais_vessels_in_window || 1;
  return (
    <>
      <h3>Event {ev.cluster_id} · attribution</h3>
      <div className="facts">
        <div className="fact"><div className="k">Area · length</div><div className="v">{(ev.area_m2 / 1e6).toFixed(2)} km²</div><div className="s">{(ev.length_m / 1e3).toFixed(1)} km long · {ev.width_proxy_m?.toFixed(0)} m wide</div></div>
        <div className="fact"><div className="k">Estimated age</div><div className="v">{ev.age_lo_h?.toFixed(0)}–{ev.age_hi_h?.toFixed(0)} h</div><div className="s">{ev.age_basis?.replace("forward best window of top supported suspect", "drift match")}</div></div>
        <div className="fact"><div className="k">Radar 'colour'</div><div className="v">{ev.damping_db?.toFixed(1) ?? "–"} dB</div><div className="s">{ev.darkness_class ?? "not measured"}</div></div>
        <div className="fact"><div className="k">Wind at T0</div><div className="v">{ev.wind_speed_m_s?.toFixed(1)} m/s</div><div className="s">{ev.wind_class} SAR window · 72 h beaching {pct(ev.p_beached_72h)}</div></div>
      </div>
      {ev.age_conflict && <div className="note">Age conflict: drift points to an older release than the slick's appearance allows — an old slick, or a young one from a source without AIS (platform, dark vessel, seep).</div>}
      <div className="funnel">
        {[["AIS vessels in window", f.ais_vessels_in_window], ["with ≥ 2 h of positions", f.positioned_ge_min_hours], ["reached by the drifted oil", f.within_gate_of_backward_cloud], ["ranked below", suspects.length]].map(([k, v]: any) => (
          <div key={k} className="bar"><div className="fill" style={{ width: `${Math.max(2, (100 * (v ?? 0)) / max)}%` }} /><span>{v ?? "–"} · {k}</span></div>
        ))}
      </div>
      <h3>Suspects (ranked leads)</h3>
      {suspects.map((s) => {
        const flag = s.v31_decision?.startsWith("high-confidence");
        return (
          <div key={s.MMSI} className={`suspect ${focus?.MMSI === s.MMSI ? "active" : ""}`} onClick={() => onFocus(s)}>
            <div className="head">
              <span className="rank">#{s.suspect_rank}</span>
              <span className="name">{s.VesselName || s.MMSI}</span>
              {flag ? <span className="tag flag">high-confidence</span> : <span className={tierTag(s.tier)}>{s.tier}</span>}
            </div>
            <div className="sub">
              {s.source_type?.replace("_", " ")} · {s.motion_state} · MMSI {s.MMSI}
              {s.fwd_F > 0 && <> · release {s.fwd_window_start_h}–{s.fwd_window_end_h} h before image</>}
            </div>
            <div className="bars">
              <Bar label="backward" v={s.bwd_point} color="#60a5fa" />
              <Bar label="forward" v={s.fwd_F} color="#a78bfa" />
              <Bar label="FSS 1 km" v={s.fss_1km} color="#2dd4bf" />
              <Bar label="ensemble" v={s.fwd_member_support} color="#94a3b8" fmt={(x) => pct(x)} />
            </div>
            <div className="flags">
              {s.flag_ais_silent_seen_by_sar_near_slick && <span className="tag flag">SAR sees it · AIS off</span>}
              {s.flag_ais_gap_in_release_window && <span className="tag warn">AIS gap in release window</span>}
              {s.flag_fresh_discharge_match && <span className="tag warn">at slick tip</span>}
              {s.flag_identity_anomaly && <span className="tag warn">identity anomaly</span>}
              {s.flag_window_at_48h_boundary && <span className="tag muted">window at 48 h edge</span>}
              {s.source_type === "fixed_infrastructure" && <span className="tag">platform (BOEM)</span>}
              <span className="tag muted">{s.direction_agreement}</span>
            </div>
          </div>
        );
      })}
    </>
  );
}

export function Timeline({ t, setT, playing, setPlaying, focus }: { t: number; setT: (x: number) => void; playing: boolean; setPlaying: (b: boolean) => void; focus: Suspect | null }) {
  const pos = (h: number) => `${((h + 48) / 120) * 100}%`;
  const phase = t < 0 ? "hindcast · oil traced back in time" : t === 0 ? "satellite image (T0)" : "forecast · where the oil goes next";
  return (
    <div className="timeline">
      <div className="clock"><div className="big">{fmtH(t)}</div><div className="small">{phase}</div></div>
      <div className="track">
        <div className="zone" style={{ left: 0, width: pos(0), background: "linear-gradient(90deg,#1e3a8a,#60a5fa)", opacity: .35 }} />
        <div className="zone" style={{ left: pos(0), right: 0, background: "linear-gradient(90deg,#fb923c,#7c2d12)", opacity: .35 }} />
        {focus && focus.fwd_F > 0 && (
          <div className="win" title="best-matching release window" style={{ left: pos(-focus.fwd_window_end_h), width: `${((focus.fwd_window_end_h - focus.fwd_window_start_h) / 120) * 100}%` }} />
        )}
        <div className="t0" style={{ left: pos(0) }} />
        <input type="range" min={-48} max={72} step={0.25} value={t} onChange={(e) => { setPlaying(false); setT(parseFloat(e.target.value)); }} />
        {[-48, -36, -24, -12, 0, 24, 48, 72].map((h) => <span key={h} className="lbl" style={{ left: pos(h) }}>{h === 0 ? "T0" : h > 0 ? `+${h}` : h}</span>)}
      </div>
      <div className="ctrls">
        <button className="btn" onClick={() => setT(-48)}>⏮</button>
        <button className="btn primary" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => { setPlaying(false); setT(0); }}>T0</button>
      </div>
    </div>
  );
}

export function Legend({ layers, set }: { layers: any; set: (k: string) => void }) {
  const items: [string, string, string][] = [
    ["backward", "#60a5fa", "hindcast particles (before T0)"], ["forecast", "#fb923c", "forecast particles (after T0)"],
    ["ais", "#2dd4bf", "AIS tracks (suspects bright)"], ["ships", "#f43f5e", "SAR ships at T0 (red = no AIS)"],
    ["platforms", "#78869f", "BOEM platforms"], ["reference", "#facc15", "Cerulean outline (post-hoc)"],
  ];
  return (
    <div className="legend">
      {items.map(([k, c, l]) => (
        <label key={k}><input type="checkbox" checked={layers[k]} onChange={() => set(k)} /><span className="sw" style={{ background: c }} />{l}</label>
      ))}
      <label><span className="sw" style={{ background: "#a78bfa" }} />selected suspect's virtual oil (at T0)</label>
    </div>
  );
}

export function MethodPanel({ meta, twin, onClose }: { meta: any; twin: any; onClose: () => void }) {
  const all = twin?.skill?.find((r: any) => r.subset === "all");
  const cal = twin?.calibration;
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <div><h2>How it works · how well it works</h2><p style={{ margin: 0 }}>Sentinel-1 SAR → oil segmentation → drift backward and forward → AIS evidence → ranked leads.</p></div>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
        <h4>Measured accuracy — {twin?.summary?.n_cases ?? "–"} blind synthetic spills from real AIS ships</h4>
        {all && (
          <div className="kpis">
            <div className="kpi"><div className="v">{pct(all.top1_fused)}</div><div className="k">true ship ranked #1</div></div>
            <div className="kpi"><div className="v">{pct(all.top3_fused)}</div><div className="k">true ship in top 3</div></div>
            <div className="kpi"><div className="v">{pct(all.top5_fused)}</div><div className="k">true ship in top 5</div></div>
            <div className="kpi"><div className="v">{all.median_window_mid_error_h?.toFixed(1)} h</div><div className="k">median release-time error</div></div>
          </div>
        )}
        <div className="grid2">
          <div>
            <h4>By age of the spill</h4>
            <table className="t"><thead><tr><th>age</th><th>n</th><th>#1</th><th>top 3</th><th>#1 forward only</th></tr></thead>
              <tbody>{twin?.skill?.filter((r: any) => /h$/.test(r.subset)).map((r: any) => (
                <tr key={r.subset}><td>{r.subset}</td><td>{r.n}</td><td>{pct(r.top1_fused)}</td><td>{pct(r.top3_fused)}</td><td>{pct(r.top1_forward_only)}</td></tr>))}</tbody></table>
          </div>
          <div>
            <h4>Leads vs. accusations</h4>
            <p>The ranking is a strong <b>lead generator</b>. Deciding that the culprit is among the AIS ships at all is harder: with the true ship's AIS hidden, some innocent ship usually still looks plausible in crowded water.</p>
            <p>A <b>high-confidence flag</b> is therefore raised only under a rule calibrated on half of the twins and tested on the other half
              {cal && <> (forward ≥ {cal.rule.fwd_F_min}, backward ≥ {cal.rule.bwd_point_min}, ensemble ≥ {pct(cal.rule.member_support_min)}): held-out false accusation <b>{pct(cal.test.false_accusation)}</b>, fires for {pct(cal.test.correct_attribution)} of true culprits</>}.</p>
          </div>
        </div>
        <h4>Method</h4>
        <ul>
          <li><b>Detection:</b> SegFormer-B2 on calibrated VV σ0 (frozen E005). Fragments within 1 km are merged into slick events; the wind at the slick must lie in the SAR oil window (2.5–10 m/s).</li>
          <li><b>Backward drift:</b> OpenDrift, 48 h, 27-member ensemble (windage 1–3 % plus Stokes drift, diffusivity 1–10 m²/s, current uncertainty). The kernel widens with age (1 km at T0).</li>
          <li><b>Forward test:</b> every candidate ship "discharges" along its own AIS track. Its virtual oil is drifted to T0 and compared with the slick (overlap F and Fractions Skill Score); the best window gives the release time.</li>
          <li><b>Extra evidence:</b> SAR ship detections versus AIS (dark or AIS-silent ships), AIS gaps, BOEM platforms, radar darkness ('colour') and slick width for age.</li>
          <li><b>Checks:</b> the frozen team pipeline reproduces bit-exactly; the image geolocation error is measured at {meta?.geolocation?.median_abs_offset_m?.toFixed(0)} m on platforms; the reference (Cerulean) is never used as an input.</li>
        </ul>
        <p style={{ fontSize: 12, color: "var(--ink-3)" }}>{meta?.disclaimer}</p>
      </div>
    </div>
  );
}
