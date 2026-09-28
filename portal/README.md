# OilWatch portal: analyst console

A dark-map web console on top of `oilwatch` runs:

- **Detected slick events:** size, wind window, radar "colour", estimated age, and the lead suspect.
- **Time slider from −48 h to +72 h:** the oil drifts back to its source (hindcast particles) while AIS ships move along their tracks. Past T0 the forecast cloud shows where the oil goes next (solid dots = still at the surface).
- **Suspect board:** ranked leads with backward, forward, FSS and ensemble evidence bars, plus flags (SAR sees it while AIS is off, AIS gap in the release window, at the slick tip, BOEM platform). The v3.1 *high-confidence flag* appears only under the twin-calibrated rule. Clicking a suspect shows its virtual oil at T0 and jumps to its release window.
- **"How accurate is this?":** measured twin-experiment skill (top-1/3/5, release-time error, the leads vs. accusations trade-off) and the method.

## Run

```bash
# 1. export a pipeline run to portal JSON
.venv/Scripts/python portal/build_data.py incident_001 e008_v3_scene
# 2a. dev (hot reload)
cd portal/web && npm install && npm run dev            # http://localhost:5173
# 2b. or build once and serve everything from the API
cd portal/web && npm run build && cd ../..
.venv/Scripts/python -m uvicorn portal.api.main:app --host 127.0.0.1 --port 8000   # http://127.0.0.1:8000
```

**Deep links:** `?event=18&focus=367655260&t=-1.5` (event, suspect MMSI, hours relative to T0) and `?method=1`.

## API (localhost only)

| method | path | what |
|---|---|---|
| GET | `/api/incidents` | incident specs in `configs/incidents/` |
| GET | `/api/incidents/{id}/runs` | runs with per-stage status and time (from the manifests) |
| POST | `/api/incidents/{id}/runs/{run_id}?stages=...` | start a pipeline run in the background |
| POST | `/api/incidents/{id}/runs/{run_id}/export` | re-export a finished run into portal JSON |
| GET | `/api/jobs` | background job status and log tail |

Stack: React 19 + Vite, MapLibre GL (CARTO dark basemap), deck.gl 9 (Trips, Scatterplot, GeoJson layers), FastAPI.
