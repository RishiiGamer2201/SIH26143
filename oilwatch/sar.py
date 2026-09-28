"""Tile-wise Sentinel-1 GRD processing straight from the measurement TIFF (radar geometry, no full-scene rasters).

Built on the frozen scripts/s1_grd_calibrate.py (sigma0 = (DN^2 - noise)/A^2 per window, GCP thin-plate-spline
geolocation). Two products:

darkness  per slick event: VV damping ratio (median sea ring - median slick, dB; histogram-median method, Quigley et
          al. 2023), edge contrast and slick texture. SAR 'colour' of oil: fresh/thick = strong contrast + sharp edges,
          weathered = weak/diffuse (KSAT labels, Bianchi et al. 2020). There is no validated SAR-age formula, so the class
          is a heuristic, anchored on the training-set oil damping median (6.36 dB, SAR_DATA_AUDIT section 4).
ships     scene-wide two-parameter CFAR on calibrated VV: pixel above local sea clutter (guard/background rings)
          by k sigma AND by >= min_db; components -> ship detections with lon/lat. Matched to AIS at acquisition
          time (moving ships allow the SAR azimuth (Doppler) shift ~ R*v_r/V_sat) and to BOEM platforms; unmatched =
          dark-vessel candidates. Platform matches also MEASURE the geolocation error of the GCP/TPS product.
"""
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage

from .frozen import _load

cal = _load("s1_grd_calibrate")
gdal.UseExceptions()


def short_safe(safe):
    """Windows MAX_PATH (260) breaks Python open() on deep SAFE paths (GDAL copes, xml.etree does not).
    Use a directory junction under %LOCALAPPDATA%/oilwatch (no admin rights needed); no-op elsewhere."""
    import hashlib
    import os
    import subprocess
    safe = Path(safe)
    if os.name != "nt" or len(str(safe.resolve())) < 150:
        return safe
    link = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "oilwatch" / ("safe_" + hashlib.md5(str(safe.resolve()).encode()).hexdigest()[:8])
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(safe.resolve())], check=True, capture_output=True)
    return link


def measurement(safe, pol):
    return glob.glob(str(Path(safe) / f"measurement/*-{pol}-*.tiff"))[0]


def annotation_gcps(safe, pol="vv"):
    xml = glob.glob(str(Path(safe) / f"annotation/s1*-{pol}-*.xml"))[0]
    pts = ET.parse(xml).getroot().findall(".//geolocationGridPoint")
    return [gdal.GCP(float(p.find("longitude").text), float(p.find("latitude").text), float(p.find("height").text),
                     float(p.find("pixel").text), float(p.find("line").text)) for p in pts]


class Scene:
    """lon/lat <-> radar pixel via GCP thin-plate spline (the E007 geocoding model)."""

    def __init__(self, safe, pol="vv"):
        self.safe, self.pol = str(short_safe(safe)), pol
        self.tif = measurement(safe, pol)
        ds = gdal.Open(self.tif)
        self.W, self.H = ds.RasterXSize, ds.RasterYSize
        gcps = ds.GetGCPs()
        self.gcp_source = "measurement tiff"
        if len(gcps) < 10:  # some COG re-encodings drop GCPs -> use the SAFE annotation geolocation grid
            gcps, self.gcp_source = annotation_gcps(safe, pol), "SAFE annotation geolocationGrid"
            vrt = gdal.Translate("/vsimem/gcp.vrt", ds, format="VRT", GCPs=gcps, outputSRS="EPSG:4326")
            ds = vrt
        self.ds = ds
        self.tr = gdal.Transformer(ds, None, ["METHOD=GCP_TPS"])
        self.n_gcps = len(gcps)

    def to_px(self, lon, lat):
        pts = self.tr.TransformPoints(1, list(zip(np.atleast_1d(lon), np.atleast_1d(lat))))[0]
        a = np.asarray(pts)
        return a[:, 0], a[:, 1]  # col, row

    def to_lonlat(self, col, row):
        pts = self.tr.TransformPoints(0, list(zip(np.atleast_1d(col).astype(float), np.atleast_1d(row).astype(float))))[0]
        a = np.asarray(pts)
        return a[:, 0], a[:, 1]

    def sigma0(self, row0, col0, h, w):
        row0, col0 = max(int(row0), 0), max(int(col0), 0)
        h, w = min(int(h), self.H - row0), min(int(w), self.W - col0)
        return cal.sigma0(self.safe, self.pol, row0, col0, h, w), row0, col0


def geom_to_radar_mask(scene, geom, row0, col0, h, w):
    polys = list(getattr(geom, "geoms", [geom]))
    shapes = []
    from shapely.geometry import Polygon
    for p in polys:
        x, y = p.exterior.xy
        c, r = scene.to_px(np.asarray(x), np.asarray(y))
        holes = []
        for ring in p.interiors:
            hx, hy = ring.xy
            hc, hr = scene.to_px(np.asarray(hx), np.asarray(hy))
            holes.append(list(zip(hc - col0, hr - row0)))
        shapes.append((Polygon(list(zip(c - col0, r - row0)), holes), 1))
    return rasterize(shapes, out_shape=(h, w), transform=Affine.identity(), all_touched=False).astype(bool)


def darkness(scene, cand, margin_px=400, ring_in=5, ring_out=60, P=None):
    rows = []
    for _, c in cand.iterrows():
        x0, y0, x1, y1 = c.geometry.bounds
        cc, rr = scene.to_px([x0, x1, x0, x1], [y0, y0, y1, y1])
        r0, c0 = rr.min() - margin_px, cc.min() - margin_px
        s0, r0, c0 = scene.sigma0(r0, c0, rr.max() - rr.min() + 2 * margin_px, cc.max() - cc.min() + 2 * margin_px)
        db = 10 * np.log10(s0)
        m = geom_to_radar_mask(scene, c.geometry, r0, c0, *db.shape)
        if m.sum() < 20:
            rows.append({"cluster_id": c.cluster_id, "darkness_status": "slick not resolved in radar window"})
            continue
        near = ndimage.binary_dilation(m, iterations=ring_in)
        ring = ndimage.binary_dilation(m, iterations=ring_out) & ~near & np.isfinite(db)
        rim = ndimage.binary_dilation(m, iterations=2) & ~m & np.isfinite(db)
        inner = m & np.isfinite(db)
        sea, oil = np.nanmedian(db[ring]), np.nanmedian(db[inner])
        core = ndimage.binary_erosion(m, iterations=2) & np.isfinite(db)
        dr = float(sea - oil)
        edge = float(np.nanmedian(db[rim]) - oil)
        cls = ("strong/fresh-looking" if dr >= P["strong_db"] else "weak/weathered-looking" if dr < P["weak_db"]
               else "intermediate")
        rows.append({"cluster_id": c.cluster_id, "darkness_status": "ok", "sea_vv_db": float(sea), "slick_vv_db": float(oil),
                     "slick_core_vv_db": float(np.nanmedian(db[core])) if core.any() else np.nan,
                     "damping_ratio_db": dr, "edge_contrast_db": edge, "edge_sharpness": edge / dr if dr > 0 else np.nan,
                     "slick_texture_db_std": float(np.nanstd(db[inner])), "slick_px": int(m.sum()), "darkness_class": cls})
    return pd.DataFrame(rows)


def cfar_tile(s0, P):
    """Two-parameter CFAR on linear sigma0 with a guard ring; returns boolean detections."""
    x = np.nan_to_num(s0, nan=0.0)
    valid = np.isfinite(s0).astype(float)
    g, b = P["guard_px"], P["background_px"]
    sum_b, sum_g = ndimage.uniform_filter(x, b) * b * b, ndimage.uniform_filter(x, g) * g * g
    sq_b, sq_g = ndimage.uniform_filter(x * x, b) * b * b, ndimage.uniform_filter(x * x, g) * g * g
    n_b, n_g = ndimage.uniform_filter(valid, b) * b * b, ndimage.uniform_filter(valid, g) * g * g
    n = np.maximum(n_b - n_g, 1)
    mu = (sum_b - sum_g) / n
    sd = np.sqrt(np.maximum((sq_b - sq_g) / n - mu ** 2, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        excess_db = 10 * np.log10(x / np.maximum(mu, 1e-12))
    return (x > mu + P["k_sigma"] * sd) & (excess_db >= P["min_excess_db"]) & (n > 0.5 * (b * b - g * g))


def ships(scene, P, log_every=50):
    T, ov = P["tile_px"], P["background_px"]
    dets = []
    tiles = [(r, c) for r in range(0, scene.H, T) for c in range(0, scene.W, T)]
    for i, (r, c) in enumerate(tiles):
        s0, r0, c0 = scene.sigma0(r - ov, c - ov, T + 2 * ov, T + 2 * ov)
        if not np.isfinite(s0).any():
            continue
        det = cfar_tile(s0, P)
        lbl, n = ndimage.label(det)
        if n:
            idx = np.arange(1, n + 1)
            size = ndimage.sum(det, lbl, idx)
            peak = ndimage.maximum(np.nan_to_num(s0), lbl, idx)
            cy, cx = np.array(ndimage.center_of_mass(det, lbl, idx)).T
            keep = (size >= P["min_px"]) & (size <= P["max_px"])
            # own the detection only if its centre lies in this tile's core (overlap de-duplication)
            gy, gx = cy + r0, cx + c0
            keep &= (gy >= r) & (gy < r + T) & (gx >= c) & (gx < c + T)
            for k in np.where(keep)[0]:
                dets.append({"row": gy[k], "col": gx[k], "size_px": int(size[k]), "peak_db": float(10 * np.log10(peak[k]))})
        if i % log_every == 0:
            print(f"  ship scan tile {i + 1}/{len(tiles)}: {len(dets)} detections so far", flush=True)
    d = pd.DataFrame(dets)
    if len(d):
        d["lon"], d["lat"] = scene.to_lonlat(d.col.values, d.row.values)
    return d


def match_detections(d, ais, t0, platforms, M, cfg21):
    """Label each detection: ais_vessel (moving/stationary) | platform (BOEM) | dark_vessel_candidate.
    Also returns geolocation offsets measured on stationary matches (known E007 limitation #4)."""
    from .ais import state_at
    from .geo import haversine_km
    w = pd.Timedelta(minutes=M["ais_window_min"])
    near = ais[(ais.BaseDateTime >= t0 - w) & (ais.BaseDateTime <= t0 + w)]
    st = []
    for m, g in near.groupby("MMSI"):
        s = state_at(ais[ais.MMSI == m], t0, cfg21)
        if not np.isfinite(s["lon"]):
            i = (g.BaseDateTime - t0).abs().argmin()
            s.update(lon=float(g.LON.iloc[i]), lat=float(g.LAT.iloc[i]))
        st.append({"MMSI": m, "VesselName": g.VesselName.dropna().iloc[0] if g.VesselName.notna().any() else "", **s})
    st = pd.DataFrame(st)
    d = d.copy()
    for col in ("label", "match_id", "match_name", "match_km", "d_lon_m", "d_lat_m"):
        d[col] = None
    for i, r in d.iterrows():
        best = None
        if len(st):
            dist = haversine_km(r.lon, r.lat, st.lon.values, st.lat.values)
            j = int(np.argmin(dist))
            moving = np.isfinite(st.sog.iloc[j]) and st.sog.iloc[j] >= M["moving_sog_kn"]
            if dist[j] <= (M["moving_km"] if moving else M["stationary_km"]):
                best = ("ais_moving" if moving else "ais_stationary", st.MMSI.iloc[j], st.VesselName.iloc[j], dist[j],
                        st.lon.iloc[j], st.lat.iloc[j])
        if best is None and platforms is not None and len(platforms):
            dist = haversine_km(r.lon, r.lat, platforms.lon.values, platforms.lat.values)
            j = int(np.argmin(dist))
            if dist[j] <= M["platform_km"]:
                best = ("platform_no_ais", platforms.id.iloc[j], platforms.name.iloc[j], dist[j], platforms.lon.iloc[j], platforms.lat.iloc[j])
        if best:
            lab, mid, nm, km, lo, la = best
            d.loc[i, ["label", "match_id", "match_name", "match_km"]] = [lab, mid, nm, km]
            d.loc[i, "d_lon_m"] = (r.lon - lo) * 111320 * np.cos(np.radians(la))
            d.loc[i, "d_lat_m"] = (r.lat - la) * 110540
        else:
            d.loc[i, "label"] = "dark_vessel_candidate"
    d = silent_vessels(d, ais, t0, M, cover_bbox=(ais.LON.min(), ais.LAT.min(), ais.LON.max(), ais.LAT.max()))
    return d, st


def silent_vessels(d, ais, t0, M, cover_bbox):
    """Unmatched detections: (a) outside AIS coverage -> 'no_ais_coverage' (cannot call it dark);
    (b) explained by a vessel whose AIS went silent before T0 (last report within silent_max_h, none within the match
    window): dead-reckoned position last_pos + SOG*dt along COG within match radius + 30% of the dead-reckoned
    distance -> 'ais_silent_at_t0' with the probable MMSI; (c) otherwise stays 'dark_vessel_candidate'."""
    from .geo import haversine_km
    w, H = pd.Timedelta(minutes=M["ais_window_min"]), pd.Timedelta(hours=M.get("silent_max_h", 3))
    before = ais[(ais.BaseDateTime >= t0 - H) & (ais.BaseDateTime < t0 - w)]
    recent = set(ais[(ais.BaseDateTime >= t0 - w) & (ais.BaseDateTime <= t0 + w)].MMSI)
    last = before[~before.MMSI.isin(recent)].sort_values("BaseDateTime").groupby("MMSI").tail(1)
    dt_h = (t0 - last.BaseDateTime).dt.total_seconds().values / 3600
    sog = np.nan_to_num(last.SOG.values.astype(float)) * 1.852  # km/h
    cog = np.radians(np.nan_to_num(last.COG.values.astype(float)))
    dr_km = sog * dt_h
    lat_dr = last.LAT.values + dr_km * np.cos(cog) / 110.54
    lon_dr = last.LON.values + dr_km * np.sin(cog) / (111.32 * np.cos(np.radians(last.LAT.values)))
    d["silent_mmsi"], d["silent_name"], d["silent_last_report_min"], d["silent_dist_km"] = None, None, np.nan, np.nan
    for i, r in d[d.label == "dark_vessel_candidate"].iterrows():
        if not (cover_bbox[0] <= r.lon <= cover_bbox[2] and cover_bbox[1] <= r.lat <= cover_bbox[3]):
            d.loc[i, "label"] = "no_ais_coverage"
            continue
        if not len(last):
            continue
        dist = haversine_km(r.lon, r.lat, lon_dr, lat_dr)
        # cap: dead reckoning a fast boat over hours gives radii of tens of km (first run matched noise at 8-25 km)
        ok = dist <= np.minimum(M["moving_km"] + 0.3 * dr_km, M.get("silent_max_radius_km", 3.0))
        if ok.any():
            j = int(np.argmin(np.where(ok, dist, np.inf)))
            d.loc[i, ["label", "silent_mmsi", "silent_name", "silent_last_report_min", "silent_dist_km"]] = [
                "ais_silent_at_t0", last.MMSI.iloc[j], last.VesselName.iloc[j], dt_h[j] * 60, dist[j]]
    return d


def geolocation_offset(d):
    """Median offset of stationary targets (AIS-stationary + platforms): measures the GCP/TPS geolocation error."""
    s = d[d.label.isin(["ais_stationary", "platform_no_ais"])]
    if len(s) < 3:
        return {"n": int(len(s)), "status": "too few stationary matches"}
    dx, dy = s.d_lon_m.astype(float), s.d_lat_m.astype(float)
    return {"n": int(len(s)), "median_dx_m": float(dx.median()), "median_dy_m": float(dy.median()),
            "median_abs_offset_m": float(np.hypot(dx, dy).median()), "p90_abs_offset_m": float(np.quantile(np.hypot(dx, dy), .9))}
