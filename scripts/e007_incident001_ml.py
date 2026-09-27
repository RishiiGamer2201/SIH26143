"""E007: Incident 001 real Sentinel-1 GRD -> calibrated, geocoded VV -> FROZEN E004 + E005 -> probability / mask rasters
-> polygons -> post-hoc Cerulean comparison. STOPS before physics (no OpenDrift, no AIS).

  python scripts/e007_incident001_ml.py preprocess   # SAFE -> sigma0 VV -> EPSG:4326 -> 20 / 40 m rasters + sanity report
  python scripts/e007_incident001_ml.py infer        # verify frozen checkpoints; overlapping-window inference, blended probability
  python scripts/e007_incident001_ml.py vector       # raw 0.60 mask -> 4-connected components -> geodesic metrics; Cerulean post hoc
  python scripts/e007_incident001_ml.py freeze       # write FROZEN.json (hashes + frozen configuration); refuses to overwrite
  python scripts/e007_incident001_ml.py verify       # re-hash every artefact listed in FROZEN.json

Preprocessing = the training chain (SAR_DATA_AUDIT.md sections 3, 9, 10):
  sigma0 = (DN^2 - N_range * N_azimuth) / A_sigma^2       (S1 IPF >= 2.9 calibration + thermal-noise LUTs, bilinear LUT interp)
  DN == 0 -> invalid (outside swath / border-noise-removed by the IPF). sigma0 <= 0 after noise subtraction stays VALID and is
  floored at 1e-13 (-130 dB): training data keeps such pixels as valid ~-128 dB values; the clip to -40 dB makes the floor moot.
  Geocoding: GDAL warp, thin-plate spline on the 210 annotation GCPs (ellipsoid heights ~0 m, ocean, no terrain), bilinear in
  LINEAR power, NaN nodata excluded from kernels, to EPSG:4326 at 8.983152841195208e-05 deg (the training tiles' exact pixel;
  square in degrees, NOT square in metres). Then linear-power block averages x2 (E005 "20 m") and x4 (E004 "40 m"), block
  valid if >= 50% valid, dB, clip [-40, +5], scale (dB + 40) / 45, float16 (the cache's storage precision), train norm_mu/sd.
The Cerulean polygon is used ONLY as a reporting ROI (bbox + margin) and for post-hoc comparison after vectorisation.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import json
import math
import sys
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import torch
from osgeo import gdal, ogr, osr
from scipy import ndimage
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e004_resnet18_cls as e004  # noqa: E402
import e005_segformer as e005  # noqa: E402
from e003_feature_baseline import CLASSES  # noqa: E402
from e006_segformer_test import git, sha256, verify_frozen  # noqa: E402
from s1_grd_calibrate import _files, _floats, sigma0 as sigma0_window  # noqa: E402

gdal.UseExceptions()
ROOT = e005.ROOT
SAFE = glob.glob(str(ROOT / "data/incident_001/sentinel1/extracted/*.SAFE"))[0]
CERULEAN = ROOT / "data/incident_001/metadata/slick_3775938.geojson"
OUT = ROOT / "results/E007_incident001_ml"
RAS, WORK = OUT / "rasters", OUT / "work"
RES = 8.983152841195208e-05  # training tile pixel (deg), SAR_DATA_AUDIT section 2
LO, HI, MIN_VALID, FLOOR = -40.0, 5.0, 0.5, 1e-13
THR = 0.60
WIN, STRIDE = 1024, 512       # E005 windows on the x2 grid (50% overlap)
WIN4, STRIDE4 = 512, 256      # the same footprints on the x4 grid for E004
ROI_MARGIN = 0.1              # deg around the Cerulean bbox; reporting / plotting only
E004_CK = ROOT / "results/E004_resnet18_cls/best.pt"
P = dict(sigma0_native=RAS / "incident001_sigma0_vv_linear_native.tif", vv=RAS / "incident001_vv_db.tif",
         valid=RAS / "incident001_valid.tif", vv40=RAS / "incident001_vv_db_40m.tif", valid40=RAS / "incident001_valid_40m.tif",
         prob=RAS / "incident001_segformer_probability.tif", mask=RAS / "incident001_segformer_mask_t060.tif")


# ---------------------------------------------------------------------------------------------------- helpers
def wgs84():
    s = osr.SpatialReference()
    s.ImportFromEPSG(4326)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def ground_spacing(lat, dlon, dlat):
    """Metres per pixel on the WGS84 ellipsoid: east-west (prime-vertical radius) and north-south (meridional radius)."""
    a, e2, s = 6378137.0, 6.69437999014e-3, math.sin(math.radians(lat))
    n, m = a / math.sqrt(1 - e2 * s * s), a * (1 - e2) / (1 - e2 * s * s) ** 1.5
    return n * math.cos(math.radians(lat)) * math.radians(dlon), m * math.radians(dlat)


def read(p):
    ds = gdal.Open(str(p))
    return ds.GetRasterBand(1).ReadAsArray(), ds.GetGeoTransform()


def write(p, a, gt, nodata=None):
    t = {np.dtype("float32"): gdal.GDT_Float32, np.dtype("uint8"): gdal.GDT_Byte}[a.dtype]
    ds = gdal.GetDriverByName("GTiff").Create(str(p), a.shape[1], a.shape[0], 1, t,
                                              ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=" + ("3" if t == gdal.GDT_Float32 else "2"),
                                               "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(wgs84().ExportToWkt())
    b = ds.GetRasterBand(1)
    if nodata is not None:
        b.SetNoDataValue(nodata)
    b.WriteArray(a)
    ds = None


def cerulean():
    return shape(json.loads(CERULEAN.read_text())["features"][0]["geometry"])


def roi_bounds():
    x0, y0, x1, y1 = cerulean().bounds
    return x0 - ROI_MARGIN, y0 - ROI_MARGIN, x1 + ROI_MARGIN, y1 + ROI_MARGIN


def px_window(gt, b, shape_):
    """(lon0, lat0, lon1, lat1) -> row/col slice on a north-up grid."""
    c0, c1 = int((b[0] - gt[0]) // gt[1]), int(math.ceil((b[2] - gt[0]) / gt[1]))
    r0, r1 = int((b[3] - gt[3]) // gt[5]), int(math.ceil((b[1] - gt[3]) / gt[5]))
    return slice(max(r0, 0), min(r1, shape_[0])), slice(max(c0, 0), min(c1, shape_[1]))


def pct(a):
    q = np.percentile(a, [0, 1, 10, 50, 90, 99, 100]) if a.size else [np.nan] * 7
    return dict(zip(["min", "p01", "p10", "median", "p90", "p99", "max"], map(float, q)))


# ---------------------------------------------------------------------------------------------------- preprocess
def lut_rows(vectors, key, width):
    """Each LUT vector interpolated along range to every column: (lines [n], values [n, width])."""
    lines = np.array([int(v.find("line").text) for v in vectors], dtype=np.float64)
    return lines, np.stack([np.interp(np.arange(width), _floats(v.find("pixel")), _floats(v.find(key))) for v in vectors])


def rows_interp(lines, along, rows):
    """Vectorised np.interp(rows, lines, along[:, j]) for all columns j (linear, clamped at the ends)."""
    i = np.clip(np.searchsorted(lines, rows, side="right") - 1, 0, len(lines) - 2)
    t = np.clip((rows - lines[i]) / (lines[i + 1] - lines[i]), 0.0, 1.0)
    return along[i] * (1 - t)[:, None] + along[i + 1] * t[:, None]


def calibrate(dst):
    tif, cal_xml, noise_xml = _files(SAFE, "vv")
    src = gdal.Open(tif)
    W, H = src.RasterXSize, src.RasterYSize
    cl, ca = lut_rows(ET.parse(cal_xml).getroot().findall(".//calibrationVector"), "sigmaNought", W)
    nz = ET.parse(noise_xml).getroot()
    nl, na = lut_rows(nz.findall(".//noiseRangeVector"), "noiseRangeLut", W)
    azb = [(int(b.find("firstAzimuthLine").text), int(b.find("lastAzimuthLine").text), int(b.find("firstRangeSample").text),
            int(b.find("lastRangeSample").text), _floats(b.find("line")), _floats(b.find("noiseAzimuthLut")))
           for b in nz.findall(".//noiseAzimuthVector")]
    ds = gdal.GetDriverByName("GTiff").Create(str(dst), W, H, 1, gdal.GDT_Float32, ["TILED=YES", "BIGTIFF=YES"])
    ds.SetGCPs(src.GetGCPs(), src.GetGCPProjection())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float("nan"))
    n_dn0 = n_floor = 0
    for r0 in range(0, H, 1024):
        h = min(1024, H - r0)
        rows = np.arange(r0, r0 + h, dtype=np.float64)
        dn = src.GetRasterBand(1).ReadAsArray(0, r0, W, h).astype(np.float64)
        az = np.ones((h, W))
        for l0, l1, s0, s1, ll, lut in azb:
            sel = (rows >= l0) & (rows <= l1)
            if sel.any():
                az[sel, s0:s1 + 1] = np.interp(rows[sel], ll, lut)[:, None]
        s = (dn ** 2 - rows_interp(nl, na, rows) * az) / rows_interp(cl, ca, rows) ** 2
        floor = (dn > 0) & (s <= 0)
        s[floor] = FLOOR
        s[dn == 0] = np.nan
        n_dn0, n_floor = n_dn0 + int((dn == 0).sum()), n_floor + int(floor.sum())
        band.WriteArray(s.astype(np.float32), 0, r0)
    ds = None
    return dict(radar_size=[H, W], dn0_frac=n_dn0 / (H * W), noise_floored_frac_of_valid=n_floor / max(H * W - n_dn0, 1),
                n_gcps=len(src.GetGCPs()))


def calib_check(radar_tif):
    """Fast full-scene calibration must equal the reference window implementation (scripts/s1_grd_calibrate.py)."""
    src_ds = gdal.Open(_files(SAFE, "vv")[0])
    r0, c0, h, w = src_ds.RasterYSize // 2 - 256, src_ds.RasterXSize // 2 - 256, 512, 512  # scene centre
    ref = sigma0_window(SAFE, "vv", r0, c0, h, w)
    rds = gdal.Open(str(radar_tif))  # keep the dataset referenced while its band is read
    new = rds.GetRasterBand(1).ReadAsArray(c0, r0, w, h).astype(np.float64)
    fin = np.isfinite(ref)
    rel = np.abs(new[fin] - ref[fin]) / ref[fin]
    assert rel.max() < 1e-5, f"calibration differs from reference implementation: {rel.max()}"
    return dict(window_rows=[r0, r0 + h], window_cols=[c0, c0 + w], max_rel_diff=float(rel.max()),
                ref_nan_px=int((~fin).sum()), reference="scripts/s1_grd_calibrate.py sigma0()")


def geocode(src, dst):
    sds = gdal.Open(str(src))
    g = np.array([(p.GCPX, p.GCPY) for p in sds.GetGCPs()])
    sds = None
    q = 4 * RES  # origin and size on multiples of 4 px so the x2 / x4 grids nest exactly
    x0, x1 = math.floor(g[:, 0].min() / q) * q, math.ceil(g[:, 0].max() / q) * q
    y0, y1 = math.floor(g[:, 1].min() / q) * q, math.ceil(g[:, 1].max() / q) * q
    width, height = round((x1 - x0) / RES), round((y1 - y0) / RES)
    gdal.Warp(str(dst), str(src), dstSRS="EPSG:4326", outputBounds=(x0, y0, x1, y1), width=width, height=height, tps=True,
              resampleAlg="bilinear", srcNodata=float("nan"), dstNodata=float("nan"), multithread=True, warpMemoryLimit=2048,
              warpOptions=["NUM_THREADS=ALL_CPUS"],
              creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"])
    return gdal.Open(str(dst)).GetGeoTransform(), (height, width)


def block_average(native):
    """Linear-power block means x2 and x4 (block valid if >= 50% valid), -> dB float32 (NaN invalid) + valid uint8."""
    ds = gdal.Open(str(native))
    H, W, gt = ds.RasterYSize, ds.RasterXSize, ds.GetGeoTransform()
    out = {f: (np.full((H // f, W // f), np.nan, np.float32), np.zeros((H // f, W // f), np.uint8)) for f in (2, 4)}
    for r0 in range(0, H, 1024):
        h = min(1024, H - r0)
        a = ds.GetRasterBand(1).ReadAsArray(0, r0, W, h).astype(np.float64)
        v = np.isfinite(a)
        a = np.where(v, a, 0.0)
        for f, (db, ok) in out.items():
            n, m = h // f, W // f
            s, c = a.reshape(n, f, m, f).sum((1, 3)), v.reshape(n, f, m, f).sum((1, 3))
            good = c >= MIN_VALID * f * f
            with np.errstate(all="ignore"):
                db[r0 // f:r0 // f + n] = np.where(good, 10 * np.log10(s / np.maximum(c, 1)), np.nan)
            ok[r0 // f:r0 // f + n] = good
    return {f: (db, ok, (gt[0], gt[1] * f, 0.0, gt[3], 0.0, gt[5] * f)) for f, (db, ok) in out.items()}


def train_distribution():
    """Clipped VV dB of E005 train tiles (20 m cache, every 4th pixel in x and y)."""
    meta = pd.read_parquet(e005.C / "meta.parquet")
    rows = meta.index[meta.role == "train"].to_numpy()
    vv, lab = np.load(e005.C / "vv20.npy", mmap_mode="r"), np.load(e005.C / "lab20.npy", mmap_mode="r")
    out, oil = [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]
        s, ok = vv[r, ::4, ::4].astype(np.float32), lab[r, ::4, ::4] != 255
        out.append(s[ok] * 45 + LO)
        o = (meta.cls[r] == "oil").to_numpy()
        oil.append(s[o][ok[o]] * 45 + LO)
    return np.concatenate(out), np.concatenate(oil), meta


def preprocess():
    t0 = time.time()
    for d in (RAS, WORK):
        d.mkdir(parents=True, exist_ok=True)
    assert not P["vv"].exists(), f"{P['vv']} exists; preprocessing already done"
    radar = WORK / "sigma0_vv_radar.tif"
    cal = calibrate(radar)
    cal["check_vs_reference"] = calib_check(radar)
    gt, size = geocode(radar, P["sigma0_native"])
    radar.unlink()  # temporary radar-geometry product; the SAFE itself is never touched
    res = block_average(P["sigma0_native"])
    db20, ok20, gt20 = res[2]
    db40, ok40, gt40 = res[4]
    write(P["vv"], db20, gt20, float("nan"))
    write(P["valid"], ok20, gt20)
    write(P["vv40"], db40, gt40, float("nan"))
    write(P["valid40"], ok40, gt40)

    lat_c = (roi_bounds()[1] + roi_bounds()[3]) / 2
    b = roi_bounds()
    rs, cs = px_window(gt20, b, db20.shape)
    roi, roi_ok = db20[rs, cs], ok20[rs, cs].astype(bool)
    scene_s = db20[::2, ::2][ok20[::2, ::2].astype(bool)]
    tr_all, tr_oil, meta = train_distribution()
    clip = lambda a: np.clip(a, LO, HI)
    stat = lambda a, ok: dict(valid_frac=float(ok.mean()), n_valid=int(ok.sum()), vv_db_unclipped=pct(a[ok]),
                              clip_lo_frac=float((a[ok] < LO).mean()), clip_hi_frac=float((a[ok] > HI).mean()))
    rep = dict(
        safe=Path(SAFE).name, polarisation="VV", ipf="003.52",
        calibration=dict(equation="sigma0_lin = (DN^2 - noiseRangeLut * noiseAzimuthLut) / sigmaNought^2",
                         lut_interpolation="bilinear: along range per LUT vector, then linear between vector lines (clamped)",
                         invalid_rule="DN == 0 (outside swath / IPF border-noise removal) -> NaN / invalid",
                         noise_floor_rule=f"sigma0 <= 0 after noise subtraction -> {FLOOR} (-130 dB), kept valid (training keeps ~-128 dB valid px)",
                         speckle_filter="none (x2 / x4 linear-power block averaging only, as in training)", **cal),
        geocoding=dict(method="GDAL warp, thin-plate spline on annotation GCP grid (ellipsoid heights ~0 m; ocean, no DEM); "
                              "absolute geolocation error of the GCP/TPS product has not been measured", resampling="bilinear in linear power; NaN nodata excluded",
                       crs="EPSG:4326 (WGS84 lon/lat, north-up)", native_pixel_deg=RES, native_geotransform=gt, native_size=list(size)),
        grids={"x2 (E005 input, rasters incident001_vv_db / valid / probability / mask)": dict(
                   geotransform=gt20, size=list(db20.shape), pixel_deg=gt20[1],
                   ground_spacing_m_at_roi_lat=dict(zip(["east_west", "north_south"], ground_spacing(lat_c, gt20[1], -gt20[5])))),
               "x4 (E004 input)": dict(geotransform=gt40, size=list(db40.shape), pixel_deg=gt40[1],
                                       ground_spacing_m_at_roi_lat=dict(zip(["east_west", "north_south"], ground_spacing(lat_c, gt40[1], -gt40[5]))))},
        block_average=f"linear-power mean of valid sub-pixels, block valid if >= {MIN_VALID:.0%} valid (as E002 cache)",
        model_scaling=f"clip dB to [{LO}, {HI}], (dB + 40) / 45, float16, then (x - norm_mu) / norm_sd on valid px, invalid -> 0",
        nodata="vv_db: NaN; valid: 0/1; probability: NaN on invalid; mask: 255 invalid, 1 oil, 0 not oil",
        bounds_lonlat=dict(west=gt20[0], north=gt20[3], east=gt20[0] + gt20[1] * db20.shape[1], south=gt20[3] + gt20[5] * db20.shape[0]),
        roi=dict(definition=f"Cerulean slick bbox +/- {ROI_MARGIN} deg (reporting only)", bounds=b, pixels=[rs.stop - rs.start, cs.stop - cs.start],
                 **stat(roi, roi_ok)),
        scene_x2_every_2nd_px=dict(valid_frac=float(ok20.mean()), vv_db_unclipped=pct(scene_s),
                                   clip_lo_frac=float((scene_s < LO).mean()), clip_hi_frac=float((scene_s > HI).mean())),
        training_comparison=dict(note="diagnostic only; clipped dB [-40, 5] because the cache stores clipped values",
                                 train_all_20m=pct(tr_all), train_oil_tiles_20m=pct(tr_oil),
                                 roi_clipped=pct(clip(roi[roi_ok])), scene_clipped=pct(clip(scene_s)),
                                 train_tile_median_db_by_class={c: pct(meta.vv_p50_db_20[(meta.role == "train") & (meta.cls == c)].dropna().to_numpy())
                                                                for c in ("oil", "lookalike", "no_oil")}),
        runtime_s=time.time() - t0)
    json.dump(rep, open(OUT / "preprocess_report.json", "w"), indent=1)
    fig_preprocess(db40, gt40, db20, gt20, b, tr_all, tr_oil, clip(roi[roi_ok]), clip(scene_s))
    print(json.dumps(rep, indent=1))


def extent(gt, rs, cs):
    return [gt[0] + gt[1] * cs.start, gt[0] + gt[1] * cs.stop, gt[3] + gt[5] * rs.stop, gt[3] + gt[5] * rs.start]


def fig_preprocess(db40, gt40, db20, gt20, b, tr_all, tr_oil, roi, scene):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.5), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    full = (slice(0, db40.shape[0]), slice(0, db40.shape[1]))
    ax[0].imshow(db40, cmap="gray", vmin=-30, vmax=-5, extent=extent(gt40, *full), interpolation="nearest")
    ax[0].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="r", lw=1.5))
    ax[0].set_title("calibrated, geocoded sigma0 VV (x4 grid, dB, -30..-5); red = reporting ROI")
    rs, cs = px_window(gt20, b, db20.shape)
    im = ax[1].imshow(db20[rs, cs], cmap="gray", vmin=-30, vmax=-5, extent=extent(gt20, rs, cs), interpolation="nearest")
    plt.colorbar(im, ax=ax[1], fraction=0.046, label="VV dB")
    ax[1].set_title("ROI, x2 grid (E005 input), dB")
    bins = np.linspace(LO, HI, 91)
    for a, lab in ((tr_all, "train tiles (all)"), (tr_oil, "train oil tiles"), (scene, "Incident scene"), (roi, "Incident ROI")):
        ax[2].hist(a, bins=bins, density=True, histtype="step", lw=1.5, label=lab)
    ax[2].set_xlabel("VV dB (clipped to [-40, 5])")
    ax[2].set_title("VV distribution: training vs incident (diagnostic)")
    ax[2].legend()
    for a in ax[:2]:
        a.set_xlabel("lon")
        a.set_ylabel("lat")
    plt.tight_layout()
    plt.savefig(OUT / "preprocess_vv.png", dpi=90)


# ---------------------------------------------------------------------------------------------------- inference
def tent(n):
    w = (np.minimum(np.arange(n), n - 1 - np.arange(n)) + 1) / (n / 2)
    return np.outer(w, w).astype(np.float32)


def pad_to(a, H, W, fill):
    return np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])), constant_values=fill)


def scaled(db, ok):
    return np.where(ok, (np.clip(db, LO, HI) - LO) / (HI - LO), 0.0).astype(np.float16)


def infer():
    t_all = time.time()
    assert not P["prob"].exists(), f"{P['prob']} exists; inference already done"
    fz = verify_frozen()
    ck5 = torch.load(e005.ROOT / "results/E005_segformer_b2/best.pt", map_location="cuda", weights_only=False)
    cfg5 = ck5["cfg"]
    assert ck5["epoch"] == fz["best_epoch"] == 9 and fz["threshold"] == THR
    e004_sha = sha256(E004_CK)
    e004_git = git("rev-parse", f"HEAD:{E004_CK.relative_to(ROOT)}") == git("hash-object", str(E004_CK.relative_to(ROOT)))
    assert e004_git, "E004 checkpoint differs from the committed one"
    ck4 = torch.load(E004_CK, map_location="cuda", weights_only=False)
    cfg4 = ck4["cfg"]
    e005.seed_all(cfg5["seed"])
    m5 = e005.build_model(cfg5, None)
    m5.load_state_dict(ck5["state_dict"])
    m5.eval()
    m4 = e004.build_model(cfg4, pretrained=False)
    m4.load_state_dict(ck4["state_dict"])
    m4.eval()
    torch.cuda.reset_peak_memory_stats()

    db, gt = read(P["vv"])
    ok = read(P["valid"])[0].astype(bool)
    db4, ok4 = read(P["vv40"])[0], read(P["valid40"])[0].astype(bool)
    H, W = db.shape
    ny, nx = max(1, math.ceil((H - WIN) / STRIDE) + 1), max(1, math.ceil((W - WIN) / STRIDE) + 1)
    Hp, Wp = (ny - 1) * STRIDE + WIN, (nx - 1) * STRIDE + WIN
    s20, v20 = pad_to(scaled(db, ok), Hp, Wp, 0), pad_to(ok, Hp, Wp, False)  # padding = invalid (validity 0, VV 0)
    s40, v40 = pad_to(scaled(db4, ok4), Hp // 2, Wp // 2, 0), pad_to(ok4, Hp // 2, Wp // 2, False)
    dbp = pad_to(db, Hp, Wp, np.nan)
    acc, wsum, wt = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32), tent(WIN)
    wins = [(iy, ix) for iy in range(ny) for ix in range(nx) if v20[iy * STRIDE:iy * STRIDE + WIN, ix * STRIDE:ix * STRIDE + WIN].any()]
    b = roi_bounds()
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(wins), 4):
            batch = wins[k:k + 4]
            sl = [(slice(iy * STRIDE, iy * STRIDE + WIN), slice(ix * STRIDE, ix * STRIDE + WIN)) for iy, ix in batch]
            sl4 = [(slice(iy * STRIDE4, iy * STRIDE4 + WIN4), slice(ix * STRIDE4, ix * STRIDE4 + WIN4)) for iy, ix in batch]
            v = torch.from_numpy(np.stack([s20[s] for s in sl])).cuda().float()
            m = torch.from_numpy(np.stack([v20[s] for s in sl])).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(m5(e005.make_input(v, m, cfg5["norm_mu"], cfg5["norm_sd"]))[:, 0].float()).cpu().numpy()
            v4 = torch.from_numpy(np.stack([s40[s] for s in sl4])).cuda().float()
            m4_ = torch.from_numpy(np.stack([v40[s] for s in sl4])).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pc = m4(e004.make_input(v4, m4_, cfg4["norm_mu"], cfg4["norm_sd"])).float().softmax(1).cpu().numpy()
            for j, ((iy, ix), (rs, cs)) in enumerate(zip(batch, sl)):
                acc[rs, cs] += p[j] * wt
                wsum[rs, cs] += wt
                vj = v20[rs, cs]
                pv = p[j][vj]
                lon0, lat1 = gt[0] + gt[1] * cs.start, gt[3] + gt[5] * rs.start
                lon1, lat0 = gt[0] + gt[1] * cs.stop, gt[3] + gt[5] * rs.stop
                rows.append(dict(iy=iy, ix=ix, row0=rs.start, col0=cs.start, lon0=lon0, lat0=lat0, lon1=lon1, lat1=lat1,
                                 overlaps_roi=bool(lon0 < b[2] and lon1 > b[0] and lat0 < b[3] and lat1 > b[1]),
                                 valid_frac=float(vj.mean()), vv_median_db=float(np.median(dbp[rs, cs][vj])),
                                 prob_p50=float(np.percentile(pv, 50)), prob_p90=float(np.percentile(pv, 90)),
                                 prob_p99=float(np.percentile(pv, 99)), prob_max=float(pv.max()),
                                 frac_prob_ge_0_5=float((pv >= 0.5).mean()), window_pred_frac_t060=float((pv >= THR).mean()),
                                 **{f"e004_p_{c}": float(pc[j, i]) for i, c in enumerate(CLASSES)},
                                 e004_pred=CLASSES[int(pc[j].argmax())]))
    infer_s = time.time() - t0
    prob = np.where(ok, acc[:H, :W] / np.maximum(wsum[:H, :W], 1e-12), np.nan).astype(np.float32)
    assert (wsum[:H, :W][ok] > 0).all()
    mask = np.where(ok, (prob >= THR).astype(np.uint8), np.uint8(255)).astype(np.uint8)
    write(P["prob"], prob, gt, float("nan"))
    write(P["mask"], mask, gt, 255)
    Wd = pd.DataFrame(rows)
    meta = pd.read_parquet(e005.C / "meta.parquet")
    tr = meta[meta.role == "train"]
    for c in ("oil", "lookalike", "no_oil"):  # where each window's median VV falls among train tiles of that class (diagnostic)
        ref = np.sort(tr.vv_p50_db_20[tr.cls == c].dropna().to_numpy())
        Wd[f"vv_median_pctile_in_train_{c}"] = np.searchsorted(ref, Wd.vv_median_db) / len(ref)
    Wd["blended_pred_frac_t060"] = [float((mask[r.row0:r.row0 + WIN, r.col0:r.col0 + WIN] == 1).sum()
                                          / max((mask[r.row0:r.row0 + WIN, r.col0:r.col0 + WIN] != 255).sum(), 1)) for r in Wd.itertuples()]
    Wd.to_parquet(OUT / "windows.parquet", index=False)
    pv = prob[ok]
    rep = dict(e005=dict(checkpoint="results/E005_segformer_b2/best.pt", sha256_verified_against="results/E005_segformer_b2/FROZEN.json",
                         sha256=fz["sha256"]["results/E005_segformer_b2/best.pt"]["sha256"], epoch=ck5["epoch"], threshold=THR,
                         norm_mu=cfg5["norm_mu"], norm_sd=cfg5["norm_sd"], amp="bf16 autocast"),
               e004=dict(checkpoint=str(E004_CK.relative_to(ROOT)), sha256=e004_sha, matches_git_head=e004_git, epoch=ck4["epoch"],
                         classes=list(CLASSES), use="reported per window only; NOT used to gate / suppress segmentation",
                         input="x4 grid 512x512 windows = the same footprint as each E005 window"),
               tiling=dict(grid="x2 (E005 input grid)", window_px=WIN, stride_px=STRIDE, overlap=1 - STRIDE / WIN, n_windows_y=ny, n_windows_x=nx,
                           n_windows_run=len(wins), skipped="windows with no valid pixel",
                           padding=f"scene padded bottom/right to {Hp}x{Wp} with invalid pixels (VV 0, validity 0), as training nodata",
                           blending="probability = sum(w * p) / sum(w) over covering windows, w = separable tent (1/512 .. 1 per axis); "
                                    "threshold applied once to the blended probability"),
               probability_valid_px=dict(n=int(pv.size), **pct(pv), frac_ge_0_5=float((pv >= 0.5).mean()), frac_ge_0_60=float((pv >= THR).mean())),
               mask=dict(oil_px=int((mask == 1).sum()), valid_px=int(ok.sum())),
               roi_windows=Wd[Wd.overlaps_roi].to_dict("records"),
               runtime_s=dict(inference=infer_s, total=time.time() - t_all),
               peak_vram_gb=dict(allocated=torch.cuda.max_memory_allocated() / 2**30, reserved=torch.cuda.max_memory_reserved() / 2**30))
    json.dump(rep, open(OUT / "inference_report.json", "w"), indent=1)
    verify_frozen()
    print(json.dumps({k: v for k, v in rep.items() if k != "roi_windows"}, indent=1))
    print(Wd[Wd.overlaps_roi].drop(columns=["lon0", "lat0", "lon1", "lat1"]).to_string(index=False))


# ---------------------------------------------------------------------------------------------------- vectorisation
def laea(lon0, lat0):
    s = osr.SpatialReference()
    s.ImportFromProj4(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(wgs84(), s)
    inv = osr.CoordinateTransformation(s, wgs84())
    fwd = lambda g: shapely.transform(g, lambda xy: np.array(ct.TransformPoints(xy.tolist()))[:, :2])
    back = lambda g: shapely.transform(g, lambda xy: np.array(inv.TransformPoints(xy.tolist()))[:, :2])
    return fwd, back


def geodesic(g):
    o = ogr.CreateGeometryFromWkb(shapely.to_wkb(g))
    o.AssignSpatialReference(wgs84())
    return o.GeodesicArea(), o.Boundary().GeodesicLength()


def shape_metrics(g_ll, fwd, back):
    """Geodesic area / perimeter on WGS84; axes and orientation from the minimum rotated rectangle in a local LAEA
    (equal-area) projection, orientation corrected to true north at the centroid (azimuth 0-180, clockwise from N)."""
    area, per = geodesic(g_ll)
    g = fwd(g_ll)
    c = g.centroid
    c_ll = back(c)
    n_ll = shapely.Point(c_ll.x, c_ll.y + 0.01)
    n = fwd(n_ll)
    north = math.degrees(math.atan2(n.x - c.x, n.y - c.y))
    r = np.array(shapely.minimum_rotated_rectangle(g).exterior.coords)[:4]
    e = [r[1] - r[0], r[2] - r[1]]
    L = [float(np.hypot(*v)) for v in e]
    k = int(np.argmax(L))
    az = (math.degrees(math.atan2(e[k][0], e[k][1])) - north) % 180
    x0, y0, x1, y1 = g_ll.bounds
    return dict(area_m2=area, perimeter_m=per, centroid_lon=c_ll.x, centroid_lat=c_ll.y, bbox_west=x0, bbox_south=y0, bbox_east=x1,
                bbox_north=y1, major_axis_m=max(L), minor_axis_m=min(L), width_proxy_m=area / max(max(L), 1e-9),
                orientation_deg_from_north=az, elongation=max(L) / max(min(L), 1e-9), compactness_polsby_popper=4 * math.pi * area / per ** 2,
                area_m2_laea=g.area)


def vector():
    t0 = time.time()
    mask, gt = read(P["mask"])
    prob = read(P["prob"])[0]
    oil = mask == 1
    lbl, n = ndimage.label(oil)  # 4-connectivity, as E005 / E006
    mem = gdal.GetDriverByName("MEM").Create("", lbl.shape[1], lbl.shape[0], 1, gdal.GDT_Int32)
    mem.SetGeoTransform(gt)
    mem.SetProjection(wgs84().ExportToWkt())
    band = mem.GetRasterBand(1)
    band.WriteArray(lbl.astype(np.int32))
    vds = ogr.GetDriverByName("Memory").CreateDataSource("m")  # keep referenced while the layer is used
    lyr = vds.CreateLayer("c", wgs84(), ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [])  # default 4-connected: one polygon (with holes) per labelled component
    geoms = {}
    for f in lyr:
        geoms.setdefault(f.GetField("id"), []).append(shapely.from_wkb(bytes(f.GetGeometryRef().ExportToWkb())))
    assert len(geoms) == n, (len(geoms), n)
    ids = np.arange(1, n + 1)
    npx = np.bincount(lbl.ravel(), minlength=n + 1)[1:]
    pmean = ndimage.mean(prob, lbl, ids) if n else []
    pmax = ndimage.maximum(prob, lbl, ids) if n else []
    H, W = mask.shape
    fwd, back = laea(gt[0] + gt[1] * W / 2, gt[3] + gt[5] * H / 2)  # scene-centred LAEA
    b = roi_bounds()
    roi_box = shapely.box(*b)
    recs = []
    for i in ids:
        g = shapely.union_all(geoms[i]) if len(geoms[i]) > 1 else geoms[i][0]
        recs.append(dict(component_id=int(i), n_px_image_space=int(npx[i - 1]), prob_mean=float(pmean[i - 1]), prob_max=float(pmax[i - 1]),
                         in_roi=bool(g.intersects(roi_box)), **shape_metrics(g, fwd, back), geometry=g))
    C = pd.DataFrame(recs)
    feats = [dict(type="Feature", geometry=shapely.geometry.mapping(r.pop("geometry")), properties=r) for r in [dict(x) for x in recs]]
    json.dump(dict(type="FeatureCollection", name="incident001_predicted_components",
                   crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")),
                   description="E007 raw frozen E005 SegFormer mask @0.60, 4-connected components, no filtering. Areas geodesic (WGS84).",
                   features=feats), open(OUT / "incident001_predicted_components.geojson", "w"))
    C.drop(columns="geometry").to_csv(OUT / "predicted_components.csv", index=False)

    # post-hoc Cerulean comparison (reference only; nothing above depends on it)
    cer = cerulean()
    props = json.loads(CERULEAN.read_text())["features"][0]["properties"]
    inc = json.loads((ROOT / "data/incident_001/metadata/incident.json").read_text())
    cm = shape_metrics(cer, fwd, back)
    R = C[C.in_roi]
    cmp = dict(note="POST-HOC diagnostics. The Cerulean polygon did not crop, select, threshold or alter the prediction.",
               cerulean=dict(slick_id=props.get("id"), our_metrics=cm,
                             cerulean_reported=dict(area_m2=inc["area_m2"], length_m=inc["length_m"], perimeter=props.get("perimeter"),
                                                    polsby_popper=props.get("polsby_popper"), machine_confidence=inc["machine_confidence"])),
               roi=dict(bounds=b, n_components=len(R)))
    if len(R):
        cer_m = fwd(cer)
        U_ll = shapely.union_all(list(R.geometry))
        U = fwd(U_ll)
        inter, uni = U.intersection(cer_m).area, U.union(cer_m).area
        um = shape_metrics(U_ll, fwd, back)
        dist = lambda g: fwd(shapely.Point(g["centroid_lon"], g["centroid_lat"])).distance(fwd(shapely.Point(cm["centroid_lon"], cm["centroid_lat"])))
        adiff = lambda a1, a2: abs((a1 - a2 + 90) % 180 - 90)
        cmp["all_roi_predictions_union"] = dict(
            predicted_area_m2=um["area_m2"], reference_area_m2=cm["area_m2"], intersection_m2=inter, union_m2=uni, iou=inter / uni,
            frac_reference_covered=inter / cer_m.area, frac_prediction_inside_reference=inter / U.area,
            centroid_distance_m=dist(um), major_axis_pred_m=um["major_axis_m"], major_axis_ref_m=cm["major_axis_m"],
            orientation_pred_deg=um["orientation_deg_from_north"], orientation_ref_deg=cm["orientation_deg_from_north"],
            orientation_diff_deg=adiff(um["orientation_deg_from_north"], cm["orientation_deg_from_north"]))
        cmp["per_component"] = [dict(component_id=int(r.component_id), area_m2=r.area_m2, n_px_image_space=int(r.n_px_image_space),
                                     prob_mean=r.prob_mean, intersection_with_reference_m2=fwd(r.geometry).intersection(cer_m).area,
                                     centroid_distance_m=dist(r._asdict()), major_axis_m=r.major_axis_m, width_proxy_m=r.width_proxy_m,
                                     orientation_deg=r.orientation_deg_from_north,
                                     orientation_diff_deg=adiff(r.orientation_deg_from_north, cm["orientation_deg_from_north"]),
                                     elongation=r.elongation) for r in R.itertuples()]
    summ = dict(n_components_scene=int(n), n_components_roi=int(len(R)), scene_predicted_area_m2=float(C.area_m2.sum()) if n else 0.0,
                component_area_m2_quantiles=pct(C.area_m2.to_numpy()) if n else None,
                components_by_area_m2={k: int(((C.area_m2 >= lo) & (C.area_m2 < hi)).sum()) for k, lo, hi in
                                       (("<0.01 km2", 0, 1e4), ("0.01-0.1 km2", 1e4, 1e5), ("0.1-1 km2", 1e5, 1e6), (">=1 km2", 1e6, 1e15))} if n else None,
                runtime_s=time.time() - t0)
    cmp["scene_summary"] = summ
    json.dump(cmp, open(OUT / "cerulean_comparison.json", "w"), indent=1, default=float)
    fig_result(C, cer, gt, prob, mask)
    print(json.dumps(cmp, indent=1, default=float))


def fig_result(C, cer, gt, prob, mask):
    db = read(P["vv"])[0]
    b = roi_bounds()
    x0, y0, x1, y1 = cer.bounds
    tight = (x0 - 0.03, y0 - 0.03, x1 + 0.03, y1 + 0.03)
    R = C[C.in_roi] if len(C) else C
    fig, ax = plt.subplots(3, 4, figsize=(22, 17))
    for r, bb in enumerate((b, tight)):
        rs, cs = px_window(gt, bb, db.shape)
        ex = extent(gt, rs, cs)
        ax[r, 0].imshow(db[rs, cs], cmap="gray", vmin=-30, vmax=-5, extent=ex, interpolation="nearest")
        ax[r, 0].set_title("VV sigma0 dB (x2 grid)")
        im = ax[r, 1].imshow(prob[rs, cs], cmap="magma", vmin=0, vmax=1, extent=ex, interpolation="nearest")
        plt.colorbar(im, ax=ax[r, 1], fraction=0.046)
        ax[r, 1].set_title("E005 probability (blended)")
        mk = mask[rs, cs].astype(np.float32)
        mk[mk == 255] = np.nan
        ax[r, 2].imshow(mk, cmap="gray", vmin=0, vmax=1, extent=ex, interpolation="nearest")
        ax[r, 2].set_title(f"raw mask @ {THR:.2f} (white = oil)")
        ax[r, 3].imshow(db[rs, cs], cmap="gray", vmin=-30, vmax=-5, extent=ex, interpolation="nearest")
        for g in R.geometry:
            for p in getattr(g, "geoms", [g]):
                ax[r, 3].plot(*p.exterior.xy, c="r", lw=1.2)
        for p in getattr(cer, "geoms", [cer]):
            ax[r, 3].plot(*p.exterior.xy, c="cyan", lw=1.5, ls="--")
        ax[r, 3].plot([], [], c="r", label="MODEL prediction (E005 frozen, raw @0.60)")
        ax[r, 3].plot([], [], c="cyan", ls="--", label="Cerulean reference (post-hoc only)")
        ax[r, 3].legend(fontsize=8, loc="lower left")
        ax[r, 3].set_title("overlay")
        for a in ax[r]:
            a.set_xlim(ex[0], ex[1])
            a.set_ylim(ex[2], ex[3])
    f = 8  # scene overview: max-pooled probability
    H, W = prob.shape
    p8 = np.nan_to_num(prob[:H // f * f, :W // f * f]).reshape(H // f, f, W // f, f).max((1, 3))
    ex = [gt[0], gt[0] + gt[1] * (W // f * f), gt[3] + gt[5] * (H // f * f), gt[3]]
    ax[2, 0].imshow(p8, cmap="magma", vmin=0, vmax=1, extent=ex, interpolation="nearest")
    ax[2, 0].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="cyan", lw=1))
    ax[2, 0].set_title("whole scene: max-pooled probability (x8), cyan = ROI")
    Wd = pd.read_parquet(OUT / "windows.parquet")
    ax[2, 1].hist(Wd.window_pred_frac_t060, bins=50, log=True)
    ax[2, 1].set_title("per-window predicted oil fraction @0.60 (all windows)")
    ax[2, 2].scatter(Wd.vv_median_db, Wd.e004_p_oil, s=6, c=Wd.overlaps_roi.map({True: "r", False: "grey"}))
    ax[2, 2].set_xlabel("window median VV dB")
    ax[2, 2].set_ylabel("E004 p(oil)")
    ax[2, 2].set_title("E004 per window (red = overlaps ROI); not used for gating")
    if len(C):
        ax[2, 3].hist(np.log10(C.area_m2), bins=50, log=True)
    ax[2, 3].set_xlabel("log10 component area (m2, geodesic)")
    ax[2, 3].set_title("predicted component areas, whole scene")
    plt.tight_layout()
    plt.savefig(OUT / "incident001_ml_result.png", dpi=80)


# ---------------------------------------------------------------------------------------------------- freeze
FROZEN_OUT = ["preprocess_report.json", "inference_report.json", "windows.parquet", "predicted_components.csv",
              "incident001_predicted_components.geojson", "cerulean_comparison.json", "preprocess_vv.png", "incident001_ml_result.png"]


def frozen_files():
    return ([Path(__file__).resolve(), E004_CK, ROOT / "results/E005_segformer_b2/best.pt", ROOT / "results/E005_segformer_b2/FROZEN.json"]
            + [OUT / f for f in FROZEN_OUT] + sorted(RAS.glob("*.tif")))


def freeze():
    dst = OUT / "FROZEN.json"
    assert not dst.exists(), f"{dst} exists; E007 is already frozen"
    verify_frozen()
    pre, inf = json.load(open(OUT / "preprocess_report.json")), json.load(open(OUT / "inference_report.json"))
    cmp_ = json.load(open(OUT / "cerulean_comparison.json"))
    files = frozen_files()
    fz = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="E007 approved as an INTEGRATION milestone (successful real-scene pipeline integration with partial/over-segmented "
               "Incident 001 recovery), not as evidence of robust scene-wide autonomous segmentation. Never overwrite these files.",
        e005=dict(checkpoint="results/E005_segformer_b2/best.pt", epoch=inf["e005"]["epoch"], threshold=THR,
                  norm_mu=inf["e005"]["norm_mu"], norm_sd=inf["e005"]["norm_sd"]),
        e004=dict(checkpoint=str(E004_CK.relative_to(ROOT)), epoch=inf["e004"]["epoch"],
                  use="reported only. NEGATIVE RESULT: the 4 windows holding the reference slick give p(oil) 0.12-0.34, argmax "
                      "lookalike, so E004 MUST NOT gate E005 segmentation (E008)."),
        preprocessing=dict(calibration=pre["calibration"]["equation"], invalid_rule=pre["calibration"]["invalid_rule"],
                           noise_floor_rule=pre["calibration"]["noise_floor_rule"], block_average=pre["block_average"],
                           model_scaling=pre["model_scaling"], clip_db=[LO, HI], min_valid_block_frac=MIN_VALID),
        geocoding=dict(**pre["geocoding"], origin_snap="origin and size on multiples of 4 native px (x2 / x4 grids nest)",
                       grids=pre["grids"], bounds_lonlat=pre["bounds_lonlat"]),
        tiling=dict(window_px=WIN, stride_px=STRIDE, e004_window_px=WIN4, e004_stride_px=STRIDE4, grid=inf["tiling"]["grid"],
                    padding=inf["tiling"]["padding"], blending=inf["tiling"]["blending"]),
        components=dict(connectivity=4, filtering="none", area="geodesic WGS84 (OGR); never pixel_count * 400 m2",
                        shape="minimum rotated rectangle in scene-centred LAEA, orientation from true north"),
        nodata=pre["nodata"],
        measured_post_hoc_diagnostics=dict(note="Cerulean used only post hoc; values must not change",
                                           **{k: cmp_["all_roi_predictions_union"][k] for k in
                                              ["iou", "frac_reference_covered", "frac_prediction_inside_reference",
                                               "predicted_area_m2", "reference_area_m2", "centroid_distance_m"]}),
        sha256={str(f.relative_to(ROOT)): dict(sha256=sha256(f), bytes=f.stat().st_size) for f in files},
        git=dict(head=git("rev-parse", "HEAD"), note="E007 files were committed after this freeze; see PROJECT_CONTEXT.md for the commit"),
    )
    json.dump(fz, open(dst, "w"), indent=1)
    print(f"froze {len(files)} files -> {dst}")


def verify():
    fz = json.load(open(OUT / "FROZEN.json"))
    bad = {k for k, v in fz["sha256"].items() if sha256(ROOT / k) != v["sha256"]}
    assert not bad, f"frozen E007 artefacts changed: {sorted(bad)}"
    print(f"E007 FROZEN verified: {len(fz['sha256'])} files match")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preprocess", "infer", "vector", "freeze", "verify"])
    {"preprocess": preprocess, "infer": infer, "vector": vector, "freeze": freeze, "verify": verify}[ap.parse_args().stage]()
