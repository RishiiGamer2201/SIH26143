"""Minimal Sentinel-1 IW GRD radiometric calibration for a pixel window (radar geometry).

sigma0 = (DN^2 - noise) / A_sigma^2, noise = rangeNoiseLUT * azimuthNoiseLUT (IPF >= 2.9),
LUTs linearly interpolated from the SAFE calibration/noise annotation. Negative results
(noise-subtracted below zero) are returned as NaN, never clipped silently.

Not done here (documented gaps): orbit refinement, terrain/ellipsoid geocoding, border-noise
masking beyond NaN of DN==0, speckle filtering. For the ocean crops this is used on, geocoding
is handled separately via the annotation GCP grid.

Run as a script: labelled VV/VH cross-check on the Incident 001 slick (Phase B evidence).
"""
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")


def _floats(e):
    return np.array(e.text.split(), dtype=np.float64)


def _grid_interp(lines, pixel_rows, values_rows, rows, cols):
    """values given on (line_i, pixel_ij) -> interpolate to rows x cols (separable linear)."""
    along_px = np.stack([np.interp(cols, p, v) for p, v in zip(pixel_rows, values_rows)])
    return np.stack([np.interp(rows, lines, along_px[:, j]) for j in range(len(cols))], axis=1)


def _files(safe, pol):
    pol = pol.lower()
    one = lambda pat: glob.glob(str(Path(safe) / pat))[0]
    return (one(f"measurement/*-{pol}-*.tiff"),
            one(f"annotation/calibration/calibration-*-{pol}-*.xml"),
            one(f"annotation/calibration/noise-*-{pol}-*.xml"))


def sigma0(safe, pol, row0, col0, h, w, denoise=True):
    tif, cal_xml, noise_xml = _files(safe, pol)
    ds = gdal.Open(tif)
    dn = ds.GetRasterBand(1).ReadAsArray(col0, row0, w, h).astype(np.float64)
    rows, cols = np.arange(row0, row0 + h), np.arange(col0, col0 + w)

    cal = ET.parse(cal_xml).getroot().findall(".//calibrationVector")
    A = _grid_interp(np.array([int(v.find("line").text) for v in cal]),
                     [_floats(v.find("pixel")) for v in cal],
                     [_floats(v.find("sigmaNought")) for v in cal], rows, cols)
    power = dn ** 2
    if denoise:
        nz = ET.parse(noise_xml).getroot()
        rv = nz.findall(".//noiseRangeVector")
        noise = _grid_interp(np.array([int(v.find("line").text) for v in rv]),
                             [_floats(v.find("pixel")) for v in rv],
                             [_floats(v.find("noiseRangeLut")) for v in rv], rows, cols)
        az = np.ones_like(noise)
        for b in nz.findall(".//noiseAzimuthVector"):
            l0, l1 = int(b.find("firstAzimuthLine").text), int(b.find("lastAzimuthLine").text)
            s0, s1 = int(b.find("firstRangeSample").text), int(b.find("lastRangeSample").text)
            rsel = (rows >= l0) & (rows <= l1)
            csel = (cols >= s0) & (cols <= s1)
            if rsel.any() and csel.any():
                lut = np.interp(rows[rsel], _floats(b.find("line")), _floats(b.find("noiseAzimuthLut")))
                az[np.ix_(rsel, csel)] = lut[:, None]
        power = power - noise * az
    s0 = power / A ** 2
    s0[(dn == 0) | (s0 <= 0)] = np.nan
    return s0


def gcp_transformer(safe, pol="vv"):
    tif, _, _ = _files(safe, pol)
    ds = gdal.Open(tif)
    return ds, gdal.Transformer(ds, None, ["METHOD=GCP_TPS"])


if __name__ == "__main__":
    import json
    from shapely import contains_xy
    from shapely.geometry import shape

    ROOT = Path(__file__).resolve().parents[1]
    SAFE = glob.glob(str(ROOT / "data/incident_001/sentinel1/extracted/*.SAFE"))[0]
    slick = shape(json.loads((ROOT / "data/incident_001/metadata/slick_3775938.geojson").read_text())["features"][0]["geometry"])

    ds, tr = gcp_transformer(SAFE)
    # lon/lat -> pixel (dst->src)
    corners = [tr.TransformPoint(1, x, y)[1][:2] for x, y in slick.envelope.exterior.coords]
    px, py = np.array(corners).T
    M = 1000  # ~10 km margin of clean sea
    col0, row0 = int(max(px.min() - M, 0)), int(max(py.min() - M, 0))
    w = int(min(px.max() + M, ds.RasterXSize) - col0)
    h = int(min(py.max() + M, ds.RasterYSize) - row0)
    print("crop rows", row0, row0 + h, "cols", col0, col0 + w)

    rr, cc = np.mgrid[row0:row0 + h, col0:col0 + w]
    pts = np.column_stack([cc.ravel() + 0.5, rr.ravel() + 0.5])
    ll = np.array(tr.TransformPoints(0, pts)[0])[:, :2]
    inside = contains_xy(slick, ll[:, 0], ll[:, 1]).reshape(h, w)
    print("slick px (Cerulean polygon, validation only):", int(inside.sum()))

    out = {}
    for pol in ("vv", "vh"):
        for dn in (True, False):
            db = 10 * np.log10(sigma0(SAFE, pol, row0, col0, h, w, denoise=dn))
            sea = db[~inside & np.isfinite(db)]
            oil = db[inside & np.isfinite(db)]
            med = np.median(sea)
            mad = np.median(np.abs(sea - med)) * 1.4826
            tgt = sea[sea - med >= 12]
            k = f"{pol}_{'denoised' if dn else 'raw'}"
            out[k] = dict(sea_med=round(float(med), 2), sea_mad=round(float(mad), 2),
                          p01=round(float(np.percentile(sea, 1)), 2), p99=round(float(np.percentile(sea, 99)), 2),
                          oil_contrast_db=round(float(med - np.median(oil)), 2),
                          oil_cnr=round(float((med - np.median(oil)) / mad), 2),
                          nan_frac=round(float(np.isnan(db).mean()), 4), n_target_px=int(tgt.size))
            print(k, out[k])
    dst = ROOT / "results/sar_audit/incident001_labelled_pol_check.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print("saved", dst)
