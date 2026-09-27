"""E008b: forcing-domain expansion (forcing v2) with the SAME products, variables, time range, resolution and processing
recipe as forcing v1. Physics parameters are not touched. Forcing v1 files are never modified.

  python scripts/e008b_prepare_forcing_v2.py provenance   # rebuild v1 from its raw files with build_ocean/build_wind, compare
  python scripts/e008b_prepare_forcing_v2.py config       # expanded domain from E008a trajectories (written BEFORE download)
  python scripts/e008b_prepare_forcing_v2.py download     # CMEMS SMOC + ERA5 raw files for the v2 domain (base env: cdsapi)
  python scripts/e008b_prepare_forcing_v2.py build        # v2 OpenDrift files with the identical recipe
  python scripts/e008b_prepare_forcing_v2.py validate     # common-domain equivalence v1 vs v2 + schema + OpenDrift reader check
  python scripts/e008b_prepare_forcing_v2.py freeze       # forcing_v2/FROZEN.json (only if the predeclared criteria hold)

Domain bounds come only from the E008a deterministic candidate trajectories (all 42). No reference-slick geometry,
candidate order, vessel data or reference agreement is read.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/home/admin_wsl/SIH26143")
V1_OCEAN_RAW = ROOT / "data/incident_001/currents/ocean_surface_velocity.nc"
V1_OCEAN = ROOT / "data/incident_001/currents/ocean_opendrift.nc"
V1_WIND_RAW = ROOT / "data/incident_001/wind/era5_wind_10m.nc"
V1_WIND = ROOT / "data/incident_001/wind/era5_wind_10m_151h.nc"
V1_WIND_DOWNLOAD = ROOT / "data/incident_001/wind/download_era5_wind.py"
V1_DESCRIPTION = ROOT / "data/incident_001/currents/dataset_description.json"
V2 = ROOT / "data/incident_001/forcing_v2"
V2_OCEAN_RAW = V2 / "raw/ocean_surface_velocity.nc"
V2_WIND_RAW = V2 / "raw/era5_wind_10m.nc"
V2_OCEAN = V2 / "ocean_opendrift.nc"
V2_WIND = V2 / "era5_wind_10m_151h.nc"
V2_DOWNLOAD_META = V2 / "download_metadata.json"
OUT = ROOT / "results/E008b_forcing_expansion"
PROVENANCE = OUT / "provenance_audit.json"
CONFIG = OUT / "forcing_expansion_config.json"
VALIDATION = OUT / "forcing_overlap_validation.json"
E008A = ROOT / "results/E008a_candidate_hindcast"
E008B = ROOT / "results/E008b_candidate_hindcast_v2"
HINDCAST_SCRIPT = ROOT / "scripts/e008b_candidate_hindcast_v2.py"

SMOC_DATASET = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
SMOC_VERSION = "202211"
SMOC_VARIABLES = ["uo", "vo", "utide", "vtide", "vsdx", "vsdy", "utotal", "vtotal"]  # = all variables of the v1 raw file
BUFFER_DEG = 2.0
SNAP_DEG = 0.25  # ERA5 native step; every 0.25 deg node is also a node of the SMOC 1/12 deg grid (-180 + k/12)
OCEAN_VARS = ["x_sea_water_velocity", "y_sea_water_velocity", "sea_surface_wave_stokes_drift_x_velocity",
              "sea_surface_wave_stokes_drift_y_velocity"]
WIND_VARS = ["u10", "v10"]
FORBIDDEN_TOKENS = ["slick_" + "3775938", "ceru" + "lean", "in_" + "roi", "incident" + ".json", "ais" + "_", "rank" + "ing"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(obj, p, overwrite=False):
    p = Path(p)
    assert overwrite or not p.exists(), f"{p} exists; not overwritten"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def rel(p):
    return str(Path(p).relative_to(ROOT))


# ------------------------------------------------------------------ the forcing recipe (reconstructed from v1, see provenance)
def build_ocean(raw_path, out):
    """SMOC -> OpenDrift ocean file. Water current = uo+utide, vo+vtide (NOT utotal); Stokes = vsdx/vsdy kept separately.
    Coordinate order time, depth, latitude, longitude; netCDF4 engine; attrs replaced wholesale."""
    raw = xr.open_dataset(raw_path)
    A = lambda sn, ln: dict(standard_name=sn, long_name=ln, units="m s-1")
    x = raw.uo + raw.utide
    x.attrs = A("x_sea_water_velocity", "Eastward ocean circulation plus tidal velocity")
    y = raw.vo + raw.vtide
    y.attrs = A("y_sea_water_velocity", "Northward ocean circulation plus tidal velocity")
    sx = raw.vsdx.copy()
    sx.attrs = A("sea_surface_wave_stokes_drift_x_velocity", "Eastward Stokes drift")
    sy = raw.vsdy.copy()
    sy.attrs = A("sea_surface_wave_stokes_drift_y_velocity", "Northward Stokes drift")
    ds = xr.Dataset(coords={c: raw[c] for c in ["time", "depth", "latitude", "longitude"]}).assign(
        {"x_sea_water_velocity": x, "y_sea_water_velocity": y,
         "sea_surface_wave_stokes_drift_x_velocity": sx, "sea_surface_wave_stokes_drift_y_velocity": sy})
    ds.attrs = {"source": "Derived from Copernicus Marine SMOC: uo+utide as water current; vsdx/vsdy retained separately."}
    ds.to_netcdf(out)
    raw.close()


def build_wind(raw_path, out):
    """ERA5 168 h file -> first 151 valid_times (2023-01-01T00 .. 2023-01-07T06, = ocean time axis); h5netcdf engine."""
    w = xr.open_dataset(raw_path)
    w.isel(valid_time=slice(0, 151)).to_netcdf(out, engine="h5netcdf")
    w.close()


def header(p):
    r = subprocess.run(["ncdump", "-hs", str(p)], capture_output=True, text=True, check=True).stdout.splitlines()
    return r[1:]  # first line carries the file name


def ncprops(p):
    return [l.strip() for l in header(p) if "_NCProperties" in l]


def values_equal(a, b, names):
    A, B = xr.open_dataset(a), xr.open_dataset(b)
    r = {n: bool(np.array_equal(A[n].values, B[n].values, equal_nan=A[n].dtype.kind == "f")) for n in names}
    A.close(), B.close()
    return r


# ------------------------------------------------------------------ stage 1: provenance
def provenance():
    src = Path(__file__).read_text().lower()
    hits = [t for t in FORBIDDEN_TOKENS if t in src]
    res = {}
    with tempfile.TemporaryDirectory() as T:
        for tag, fn, raw, v1, names in [("ocean", build_ocean, V1_OCEAN_RAW, V1_OCEAN, None),
                                        ("wind", build_wind, V1_WIND_RAW, V1_WIND, None)]:
            p = Path(T) / f"{tag}.nc"
            fn(raw, p)
            names = list(xr.open_dataset(v1).variables)
            hv, hr = header(v1), header(p)
            diff = [(a, b) for a, b in zip(hv, hr) if a != b]
            res[tag] = dict(raw=rel(raw), raw_sha256=sha256(raw), v1=rel(v1), v1_sha256=sha256(v1), rebuilt_sha256=sha256(p),
                            byte_identical=sha256(p) == sha256(v1), header_lines_equal_len=len(hv) == len(hr),
                            header_differences=diff, values_bit_exact=values_equal(v1, p, names))
            nc_only = all("_NCProperties" in a for a, _ in diff)
            res[tag]["reproduced"] = bool(res[tag]["byte_identical"] or (
                res[tag]["header_lines_equal_len"] and nc_only and all(res[tag]["values_bit_exact"].values())))
    desc = json.load(open(V1_DESCRIPTION))["products"][0]
    ds0 = desc["datasets"][0]
    raw = xr.open_dataset(V1_OCEAN_RAW)
    w = xr.open_dataset(V1_WIND_RAW)
    audit = dict(
        forbidden_token_hits=hits,
        ocean=dict(
            product_id=desc["product_id"], dataset_id=ds0["dataset_id"], dataset_version=ds0["versions"][0]["label"],
            dataset_name=ds0["dataset_name"], description_file=rel(V1_DESCRIPTION), description_sha256=sha256(V1_DESCRIPTION),
            raw_global_attrs=dict(raw.attrs), raw_variables=list(raw.data_vars),
            raw_time=[str(raw.time.values[0]), str(raw.time.values[-1]), int(raw.sizes["time"])],
            raw_depth=[float(d) for d in raw.depth.values],
            raw_lon=[float(raw.longitude.values[0]), float(raw.longitude.values[-1]), int(raw.sizes["longitude"])],
            raw_lat=[float(raw.latitude.values[0]), float(raw.latitude.values[-1]), int(raw.sizes["latitude"])],
            download_tool=f"copernicusmarine {raw.attrs.get('copernicusmarine_version')} subset (command not preserved; "
                          "request reconstructed from the file)",
            reconstructed_request=dict(dataset_id=ds0["dataset_id"], variables="all 8 (= " + ",".join(SMOC_VARIABLES) + ")",
                                       start_datetime="2023-01-01T00:00:00", end_datetime="2023-01-07T06:00:00",
                                       bbox_same_as_era5_area=True, depth="not requested (dataset has one level, 0.494025 m)"),
            processing=build_ocean.__doc__.strip(), rebuild=res["ocean"]),
        wind=dict(
            dataset="reanalysis-era5-single-levels (product_type reanalysis)", download_script=rel(V1_WIND_DOWNLOAD),
            download_script_sha256=sha256(V1_WIND_DOWNLOAD), download_script_text=V1_WIND_DOWNLOAD.read_text(),
            raw_valid_time=[str(w.valid_time.values[0]), str(w.valid_time.values[-1]), int(w.sizes["valid_time"])],
            raw_lon=[float(w.longitude.values[0]), float(w.longitude.values[-1]), int(w.sizes["longitude"])],
            raw_lat=[float(w.latitude.values[0]), float(w.latitude.values[-1]), int(w.sizes["latitude"])],
            raw_global_attrs=dict(w.attrs), processing=build_wind.__doc__.strip(), rebuild=res["wind"]),
        v1_request_bbox=dict(
            west=-92.477933, south=25.106677, east=-88.379742, north=29.139636,
            origin="download_era5_wind.py area; SMOC raw grid extent is exactly the SMOC nodes inside this box. The box equals "
                   "the reference slick polygon bounds +- 2.0 deg (reference-derived; recorded for provenance only; forcing v2 "
                   "bounds are NOT derived from it)"),
        library_note="Wind v1 was written by h5netcdf 1.8.1 with HDF5 2.0.0; the current env has HDF5 1.14.6, so the "
                     "_NCProperties provenance string is the only byte-level difference of the rebuilt wind file.",
    )
    raw.close(), w.close()
    audit["provenance_reproduced"] = bool(res["ocean"]["reproduced"] and res["wind"]["reproduced"] and not hits)
    dump(audit, PROVENANCE)
    print(json.dumps(dict(ocean=res["ocean"], wind=res["wind"], reproduced=audit["provenance_reproduced"], hits=hits),
                     indent=1, default=str))


# ------------------------------------------------------------------ stage 2: domain (before any download)
def config():
    assert json.load(open(PROVENANCE))["provenance_reproduced"], "provenance not reproduced; stop"
    assert not V2.exists(), "forcing_v2 already exists; the config must be written before any download"
    lo = [np.inf, np.inf]
    hi = [-np.inf, -np.inf]
    files = {}
    dead = {}
    cands = sorted([p for p in E008A.iterdir() if p.is_dir() and p.name.startswith("E007-C")], key=lambda p: int(p.name[6:]))
    assert len(cands) == 42
    for c in cands:
        d = xr.open_dataset(c / "hindcast_48h.nc")
        lon, lat = d.lon.values, d.lat.values
        ok = np.isfinite(lon) & np.isfinite(lat)
        lo = [min(lo[0], lon[ok].min()), min(lo[1], lat[ok].min())]
        hi = [max(hi[0], lon[ok].max()), max(hi[1], lat[ok].max())]
        d.close()
        files[c.name] = sha256(c / "hindcast_48h.nc")
        g = json.load(open(c / "coverage_diagnostics.json"))
        if g["n_deactivated"]:  # in-memory deactivation positions (not in the file when the run stopped early)
            dead[c.name] = dict(lon=g["deactivation_lon_range"], lat=g["deactivation_lat_range"])
            lo = [min(lo[0], g["deactivation_lon_range"][0]), min(lo[1], g["deactivation_lat_range"][0])]
            hi = [max(hi[0], g["deactivation_lon_range"][1]), max(hi[1], g["deactivation_lat_range"][1])]
    env = dict(west=float(lo[0]), south=float(lo[1]), east=float(hi[0]), north=float(hi[1]))
    buf = dict(west=env["west"] - BUFFER_DEG, south=env["south"] - BUFFER_DEG, east=env["east"] + BUFFER_DEG, north=env["north"] + BUFFER_DEG)
    snap = dict(west=float(np.floor(buf["west"] / SNAP_DEG) * SNAP_DEG), south=float(np.floor(buf["south"] / SNAP_DEG) * SNAP_DEG),
                east=float(np.ceil(buf["east"] / SNAP_DEG) * SNAP_DEG), north=float(np.ceil(buf["north"] / SNAP_DEG) * SNAP_DEG))
    n12 = lambda a, b: int(round((b - a) * 12)) + 1
    n4 = lambda a, b: int(round((b - a) * 4)) + 1
    cfg = dict(
        experiment="E008b forcing-domain expansion (forcing v2)",
        bounds_source="envelope of all finite positions of all 42 E008a deterministic candidate trajectories (every output "
                      "time, incl. C12/C14) plus in-memory deactivation positions; nothing else",
        not_used=["reference slick geometry", "candidate rank", "vessel data", "reference agreement"],
        e008a_hindcast_sha256=files, deactivation_ranges_included=dead,
        trajectory_envelope=env, buffer_deg=BUFFER_DEG, buffered_box=buf,
        snapping=f"outward to multiples of {SNAP_DEG} deg (ERA5 native grid; also nodes of the SMOC -180+k/12 grid), so the "
                 "requested bounds are identical for both products and lie on both native grids",
        requested_bounds=snap,
        expected_native=dict(
            smoc=dict(lon=[snap["west"], snap["east"]], lat=[snap["south"], snap["north"]], n_lon=n12(snap["west"], snap["east"]),
                      n_lat=n12(snap["south"], snap["north"]), step="1/12 deg"),
            era5=dict(lon=[snap["west"], snap["east"]], lat=[snap["north"], snap["south"]], n_lon=n4(snap["west"], snap["east"]),
                      n_lat=n4(snap["south"], snap["north"]), step="0.25 deg")),
        v1_bounds=dict(smoc=dict(lon=[-92.416664, -88.416664], lat=[25.166666, 29.083334]), era5=dict(lon=[-92.25, -88.5], lat=[29.0, 25.25])),
        smoc_request=dict(dataset_id=SMOC_DATASET, dataset_version=SMOC_VERSION, variables=SMOC_VARIABLES,
                          start_datetime="2023-01-01T00:00:00", end_datetime="2023-01-07T06:00:00",
                          minimum_longitude=snap["west"], maximum_longitude=snap["east"], minimum_latitude=snap["south"],
                          maximum_latitude=snap["north"], depth="not requested (as v1)", coordinates_selection_method="default (as v1)"),
        era5_request=dict(dataset="reanalysis-era5-single-levels", product_type=["reanalysis"],
                          variable=["10m_u_component_of_wind", "10m_v_component_of_wind"], year=["2023"], month=["01"],
                          day=["01", "02", "03", "04", "05", "06", "07"], time=[f"{h:02d}:00" for h in range(24)],
                          area=[snap["north"], snap["west"], snap["south"], snap["east"]], data_format="netcdf",
                          download_format="unarchived"),
        temporal_domain="identical to v1: ocean 2023-01-01T00..2023-01-07T06 hourly (151); wind 168 h downloaded, first 151 kept",
        recipe=dict(ocean=build_ocean.__doc__.strip(), wind=build_wind.__doc__.strip()),
        outputs=dict(ocean_raw=rel(V2_OCEAN_RAW), wind_raw=rel(V2_WIND_RAW), ocean=rel(V2_OCEAN), wind=rel(V2_WIND)),
        predeclared_equivalence_criterion=(
            "PASS only if, on the common domain (coordinate values present in both files, no resampling), every one of the 4 "
            "ocean and 2 wind variables is exactly equal wherever finite in both AND the NaN masks agree everywhere AND the "
            "common coordinates are identical. Any other outcome = STOP (no tolerance)."),
        predeclared_freeze_criterion=("equivalence PASS AND 42/42 runs completed AND zero active particle-hours outside either "
                                      "reader and zero deactivations outside either reader in all 42 runs"),
    )
    dump(cfg, CONFIG)
    print(json.dumps({k: cfg[k] for k in ("trajectory_envelope", "buffered_box", "requested_bounds", "expected_native")}, indent=1))


# ------------------------------------------------------------------ stage 3: download (run with the base env python: cdsapi)
def download():
    import copernicusmarine
    import cdsapi
    cfg = json.load(open(CONFIG))
    assert not V2_OCEAN_RAW.exists() and not V2_WIND_RAW.exists(), "raw v2 files exist; not overwritten"
    V2_OCEAN_RAW.parent.mkdir(parents=True, exist_ok=True)
    s = cfg["smoc_request"]
    r = copernicusmarine.subset(
        dataset_id=s["dataset_id"], dataset_version=s["dataset_version"], variables=s["variables"],
        minimum_longitude=s["minimum_longitude"], maximum_longitude=s["maximum_longitude"],
        minimum_latitude=s["minimum_latitude"], maximum_latitude=s["maximum_latitude"],
        start_datetime=s["start_datetime"], end_datetime=s["end_datetime"],
        output_directory=str(V2_OCEAN_RAW.parent), output_filename=V2_OCEAN_RAW.name, disable_progress_bar=True)
    smoc_resp = r.model_dump() if hasattr(r, "model_dump") else str(r)
    e = dict(cfg["era5_request"])
    ds_name = e.pop("dataset")
    cdsapi.Client().retrieve(ds_name, e, str(V2_WIND_RAW))
    meta = dict(config_sha256=sha256(CONFIG), smoc_request=s, smoc_response=smoc_resp,
                copernicusmarine_version=copernicusmarine.__version__, era5_request=cfg["era5_request"],
                cdsapi_version=getattr(cdsapi, "__version__", None),
                raw_sha256={rel(V2_OCEAN_RAW): sha256(V2_OCEAN_RAW), rel(V2_WIND_RAW): sha256(V2_WIND_RAW)},
                python=sys.version)
    dump(meta, V2_DOWNLOAD_META)
    print(json.dumps(meta, indent=1, default=str)[:3000])


# ------------------------------------------------------------------ stage 4: build
def build():
    cfg = json.load(open(CONFIG))
    assert json.load(open(V2_DOWNLOAD_META))["config_sha256"] == sha256(CONFIG)
    assert not V2_OCEAN.exists() and not V2_WIND.exists(), "v2 forcing exists; not overwritten"
    raw, w = xr.open_dataset(V2_OCEAN_RAW), xr.open_dataset(V2_WIND_RAW)
    ex = cfg["expected_native"]
    got = dict(smoc=dict(lon=[float(raw.longitude[0]), float(raw.longitude[-1])], lat=[float(raw.latitude[0]), float(raw.latitude[-1])],
                         n_lon=raw.sizes["longitude"], n_lat=raw.sizes["latitude"]),
               era5=dict(lon=[float(w.longitude[0]), float(w.longitude[-1])], lat=[float(w.latitude[0]), float(w.latitude[-1])],
                         n_lon=w.sizes["longitude"], n_lat=w.sizes["latitude"]))
    print(json.dumps(got))
    for k in ("smoc", "era5"):
        assert got[k]["n_lon"] == ex[k]["n_lon"] and got[k]["n_lat"] == ex[k]["n_lat"], (k, got[k], ex[k])
        assert np.allclose(got[k]["lon"], ex[k]["lon"], atol=1e-4) and np.allclose(got[k]["lat"], ex[k]["lat"], atol=1e-4), (k, got[k])
    assert raw.sizes["time"] == 151 and w.sizes["valid_time"] == 168
    assert list(raw.data_vars) == json.load(open(PROVENANCE))["ocean"]["raw_variables"]
    raw.close(), w.close()
    build_ocean(V2_OCEAN_RAW, V2_OCEAN)
    build_wind(V2_WIND_RAW, V2_WIND)
    print(rel(V2_OCEAN), sha256(V2_OCEAN), rel(V2_WIND), sha256(V2_WIND))


# ------------------------------------------------------------------ stage 5: validation
def overlap(v1p, v2p, names, dims):
    A, B = xr.open_dataset(v1p), xr.open_dataset(v2p)
    common = {}
    for d in dims:
        a, b = A[d].values, B[d].values
        c = np.intersect1d(a, b)  # exact coordinate values; no resampling / nearest matching
        common[d] = dict(n_v1=len(a), n_v2=len(b), n_common=len(c), v1_fully_contained=bool(len(c) == len(a)),
                         v1_first_last=[str(a[0]), str(a[-1])], v2_first_last=[str(b[0]), str(b[-1])])
        A, B = A.sel({d: c}), B.sel({d: c})
    stats = {}
    for n in names:
        a, b = A[n].values.astype("f8"), B[n].values.astype("f8")
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        d = np.abs(a[both] - b[both])
        stats[n] = dict(n_compared=int(a.size), n_finite_both=int(both.sum()),
                        exact_equality_fraction=float((d == 0).mean()) if d.size else None,
                        max_abs_diff=float(d.max()) if d.size else None, mean_abs_diff=float(d.mean()) if d.size else None,
                        rmse=float(np.sqrt((d ** 2).mean())) if d.size else None,
                        nan_mask_disagreement=int((fa != fb).sum()), n_nan_v1=int((~fa).sum()), n_nan_v2=int((~fb).sum()),
                        bitwise_equal_float32=bool(np.array_equal(A[n].values, B[n].values, equal_nan=True)))
    A.close(), B.close()
    return common, stats


def schema(p):
    d = xr.open_dataset(p)
    enc_keys = ("dtype", "_FillValue", "least_significant_digit", "units", "calendar")
    r = {k: dict(dims=list(v.dims), dtype=str(v.dtype), attrs={a: str(x) for a, x in v.attrs.items()},
                 encoding={e: str(v.encoding.get(e)) for e in enc_keys}) for k, v in d.variables.items()}
    tm = "time" if "time" in d else "valid_time"
    extra = dict(time=[str(t) for t in d[tm].values], depth=[float(x) for x in d.depth.values] if "depth" in d else None,
                 global_attrs={a: str(x) for a, x in d.attrs.items()})
    d.close()
    return r, extra


def reader_check():
    from opendrift.readers import reader_netCDF_CF_generic
    from opendrift.models.openoil import OpenOil
    out = {}
    for tag, p1, p2 in [("ocean", V1_OCEAN, V2_OCEAN), ("wind", V1_WIND, V2_WIND)]:
        r = {}
        for ver, p in (("v1", p1), ("v2", p2)):
            R = reader_netCDF_CF_generic.Reader(str(p))
            r[ver] = dict(variables=sorted(R.variables), variable_mapping={k: str(v) for k, v in sorted(R.variable_mapping.items())},
                          proj=str(R.proj4), delta_x=R.delta_x, delta_y=R.delta_y, time_step=str(R.time_step),
                          start_time=str(R.start_time), end_time=str(R.end_time), zmin=R.zmin, zmax=R.zmax,
                          xmin=float(R.xmin), xmax=float(R.xmax), ymin=float(R.ymin), ymax=float(R.ymax))
        same = {k: r["v1"][k] == r["v2"][k] for k in r["v1"]}
        out[tag] = dict(v1=r["v1"], v2=r["v2"], equal=same,
                        identical_except_bounds=all(v for k, v in same.items() if k not in ("xmin", "xmax", "ymin", "ymax")))
    req = list(OpenOil(loglevel=50).required_variables)
    for tag in out:
        out[tag]["required_variables_provided"] = {ver: sorted(set(req) & set(out[tag][ver]["variables"])) for ver in ("v1", "v2")}
    return out


def validate():
    assert V2_OCEAN.exists() and V2_WIND.exists()
    oc_common, oc = overlap(V1_OCEAN, V2_OCEAN, OCEAN_VARS, ["time", "depth", "latitude", "longitude"])
    wd_common, wd = overlap(V1_WIND, V2_WIND, WIND_VARS, ["valid_time", "latitude", "longitude"])
    sch = {}
    for tag, a, b, ignore in [("ocean", V1_OCEAN, V2_OCEAN, ()), ("wind", V1_WIND, V2_WIND, ())]:
        (s1, e1), (s2, e2) = schema(a), schema(b)
        diff = {k: dict(v1=s1.get(k), v2=s2.get(k)) for k in sorted(set(s1) | set(s2)) if s1.get(k) != s2.get(k)}
        sch[tag] = dict(variables_equal=list(s1) == list(s2), per_variable_differences=diff, time_equal=e1["time"] == e2["time"],
                        depth_equal=e1["depth"] == e2["depth"],
                        global_attr_differences={k: [e1["global_attrs"].get(k), e2["global_attrs"].get(k)]
                                                 for k in sorted(set(e1["global_attrs"]) | set(e2["global_attrs"]))
                                                 if e1["global_attrs"].get(k) != e2["global_attrs"].get(k)})
    rc = reader_check()
    coords_ok = all(c["v1_fully_contained"] for c in list(oc_common.values()) + list(wd_common.values()))
    values_ok = all(s["nan_mask_disagreement"] == 0 and (s["exact_equality_fraction"] == 1.0) for s in list(oc.values()) + list(wd.values()))
    res = dict(
        criterion=json.load(open(CONFIG))["predeclared_equivalence_criterion"],
        method="coordinates intersected on exact values (np.intersect1d), selection by those values; no interpolation",
        ocean=dict(common_domain=oc_common, stats=oc), wind=dict(common_domain=wd_common, stats=wd),
        schema_check=sch, opendrift_reader_check=rc,
        coordinates_v1_fully_contained=coords_ok, values_exact=values_ok,
        schema_same=all(not sch[t]["per_variable_differences"] and sch[t]["variables_equal"] and sch[t]["time_equal"]
                        and sch[t]["depth_equal"] for t in sch),
        readers_same_except_bounds=all(rc[t]["identical_except_bounds"] for t in rc),
        file_sha256={rel(p): sha256(p) for p in (V1_OCEAN, V1_WIND, V2_OCEAN, V2_WIND)})
    res["equivalence_pass"] = bool(coords_ok and values_ok)
    dump(res, VALIDATION)
    print(json.dumps(dict(ocean=oc, wind=wd, oc_common=oc_common, wd_common=wd_common, schema=sch,
                          readers_same=res["readers_same_except_bounds"], PASS=res["equivalence_pass"]), indent=1, default=str))


# ------------------------------------------------------------------ stage 6: freeze
def freeze():
    import pandas as pd
    val = json.load(open(VALIDATION))
    S = pd.read_csv(E008B / "candidate_hindcast_summary.csv")
    edge = [json.load(open(E008B / c / "coverage_diagnostics.json")) for c in S.candidate_id]
    edge_ok = all(g["max_current_outside_frac"] == 0 and g["max_wind_outside_frac"] == 0 for g in edge)
    ok = dict(equivalence_pass=val["equivalence_pass"], completed_42=bool(len(S) == 42 and (S.run_status == "completed").all()),
              zero_forcing_edge_exits=edge_ok)
    print(ok)
    assert all(ok.values()), "freeze criteria not met; not frozen"
    files = [V1_DESCRIPTION, V2_DOWNLOAD_META, V2_OCEAN_RAW, V2_WIND_RAW, V2_OCEAN, V2_WIND, Path(__file__), HINDCAST_SCRIPT,
             PROVENANCE, CONFIG, VALIDATION]
    dump(dict(name="forcing v2", description="same frozen physics recipe with expanded environmental spatial coverage",
              not_a_claim="not a physics change and not 'better physics': products, variables, times, resolution and "
                          "processing are those of forcing v1; only the horizontal extent differs",
              criteria=ok, predeclared_freeze_criterion=json.load(open(CONFIG))["predeclared_freeze_criterion"],
              sha256={rel(p): sha256(p) for p in files}), V2 / "FROZEN.json")
    print("frozen", rel(V2 / "FROZEN.json"))


if __name__ == "__main__":
    {"provenance": provenance, "config": config, "download": download, "build": build, "validate": validate,
     "freeze": freeze}[sys.argv[1]]()
