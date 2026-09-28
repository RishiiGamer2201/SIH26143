"""Offshore infrastructure (BOEM Platform Structures, data.boem.gov) and the source_type rule.

Structures in place at T0 (installed <= T0, not removed before T0). Coordinates are NAD27 in the BOEM table
(NAD_YEAR_CD = 27) -> converted to WGS84 with pyproj (tens of metres in the Gulf; matters at 0.5 km match radius).

source_type (E102 kept motion_state and source_type separate; this fills source_type with an explicit basis):
  fixed_infrastructure : AIS track is stationary (E102) AND a BOEM structure lies within match_km of its median position
  vessel               : AIS VesselType is a ship code (20-89: WIG/fishing/tug/sailing/HSC/pilot/SAR/passenger/cargo/tanker),
                         or the track is moving/loitering/mixed and no BOEM structure is within match_km
  unknown              : otherwise (e.g. stationary with no structure nearby: DP drillship, anchored ship, unlisted asset)
  dark_vessel          : SAR ship detection with no AIS match (sar stage), never assigned to an AIS track
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import ROOT
from .geo import haversine_km

SHIP_TYPES = set(range(20, 90))


def load_platforms(inc):
    d = inc.spec.get("inputs", {}).get("platforms_dir")
    f = next((ROOT / d).rglob("mv_platstruc_structures.txt"), None) if d and (ROOT / d).exists() else None
    if f is None:
        return None
    s = pd.read_csv(f, dtype=str)
    s["install"] = pd.to_datetime(s.INSTALL_DATE, errors="coerce", format="mixed")
    s["removal"] = pd.to_datetime(s.REMOVAL_DATE, errors="coerce", format="mixed")
    t0 = inc.t0.tz_convert(None)
    s = s[(s.install.isna() | (s.install <= t0)) & (s.removal.isna() | (s.removal > t0))].copy()
    lon27, lat27 = pd.to_numeric(s.LONGITUDE, errors="coerce"), pd.to_numeric(s.LATITUDE, errors="coerce")
    ok = lon27.notna() & lat27.notna()
    s, lon27, lat27 = s[ok], lon27[ok], lat27[ok]
    tr = Transformer.from_crs("EPSG:4267", "EPSG:4326", always_xy=True)
    s["lon"], s["lat"] = tr.transform(lon27.values, lat27.values)
    s["id"] = s.COMPLEX_ID_NUM + "-" + s.STRUCTURE_NUMBER
    s["name"] = (s.AREA_CODE.str.strip() + " " + s.BLOCK_NUMBER.str.strip() + " " + s.STRUCTURE_NAME.fillna("").str.strip()
                 + " (" + s.BUS_ASC_NAME.fillna("") + ")")
    return s[["id", "name", "lon", "lat", "STRUC_TYPE_CODE", "WATER_DEPTH", "MAJ_STRUC_FLAG", "BUS_ASC_NAME", "install"]] \
        .reset_index(drop=True)


def resolve_source_type(motion, ais, platforms, match_km=0.5):
    """Adds source_type + source_type_basis + nearest BOEM structure to the motion table (one row per MMSI)."""
    out = motion.copy()
    types = ais.groupby(ais.MMSI.astype(str)).VesselType.agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else np.nan)
    med = ais.groupby(ais.MMSI.astype(str))[["LON", "LAT"]].median()
    st, basis, near_id, near_km = [], [], [], []
    for _, r in out.iterrows():
        m = str(r.MMSI)
        vt = types.get(m, np.nan)
        km, pid = np.nan, None
        if platforms is not None and m in med.index:
            d = haversine_km(med.loc[m, "LON"], med.loc[m, "LAT"], platforms.lon.values, platforms.lat.values)
            j = int(np.argmin(d))
            km, pid = float(d[j]), platforms.name.iloc[j]
        at_struct = np.isfinite(km) and km <= match_km
        if r.motion_state == "stationary" and at_struct:
            st.append("fixed_infrastructure"), basis.append(f"stationary AIS within {km * 1000:.0f} m of BOEM structure {pid}")
        elif np.isfinite(vt) and int(vt) in SHIP_TYPES:
            st.append("vessel"), basis.append(f"AIS VesselType {int(vt)} (ship code)")
        elif r.motion_state in ("moving", "loitering", "mixed") and not at_struct:
            st.append("vessel"), basis.append(f"motion_state {r.motion_state}, no BOEM structure within {match_km} km")
        else:
            st.append("unknown"), basis.append(f"motion_state {r.motion_state}, VesselType {vt}, nearest structure "
                                               f"{'n/a' if not np.isfinite(km) else f'{km:.2f} km'}")
        near_id.append(pid), near_km.append(km)
    out["source_type"], out["source_type_basis"] = st, basis
    out["nearest_boem_structure"], out["nearest_boem_km"] = near_id, near_km
    return out
