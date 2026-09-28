"""AIS loading and vessel positions at model times.

positions_v2   : frozen V2 rule (score_ais_density_v2.py): nearest report within +/-tolerance, no interpolation.
positions_v2_1 : frozen V2.1 rule (scripts/score_ais_density_v2_1.py, E101): exact / linear interp / edge-nearest.
"""
import numpy as np
import pandas as pd

from .geo import haversine_km, wrap180


def load_ais(path):
    """Same cleaning as the frozen scoring scripts."""
    ais = pd.read_parquet(path)
    ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    ais["MMSI"] = ais["MMSI"].astype(str).str.replace(r"\.0$", "", regex=True)
    ais["LAT"] = pd.to_numeric(ais["LAT"], errors="coerce")
    ais["LON"] = pd.to_numeric(ais["LON"], errors="coerce")
    return ais.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])


def vessel_name(g):
    if "VesselName" not in g:
        return ""
    names = g.VesselName.dropna().astype(str).str.strip()
    names = names[names != ""]
    return names.mode().iloc[0] if len(names) else ""


def positions_v2(model_times, g, tolerance):
    """Frozen V2: merge_asof nearest within tolerance. g must be one MMSI, sorted, first duplicate kept."""
    matched = pd.merge_asof(pd.DataFrame({"model_time": model_times}), g, left_on="model_time",
                            right_on="BaseDateTime", direction="nearest", tolerance=tolerance)
    return matched.dropna(subset=["LAT", "LON"])


def positions_v2_1(t_model, t_obs, lon, lat, cfg):
    """Frozen V2.1 rule (E101), imported from scripts/score_ais_density_v2_1.py. t_* are int64 ns."""
    from .frozen import v2_1
    return v2_1.vessel_positions(t_model, t_obs, lon, lat, cfg)


def ns(ts):
    """tz-aware timestamps -> int64 ns (UTC)."""
    return pd.DatetimeIndex(ts).tz_convert(None).values.astype("datetime64[ns]").astype("int64")


def track_arrays(g):
    g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
    return g, ns(g.BaseDateTime), g.LON.to_numpy(float), g.LAT.to_numpy(float)


def state_at(g, t, cfg):
    """V2.1 position at one time + SOG/COG of the nearest report (for heading tests)."""
    g, t_obs, lon, lat = track_arrays(g)
    lo, la, m, gap, off = positions_v2_1(ns([t]), t_obs, lon, lat, cfg)
    i = int(np.argmin(np.abs(t_obs - ns([t])[0])))
    return dict(lon=lo[0], lat=la[0], method=m[0], offset_min=off[0],
                sog=float(g.SOG.iloc[i]) if "SOG" in g else np.nan,
                cog=float(g.COG.iloc[i]) if "COG" in g else np.nan,
                nearest_report_min=abs(t_obs[i] - ns([t])[0]) / 60e9)


def ais_gaps(g, t_start, t_end, min_gap_min):
    """Reporting gaps >= min_gap_min inside [t_start, t_end] (candidate 'dark' periods)."""
    g = g[(g.BaseDateTime >= t_start) & (g.BaseDateTime <= t_end)].sort_values("BaseDateTime")
    if len(g) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_min", "jump_km"])
    t = g.BaseDateTime.to_numpy()
    dt = np.diff(t).astype("timedelta64[s]").astype(float) / 60
    k = np.where(dt >= min_gap_min)[0]
    lon, lat = g.LON.to_numpy(), g.LAT.to_numpy()
    return pd.DataFrame({"gap_start": g.BaseDateTime.iloc[k].values, "gap_end": g.BaseDateTime.iloc[k + 1].values,
                         "gap_min": dt[k], "jump_km": haversine_km(lon[k], lat[k], lon[k + 1], lat[k + 1])})
