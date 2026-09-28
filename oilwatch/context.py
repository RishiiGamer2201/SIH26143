"""External factors: wind / current / Stokes sampled from the incident forcing at any (lon, lat, time).

Used for (1) SAR oil-detectability window at the slick (wind), (2) the environmental context shown to the analyst
(wind and current history over the backward window), (3) domain-coverage checks before physics runs.
"""
import numpy as np
import pandas as pd
import xarray as xr


def _dir_to(u, v):
    return float(np.degrees(np.arctan2(u, v)) % 360)  # direction the flow goes TO, deg from north


class Forcing:
    def __init__(self, ocean_nc, wind_nc):
        self.w = xr.open_dataset(wind_nc)
        self.o = xr.open_dataset(ocean_nc)
        self.wt = "valid_time" if "valid_time" in self.w.coords else "time"
        if "depth" in self.o.dims:
            self.o = self.o.isel(depth=0)

    @staticmethod
    def _t(t):
        return np.datetime64(pd.Timestamp(t).tz_convert(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t), "ns")

    def bounds(self):
        def bb(ds):
            return (float(ds.longitude.min()), float(ds.latitude.min()), float(ds.longitude.max()), float(ds.latitude.max()))
        w, o = bb(self.w), bb(self.o)
        return (max(w[0], o[0]), max(w[1], o[1]), min(w[2], o[2]), min(w[3], o[3]))

    def time_range(self):
        return (pd.Timestamp(max(self.w[self.wt].values[0], self.o.time.values[0]), tz="UTC"),
                pd.Timestamp(min(self.w[self.wt].values[-1], self.o.time.values[-1]), tz="UTC"))

    def wind(self, lon, lat, t):
        s = self.w.interp({self.wt: self._t(t), "longitude": lon, "latitude": lat})
        u, v = float(s.u10), float(s.v10)
        # meteorological convention: direction the wind comes FROM
        return {"wind_u": u, "wind_v": v, "wind_speed_m_s": float(np.hypot(u, v)),
                "wind_from_deg": float((np.degrees(np.arctan2(u, v)) + 180) % 360)}

    def current(self, lon, lat, t):
        s = self.o.interp(time=self._t(t), longitude=lon, latitude=lat)
        u, v = float(s.x_sea_water_velocity), float(s.y_sea_water_velocity)
        su, sv = float(s.sea_surface_wave_stokes_drift_x_velocity), float(s.sea_surface_wave_stokes_drift_y_velocity)
        return {"current_speed_m_s": float(np.hypot(u, v)), "current_to_deg": _dir_to(u, v),
                "stokes_speed_m_s": float(np.hypot(su, sv)), "stokes_to_deg": _dir_to(su, sv)}

    def history(self, lon, lat, t_end, hours, step_h=1):
        """Hourly wind/current at a fixed point over the preceding `hours` (context panel; wind history matters for
        slick appearance, Espedal 1999)."""
        rows = []
        for h in np.arange(0, hours + 1e-9, step_h):
            t = pd.Timestamp(t_end) - pd.Timedelta(hours=float(h))
            rows.append({"hours_before_t0": float(h), "time": t, **self.wind(lon, lat, t), **self.current(lon, lat, t)})
        return pd.DataFrame(rows)


def wind_class(speed, P):
    lo, hi = P["wind_valid_m_s"]
    mlo, mhi = P["wind_marginal_m_s"]
    if lo <= speed <= hi:
        return "valid"
    if mlo <= speed <= mhi:
        return "marginal"
    return "invalid"
