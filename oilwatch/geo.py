"""Geodesy helpers. haversine_km keeps the exact operation order of the frozen scoring scripts (bit-exact parity)."""
import numpy as np
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform

R_EARTH_KM = 6371.0088
GEOD = Geod(ellps="WGS84")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def laea(lon0, lat0):
    """Local equal-area metric CRS (metres) centred on the slick, as E007 uses for axes/orientation."""
    crs = CRS.from_proj4(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return fwd, back


def to_metric(geom, fwd):
    return transform(fwd.transform, geom)


def geodesic_area_m2(geom):
    return abs(GEOD.geometry_area_perimeter(geom)[0])


def axis_endpoints(geom_m):
    """Long axis of the minimum rotated rectangle (metric CRS): two end points, length, bearing (deg from north)."""
    rect = geom_m.minimum_rotated_rectangle
    xy = np.asarray(rect.exterior.coords)[:4]
    edges = [(xy[i], xy[(i + 1) % 4]) for i in range(4)]
    lens = [np.hypot(*(b - a)) for a, b in edges]
    i = int(np.argmax(lens))
    # the long axis runs through the mid-points of the two short edges
    s1, s2 = edges[(i + 1) % 4], edges[(i + 3) % 4]
    p1, p2 = (s1[0] + s1[1]) / 2, (s2[0] + s2[1]) / 2
    # clip to the actual geometry so the "ends" are on the slick, not on the rectangle
    line = LineString([p1, p2])
    inter = line.intersection(geom_m.convex_hull)
    if not inter.is_empty and inter.geom_type == "LineString":
        p1, p2 = np.asarray(inter.coords[0]), np.asarray(inter.coords[-1])
    d = p2 - p1
    bearing = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)  # axis: undirected, 0-180
    return Point(p1), Point(p2), float(np.hypot(*d)), bearing, float(min(lens))


def angle_diff_axis(heading_deg, axis_deg):
    """Smallest angle between a directed heading and an undirected axis, 0-90 deg."""
    d = abs((heading_deg - axis_deg) % 180.0)
    return min(d, 180.0 - d)
