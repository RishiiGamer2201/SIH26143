"""E008 slick-candidate policy v1: which detected geometry seeds the physics. Reference-independent.

1. Detected components (E007, raw 0.60 mask polygons) are merged into slick events when their gap <= merge_distance
   (fragments of one spill; Cerulean closing buffer).
2. Per event: geodesic area, parts, area-weighted model confidence, long axis (ends, length, bearing), width proxy,
   wind at the slick at T0 and its SAR-detectability class.
3. slick_potential = confidence x wind_factor x size_factor (size_factor = 1 - exp(-A / A_ref)).
4. Attribution-eligible: area >= min_cluster_area, inside forcing domain and inside AIS coverage. Ineligible
   events are kept in the table with the reason (never silently dropped).
5. Ranked by slick_potential; top max_clusters_attributed go to physics. An analyst may override in the
   incident spec ("seed": {"clusters": [...]}) - logged in the manifest.
Cerulean / the reference polygon is not read here.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from .context import wind_class
from .geo import axis_endpoints, geodesic_area_m2, laea, to_metric


def cluster_components(comps, merge_m):
    """Single-linkage clustering of polygons with gap <= merge_m (metric LAEA at the scene centre)."""
    c = comps.unary_union.centroid
    fwd, back = laea(c.x, c.y)
    gm = [to_metric(g, fwd) for g in comps.geometry]
    n = len(gm)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if gm[i].distance(gm[j]) <= merge_m:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    order = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([order[r] + 1 for r in roots])


def build_candidates(comps, forcing, ais_bbox, t0, P):
    comps = comps[comps.area_m2 >= P["min_component_area_m2"]].copy()
    comps["cluster_id"] = cluster_components(comps, P["merge_distance_m"])
    fb = box(*forcing.bounds())
    rows, geoms = [], []
    for cid, g in comps.groupby("cluster_id"):
        geom = unary_union(list(g.geometry))
        cen = geom.centroid
        fwd, back = laea(cen.x, cen.y)
        p1, p2, length_m, bearing, rect_short = axis_endpoints(to_metric(geom, fwd))
        e1, e2 = back.transform(p1.x, p1.y), back.transform(p2.x, p2.y)
        area = geodesic_area_m2(geom)
        w = forcing.wind(cen.x, cen.y, t0)
        wc = wind_class(w["wind_speed_m_s"], P)
        conf = float(np.average(g.prob_mean, weights=g.area_m2))
        size_f = 1 - np.exp(-area / P["size_ref_m2"])
        reasons = []
        if area < P["min_cluster_area_m2"]:
            reasons.append(f"area {area:.0f} m2 < {P['min_cluster_area_m2']} (below detector's reliable size)")
        if not fb.buffer(-0.2).contains(geom):
            reasons.append("outside forcing domain (0.2 deg margin)")
        if ais_bbox is not None and not box(*ais_bbox).contains(cen):
            reasons.append("outside AIS coverage")
        rows.append({"cluster_id": int(cid), "component_ids": ",".join(map(str, g.component_id)), "n_parts": len(g),
                     "area_m2": area, "confidence": conf, "prob_max": float(g.prob_max.max()),
                     "centroid_lon": cen.x, "centroid_lat": cen.y, "length_m": length_m, "axis_bearing_deg": bearing,
                     "width_proxy_m": area / max(length_m, 1.0), "elongation": length_m / max(rect_short, 1.0),
                     "end1_lon": e1[0], "end1_lat": e1[1], "end2_lon": e2[0], "end2_lat": e2[1],
                     **w, "wind_class": wc, "wind_factor": P["wind_factor"][wc], "size_factor": size_f,
                     "slick_potential": conf * P["wind_factor"][wc] * size_f,
                     "eligible": not reasons, "ineligible_reason": "; ".join(reasons)})
        geoms.append(geom)
    cand = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    cand = cand.sort_values("slick_potential", ascending=False).reset_index(drop=True)
    cand["potential_rank"] = np.arange(1, len(cand) + 1)
    el = cand[cand.eligible]
    cand["selected"] = cand.cluster_id.isin(el.head(P["max_clusters_attributed"]).cluster_id)
    cand["selection_basis"] = np.where(cand.selected, "policy_v1_top_potential", "")
    return cand, comps


def apply_override(cand, spec):
    ids = spec.get("seed", {}).get("clusters")
    if ids:
        cand["selected"] = cand.cluster_id.isin(ids)
        cand["selection_basis"] = np.where(cand.selected, "analyst_override", "")
    return cand
