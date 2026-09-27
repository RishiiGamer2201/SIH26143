"""E007b: MODEL-ONLY structure of the frozen E007 predicted components (descriptive; prepares E008 selection policy).

  python scripts/e007b_component_structure.py model_only   # per-component table + clusters at predeclared distances + figure
  python scripts/e007b_component_structure.py posthoc      # AFTER model_only: which clusters hold the E007 reporting-ROI components

model_only reads ONLY the E007 predicted-components GeoJSON (its post-hoc `in_roi` flag is dropped on load), the E007 x4 VV raster
and mask geotransform. It never opens the Cerulean polygon, bbox or reporting ROI; nothing is selected, filtered or ranked by
agreement with any reference. Cluster distance thresholds are PREDECLARED (below) and descriptive only; none is chosen here.
Distances are edge-to-edge in the same scene-centred LAEA (metres) as E007 vector().
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.sparse.csgraph import connected_components
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e007_incident001_ml as e007  # noqa: E402

ROOT = e007.ROOT
SRC = e007.OUT / "incident001_predicted_components.geojson"
OUT = ROOT / "results/E007b_component_structure"
LINK_M = [250, 500, 1000, 2000, 5000]  # predeclared single-linkage gaps (m): ~12, 25, 50, 100, 250 px at 20 m; descriptive only
MIN_ELONG = 2.0                        # predeclared: orientation used only for components with elongation >= 2 (blocky ones quantise)
DISPLAY_GAP = 2000                     # figure only (which clustering is drawn); selects nothing


def adiff(a, b):
    return abs((a - b + 90) % 180 - 90)


def coherence(theta_deg, w):
    """Axial (mod 180) resultant length in [0, 1] of orientations weighted by w; 1 = all parallel. NaN if < 2 oriented members."""
    if len(theta_deg) < 2:
        return float("nan")
    z = np.sum(w * np.exp(2j * np.radians(theta_deg))) / np.sum(w)
    return float(abs(z))


def load():
    fc = json.load(open(SRC))
    rec = []
    for f in fc["features"]:
        p = {k: v for k, v in f["properties"].items() if k != "in_roi"}  # post-hoc ROI flag dropped
        rec.append(dict(**p, geometry=shape(f["geometry"])))
    C = pd.DataFrame(rec).sort_values("component_id").reset_index(drop=True)
    mask_ds = e007.gdal.Open(str(e007.P["mask"]))
    gt, H, W = mask_ds.GetGeoTransform(), mask_ds.RasterYSize, mask_ds.RasterXSize
    mask_ds = None
    fwd, back = e007.laea(gt[0] + gt[1] * W / 2, gt[3] + gt[5] * H / 2)  # same projection as E007 vector()
    C["g_m"] = [fwd(g) for g in C.geometry]
    return C, fwd, back


def model_only():
    OUT.mkdir(parents=True, exist_ok=True)
    C, fwd, back = load()
    n = len(C)
    G = list(C.g_m)
    D = np.array([[G[i].distance(G[j]) if i != j else np.inf for j in range(n)] for i in range(n)])
    nn = D.argmin(1)
    C["oriented"] = C.elongation >= MIN_ELONG
    C["nearest_component_id"] = C.component_id.to_numpy()[nn]
    C["nearest_component_gap_m"] = D.min(1)
    C["nearest_orientation_diff_deg"] = [adiff(C.orientation_deg_from_north[i], C.orientation_deg_from_north[j]) for i, j in enumerate(nn)]
    C["nearest_both_oriented"] = C.oriented.to_numpy() & C.oriented.to_numpy()[nn]
    # nearest ORIENTED neighbour (orientation similarity meaningful only there)
    Do = np.where(C.oriented.to_numpy()[None, :], D, np.inf)
    j_o = Do.argmin(1)
    ok_o = np.isfinite(Do.min(1)) & C.oriented.to_numpy()
    C["nearest_oriented_id"] = np.where(ok_o, C.component_id.to_numpy()[j_o], -1)
    C["nearest_oriented_gap_m"] = np.where(ok_o, Do.min(1), np.nan)
    C["nearest_oriented_orientation_diff_deg"] = [adiff(C.orientation_deg_from_north[i], C.orientation_deg_from_north[j]) if ok_o[i] else np.nan
                                                 for i, j in enumerate(j_o)]
    cols = ["component_id", "area_m2", "n_px_image_space", "centroid_lon", "centroid_lat", "major_axis_m", "width_proxy_m", "elongation",
            "orientation_deg_from_north", "oriented", "prob_mean", "prob_max", "nearest_component_id", "nearest_component_gap_m",
            "nearest_orientation_diff_deg", "nearest_both_oriented", "nearest_oriented_id", "nearest_oriented_gap_m",
            "nearest_oriented_orientation_diff_deg"]
    C[cols].to_csv(OUT / "components_model_only.csv", index=False)

    summary = dict(source=str(SRC.relative_to(ROOT)), n_components=n, link_gaps_m=LINK_M, min_elongation_for_orientation=MIN_ELONG,
                   note="MODEL-ONLY. No Cerulean geometry / bbox / ROI read. Thresholds predeclared, descriptive only; none selected.",
                   clusterings={})
    allc = []
    for d in LINK_M:
        k, lab = connected_components(D <= d, directed=False)
        rows = []
        for c in range(k):
            m = lab == c
            M = C[m]
            U_ll = shapely.union_all(list(M.geometry))
            um = e007.shape_metrics(U_ll, fwd, back)
            hull = shapely.convex_hull(shapely.union_all(list(M.g_m)))
            O = M[M.oriented]
            outside = D[np.ix_(m, ~m)]
            rows.append(dict(link_gap_m=d, cluster=c, n_members=int(m.sum()), member_ids=" ".join(map(str, M.component_id)),
                             area_m2=float(M.area_m2.sum()), largest_member_area_m2=float(M.area_m2.max()),
                             centroid_lon=um["centroid_lon"], centroid_lat=um["centroid_lat"], extent_major_m=um["major_axis_m"],
                             extent_minor_m=um["minor_axis_m"], extent_orientation_deg=um["orientation_deg_from_north"],
                             hull_fill=float(sum(g.area for g in M.g_m) / max(hull.area, 1e-9)),
                             n_oriented=len(O), orientation_coherence=coherence(O.orientation_deg_from_north.to_numpy(), O.major_axis_m.to_numpy()),
                             prob_mean_areaw=float(np.average(M.prob_mean, weights=M.area_m2)), prob_max=float(M.prob_max.max()),
                             isolation_gap_m=float(outside.min()) if outside.size else float("nan")))
        R = pd.DataFrame(rows).sort_values("area_m2", ascending=False).reset_index(drop=True)
        R["cluster"] = np.arange(len(R))  # renumber by area (model-only ordering)
        allc.append(R)
        summary["clusterings"][str(d)] = dict(n_clusters=len(R), n_singletons=int((R.n_members == 1).sum()),
                                              largest_5=R.head(5)[["n_members", "area_m2", "centroid_lon", "centroid_lat", "extent_major_m",
                                                                   "orientation_coherence", "isolation_gap_m"]].round(4).to_dict("records"))
    A = pd.concat(allc, ignore_index=True)
    A.to_csv(OUT / "clusters_model_only.csv", index=False)
    json.dump(summary, open(OUT / "structure_summary.json", "w"), indent=1, default=float)
    figure(C, A)
    print(json.dumps(summary, indent=1, default=float))


def figure(C, A):
    db, gt = e007.read(e007.P["vv40"])
    s = 4
    img = db[::s, ::s]
    ext = [gt[0], gt[0] + gt[1] * db.shape[1], gt[3] + gt[5] * db.shape[0], gt[3]]
    R = A[A.link_gap_m == DISPLAY_GAP]
    top = R.head(10)
    fig = plt.figure(figsize=(26, 13))
    gs = fig.add_gridspec(2, 7)
    ax0 = fig.add_subplot(gs[:, :2])
    ax0.imshow(img, extent=ext, cmap="gray", vmin=-28, vmax=-8, interpolation="nearest")
    ax0.scatter(C.centroid_lon, C.centroid_lat, s=4 + np.sqrt(C.area_m2) / 20, facecolors="none", edgecolors="red", lw=0.8)
    for r in top.itertuples():
        ax0.annotate(f"c{r.cluster}", (r.centroid_lon, r.centroid_lat), color="yellow", fontsize=11, xytext=(6, 6), textcoords="offset points")
    ax0.set_title(f"MODEL ONLY: all 49 components (red, size ~ area);\nc# = clusters at {DISPLAY_GAP} m gap by area (no reference shown)")
    for k, r in enumerate(top.itertuples()):
        ax = fig.add_subplot(gs[k // 5, 2 + k % 5])
        ids = set(map(int, r.member_ids.split()))
        M = C[C.component_id.isin(ids)]
        x0, y0, x1, y1 = shapely.union_all(list(M.geometry)).bounds
        pad = max(x1 - x0, y1 - y0) * 0.25 + 0.01
        c0, c1 = int((x0 - pad - gt[0]) / gt[1]), int((x1 + pad - gt[0]) / gt[1])
        r0, r1 = int((y1 + pad - gt[3]) / gt[5]), int((y0 - pad - gt[3]) / gt[5])
        c0, r0 = max(c0, 0), max(r0, 0)
        ax.imshow(db[r0:r1, c0:c1], extent=[gt[0] + c0 * gt[1], gt[0] + c1 * gt[1], gt[3] + r1 * gt[5], gt[3] + r0 * gt[5]],
                  cmap="gray", vmin=-28, vmax=-8, interpolation="nearest")
        for g in C.geometry:  # all components in view (members and non-members) as red outlines
            for poly in getattr(g, "geoms", [g]):
                ax.plot(*poly.exterior.xy, color="red", lw=0.8)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_title(f"c{r.cluster}: {r.n_members} comp, {r.area_m2 / 1e6:.2f} km2, extent {r.extent_major_m / 1e3:.1f} km,\n"
                     f"coherence {r.orientation_coherence:.2f}, isolation {r.isolation_gap_m / 1e3:.1f} km", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "model_only_clusters.png", dpi=80)


def posthoc():
    """POST-HOC ONLY (after model_only is written): map E007's reporting-ROI flag onto the model-only clusters. Never feeds back."""
    A = pd.read_csv(OUT / "clusters_model_only.csv")
    roi_ids = set(pd.read_csv(e007.OUT / "predicted_components.csv").query("in_roi").component_id)
    out = {}
    for d, R in A.groupby("link_gap_m"):
        hit = []
        for r in R.itertuples():
            ids = set(map(int, r.member_ids.split()))
            if ids & roi_ids:
                hit.append(dict(cluster=int(r.cluster), area_rank=int(r.cluster) + 1, n_members=int(r.n_members),
                                n_roi_members=len(ids & roi_ids), n_non_roi_members=len(ids - roi_ids), area_m2=float(r.area_m2),
                                extent_major_m=float(r.extent_major_m), orientation_coherence=float(r.orientation_coherence),
                                isolation_gap_m=float(r.isolation_gap_m)))
        out[str(d)] = dict(n_clusters_total=len(R), clusters_holding_roi_components=hit)
    json.dump(dict(note="POST-HOC evaluation only. Uses E007's reporting-ROI flag (Cerulean bbox + 0.1 deg). Must not select E008 geometry.",
                   n_roi_components=len(roi_ids), by_link_gap_m=out), open(OUT / "posthoc_roi_vs_clusters.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["model_only", "posthoc"])
    {"model_only": model_only, "posthoc": posthoc}[ap.parse_args().stage]()
