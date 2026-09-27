"""E007c: learn the E008 component-grouping policy (link gap G, minimum component area A_min, optional orientation guard)
from the E005 VALIDATION split only, freeze it, then apply it ONCE to the 49 frozen E007 components.

  python scripts/e007c_validate_clustering.py extract   # frozen E005 @0.60 on val -> pred / GT components, overlaps, pair gaps
  python scripts/e007c_validate_clustering.py analyse   # gap distributions, G sweep, A_min sweep, orientation guard, rule outcome
  python scripts/e007c_validate_clustering.py freeze    # configs/e008_candidate_policy_v1.json (+ .sha256); refuses to overwrite
  python scripts/e007c_validate_clustering.py apply     # policy -> E007 components (once); then Cerulean POST HOC only

Never used here: official test, Incident 001 / E007 geometry (until `apply`), Cerulean (until the post-hoc step of `apply`),
the E007 ROI, AIS. E005 is frozen (hash-verified), threshold 0.60, no retraining.

Definitions (fixed before any result was computed)
  Grid: E005 val cache tiles (1024 px, x2 of the native EPSG:4326 tile). Local metric frame per tile: dx = 2 * px_x_m,
    dy = 2 * px_y_m (meta.parquet, from the tile geotransform at the tile mid-latitude; < 0.5 % distance/area error in-tile).
  Components: 4-connected (scipy default), prediction = (prob >= 0.60) & valid, GT = label == 1. GT_8 (8-connected) = sensitivity only.
  Polygons: GDAL Polygonize of the label raster in the local metric frame. Distance = shapely edge-to-edge polygon distance (m),
    the same definition E008 applies to E007 polygons (scene-centred LAEA). Corner-touching components have distance 0.
  Association (overlap only, never nearest distance): pred component k overlaps GT component g if they share >= 1 pixel.
    status = fp (no GT overlap) | tp (exactly one GT) | ambiguous (>= 2 GT; dominant GT = largest pixel overlap).
    "TP-associated" = tp or ambiguous; its dominant GT is its GT object for cluster metrics.
  Fragmented GT slick: GT component overlapped by >= 2 predicted components. Its fragment weight = overlap area (TP area) with it.
  Clustering: single linkage, edge iff gap <= G (and, with a guard, orientation-compatible), within a tile.
  Recovery at G: r_area = sum over fragmented GT of fragment weight in its dominant cluster / total fragment weight;
    r_count = fraction of fragmented GT whose fragments all fall in one cluster.
  Wrong merge: a cluster whose TP-associated members have > 1 distinct dominant GT. wrong_rate = wrong clusters / clusters
    holding >= 1 TP-associated component. Mixed: cluster with TP-associated AND fp members (same denominator).
  Purity of a TP cluster: area of members whose dominant GT = the cluster's dominant GT (largest area) / cluster area (fp included).
  Orientation guard (exploratory): an edge is dropped if BOTH components have elongation >= 2 and axial orientation difference > theta.
  A_min: components with area < A_min are removed BEFORE clustering (they cannot bridge).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e005_segformer as e005  # noqa: E402
from e006_segformer_test import git, sha256, verify_frozen  # noqa: E402

gdal.UseExceptions()
ROOT = e005.ROOT
E005 = ROOT / "results/E005_segformer_b2"
OUT = ROOT / "results/E007c_validation_clustering"
CFG_OUT = ROOT / "configs/e008_candidate_policy_v1.json"
THR = 0.60

# ------------------------------------------------ PREDECLARED (fixed before extract / analyse were run; do not edit after)
G_GRID = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]    # m; G = 0 is reported as the no-grouping baseline only
R_AREA_MIN, WRONG_MAX = 0.95, 0.05                              # rule: smallest G with r_area >= 0.95 and wrong_rate <= 0.05
A_GRID = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]  # m2 (one 20 m val pixel ~ 300-400 m2)
TP_RETAIN_MIN, FP_REDUCTION_MATERIAL = 0.99, 0.20               # rule: smallest A_min removing >= 20 % of fp components, >= 99 % TP area kept
MIN_ELONG, THETA_GRID = 2.0, [20, 30, 45]                       # orientation guard (exploratory)
GUARD_ADOPT = dict(min_rel_wrong_reduction=0.25, min_abs_wrong_reduction=0.01, max_r_area_drop=0.01)
N_BOOT, SEED = 1000, 26143
GT_AREA_BINS = [0, 5e4, 5e5, 5e6, np.inf]                       # m2, stratification only
ELONG_BINS = [0, 2, 5, np.inf]
FRAG_AREA_BINS = [0, 1e4, 1e5, 1e6, np.inf]


def srs_wgs84():
    s = osr.SpatialReference()
    s.ImportFromEPSG(4326)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def polygons(lbl, n, gt):
    """One (multi)polygon per label 1..n from GDAL Polygonize (4-connected), in the frame of geotransform gt."""
    mem = gdal.GetDriverByName("MEM").Create("", lbl.shape[1], lbl.shape[0], 1, gdal.GDT_Int32)
    mem.SetGeoTransform(gt)
    band = mem.GetRasterBand(1)
    band.WriteArray(lbl.astype(np.int32))
    vds = ogr.GetDriverByName("Memory").CreateDataSource("m")  # keep referenced while the layer is used
    lyr = vds.CreateLayer("c", None, ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [])
    parts = {}
    for f in lyr:
        parts.setdefault(f.GetField("id"), []).append(shapely.from_wkb(bytes(f.GetGeometryRef().ExportToWkb())))
    assert len(parts) == n, (len(parts), n)
    return [shapely.union_all(parts[i]) if len(parts[i]) > 1 else parts[i][0] for i in range(1, n + 1)]


def mrr(g):
    """Minimum rotated rectangle in a north-up metric frame: major, minor, elongation, axial orientation from north (0-180)."""
    r = np.array(shapely.minimum_rotated_rectangle(g).exterior.coords)[:4]
    e = [r[1] - r[0], r[2] - r[1]]
    L = [float(np.hypot(*v)) for v in e]
    k = int(np.argmax(L))
    return max(L), min(L), max(L) / max(min(L), 1e-9), math.degrees(math.atan2(e[k][0], e[k][1])) % 180


def adiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 90) % 180 - 90)


def dist_stats(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(n=0)
    q = np.percentile(x, [0, 25, 50, 75, 90, 95, 99, 100])
    return dict(n=int(len(x)), **dict(zip(["min", "p25", "median", "p75", "p90", "p95", "p99", "max"], map(float, q))))


# ---------------------------------------------------------------------------------------------------- extract
def tile_tables(r, tid, cls, m, pred, gtm, prob):
    dx, dy = 2 * m.px_x_m, 2 * m.px_y_m
    ds = gdal.Open(str(ROOT / m.img))
    g0 = ds.GetGeoTransform()
    ds = None
    to_ll = lambda x, y: (g0[0] + (x / dx) * 2 * g0[1], g0[3] + (-y / dy) * 2 * g0[5])
    mgt = (0.0, dx, 0.0, 0.0, 0.0, -dy)
    plab, npc = ndimage.label(pred)
    glab, ng = ndimage.label(gtm)
    glab8, _ = ndimage.label(gtm, structure=np.ones((3, 3), int))
    P, G, O, PR = [], [], [], []
    if ng:
        gpoly = polygons(glab, ng, mgt)
        gnpx = np.bincount(glab.ravel(), minlength=ng + 1)[1:]
        g8 = ndimage.maximum(glab8, glab, np.arange(1, ng + 1))  # each 4-comp lies inside exactly one 8-comp
        for g in range(1, ng + 1):
            mj, mn, el, az = mrr(gpoly[g - 1])
            G.append(dict(row=r, tile_id=tid, gt=g, gt8=int(g8[g - 1]), npx=int(gnpx[g - 1]), area_m2=float(gnpx[g - 1] * dx * dy),
                          major_m=mj, minor_m=mn, elongation=el, orientation_deg=az))
    if not npc:
        return P, G, O, PR
    ppoly = polygons(plab, npc, mgt)
    ids = np.arange(1, npc + 1)
    npx = np.bincount(plab.ravel(), minlength=npc + 1)[1:]
    pmean, pmax = ndimage.mean(prob, plab, ids), ndimage.maximum(prob, plab, ids)
    ov = {}
    if ng:
        both = (plab > 0) & (glab > 0)
        key = plab[both].astype(np.int64) * (ng + 1) + glab[both]
        u, c = np.unique(key, return_counts=True)
        for kk, cc in zip(u, c):
            k, g = divmod(int(kk), ng + 1)
            ov.setdefault(k, {})[g] = int(cc)
            O.append(dict(row=r, comp=k, gt=g, px=int(cc), area_m2=float(cc * dx * dy)))
    for k in ids:
        poly = ppoly[k - 1]
        mj, mn, el, az = mrr(poly)
        o = ov.get(int(k), {})
        dom = max(o, key=lambda g: (o[g], -g)) if o else 0
        c = poly.centroid
        x0, y0, x1, y1 = poly.bounds
        (lon, lat), (w_, s_), (e_, n_) = to_ll(c.x, c.y), to_ll(x0, y0), to_ll(x1, y1)
        P.append(dict(row=r, tile_id=tid, cls=cls, comp=int(k), npx=int(npx[k - 1]), area_m2=float(npx[k - 1] * dx * dy),
                      centroid_lon=lon, centroid_lat=lat, bbox_west=w_, bbox_south=s_, bbox_east=e_, bbox_north=n_,
                      major_m=mj, minor_m=mn, elongation=el, orientation_deg=az, prob_mean=float(pmean[k - 1]), prob_max=float(pmax[k - 1]),
                      n_gt_overlap=len(o), dominant_gt=int(dom), dominant_overlap_px=int(o.get(dom, 0)),
                      overlap_px_total=int(sum(o.values())), overlaps=";".join(f"{g}:{o[g]}" for g in sorted(o)),
                      status="fp" if not o else ("tp" if len(o) == 1 else "ambiguous")))
    if npc > 1:
        a, b = np.triu_indices(npc, 1)
        d = shapely.distance(np.array(ppoly, dtype=object)[a], np.array(ppoly, dtype=object)[b])
        PR = [dict(row=r, a=int(i + 1), b=int(j + 1), gap_m=float(v)) for i, j, v in zip(a, b, d)]
    return P, G, O, PR


def extract():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fz = verify_frozen()
    ck = torch.load(E005 / "best.pt", map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    assert ck["epoch"] == fz["best_epoch"] == 9 and fz["threshold"] == THR
    e005.seed_all(cfg["seed"])
    model = e005.build_model(cfg, None)
    model.load_state_dict(ck["state_dict"])
    rows, cls, ids, vv, lab = e005.load("val")
    meta = pd.read_parquet(e005.C / "meta.parquet")
    VT = pd.read_parquet(E005 / "val_tiles.parquet").set_index("row")
    vm = json.load(open(E005 / "val_metrics.json"))
    tot = np.zeros(3)
    P, G, O, PR = [], [], [], []
    for i, p, valid, y in e005.infer(model, vv, lab, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]):
        pr = (p >= THR) & valid
        tot += np.array(e005.counts(pr, y, valid).tolist())
        prn, yn, pn = pr.cpu().numpy(), y.cpu().numpy(), p.float().cpu().numpy()
        for j in range(len(prn)):
            n = i + j
            r = int(rows[n])
            t = tile_tables(r, ids[n], cls[n], meta.loc[r], prn[j], yn[j], pn[j])
            assert len(t[0]) == VT.loc[r, "tsel_n_comp"], ("component count differs from E005 val_tiles", r)
            for acc, x in zip((P, G, O, PR), t):
                acc.extend(x)
    iou = e005.prf(*tot.tolist())["iou"]
    ref = vm["at_selected"]["global_pixel"]["iou"]
    assert abs(iou - ref) < 1e-6, ("val IoU @0.60 not reproduced", iou, ref)
    P, G, O, PR = map(pd.DataFrame, (P, G, O, PR))
    P.to_parquet(OUT / "validation_components.parquet", index=False)
    G.to_parquet(OUT / "validation_gt_components.parquet", index=False)
    O.to_parquet(OUT / "validation_overlaps.parquet", index=False)
    PR.to_parquet(OUT / "validation_pairs.parquet", index=False)
    rep = dict(e005_checkpoint_sha256=fz["sha256"]["results/E005_segformer_b2/best.pt"]["sha256"], threshold=THR,
               val_tiles=int(len(rows)), val_iou_reproduced=iou, val_iou_reference=ref,
               n_pred_components=len(P), n_gt_components=len(G), n_overlaps=len(O), n_pred_pairs=len(PR),
               pred_components_by_status=P.status.value_counts().to_dict(), pred_components_by_class=P.cls.value_counts().to_dict(),
               runtime_s=time.time() - t0, peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
    json.dump(rep, open(OUT / "extract_report.json", "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------- analyse
def load_tables():
    P = pd.read_parquet(OUT / "validation_components.parquet")
    G = pd.read_parquet(OUT / "validation_gt_components.parquet")
    O = pd.read_parquet(OUT / "validation_overlaps.parquet")
    PR = pd.read_parquet(OUT / "validation_pairs.parquet")
    P["uid"] = np.arange(len(P))
    uid = P.set_index(["row", "comp"]).uid
    PR["ua"] = uid.loc[list(zip(PR.row, PR.a))].to_numpy()
    PR["ub"] = uid.loc[list(zip(PR.row, PR.b))].to_numpy()
    O["uid"] = uid.loc[list(zip(O.row, O.comp))].to_numpy()
    O = O.merge(G[["row", "gt", "gt8"]], on=["row", "gt"])
    return P, G, O, PR


def cluster(P, PR, G_m, keep=None, theta=None):
    """Single-linkage labels over all val components (edges exist only within a tile)."""
    keep = np.ones(len(P), bool) if keep is None else keep
    e = PR[(PR.gap_m <= G_m) & keep[PR.ua] & keep[PR.ub]]
    if theta is not None:
        el, az = P.elongation.to_numpy(), P.orientation_deg.to_numpy()
        both = (el[e.ua] >= MIN_ELONG) & (el[e.ub] >= MIN_ELONG)
        e = e[~both | (adiff(az[e.ua], az[e.ub]) <= theta)]
    A = coo_matrix((np.ones(len(e)), (e.ua, e.ub)), shape=(len(P), len(P)))
    return connected_components(A, directed=False)[1]


def per_tile_metrics(P, O, lab, keep, gtcol):
    """Per-tile additive counts (so tile bootstrap is a re-weighting) + global extras."""
    Pk = P[keep].assign(cl=lab[keep])
    Ok = O[keep[O.uid]].copy()
    Ok["gtk"] = Ok[gtcol]
    Ok = Ok.groupby(["row", "uid", "gtk"], as_index=False).area_m2.sum()
    dom = Ok.sort_values(["uid", "area_m2", "gtk"], ascending=[True, False, True]).drop_duplicates("uid").set_index("uid").gtk
    Pk["dom"] = Pk.uid.map(dom).fillna(0).astype(int)
    Pk["tpa"] = Pk.dom > 0
    # recovery over fragmented GT (>= 2 kept overlapping components)
    Ok["cl"] = lab[Ok.uid]
    nfr = Ok.groupby(["row", "gtk"]).uid.nunique()
    F = Ok.set_index(["row", "gtk"]).loc[nfr[nfr >= 2].index].reset_index()
    byc = F.groupby(["row", "gtk", "cl"]).area_m2.sum()
    g = byc.groupby(level=[0, 1]).agg(["max", "sum", "count"])
    rec = g.groupby(level=0).agg(rec_num=("max", "sum"), rec_den=("sum", "sum"),
                                 n_complete=("count", lambda c: int((c == 1).sum())), n_frag=("count", "size"))
    # clusters holding >= 1 TP-associated component
    C = Pk.groupby("cl").agg(row=("row", "first"), area=("area_m2", "sum"), n_tp=("tpa", "sum"), n_fp=("tpa", lambda s: int((~s).sum())),
                             n_dom=("dom", lambda s: s[s > 0].nunique()))
    T = C[C.n_tp > 0].copy()
    domarea = Pk[Pk.tpa].groupby(["cl", "dom"]).area_m2.sum().groupby(level=0).max()
    T["pure_area"] = domarea.reindex(T.index).fillna(0)
    T["wrong"] = T.n_dom > 1
    T["mixed"] = T.n_fp > 0
    fp_in_tp = Pk[~Pk.tpa & Pk.cl.isin(T.index)].groupby("row").area_m2.sum()
    cl_t = T.groupby("row").agg(n_tpc=("area", "size"), n_wrong=("wrong", "sum"), n_mixed=("mixed", "sum"),
                                wrong_area=("area", lambda a: float(a[T.loc[a.index, "wrong"]].sum())),
                                pure_area=("pure_area", "sum"), tpc_area=("area", "sum"),
                                purity_sum=("pure_area", lambda a: float((a / T.loc[a.index, "area"]).sum())))
    tiles = pd.DataFrame(index=pd.Index(sorted(P.row.unique()), name="row"))
    out = tiles.join(rec).join(cl_t).join(fp_in_tp.rename("fp_area_in_tp_clusters")).fillna(0)
    extra = dict(n_clusters=int(C.shape[0]), n_fp_only_clusters=int((C.n_tp == 0).sum()))
    return out, extra


def summarise(t, extra):
    s = t.sum()
    return dict(**extra, n_frag_gt=int(s.n_frag), r_area=s.rec_num / s.rec_den if s.rec_den else np.nan,
                r_count=s.n_complete / s.n_frag if s.n_frag else np.nan, n_tp_clusters=int(s.n_tpc),
                wrong_n=int(s.n_wrong), wrong_rate=s.n_wrong / s.n_tpc if s.n_tpc else np.nan, wrong_area_m2=float(s.wrong_area),
                mixed_n=int(s.n_mixed), mixed_rate=s.n_mixed / s.n_tpc if s.n_tpc else np.nan,
                fp_area_in_tp_clusters_m2=float(s.fp_area_in_tp_clusters),
                purity_mean=s.purity_sum / s.n_tpc if s.n_tpc else np.nan, purity_areaw=s.pure_area / s.tpc_area if s.tpc_area else np.nan)


def boot_ci(t, W):
    """95 % tile-bootstrap intervals for r_area and wrong_rate (ratios of per-tile sums)."""
    num_r, den_r = W @ t.rec_num.to_numpy(), W @ t.rec_den.to_numpy()
    num_w, den_w = W @ t.n_wrong.to_numpy(), W @ t.n_tpc.to_numpy()
    with np.errstate(all="ignore"):
        r, w = num_r / den_r, num_w / den_w
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return dict(r_area_ci=q(r), wrong_rate_ci=q(w))


def evaluate(P, O, PR, G_m, keep=None, theta=None, gtcol="gt", W=None):
    keep = np.ones(len(P), bool) if keep is None else keep
    lab = cluster(P, PR, G_m, keep, theta)
    t, extra = per_tile_metrics(P, O, lab, keep, gtcol)
    res = summarise(t, extra)
    if W is not None:
        res.update(boot_ci(t, W))
    return res


def analyse():
    t0 = time.time()
    P, G, O, PR = load_tables()
    st = P.status.to_numpy()
    tpa = st != "fp"
    tiles = np.array(sorted(P.row.unique()))
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(tiles), np.full(len(tiles), 1 / len(tiles)), size=N_BOOT).astype(float)  # tile bootstrap weights

    # ---- 2. intra-slick fragment gaps (GT overlapped by >= 2 predicted components; any overlap)
    gap = {(a, b): d for a, b, d in zip(PR.ua, PR.ub, PR.gap_m)}
    frag = O.groupby(["row", "gt"]).uid.apply(lambda s: sorted(set(s)))
    frag = frag[frag.map(len) >= 2]
    Gi = G.set_index(["row", "gt"])
    area = P.area_m2.to_numpy()
    rows_ = []
    for (r, g), us in frag.items():
        n = len(us)
        D = np.array([[gap[(min(a, b), max(a, b))] if a != b else 0.0 for b in us] for a in us])
        mst = minimum_spanning_tree(D + 1e-6 * (1 - np.eye(n)))  # +eps: touching fragments (gap 0) remain edges
        height = float(mst.data.max() - 1e-6)
        for i in range(n):
            for j in range(i + 1, n):
                rows_.append(dict(row=r, tile_id=P.tile_id[us[i]], gt=g, gt_area_m2=Gi.loc[(r, g), "area_m2"],
                                  gt_elongation=Gi.loc[(r, g), "elongation"], gt_major_m=Gi.loc[(r, g), "major_m"], n_fragments=n,
                                  comp_a=int(P.comp[us[i]]), comp_b=int(P.comp[us[j]]), area_a_m2=area[us[i]], area_b_m2=area[us[j]],
                                  gap_m=D[i, j], gt_merge_height_m=height))
    I = pd.DataFrame(rows_)
    I.to_csv(OUT / "intra_slick_gaps.csv", index=False)
    Ig = I.drop_duplicates(["row", "gt"])
    strat = lambda df, col, bins, v: {f"[{lo:g}, {hi:g})": dist_stats(df[(df[col] >= lo) & (df[col] < hi)][v])
                                      for lo, hi in zip(bins[:-1], bins[1:])}
    I["min_frag_area_m2"] = I[["area_a_m2", "area_b_m2"]].min(1)
    intra = dict(n_fragmented_gt=int(len(Ig)), pairwise_gap_m=dist_stats(I.gap_m), gt_merge_height_m=dist_stats(Ig.gt_merge_height_m),
                 merge_height_by_gt_area_m2=strat(Ig, "gt_area_m2", GT_AREA_BINS, "gt_merge_height_m"),
                 merge_height_by_gt_elongation=strat(Ig, "gt_elongation", ELONG_BINS, "gt_merge_height_m"),
                 pairwise_gap_by_smaller_fragment_area_m2=strat(I, "min_frag_area_m2", FRAG_AREA_BINS, "gap_m"))

    # ---- 3. inter-slick / tp-fp / fp-fp gaps (all within-tile pairs except same-GT pairs)
    dom = P.dominant_gt.to_numpy()
    ta, tb = tpa[PR.ua], tpa[PR.ub]
    typ = np.where(ta & tb, np.where(dom[PR.ua] == dom[PR.ub], "tp_tp_same_gt", "tp_tp_diff_gt"), np.where(ta | tb, "tp_fp", "fp_fp"))
    X = PR.assign(pair_type=typ, cls=P.cls.to_numpy()[PR.ua], dom_a=dom[PR.ua], dom_b=dom[PR.ub], comp_a=PR.a, comp_b=PR.b)
    X = X[X.pair_type != "tp_tp_same_gt"]
    X[["row", "cls", "comp_a", "comp_b", "pair_type", "dom_a", "dom_b", "gap_m"]].to_csv(OUT / "inter_slick_gaps.csv", index=False)
    dd = X[X.pair_type == "tp_tp_diff_gt"]
    gtpair = dd.assign(g1=dd[["dom_a", "dom_b"]].min(1), g2=dd[["dom_a", "dom_b"]].max(1)).groupby(["row", "g1", "g2"]).gap_m.min()

    def nearest(mask_self, mask_other, types):
        s = X[X.pair_type.isin(types)]
        v = pd.concat([s[["ua", "ub", "gap_m"]].rename(columns={"ua": "u", "ub": "o"}), s[["ub", "ua", "gap_m"]].rename(columns={"ub": "u", "ua": "o"})])
        v = v[mask_self[v.u] & mask_other[v.o]]
        if types == ["tp_tp_diff_gt"]:
            v = v[dom[v.u] != dom[v.o]]
        return dist_stats(v.groupby("u").gap_m.min())
    inter = dict(gt_pair_nearest_gap_m=dist_stats(gtpair),
                 all_pairs_gap_m={k: dist_stats(v.gap_m) for k, v in X.groupby("pair_type")},
                 nearest_per_component_m=dict(tp_to_other_gt_tp=nearest(tpa, tpa, ["tp_tp_diff_gt"]),
                                              tp_to_fp=nearest(tpa, ~tpa, ["tp_fp"]), fp_to_tp=nearest(~tpa, tpa, ["tp_fp"]),
                                              fp_to_fp=nearest(~tpa, ~tpa, ["fp_fp"])),
                 frac_gt_pairs_within_G={str(g_): float((gtpair <= g_).mean()) for g_ in G_GRID})

    # ---- 4. G sweep (+ orientation guard, + GT_8 sensitivity)
    sweep = []
    for g_ in [0] + G_GRID:
        sweep.append(dict(G_m=g_, variant="none", **evaluate(P, O, PR, g_, W=W)))
        for th in THETA_GRID:
            sweep.append(dict(G_m=g_, variant=f"theta{th}", **evaluate(P, O, PR, g_, theta=th, W=W)))
        sweep.append(dict(G_m=g_, variant="gt8_sensitivity", **evaluate(P, O, PR, g_, gtcol="gt8", W=W)))
    S = pd.DataFrame(sweep)
    for c in ("r_area_ci", "wrong_rate_ci"):
        S[c + "_lo"], S[c + "_hi"] = S[c].str[0], S[c].str[1]
    S = S.drop(columns=["r_area_ci", "wrong_rate_ci"])
    S.to_csv(OUT / "gap_sweep.csv", index=False)
    base = S[(S.variant == "none") & (S.G_m > 0)]
    ok = base[(base.r_area >= R_AREA_MIN) & (base.wrong_rate <= WRONG_MAX)]
    G_sel = int(ok.G_m.min()) if len(ok) else None
    g_rule = dict(rule=f"smallest G in {G_GRID} with r_area >= {R_AREA_MIN} and wrong_rate <= {WRONG_MAX} (variant none, A_min 0)",
                  selected_G_m=G_sel,
                  outcome="selected" if G_sel is not None else "NO defensible single global G: no grid value meets both constraints",
                  max_G_with_wrong_ok=int(base[base.wrong_rate <= WRONG_MAX].G_m.max()) if (base.wrong_rate <= WRONG_MAX).any() else None,
                  min_G_with_r_area_ok=int(base[base.r_area >= R_AREA_MIN].G_m.min()) if (base.r_area >= R_AREA_MIN).any() else None)
    if G_sel is not None:
        s = base[base.G_m == G_sel].iloc[0]
        g_rule["ci_robust"] = bool(s.r_area_ci_lo >= R_AREA_MIN and s.wrong_rate_ci_hi <= WRONG_MAX)
        g_rule["at_selected"] = s.to_dict()

    # ---- 6. A_min
    tp_area = P.overlap_px_total.to_numpy() * (P.area_m2 / P.npx).to_numpy()
    fp = st == "fp"
    arows = []
    for a in A_GRID:
        k = area >= a
        arows.append(dict(A_min_m2=a, n_components_kept=int(k.sum()), fp_components_removed=int((fp & ~k).sum()),
                          fp_reduction=float((fp & ~k).sum() / fp.sum()), tp_components_removed=int((tpa & ~k).sum()),
                          tp_area_retained=float(tp_area[k].sum() / tp_area.sum()),
                          fp_reduction_lookalike_noil_tiles=float((fp & ~k & (P.cls != "oil")).sum() / max((fp & (P.cls != "oil")).sum(), 1))))
    AS = pd.DataFrame(arows)
    edges = A_GRID + [np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (area >= lo) & (area < hi)
        bins.append(dict(bin_lo_m2=lo, bin_hi_m2=hi, n_components=int(b.sum()), n_overlapping_gt=int((b & tpa).sum()),
                         component_precision=float((b & tpa).sum() / b.sum()) if b.sum() else np.nan,
                         tp_area_share=float(tp_area[b].sum() / tp_area.sum()),
                         tp_area_retained_if_removed_below_bin=float(tp_area[area >= lo].sum() / tp_area.sum())))
    AB = pd.DataFrame(bins)
    AS.to_csv(OUT / "area_threshold_sweep.csv", index=False)
    AB.to_csv(OUT / "area_bins.csv", index=False)
    feas = AS[(AS.A_min_m2 > 0) & (AS.tp_area_retained >= TP_RETAIN_MIN) & (AS.fp_reduction >= FP_REDUCTION_MATERIAL)]
    A_sel = int(feas.A_min_m2.min()) if len(feas) else 0
    a_rule = dict(rule=f"smallest A_min in {A_GRID[1:]} with fp_reduction >= {FP_REDUCTION_MATERIAL} and tp_area_retained >= {TP_RETAIN_MIN}; else 0",
                  selected_A_min_m2=A_sel, outcome="selected" if len(feas) else "no useful threshold: A_min = 0 (no filtering)")

    # ---- 7. orientation guard (exploratory; adoption only by the predeclared GUARD_ADOPT rule at the selected G)
    guard = dict(rule=GUARD_ADOPT, adopted_theta_deg=None)
    if G_sel is not None:
        b0 = S[(S.variant == "none") & (S.G_m == G_sel)].iloc[0]
        cand = []
        for th in THETA_GRID:
            s = S[(S.variant == f"theta{th}") & (S.G_m == G_sel)].iloc[0]
            red = b0.wrong_rate - s.wrong_rate
            q = dict(theta=th, wrong_rate=s.wrong_rate, r_area=s.r_area, purity_areaw=s.purity_areaw, abs_wrong_reduction=red,
                     rel_wrong_reduction=red / b0.wrong_rate if b0.wrong_rate else 0.0, r_area_drop=b0.r_area - s.r_area)
            q["qualifies"] = bool(q["rel_wrong_reduction"] >= GUARD_ADOPT["min_rel_wrong_reduction"] and red >= GUARD_ADOPT["min_abs_wrong_reduction"]
                                  and q["r_area_drop"] <= GUARD_ADOPT["max_r_area_drop"] and s.r_area >= R_AREA_MIN and s.wrong_rate <= WRONG_MAX)
            cand.append(q)
        guard["at_selected_G"] = cand
        qual = [c for c in cand if c["qualifies"]]
        if qual:
            guard["adopted_theta_deg"] = sorted(qual, key=lambda c: (c["r_area_drop"], -c["theta"]))[0]["theta"]

    # ---- final combined check on validation (filter then cluster)
    combined = None
    if G_sel is not None:
        combined = evaluate(P, O, PR, G_sel, keep=area >= A_sel, theta=guard["adopted_theta_deg"], W=W)

    rep = dict(note="VALIDATION ONLY (E005 val split, frozen E005 @0.60). No test / Incident 001 / Cerulean / E007 / AIS used.",
               predeclared=dict(G_GRID=G_GRID, R_AREA_MIN=R_AREA_MIN, WRONG_MAX=WRONG_MAX, A_GRID=A_GRID, TP_RETAIN_MIN=TP_RETAIN_MIN,
                                FP_REDUCTION_MATERIAL=FP_REDUCTION_MATERIAL, MIN_ELONG=MIN_ELONG, THETA_GRID=THETA_GRID,
                                GUARD_ADOPT=GUARD_ADOPT, N_BOOT=N_BOOT, SEED=SEED),
               counts=dict(val_tiles=int(len(tiles)), pred_components=int(len(P)), by_status=P.status.value_counts().to_dict(),
                           by_class=P.cls.value_counts().to_dict(), gt_components=int(len(G)), fragmented_gt=int(len(Ig))),
               intra_slick=intra, inter_slick=inter, G_rule=g_rule, A_min_rule=a_rule, orientation_guard=guard,
               combined_policy_on_validation=combined, runtime_s=time.time() - t0)
    json.dump(rep, open(OUT / "validation_clustering_report.json", "w"), indent=1, default=float)
    curves(I, Ig, gtpair, X, S, AS)
    print(json.dumps(dict(counts=rep["counts"], intra=intra["gt_merge_height_m"], inter=inter["gt_pair_nearest_gap_m"], G_rule=g_rule,
                          A_min_rule=a_rule, guard=guard, combined=combined), indent=1, default=float))


def curves(I, Ig, gtpair, X, S, AS):
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    cdf = lambda a, x, **k: a.plot(np.sort(x), np.arange(1, len(x) + 1) / len(x), **k)
    a = ax[0, 0]
    cdf(a, Ig.gt_merge_height_m.to_numpy(), label=f"intra: GT merge height (n={len(Ig)})", color="C0")
    cdf(a, I.gap_m.to_numpy(), label=f"intra: pairwise fragment gap (n={len(I)})", color="C0", ls="--")
    cdf(a, gtpair.to_numpy(), label=f"inter: nearest gap between GT objects (n={len(gtpair)})", color="C3")
    for k, c in (("tp_fp", "C2"), ("fp_fp", "C7")):
        v = X[X.pair_type == k].gap_m.to_numpy()
        if len(v):
            cdf(a, v, label=f"{k} all pairs (n={len(v)})", color=c, ls=":")
    a.set_xscale("symlog", linthresh=10)
    for g_ in G_GRID:
        a.axvline(g_, color="k", lw=0.3)
    a.set_xlabel("edge-to-edge gap (m)")
    a.set_ylabel("CDF")
    a.legend(fontsize=8)
    a.set_title("validation: intra- vs inter-slick gaps")
    b = S[S.variant == "none"]
    a = ax[0, 1]
    a.errorbar(b.G_m, b.r_area, yerr=[b.r_area - b.r_area_ci_lo, b.r_area_ci_hi - b.r_area], marker="o", label="r_area (95% CI)")
    a.plot(b.G_m, b.r_count, marker="s", label="r_count")
    a.axhline(R_AREA_MIN, color="k", ls="--", lw=0.8)
    a.set_xscale("symlog", linthresh=100)
    a.set_title("fragment recovery vs G")
    a.legend()
    a = ax[0, 2]
    a.errorbar(b.G_m, b.wrong_rate, yerr=[b.wrong_rate - b.wrong_rate_ci_lo, b.wrong_rate_ci_hi - b.wrong_rate], marker="o", label="wrong-GT merge rate (95% CI)")
    a.plot(b.G_m, b.mixed_rate, marker="s", label="TP+FP mixed rate")
    g8 = S[S.variant == "gt8_sensitivity"]
    a.plot(g8.G_m, g8.wrong_rate, marker="^", ls=":", label="wrong rate, GT 8-conn (sensitivity)")
    a.axhline(WRONG_MAX, color="k", ls="--", lw=0.8)
    a.set_xscale("symlog", linthresh=100)
    a.set_title("wrong merges vs G")
    a.legend()
    a = ax[1, 0]
    a.plot(b.G_m, b.purity_areaw, marker="o", label="purity (area-weighted)")
    a.plot(b.G_m, b.purity_mean, marker="s", label="purity (cluster mean)")
    a.set_xscale("symlog", linthresh=100)
    a.set_title("cluster purity vs G")
    a.legend()
    a = ax[1, 1]
    for v in ["none"] + [f"theta{t}" for t in THETA_GRID]:
        s = S[(S.variant == v) & (S.G_m > 0)]
        a.plot(s.wrong_rate, s.r_area, marker="o", label=v)
    a.axhline(R_AREA_MIN, color="k", ls="--", lw=0.8)
    a.axvline(WRONG_MAX, color="k", ls="--", lw=0.8)
    a.set_xlabel("wrong-GT merge rate")
    a.set_ylabel("r_area")
    a.set_title("orientation guard: recovery vs wrong merges (points = G grid)")
    a.legend()
    a = ax[1, 2]
    a.plot(AS.A_min_m2, AS.fp_reduction, marker="o", label="fp components removed")
    a.plot(AS.A_min_m2, AS.tp_area_retained, marker="s", label="TP area retained")
    a.axhline(TP_RETAIN_MIN, color="k", ls="--", lw=0.8)
    a.axhline(FP_REDUCTION_MATERIAL, color="k", ls=":", lw=0.8)
    a.set_xscale("symlog", linthresh=500)
    a.set_xlabel("A_min (m2)")
    a.set_title("minimum component area")
    a.legend()
    plt.tight_layout()
    plt.savefig(OUT / "validation_clustering_curves.png", dpi=80)


# ---------------------------------------------------------------------------------------------------- freeze / apply
def freeze():
    assert not CFG_OUT.exists(), f"{CFG_OUT} exists; policy v1 is frozen"
    rep = json.load(open(OUT / "validation_clustering_report.json"))
    G_sel = rep["G_rule"]["selected_G_m"]
    assert G_sel is not None, "no defensible single global G on validation: nothing to freeze (report and stop)"
    th = rep["orientation_guard"]["adopted_theta_deg"]
    cfg = dict(
        policy="e008_candidate_policy_v1", created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="FROZEN. Derived from the E005 validation split only. Never change it based on Incident 001 / Cerulean / E007 results.",
        input="frozen E005 SegFormer mask @0.60 (results/E005_segformer_b2/FROZEN.json)", threshold=THR,
        connectivity=4, component_extraction="scipy.ndimage.label default (4-connectivity) + GDAL Polygonize",
        A_min_m2=rep["A_min_rule"]["selected_A_min_m2"], A_min_order="components with area < A_min removed before clustering",
        distance="edge-to-edge polygon distance (shapely) in metres, in a local metric/equal-area frame (E007: scene-centred LAEA)",
        algorithm="single linkage: components joined when distance <= G_m (transitively)", G_m=G_sel,
        orientation_rule=(None if th is None else dict(theta_deg=th, min_elongation=MIN_ELONG,
                                                       rule="edge dropped if both components elongation >= min and axial orientation difference > theta")),
        rationale=dict(G=rep["G_rule"], A_min=rep["A_min_rule"], orientation=rep["orientation_guard"]),
        validation_metrics_combined=rep["combined_policy_on_validation"],
        provenance=dict(script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha256(Path(__file__).resolve()),
                        report_sha256=sha256(OUT / "validation_clustering_report.json"),
                        validation_tables_sha256={f: sha256(OUT / f) for f in ["validation_components.parquet", "validation_gt_components.parquet",
                                                                                "validation_overlaps.parquet", "validation_pairs.parquet"]},
                        e005_frozen_sha256=sha256(E005 / "FROZEN.json"), git_head=git("rev-parse", "HEAD")),
    )
    CFG_OUT.parent.mkdir(exist_ok=True)
    json.dump(cfg, open(CFG_OUT, "w"), indent=1, default=float)
    h = sha256(CFG_OUT)
    CFG_OUT.with_suffix(".sha256").write_text(f"{h}  {CFG_OUT.name}\n")
    print(json.dumps(cfg, indent=1, default=float))
    print("sha256", h)


def apply():
    import e007_incident001_ml as e007
    import e007b_component_structure as e007b
    h = CFG_OUT.with_suffix(".sha256").read_text().split()[0]
    assert sha256(CFG_OUT) == h, "policy file changed since freeze"
    pol = json.load(open(CFG_OUT))
    dst = OUT / "e007_candidates_v1.json"
    assert not dst.exists(), f"{dst} exists; the policy was already applied once"
    e007.verify()
    C, fwd, back = e007b.load()  # frozen E007 components; post-hoc in_roi flag dropped on load
    keep = (C.area_m2 >= pol["A_min_m2"]).to_numpy()
    K = C[keep].reset_index(drop=True)
    n = len(K)
    D = np.array([[K.g_m[i].distance(K.g_m[j]) if i != j else np.inf for j in range(n)] for i in range(n)])
    E = D <= pol["G_m"]
    if pol["orientation_rule"]:
        el, az = K.elongation.to_numpy(), K.orientation_deg_from_north.to_numpy()
        both = (el[:, None] >= pol["orientation_rule"]["min_elongation"]) & (el[None, :] >= pol["orientation_rule"]["min_elongation"])
        E &= ~both | (adiff(az[:, None], az[None, :]) <= pol["orientation_rule"]["theta_deg"])
    _, lab = connected_components(E, directed=False)
    rows_, feats = [], []
    for c in np.unique(lab):
        m = lab == c
        M = K[m]
        U = shapely.union_all(list(M.geometry))
        um = e007.shape_metrics(U, fwd, back)
        O_ = M[M.elongation >= MIN_ELONG]
        rows_.append(dict(n_members=int(m.sum()), member_ids=" ".join(map(str, sorted(M.component_id))), area_m2=float(M.area_m2.sum()),
                          centroid_lon=um["centroid_lon"], centroid_lat=um["centroid_lat"], extent_major_m=um["major_axis_m"],
                          extent_minor_m=um["minor_axis_m"], extent_orientation_deg=um["orientation_deg_from_north"],
                          bbox=[um["bbox_west"], um["bbox_south"], um["bbox_east"], um["bbox_north"]],
                          n_elongated=len(O_), orientation_coherence=e007b.coherence(O_.orientation_deg_from_north.to_numpy(), O_.major_axis_m.to_numpy()),
                          prob_mean_areaw=float(np.average(M.prob_mean, weights=M.area_m2)),
                          isolation_gap_m=float(D[np.ix_(m, ~m)].min()) if (~m).any() else float("nan"), geometry=U))
    R = pd.DataFrame(rows_).sort_values("area_m2", ascending=False).reset_index(drop=True)
    R.insert(0, "candidate_id", np.arange(1, len(R) + 1))
    R.drop(columns="geometry").to_csv(OUT / "e007_candidates_v1.csv", index=False)
    json.dump(dict(type="FeatureCollection", name="e007_candidates_v1", crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")),
                   features=[dict(type="Feature", geometry=shapely.geometry.mapping(r.geometry),
                                  properties={k: v for k, v in r._asdict().items() if k not in ("geometry", "Index")}) for r in R.itertuples()]),
              open(OUT / "e007_candidates_v1.geojson", "w"), default=float)
    res = dict(policy_sha256=h, policy=dict(G_m=pol["G_m"], A_min_m2=pol["A_min_m2"], orientation_rule=pol["orientation_rule"]),
               n_components_in=len(C), n_components_removed_by_A_min=int((~keep).sum()),
               removed_component_ids=sorted(map(int, C.component_id[~keep])), n_candidates=len(R),
               candidates=R.drop(columns="geometry").to_dict("records"))
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1, default=float))
    print(R.drop(columns=["geometry", "bbox"]).round(3).to_string(index=False))

    # POST HOC ONLY (after the candidates are written): which frozen candidate corresponds to the Cerulean reference
    cer = e007.cerulean()
    cm = fwd(cer)
    roi_ids = set(pd.read_csv(e007.OUT / "predicted_components.csv").query("in_roi").component_id)
    ph = []
    for r in R.itertuples():
        g = fwd(r.geometry)
        ids = set(map(int, r.member_ids.split()))
        inter = g.intersection(cm).area
        ph.append(dict(candidate_id=int(r.candidate_id), intersection_with_reference_m2=inter, frac_reference_covered=inter / cm.area,
                       iou=inter / g.union(cm).area, n_roi_members=len(ids & roi_ids), n_non_roi_members=len(ids - roi_ids)))
    P_ = pd.DataFrame(ph)
    hit = P_[P_.intersection_with_reference_m2 > 0]
    json.dump(dict(note="POST HOC only. Cerulean did not select, filter, group or tune anything above.", per_candidate=ph,
                   candidates_intersecting_reference=hit.candidate_id.tolist(),
                   n_roi_components_total=len(roi_ids)), open(OUT / "e007_candidates_v1_posthoc_cerulean.json", "w"), indent=1, default=float)
    print(P_[(P_.intersection_with_reference_m2 > 0) | (P_.n_roi_members > 0)].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "analyse", "freeze", "apply"])
    {"extract": extract, "analyse": analyse, "freeze": freeze, "apply": apply}[ap.parse_args().stage]()
