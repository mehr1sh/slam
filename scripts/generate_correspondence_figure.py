#!/usr/bin/env python3
"""Publication figures of the ACTUAL sparse ORB feature correspondences of one
frame pair of the synthetic Stanford Bunny sequence.

Run (after `pixi run build`):
    pixi run -e results correspondences                 # pair 0 -> 1
    pixi run -e results correspondences --pair 17       # pair 17 -> 18
    pixi run -e results correspondences --skip-cpp      # reuse the last export

Writes
    results/figures/bunny_correspondences_annotated_<i>_<j>.{png,pdf,svg}
    results/figures/bunny_pnp_icp_correspondences.{png,pdf,svg}      (pair 0 -> 1)
    results/figures/bunny_pnp_icp_correspondences_<i>_<j>.{png,pdf,svg} (other pairs)

Data source: build/slam_trajectory_test is run on a temporary copy of
data/synthetic_bunny/ with its export-only flags (--export-dir, --export-pair),
and must reproduce the committed trajectory files byte for byte. Every
keypoint, match, depth value, RANSAC inlier flag and 3D point drawn here is
read from that export and cross-checked against the dataset images; nothing
is generated or resampled. Visualization only: no algorithm is changed.
"""

import argparse
import csv
import filecmp
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "synthetic_bunny"
BUILD = ROOT / "build"
FIG = ROOT / "results" / "figures"
EXPORT = ROOT / "results" / "data" / "correspondence_export"

# Colour system. Method hues match the Blender scene / results figures
# (visualization/scripts/build_scene.py METHOD_STYLES); depth uses viridis
# reversed (near = yellow, far = purple); outliers are a magenta that the
# depth colormap never produces.
PNP = (1.00, 0.62, 0.10)
ICP = (0.85, 0.40, 0.95)
GT = (0.15, 0.82, 0.45)
OUTLIER = (1.00, 0.16, 0.55)
ZOOM = (0.45, 1.00, 0.10)
ZOOM_LINK = (0.30, 0.62, 0.05)
CMAP = plt.get_cmap("viridis_r")
INK = "0.12"


# ----------------------------------------------------------------- data ----

def run_export(pair):
    exe = BUILD / "slam_trajectory_test"
    if not exe.exists():
        sys.exit(f"{exe} not found -- run `pixi run build` first")
    EXPORT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "synthetic_bunny"
        shutil.copytree(DATASET, ds)
        out = subprocess.run([str(exe), str(ds), "--export-dir", str(EXPORT), "--export-pair", str(pair)],
                             cwd=BUILD, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(out.stdout[-2000:] + out.stderr[-2000:])
        (EXPORT / "slam_trajectory_test.log").write_text(out.stdout)
        for f in ("pnp_trajectory.txt", "icp_trajectory.txt", "slam_trajectory.csv"):
            if not filecmp.cmp(ds / f, DATASET / f, shallow=False):
                sys.exit(f"the export run did not reproduce {f}; regenerate the dataset first (COMMANDS.md)")
    print(f"[export] slam_trajectory_test --export-pair {pair}: trajectory files reproduced byte for byte")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def check(label, ok, detail=""):
    print(f"[check] {'OK  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        sys.exit(f"validation failed: {label}")


def load(pair):
    i, j = pair, pair + 1
    prefix = EXPORT / f"pair_{i}_{j}"
    if not (prefix.parent / f"pair_{i}_{j}_correspondences.csv").exists():
        sys.exit(f"no export for pair {i}->{j}; run without --skip-cpp")
    w, h, fx, fy, cx, cy = (float(v) for v in (DATASET / "intrinsics.txt").read_text().split())
    K = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=int(w), h=int(h))
    rgb = [np.asarray(Image.open(DATASET / f"{k:06d}.png").convert("RGB")) for k in (i, j)]
    dep = [np.asarray(Image.open(DATASET / f"{k:06d}_depth.png")).astype(np.int64) for k in (i, j)]
    kps = read_csv(f"{prefix}_keypoints.csv")
    raw = read_csv(f"{prefix}_raw_matches.csv")
    cor = read_csv(f"{prefix}_correspondences.csv")
    pm = next(r for r in read_csv(EXPORT / "pair_metrics.csv") if int(r["pair"]) == i)
    kp = {f: np.array([[float(r["u"]), float(r["v"])] for r in kps if r["frame"] == f]) for f in ("i", "j")}

    c = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in cor])
         for k in ("u_i", "v_i", "u_j", "v_j", "hamming_distance", "depth_raw_i", "depth_raw_j",
                   "X_i", "Y_i", "Z_i", "X_j", "Y_j", "Z_j")}
    for k in ("query_index", "train_index", "pnp_used", "pnp_inlier", "icp_used"):
        c[k] = np.array([int(r[k]) for r in cor])
    c["pnp_used"] = c["pnp_used"].astype(bool)
    c["pnp_inlier"] = c["pnp_inlier"].astype(bool)
    c["icp_used"] = c["icp_used"].astype(bool)

    # ---- validation: everything shown must be the pipeline's own data ----
    n = dict(kp_i=len(kp["i"]), kp_j=len(kp["j"]), raw=len(raw), filtered=len(cor),
             pnp=int(c["pnp_used"].sum()), inliers=int(c["pnp_inlier"].sum()), icp=int(c["icp_used"].sum()))
    check("ORB keypoint counts = C++ run", n["kp_i"] == int(pm["orb_keypoints_i"]) and n["kp_j"] == int(pm["orb_keypoints_j"]),
          f"{n['kp_i']} / {n['kp_j']}")
    check("raw / filtered match counts = C++ run",
          n["raw"] == int(pm["raw_matches"]) and n["filtered"] == int(pm["filtered_matches"]), f"{n['raw']} / {n['filtered']}")
    check("PnP correspondences / RANSAC inliers / ICP pairs = C++ run",
          n["pnp"] == int(pm["pnp_correspondences"]) and n["inliers"] == int(pm["pnp_inliers"])
          and n["icp"] == int(pm["icp_correspondences"]), f"{n['pnp']} / {n['inliers']} / {n['icp']}")
    ok = np.allclose(kp["i"][c["query_index"]], np.c_[c["u_i"], c["v_i"]], atol=2e-3) and \
        np.allclose(kp["j"][c["train_index"]], np.c_[c["u_j"], c["v_j"]], atol=2e-3)
    check("every match end point is a detected ORB keypoint", ok)
    kept = {(int(r["query_index"]), int(r["train_index"])) for r in raw if r["filtered"] == "1"}
    check("every drawn match is a descriptor match kept by the filter",
          kept == set(zip(c["query_index"].tolist(), c["train_index"].tolist())))
    d = np.array([float(r["hamming_distance"]) for r in raw])
    thr = max(2 * d.min(), 30.0)
    check("filter rule distance <= max(2*d_min, 30) reproduces the kept set",
          {(int(r["query_index"]), int(r["train_index"])) for r in raw if float(r["hamming_distance"]) <= thr} == kept,
          f"(threshold {thr:g})")
    ui, vi = c["u_i"].astype(int), c["v_i"].astype(int)
    uj, vj = c["u_j"].astype(int), c["v_j"].astype(int)
    check("depth values = depth PNG at the (truncated) keypoint pixel",
          np.array_equal(dep[0][vi, ui], c["depth_raw_i"]) and np.array_equal(dep[1][vj, uj], c["depth_raw_j"]))
    check("PnP use = valid depth in frame i; ICP use = valid depth in both",
          np.array_equal(c["pnp_used"], c["depth_raw_i"] > 0)
          and np.array_equal(c["icp_used"], (c["depth_raw_i"] > 0) & (c["depth_raw_j"] > 0)))
    Z = c["depth_raw_i"] / 5000.0
    Xb = (c["u_i"] - cx) / fx * Z
    Yb = (c["v_i"] - cy) / fy * Z
    m = c["pnp_used"]
    check("3D points = back-projection of (u, v, depth) with intrinsics.txt",
          np.allclose(np.c_[Xb, Yb, Z][m], np.c_[c["X_i"], c["Y_i"], c["Z_i"]][m], atol=2e-5))
    check("RANSAC inliers are a subset of the PnP correspondences", not np.any(c["pnp_inlier"] & ~c["pnp_used"]))
    return dict(i=i, j=j, K=K, rgb=rgb, dep=dep, kp=kp, c=c, n=n, thr=thr, pm=pm)


def gt_step(i, j):
    rows = [l.split() for l in (DATASET / "groundtruth.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    a, b = np.array(rows[i][1:4], float), np.array(rows[j][1:4], float)
    return float(np.linalg.norm(b - a))


# ------------------------------------------------------------- helpers ----

def bunny_box(D, margin):
    m = (D["dep"][0] > 0) | (D["dep"][1] > 0)
    ys, xs = np.nonzero(m)
    x0, x1 = max(0, xs.min() - margin), min(m.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(m.shape[0], ys.max() + margin + 1)
    return x0, x1, y0, y1


def depth_norm(c):
    z = np.r_[c["Z_i"][c["pnp_used"]]]
    lo, hi = np.nanmin(z), np.nanmax(z)
    pad = 0.04 * (hi - lo)
    return Normalize(lo - pad, hi + pad)


def pick_zooms(D, size, count=3):
    """Choose zoom windows (frame-i pixel boxes) with the most PnP
    correspondences that stay inside the window in frame i+1 after the
    window is shifted by those features' median image motion. Prefer one
    window per vertical third of the bunny (head/ears, body, base)."""
    c = D["c"]
    m = c["pnp_used"]
    P = np.c_[c["u_i"], c["v_i"]][m]
    Q = np.c_[c["u_j"], c["v_j"]][m]
    ys, xs = np.nonzero(D["dep"][0] > 0)
    top, bot = ys.min(), ys.max()
    bands = [(top, top + (bot - top) / 3), (top + (bot - top) / 3, top + 2 * (bot - top) / 3),
             (top + 2 * (bot - top) / 3, bot)]
    cands = []
    for cy_ in range(int(top), int(bot) + 1, 3):
        for cx_ in range(int(xs.min()), int(xs.max()) + 1, 3):
            x0, y0 = cx_ - size // 2, cy_ - size // 2
            inside = (P[:, 0] >= x0) & (P[:, 0] < x0 + size) & (P[:, 1] >= y0) & (P[:, 1] < y0 + size)
            if inside.sum() < 5:
                continue
            shift = np.median(Q[inside] - P[inside], axis=0)
            q = Q[inside] - shift
            both = (q[:, 0] >= x0) & (q[:, 0] < x0 + size) & (q[:, 1] >= y0) & (q[:, 1] < y0 + size)
            cands.append((int(both.sum()), x0, y0, shift, cy_))
    chosen = []

    def free(x0, y0):
        return all(abs(x0 - a[1]) >= size or abs(y0 - a[2]) >= size for a in chosen)

    for lo, hi in bands:
        best = max((cd for cd in cands if lo <= cd[4] < hi and free(cd[1], cd[2])), default=None, key=lambda t: t[0])
        if best and best[0] >= 6:
            chosen.append(best)
    for cd in sorted(cands, key=lambda t: -t[0]):  # fill up if a band was too sparse
        if len(chosen) >= count:
            break
        if free(cd[1], cd[2]):
            chosen.append(cd)
    chosen.sort(key=lambda t: t[4])  # top to bottom of the bunny -> insets A, B, C
    names = []
    for cd in chosen:
        band = next(k for k, (lo, hi) in enumerate(bands) if lo <= cd[4] <= hi + 1)
        names.append(["head / ears", "body", "lower body / feet"][band])
    return [dict(x0=cd[1], y0=cd[2], size=size, shift=cd[3], n=cd[0], name=nm) for cd, nm in zip(chosen, names)]


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=220 if ext == "png" else None, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[figure] results/figures/{stem}.png|pdf|svg")


def chip(fig, x, y, num, label, color, w=0.108):
    fig.patches.append(patches.FancyBboxPatch((x, y), w, 0.05, boxstyle="round,pad=0.004,rounding_size=0.008",
                                              transform=fig.transFigure, facecolor=color, edgecolor="none",
                                              alpha=0.95, zorder=1))
    fig.text(x + w / 2, y + 0.032, num, ha="center", va="center", fontsize=15, weight="bold", color="white", zorder=2)
    fig.text(x + w / 2, y + 0.011, label, ha="center", va="center", fontsize=7.4, color="white", zorder=2)


# --------------------------------------------------------- figure 1 -------

def figure_correspondences(D, stem):
    c, n, i, j = D["c"], D["n"], D["i"], D["j"]
    x0, x1, y0, y1 = bunny_box(D, 26)
    W, H = x1 - x0, y1 - y0
    gap = 0.16 * W
    off = W + gap  # x offset of frame i+1 in the shared panel coordinates
    norm = depth_norm(c)
    col = CMAP(norm(c["Z_i"]))
    pnp, inl = c["pnp_used"], c["pnp_inlier"]
    outl = pnp & ~inl
    nodepth = ~pnp

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(15, 13.6), facecolor="white")
    ax = fig.add_axes([0.035, 0.395, 0.93, 0.475])
    ax.set_xlim(-2, off + W + 2)
    ax.set_ylim(H + 2, -2)
    ax.set_aspect("equal")
    ax.axis("off")
    for k, xo in ((0, 0), (1, off)):
        ax.imshow(D["rgb"][k][y0:y1, x0:x1], extent=(xo, xo + W, H, 0), interpolation="lanczos", zorder=0)
        ax.add_patch(patches.Rectangle((xo, 0), W, H, fill=False, lw=1.2, ec=INK, zorder=1))
        ax.text(xo + 6, 7, f"Frame {D['i'] if k == 0 else D['j']}", color="white", fontsize=15, weight="bold",
                va="top", zorder=8)

    def P(u, v, k):
        return u - x0 + (off if k else 0), v - y0

    # all detected ORB keypoints (context), then the match structure on top
    for k, f in ((0, "i"), (1, "j")):
        u, v = D["kp"][f][:, 0], D["kp"][f][:, 1]
        sel = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
        pu, pv = P(u[sel], v[sel], k)
        ax.scatter(pu, pv, s=9, facecolors="none", edgecolors="white", linewidths=0.55, alpha=0.75, zorder=2)
    a = np.c_[P(c["u_i"], c["v_i"], 0)]
    b = np.c_[P(c["u_j"], c["v_j"], 1)]
    ax.add_collection(LineCollection(np.stack([a[inl], b[inl]], 1), colors=col[inl], linewidths=0.75, alpha=0.5, zorder=3))
    ax.add_collection(LineCollection(np.stack([a[outl], b[outl]], 1), colors=[OUTLIER], linewidths=1.0,
                                     linestyles=(0, (3, 2)), alpha=0.9, zorder=4))
    for pts in (a, b):
        ax.scatter(pts[nodepth, 0], pts[nodepth, 1], s=16, marker="D", facecolors="0.72", edgecolors=INK,
                   linewidths=0.4, alpha=0.9, zorder=5)
        ax.scatter(pts[inl, 0], pts[inl, 1], s=30, c=col[inl], edgecolors="black", linewidths=0.45, zorder=6)
        ax.scatter(pts[outl, 0], pts[outl, 1], s=34, c=col[outl], edgecolors=OUTLIER, linewidths=1.4, zorder=6)

    # ---- zoomed insets ----
    zooms = pick_zooms(D, size=round(0.24 * min(W, H)))
    iw, ih, ig, ibot = 0.285, 0.215, 0.037, 0.150
    for z_idx, z in enumerate(zooms):
        lab = "ABC"[z_idx]
        s = z["size"]
        sx, sy = np.round(z["shift"]).astype(int)
        boxes = [(z["x0"], z["y0"]), (z["x0"] + sx, z["y0"] + sy)]
        for k, (bx, by) in enumerate(boxes):
            px, py = P(bx, by, k)
            ax.add_patch(patches.Rectangle((px, py), s, s, fill=False, lw=2.2, ec=ZOOM, zorder=9))
            ax.add_patch(patches.Rectangle((px, py), s, s, fill=False, lw=0.6, ec="black", zorder=9))
            ax.text(px + 1.5, py - 1.5, lab, color=ZOOM, fontsize=13, weight="bold", va="bottom", zorder=9,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2, foreground="black")])
        left = 0.035 + z_idx * (iw + ig)
        zax = fig.add_axes([left, ibot, iw, ih])
        zgap = 0.10 * s
        zax.set_xlim(0, 2 * s + zgap)
        zax.set_ylim(s, 0)
        zax.set_aspect("equal")
        zax.set_xticks([])
        zax.set_yticks([])
        for sp in zax.spines.values():
            sp.set_visible(False)
        for k, (bx, by) in enumerate(boxes):
            xo = k * (s + zgap)
            crop = D["rgb"][k][by:by + s, bx:bx + s]
            zax.imshow(crop, extent=(xo, xo + s, s, 0), interpolation="bicubic", zorder=0)
            zax.add_patch(patches.Rectangle((xo, 0), s, s, fill=False, lw=3.2, ec=ZOOM, zorder=7, clip_on=False))
            zax.text(xo + 1, s - 1.2, f"frame {D['i'] if k == 0 else D['j']}", color="white", fontsize=8.5,
                     weight="bold", zorder=8,
                     path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
            f = "ij"[k]
            u, v = D["kp"][f][:, 0] - bx, D["kp"][f][:, 1] - by
            sel = (u >= 0) & (u < s) & (v >= 0) & (v < s)
            zax.scatter(u[sel] + xo, v[sel], s=30, facecolors="none", edgecolors="white", linewidths=0.9, zorder=2)
        ua, va = c["u_i"] - boxes[0][0], c["v_i"] - boxes[0][1]
        ub, vb = c["u_j"] - boxes[1][0] + s + zgap, c["v_j"] - boxes[1][1]
        ina = (ua >= 0) & (ua < s) & (va >= 0) & (va < s)
        inb = (ub >= s + zgap) & (ub < 2 * s + zgap) & (vb >= 0) & (vb < s)
        both = ina & inb & pnp
        for msk, kw in ((both & inl, dict(lw=1.9, alpha=0.95, ls="-")), (both & outl, dict(lw=1.8, alpha=1.0, ls=(0, (3, 2))))):
            for idx in np.nonzero(msk)[0]:
                zax.plot([ua[idx], ub[idx]], [va[idx], vb[idx]], color=col[idx] if inl[idx] else OUTLIER,
                         solid_capstyle="round", zorder=4, **kw)
        for msk_pts, xs_, ys_ in ((ina, ua, va), (inb, ub, vb)):
            sel = msk_pts & inl
            zax.scatter(xs_[sel], ys_[sel], s=90, c=col[sel], edgecolors="black", linewidths=0.8, zorder=6)
            sel = msk_pts & outl
            zax.scatter(xs_[sel], ys_[sel], s=95, c=col[sel], edgecolors=OUTLIER, linewidths=2.0, zorder=6)
            sel = msk_pts & nodepth
            zax.scatter(xs_[sel], ys_[sel], s=40, marker="D", facecolors="0.72", edgecolors=INK, linewidths=0.6, zorder=5)
        nin, nout = int((both & inl).sum()), int((both & outl).sum())
        zax.text(0, -0.035, f"{lab}   {z['name']}   ·   {nin} inlier{'s' if nin != 1 else ''}"
                 + (f", {nout} outlier{'s' if nout != 1 else ''}" if nout else ""),
                 transform=zax.transAxes, fontsize=10, color=INK, weight="bold", va="top")
        # connectors from the main-image boxes to the inset halves
        for k, (bx, by) in enumerate(boxes):
            px, py = P(bx, by, k)
            con = ConnectionPatch(xyA=(px + s / 2, py + s), coordsA=ax.transData,
                                  xyB=(k * (s + zgap) + s / 2, 0), coordsB=zax.transData,
                                  color=ZOOM_LINK, lw=1.0, ls=(0, (4, 3)), alpha=0.8, zorder=20)
            fig.add_artist(con)

    # ---- header ----
    fig.text(0.035, 0.968, f"Sparse ORB feature correspondences  ·  frame {i} → frame {j}", fontsize=21,
             weight="bold", color=INK)
    fig.text(0.035, 0.943,
             f"Synthetic Stanford Bunny RGB-D sequence  ·  true motion 10° / {gt_step(i, j) * 100:.1f} cm  ·  crops of the "
             "640×480 renders  ·  every point and line is an actual ORB keypoint or match of the C++ run",
             fontsize=10.5, color="0.30")

    # ---- legend + colorbar ----
    cax = fig.add_axes([0.745, 0.900, 0.22, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(f"Depth in frame {i} (m)   near → far", fontsize=9, labelpad=3)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_linewidth(0.5)
    hand = [
        Line2D([], [], ls="none", marker="o", ms=7, mfc=CMAP(0.35), mec="black", mew=0.5,
               label="PnP correspondence (colour = depth)"),
        Line2D([], [], color=CMAP(0.35), lw=2.2, label="RANSAC inlier match"),
        Line2D([], [], color=OUTLIER, lw=1.8, ls=(0, (3, 2)), marker="o", ms=7, mfc=CMAP(0.6), mec=OUTLIER, mew=1.4,
               label="RANSAC outlier (rejected)"),
        Line2D([], [], ls="none", marker="D", ms=5.5, mfc="0.72", mec=INK, mew=0.5,
               label="filtered match without depth (dropped)"),
        Line2D([], [], ls="none", marker="o", ms=5.5, mfc="none", mec="0.45", mew=0.9,
               label="other ORB keypoint (white)"),
    ]
    fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.03, 0.927), ncol=3, frameon=False, fontsize=9,
               handlelength=2.6, columnspacing=1.4, labelspacing=0.5)

    # ---- funnel ----
    fy = 0.025
    steps = [(f"{n['kp_i']} / {n['kp_j']}", "ORB keypoints (frame i / i+1)", "0.30"),
             (f"{n['raw']}", "raw descriptor matches", "0.42"),
             (f"{n['filtered']}", f"Hamming filter (≤ {D['thr']:g})", "0.55"),
             (f"{n['pnp']}", "PnP 3D→2D (depth in frame i)", PNP),
             (f"{n['inliers']}", "RANSAC inliers (8 px)", (0.80, 0.42, 0.0)),
             (f"{n['icp']}", "ICP 3D→3D (depth in both)", ICP)]
    xs = [0.035, 0.155, 0.275, 0.415, 0.535, 0.675]
    for (num, lab, colr), x in zip(steps, xs):
        chip(fig, x, fy, num, lab, colr)
    for xa, xb in ((0.143, 0.155), (0.263, 0.275), (0.383, 0.415), (0.523, 0.535)):
        fig.add_artist(patches.FancyArrowPatch((xa + 0.001, fy + 0.025), (xb - 0.001, fy + 0.025),
                                               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                                               color="0.35", lw=1))
    fig.add_artist(patches.FancyArrowPatch((0.33, fy + 0.05), (0.73, fy + 0.05), transform=fig.transFigure,
                                           connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=10,
                                           color=ICP, lw=1.1))
    fig.text(0.80, fy + 0.042, "ORB match  +  depth at frame-i pixel", fontsize=9, color=INK, weight="bold")
    fig.text(0.80, fy + 0.025, "↓  back-project with K:  X = Z·K⁻¹(u, v, 1)", fontsize=8.5, color="0.3")
    fig.text(0.80, fy + 0.008, "3D → 2D for PnP  ·  3D → 3D for ICP", fontsize=9, color=INK, weight="bold")
    save(fig, stem)
    return zooms


# --------------------------------------------------------- figure 2 -------

def oblique(P, yaw=-38, pitch=18):
    """Fixed orthographic oblique view of camera-frame points (X right, Y down,
    Z forward) -> 2D drawing coordinates. Display only."""
    x, y, z = P[:, 0], -P[:, 1], P[:, 2]
    a, b = math.radians(yaw), math.radians(pitch)
    x1 = math.cos(a) * x + math.sin(a) * z
    z1 = -math.sin(a) * x + math.cos(a) * z
    y2 = math.cos(b) * y - math.sin(b) * z1
    return np.c_[x1, y2]


def depth_cloud(D, k, step=3):
    dep = D["dep"][k]
    v, u = np.mgrid[0:dep.shape[0]:step, 0:dep.shape[1]:step]
    z = dep[v, u] / 5000.0
    m = z > 0
    K = D["K"]
    return np.c_[(u[m] - K["cx"]) / K["fx"] * z[m], (v[m] - K["cy"]) / K["fy"] * z[m], z[m]]


def triad(ax, k):
    """Small camera-frame axes glyph (same oblique view) in the panel corner."""
    o = np.array([0.90, 0.12])
    for vec, lab in (((1, 0, 0), "X"), ((0, 1, 0), "Y"), ((0, 0, 1), "Z")):
        d = oblique(np.array([vec], float))[0]
        d = d / max(np.linalg.norm(d), 1e-9) * 0.06
        ax.annotate("", xy=o + d, xytext=o, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0, mutation_scale=8))
        ax.text(*(o + d * 1.35), lab, transform=ax.transAxes, fontsize=7.5, color="0.35", ha="center", va="center")
    ax.text(o[0], o[1] + 0.07, f"camera-{k} axes", transform=ax.transAxes, fontsize=7, color="0.45", ha="center")


def figure_pnp_icp(D, stem, n_examples=4):
    c, i, j = D["c"], D["i"], D["j"]
    norm = depth_norm(c)
    m_pnp, m_inl, m_icp = c["pnp_used"], c["pnp_inlier"], c["icp_used"]
    # examples: PnP inliers that are also ICP pairs, spread over the bunny (farthest-point sampling)
    cand = np.nonzero(m_inl & m_icp)[0]
    uv = np.c_[c["u_i"], c["v_i"]][cand]
    pick = [int(np.argmin(uv[:, 1]))]
    while len(pick) < min(n_examples, len(cand)):
        d = np.min(np.linalg.norm(uv[:, None] - uv[pick][None], axis=2), axis=1)
        pick.append(int(np.argmax(d)))
    ex = cand[pick]
    ex = ex[np.argsort(c["v_i"][ex])]

    P3i = np.c_[c["X_i"], c["Y_i"], c["Z_i"]]
    P3j = np.c_[c["X_j"], c["Y_j"], c["Z_j"]]
    x0, x1, y0, y1 = bunny_box(D, 18)

    fig = plt.figure(figsize=(13, 11.5), facecolor="white")
    fig.text(0.05, 0.955, f"What PnP and our ICP receive from frame pair {i} → {j}", fontsize=19, weight="bold", color=INK)
    fig.text(0.05, 0.928, "Actual correspondences of the C++ run. Numbered points are the same physical matches in "
             "every panel; colours encode depth in frame i (same scale as the main figure).", fontsize=10, color="0.3")
    fig.text(0.285, 0.885, f"FRAME i = {i}", fontsize=13, weight="bold", ha="center", color=INK)
    fig.text(0.745, 0.885, f"FRAME i+1 = {j}", fontsize=13, weight="bold", ha="center", color=INK)
    rows = [(0.535, "PnP", PNP), (0.115, "ICP", ICP)]
    axs = {}
    for yb, name, colr in rows:
        h_ = 0.33 if name == "PnP" else 0.31
        aL = fig.add_axes([0.07, yb, 0.43, h_])
        aR = fig.add_axes([0.53, yb, 0.43, h_])
        axs[name] = (aL, aR)
        fig.text(0.025, yb + 0.165, name, rotation=90, fontsize=20, weight="bold", color=colr, ha="center", va="center")
        for a in (aL, aR):
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("0.8")

    def cloud_panel(a, k, P3, mask, title, colr):
        ctx = oblique(depth_cloud(D, k))
        a.scatter(ctx[:, 0], ctx[:, 1], s=1.4, c="0.72", linewidths=0, zorder=0, rasterized=True)
        q = oblique(P3[mask])
        a.scatter(q[:, 0], q[:, 1], s=18, c=CMAP(norm(c["Z_i"][mask])), edgecolors="black", linewidths=0.3, zorder=2)
        triad(a, D["i"] if k == 0 else D["j"])
        a.set_aspect("equal")
        a.margins(0.05)
        a.text(0.0, -0.025, title, transform=a.transAxes, fontsize=10.5, weight="bold", color=colr, va="top")
        a.text(0.0, -0.075, "grey: every depth pixel of the frame, oblique 3D view, for context only (not used)",
               transform=a.transAxes, fontsize=7.5, color="0.45", va="top")
        return oblique(P3)

    # PnP row
    aL, aR = axs["PnP"]
    qL = cloud_panel(aL, 0, P3i, m_pnp, f"● 3D point from RGB-D depth  ({int(m_pnp.sum())}, camera-{i} frame)", PNP)
    aR.imshow(D["rgb"][1][y0:y1, x0:x1], extent=(x0, x1, y1, y0), interpolation="lanczos")
    aR.scatter(c["u_j"][m_pnp & ~m_inl], c["v_j"][m_pnp & ~m_inl], marker="x", s=30, c=[OUTLIER], linewidths=1.3, zorder=3)
    aR.scatter(c["u_j"][m_inl], c["v_j"][m_inl], marker="x", s=26, c=CMAP(norm(c["Z_i"][m_inl])), linewidths=1.4, zorder=3)
    aR.text(0.0, -0.025, "× 2D matched feature in the RGB image", transform=aR.transAxes, fontsize=10.5,
            weight="bold", color=PNP, va="top")
    aR.text(0.0, -0.075, f"{int(m_inl.sum())} RANSAC inliers (colour = depth of the 3D point), "
            f"{int((m_pnp & ~m_inl).sum())} outliers (magenta)", transform=aR.transAxes, fontsize=7.5, color="0.45", va="top")
    fig.text(0.515, 0.468, "solvePnPRansac:  find T₍ᵢ₊₁←ᵢ₎ minimising  Σ ‖uⱼ − π(K(R·Xᵢ + t))‖²   over the inliers",
             ha="center", fontsize=10, color=INK, bbox=dict(fc=(1, 0.95, 0.88), ec=PNP, boxstyle="round,pad=0.4"))
    # ICP row
    bL, bR = axs["ICP"]
    rL = cloud_panel(bL, 0, P3i, m_icp, f"● 3D point from frame i  ({int(m_icp.sum())} pairs, camera-{i} frame)", ICP)
    rR = cloud_panel(bR, 1, P3j, m_icp, f"● 3D point from frame i+1  (camera-{j} frame)", ICP)
    fig.text(0.515, 0.035, "SVD alignment + g2o refinement:  find (R, t) minimising  Σ ‖pᵢ − (R·qᵢ₊₁ + t)‖²   "
             "(all pairs, no outlier rejection)", ha="center", fontsize=10, color=INK,
             bbox=dict(fc=(0.98, 0.92, 0.99), ec=ICP, boxstyle="round,pad=0.4"))

    for num, e in enumerate(ex, 1):
        for (A, pa), (B, pb), colr in (((aL, qL[e]), (aR, (c["u_j"][e], c["v_j"][e])), PNP),
                                       ((bL, rL[e]), (bR, rR[e]), ICP)):
            fig.add_artist(ConnectionPatch(xyA=pa, coordsA=A.transData, xyB=pb, coordsB=B.transData, color=colr,
                                           lw=1.8, alpha=0.9, arrowstyle="-|>", mutation_scale=13,
                                           connectionstyle=f"arc3,rad={-0.12 if colr == PNP else 0.12}", zorder=30))
            for ax_, p in ((A, pa), (B, pb)):
                ax_.scatter([p[0]], [p[1]], s=150, facecolors="none", edgecolors=colr, linewidths=2.2, zorder=5)
                ax_.annotate(str(num), p, xytext=(7, 5), textcoords="offset points", fontsize=11, weight="bold",
                             color=colr, zorder=6,
                             path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")])
    cax = fig.add_axes([0.80, 0.962, 0.16, 0.010])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(f"Depth in frame {i} (m)   near → far", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    save(fig, stem)
    return ex


# ---------------------------------------------------------------- main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0, help="frame pair i -> i+1 (default 0)")
    ap.add_argument("--skip-cpp", action="store_true", help="reuse results/data/correspondence_export/")
    args = ap.parse_args()
    if not args.skip_cpp:
        run_export(args.pair)
    D = load(args.pair)
    i, j = D["i"], D["j"]
    zooms = figure_correspondences(D, f"bunny_correspondences_annotated_{i}_{j}")
    for z in zooms:
        print(f"[zoom] {z['name']}: frame-{i} box x={z['x0']:.0f} y={z['y0']:.0f} size={z['size']} "
              f"({z['n']} correspondences inside in both frames)")
    stem2 = "bunny_pnp_icp_correspondences" + ("" if i == 0 else f"_{i}_{j}")
    figure_pnp_icp(D, stem2)
    n = D["n"]
    print(f"[counts] pair {i}->{j}: ORB {n['kp_i']}/{n['kp_j']}, raw {n['raw']}, filtered {n['filtered']}, "
          f"PnP {n['pnp']}, inliers {n['inliers']}, ICP {n['icp']}")


if __name__ == "__main__":
    main()
