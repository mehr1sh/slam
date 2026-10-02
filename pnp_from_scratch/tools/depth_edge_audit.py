#!/usr/bin/env python3
"""Read-only depth-edge audit of the scratch correspondences (NumPy + Matplotlib).

Input: the output of pnp_from_scratch/build/depth_edge_experiment
(pnp_from_scratch/results/depth_edge/), the dataset images and ground truth.
The production pipeline is not run or changed.

1. Statistics of the depth neighbourhood of every PnP input / RANSAC inlier,
   split by the experiment's depth-edge flag: valid pixels in the window,
   depth range, largest adjacent jump, gradient at the sample.
2. The fixed-3D-point test, with ground truth: the 3D point X_i (frame-i depth
   at the frame-i keypoint) is moved into camera j with the TRUE motion and
   compared with X_j, the back-projection of the matched frame-j keypoint with
   frame-j depth. For a fixed surface point the two agree to within the depth
   and localisation noise; a point that slides along the silhouette does not.
3. Figures (--out, default pnp_from_scratch/results/figures/depth_edge/):
     depth_edge_correspondences_0_1   all RANSAC inliers by class; interior only; edge only
     depth_edge_map_0                  frame-0 depth, the edge rule's pixels (level 0 and 7 radii), inliers
     depth_edge_statistics             neighbourhood statistics and the fixed-point test, by class
     depth_edge_experiment             per-pair signed rotation error and trajectory error: all vs interior

Usage (repository root): python3 pnp_from_scratch/tools/depth_edge_audit.py
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import make_figures as mf  # noqa: E402
from png_reader import read_png  # noqa: E402

ROOT = mf.ROOT
EXP = ROOT / "pnp_from_scratch" / "results" / "depth_edge"
DS = ROOT / "data" / "synthetic_bunny"
FX, FY, CX, CY = 520.9, 521.0, 325.1, 249.7
INTERIOR = (0.10, 0.75, 0.55)
EDGE = (1.00, 0.45, 0.10)


def gt_poses():
    gt = [list(map(float, l.split())) for l in open(DS / "groundtruth.txt") if not l.startswith("#")]
    out = []
    for g in gt:
        x, y, z, w = g[4:8]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        out.append((R, np.array(g[1:4])))
    return out


def load_pair(i):
    rows = mf.rows(EXP / "correspondences" / f"pair_{i}_{i + 1}.csv")
    a = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}
    for k in ("edge", "inlier_all", "inlier_interior", "level_i", "radius_px", "window_pixels", "valid_pixels"):
        a[k] = a[k].astype(int)
    return a


def fixed_point_test(a, i, T):
    """3D distance (mm) between the true motion of X_i and the frame-j back-projection,
    and the image residual (observed - predicted) in px."""
    (Ri, ti), (Rj, tj) = T[i], T[i + 1]
    R, t = Rj.T @ Ri, Rj.T @ (ti - tj)
    X = np.c_[a["X_i"], a["Y_i"], a["Z_i"]]
    P = X @ R.T + t
    zj = a["depth_raw_j"] / 5000
    Q = np.c_[(a["u_j"] - CX) / FX * zj, (a["v_j"] - CY) / FY * zj, zj]
    d3 = np.where(zj > 0, np.linalg.norm(P - Q, axis=1) * 1000, np.nan)
    pu, pv = FX * P[:, 0] / P[:, 2] + CX, FY * P[:, 1] / P[:, 2] + CY
    return d3, a["u_j"] - pu, a["v_j"] - pv


def pct(v, q=(10, 50, 90)):
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.2f}" for x in np.percentile(v, q)) if len(v) else "-"


def statistics(T):
    cols = {k: [] for k in ("edge", "inl", "valid_frac", "range_mm", "jump_mm", "grad_mm", "radius", "d3", "eu", "ev",
                            "level")}
    for i in range(35):
        a = load_pair(i)
        d3, eu, ev = fixed_point_test(a, i, T)
        cols["edge"].append(a["edge"])
        cols["inl"].append(a["inlier_all"])
        cols["valid_frac"].append(a["valid_pixels"] / a["window_pixels"])
        cols["range_mm"].append((a["z_max_m"] - a["z_min_m"]) * 1000)
        cols["jump_mm"].append(a["max_jump_m"] * 1000)
        cols["grad_mm"].append(a["gradient_m_per_px"] * 1000)
        cols["radius"].append(a["radius_px"])
        cols["d3"].append(d3)
        cols["eu"].append(eu)
        cols["ev"].append(ev)
        cols["level"].append(a["level_i"])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    lines = ["Depth neighbourhood of the RANSAC inliers (all 35 pairs, 'all' run), percentiles 10 / 50 / 90", ""]
    inl = c["inl"] == 1
    for name, sel in (("interior", inl & (c["edge"] == 0)), ("edge", inl & (c["edge"] == 1))):
        lines += [f"{name} inliers: {sel.sum()} ({100 * sel.sum() / inl.sum():.1f} %)",
                  f"  valid pixels in the window (fraction): {pct(c['valid_frac'][sel])}",
                  f"  depth range in the window (mm):        {pct(c['range_mm'][sel])}",
                  f"  largest adjacent depth jump (mm):      {pct(c['jump_mm'][sel])}",
                  f"  |depth gradient| at the sample (mm/px): {pct(c['grad_mm'][sel])}"
                  f"  (undefined for {np.isnan(c['grad_mm'][sel]).sum()}: a 4-neighbour has no depth)",
                  f"  window radius (px):                     {pct(c['radius'][sel].astype(float))}",
                  f"  keypoint level, mean:                   {c['level'][sel].mean():.2f}", ""]
    lines += ["Fixed-3D-point test with the TRUE motion (all inliers; d3 = |T_gt X_i - backproject(u_j, depth_j)|)", ""]
    for name, sel in (("interior", inl & (c["edge"] == 0)), ("edge", inl & (c["edge"] == 1))):
        d3 = c["d3"][sel]
        lines += [f"{name}: d3 (mm) percentiles 10/50/90: {pct(d3)};  d3 > 5 mm: {np.mean(d3[np.isfinite(d3)] > 5) * 100:.1f} %;"
                  f"  no frame-j depth at the match: {np.isnan(d3).sum()}",
                  f"   image residual observed - true projection (px): mean u {np.nanmean(c['eu'][sel]):+.3f}, "
                  f"mean v {np.nanmean(c['ev'][sel]):+.3f}; median |r| {np.nanmedian(np.hypot(c['eu'][sel], c['ev'][sel])):.2f}"]
    return c, lines


def correspondence_figure(T, rgb, box):
    a = load_pair(0)
    inl = a["inlier_all"] == 1
    m = {"ui": a["u_i"], "vi": a["v_i"], "uj": a["u_j"], "vj": a["v_j"], "used": np.ones(len(inl), bool)}
    ident = mf.id_colours(m)
    W, H = box[1] - box[0], box[3] - box[2]
    fig = plt.figure(figsize=(13, 3 * (13 * H / (2.08 * W)) + 1.6))
    rows_ = (("All RANSAC inliers, by class", inl, None),
             ("Interior inliers only (identity colours)", inl & (a["edge"] == 0), ident),
             ("Depth-edge inliers only (identity colours)", inl & (a["edge"] == 1), ident))
    for r, (title, sel, col) in enumerate(rows_):
        ax = fig.add_axes([0.01, 1 - (r + 1) * 0.325 + 0.005, 0.98, 0.29])
        P = mf.PairPanel(ax, rgb, box)
        cc = np.array([EDGE if e else INTERIOR for e in a["edge"]]) if col is None else col
        pa, pb = P.a(m, sel), P.b(m, sel)
        ax.add_collection(LineCollection(np.stack([pa, pb], 1), colors=cc[sel], linewidths=0.7, alpha=0.55, zorder=3))
        for pts in (pa, pb):
            ax.scatter(pts[:, 0], pts[:, 1], s=16, c=cc[sel], edgecolors="black", linewidths=0.4, zorder=5)
        ax.set_title(f"{title}: {sel.sum()}", fontsize=11, loc="left")
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", mfc=INTERIOR, mec="black", label="interior (depth-stable)"),
                        Line2D([], [], ls="none", marker="o", mfc=EDGE, mec="black", label="depth edge / silhouette")],
               loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, 0.003, "Depth edge: within the FAST circle radius (3 × level scale) of the depth sample, a pixel has no "
             "depth or two adjacent pixels differ by more than (Z/fx)·tan 80°.  Frame-0 RANSAC inliers of the default "
             "pipeline; lines join the two ends of a match.", fontsize=7.5, color="0.35")
    mf.save(fig, "depth_edge_correspondences_0_1")


def edge_mask(depth, radius, incidence=80.0):
    """Pixel-wise version of the experiment's rule (used for display only)."""
    z = depth.astype(float) / 5000
    H, W = z.shape
    bad = np.zeros_like(z, bool)  # pixels adjacent to a jump or without depth
    tau = z / FX * math.tan(math.radians(incidence))
    jx = (z[:, 1:] > 0) & (z[:, :-1] > 0) & (np.abs(z[:, 1:] - z[:, :-1]) > np.minimum(tau[:, 1:], tau[:, :-1]))
    jy = (z[1:, :] > 0) & (z[:-1, :] > 0) & (np.abs(z[1:, :] - z[:-1, :]) > np.minimum(tau[1:, :], tau[:-1, :]))
    bad[:, 1:] |= jx
    bad[:, :-1] |= jx
    bad[1:, :] |= jy
    bad[:-1, :] |= jy
    bad |= z == 0
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = xx ** 2 + yy ** 2 <= radius ** 2
    out = np.zeros_like(bad)
    pad = np.pad(bad, radius, constant_values=True)
    for dy, dx in zip(yy[disc], xx[disc]):
        out |= pad[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
    return out & (z > 0)


def depth_map_figure(box):
    d = read_png(DS / "000000_depth.png")
    a = load_pair(0)
    inl = a["inlier_all"] == 1
    x0, x1, y0, y1 = box
    z = np.where(d > 0, d / 5000, np.nan)[y0:y1, x0:x1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    im = axs[0].imshow(z, cmap="viridis")
    fig.colorbar(im, ax=axs[0], fraction=0.04, label="depth (m)")
    axs[0].set_title("Frame 0 depth", fontsize=10)
    for ax, r, lvl in ((axs[1], 3, 0), (axs[2], 11, 7)):
        m = edge_mask(d, r)[y0:y1, x0:x1]
        show = np.zeros(m.shape + (3,))
        show[~np.isnan(z)] = (0.30, 0.30, 0.32)
        show[m] = EDGE
        ax.imshow(show)
        ax.set_title(f"Depth-edge pixels for a level-{lvl} keypoint (radius {r} px)", fontsize=10)
    for ax in axs[1:]:
        for e, colr in ((0, INTERIOR), (1, EDGE)):
            sel = inl & (a["edge"] == e)
            ax.scatter(a["u_i"][sel] - x0, a["v_i"][sel] - y0, s=14, c=[colr], edgecolors="white" if e else "black",
                       linewidths=0.5, zorder=3)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"Where the depth-edge rule applies (frame 0). Points: the {inl.sum()} RANSAC inliers of pair 0→1; "
                 f"{(inl & (a['edge'] == 1)).sum()} edge (orange, white ring) / {(inl & (a['edge'] == 0)).sum()} interior "
                 "(green). The rule uses each keypoint's own radius.", fontsize=10)
    fig.tight_layout()
    mf.save(fig, "depth_edge_map_0")


def statistics_figure(c):
    inl = c["inl"] == 1
    sel = {"interior": inl & (c["edge"] == 0), "edge": inl & (c["edge"] == 1)}
    colr = {"interior": INTERIOR, "edge": EDGE}
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.2))
    spec = ((axs[0, 0], "valid_frac", "valid depth pixels in the window (fraction)", np.linspace(0, 1, 26), False),
            (axs[0, 1], "range_mm", "depth range in the window (mm)", np.logspace(-1, 2.5, 36), True),
            (axs[0, 2], "jump_mm", "largest adjacent depth jump (mm)", np.logspace(-1, 2.5, 36), True),
            (axs[1, 0], "grad_mm", "|depth gradient| at the sample (mm/px; 0 drawn at 0.01)", np.logspace(-2, 2, 36), True),
            (axs[1, 1], "d3", "fixed-point test: |T_gt X_i − X_j| (mm)", np.logspace(-1, 2.5, 36), True))
    for ax, k, lab, bins, logx in spec:
        for n, s in sel.items():
            v = c[k][s]
            v = v[np.isfinite(v)]
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color=colr[n], alpha=0.6, label=f"{n} (n = {len(v)})",
                    density=True)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 2].axvline(0.931 * math.tan(math.radians(80)), color="0.2", ls="--", lw=1)
    axs[0, 2].text(0.931 * math.tan(math.radians(80)) * 1.08, axs[0, 2].get_ylim()[1] * 0.9,
                   "(Z/fx)·tan 80°\nat Z = 0.486 m", fontsize=7.5)
    ax = axs[1, 2]
    for n, s in sel.items():
        ax.scatter(c["eu"][s], c["ev"][s], s=3, color=colr[n], alpha=0.35, label=n)
        ax.scatter([np.nanmean(c["eu"][s])], [np.nanmean(c["ev"][s])], s=80, marker="X", color=colr[n],
                   edgecolors="black", zorder=5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("image residual u: observed − true projection (px)")
    ax.set_ylabel("v (px)")
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    fig.suptitle("Depth neighbourhood of the RANSAC inliers, all 35 pairs (default pipeline): interior vs depth edge",
                 fontsize=11)
    fig.tight_layout()
    mf.save(fig, "depth_edge_statistics")


def experiment_figure():
    p = mf.rows(EXP / "pairs.csv")
    ta, ti = mf.rows(EXP / "trajectory_all.csv"), mf.rows(EXP / "trajectory_interior.csv")
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(35)
    sa = np.array([float(r["all_signed_rot_deg"]) for r in p])
    si = np.array([float(r["interior_signed_rot_deg"]) for r in p])
    axs[0].bar(x - 0.2, sa, 0.4, color=mf.SCRATCH_REF, label=f"all depth correspondences (sum {sa.sum():+.1f}°)")
    axs[0].bar(x + 0.2, np.clip(si, -6, 6), 0.4, color=INTERIOR, label=f"interior only (sum {si.sum():+.1f}°)")
    for k in np.nonzero(np.abs(si) > 6)[0]:
        axs[0].text(k + 0.2, 6.1, f"{si[k]:+.0f}°\n({p[k]['interior_inliers']}/{p[k]['interior_correspondences']})",
                    ha="center", fontsize=7, color="0.2")
    axs[0].axhline(0, color="0.3", lw=0.8)
    axs[0].set_ylim(-6, 7.5)
    axs[0].set_xlabel("pair i → i+1 (i)")
    axs[0].set_ylabel("estimated − true rotation (°)")
    axs[0].set_title("Signed per-pair rotation error (bars clipped at ±6°; label = inliers / inputs)", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8)
    f = np.arange(36)
    for rows_, colr, lab in ((ta, mf.SCRATCH_REF, "all depth correspondences"), (ti, INTERIOR, "interior only")):
        axs[1].plot(f, [float(r["translation_error_m"]) for r in rows_], "-o", ms=3, color=colr, label=lab)
    axs[1].set_xlabel("frame")
    axs[1].set_ylabel("camera-centre error (m)")
    axs[1].set_title("Accumulated trajectory position error", fontsize=10)
    axs[1].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    mf.save(fig, "depth_edge_experiment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pnp_from_scratch/results/figures/depth_edge")
    mf.OUT = (ROOT / ap.parse_args().out).resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    T = gt_poses()
    c, lines = statistics(T)
    print("\n".join(lines))
    rgb = [read_png(DS / f"{k:06d}.png") for k in (0, 1)]
    box = mf.bunny_box([read_png(DS / f"{k:06d}_depth.png") for k in (0, 1)])
    correspondence_figure(T, rgb, box)
    depth_map_figure(box)
    statistics_figure(c)
    experiment_figure()


if __name__ == "__main__":
    main()
