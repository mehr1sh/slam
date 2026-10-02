#!/usr/bin/env python3
"""Figures of the scratch pipeline and its comparison with the reference
(baseline ORB + RANSAC PnP) pipeline. NumPy + Matplotlib + the pure-Python
PNG reader in this directory; no computer-vision or imaging library, and no
reference program is run.

Inputs
  scratch    pnp_from_scratch/results/pipeline/  (run build/scratch_pipeline first):
             keypoints/frame_*.csv, correspondences/pair_0_1.csv, pairs.csv, trajectory.csv
  reference  docs/migration/baseline/  (frozen, tracked):
             reference_features/pair_0_1_{keypoints,raw_matches}.csv,
             correspondences/pair_0_1_correspondences.csv, metrics.json, slam_trajectory.csv
  images     data/synthetic_bunny/00000{0,1}.png, _depth.png

Outputs (generated, gitignored): pnp_from_scratch/results/figures/
  bunny_correspondences_0_1{,_inliers,_outliers}.png   scratch correspondences, RANSAC inliers/outliers
  fig03_orb_keypoints.png                              reference ORB keypoints, frames 0 and 1
  feature_comparison_0_1.png                           scratch vs reference: keypoints and matches
  pnp_comparison.png, pnp_comparison.md                scratch vs reference PnP (and ICP), all pairs
(+ .pdf of every figure)

Usage (from the repository root):  python3 pnp_from_scratch/tools/make_figures.py
"""

import colorsys
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from png_reader import read_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "synthetic_bunny"
PIPE = ROOT / "pnp_from_scratch" / "results" / "pipeline"
BASE = ROOT / "docs" / "migration" / "baseline"
OUT = ROOT / "pnp_from_scratch" / "results" / "figures"

SCRATCH = (0.10, 0.78, 1.00)   # same colours as the Blender scene
REF_PNP = (1.00, 0.62, 0.10)
ICP = (0.85, 0.40, 0.95)
GT = (0.15, 0.82, 0.45)
OUTLIER = (1.00, 0.10, 0.55)
NODEPTH = (0.70, 0.70, 0.70)
STROKE = [pe.withStroke(linewidth=2.2, foreground="black")]
INK = "0.12"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ data ---

def load_scratch_pair():
    kp = {k: rows(PIPE / "keypoints" / f"frame_{k:06d}.csv") for k in (0, 1)}
    c = rows(PIPE / "correspondences" / "pair_0_1.csv")
    m = dict(ui=np.array([fl(r["u_i"]) for r in c]), vi=np.array([fl(r["v_i"]) for r in c]),
             uj=np.array([fl(r["u_j"]) for r in c]), vj=np.array([fl(r["v_j"]) for r in c]),
             z=np.array([fl(r["Z_i"]) for r in c]),
             used=np.array([r["pnp_used"] == "1" for r in c]), inlier=np.array([r["pnp_inlier"] == "1" for r in c]))
    m["kp"] = {k: np.array([[fl(r["x"]), fl(r["y"]), fl(r["angle_deg"])] for r in kp[k]]) for k in (0, 1)}
    return m


def load_reference_pair():
    kp = rows(BASE / "reference_features" / "pair_0_1_keypoints.csv")
    c = rows(BASE / "correspondences" / "pair_0_1_correspondences.csv")
    m = dict(ui=np.array([fl(r["u_i"]) for r in c]), vi=np.array([fl(r["v_i"]) for r in c]),
             uj=np.array([fl(r["u_j"]) for r in c]), vj=np.array([fl(r["v_j"]) for r in c]),
             z=np.array([fl(r["Z_i"]) for r in c]),
             used=np.array([r["pnp_used"] == "1" for r in c]), inlier=np.array([r["pnp_inlier"] == "1" for r in c]))
    m["kp"] = {k: np.array([[fl(r["u"]), fl(r["v"]), fl(r["angle"])] for r in kp if r["frame"] == f])
               for k, f in ((0, "i"), (1, "j"))}
    m["raw"] = len(rows(BASE / "reference_features" / "pair_0_1_raw_matches.csv"))
    return m


# --------------------------------------------------------------- drawing ---

def id_colours(m):
    """Correspondence identity colour from the frame-0 position: hue = angle
    around the centroid (0 .. 0.8, skipping the outlier magenta), saturation
    = distance from it. The same colour marks both ends of a match."""
    u, v, use = m["ui"], m["vi"], m["used"]
    cu, cv = np.median(u[use]), np.median(v[use])
    ang = np.arctan2(v - cv, u - cu)
    r = np.hypot(u - cu, v - cv)
    r = np.clip(r / np.percentile(r[use], 95), 0, 1)
    return np.array([colorsys.hsv_to_rgb(0.8 * (a + math.pi) / (2 * math.pi), 0.5 + 0.5 * s, 1.0) for a, s in zip(ang, r)])


def bunny_box(depths, margin=12):
    mask = np.zeros_like(depths[0], dtype=bool)
    for d in depths:
        mask |= d > 0
    ys, xs = np.nonzero(mask)
    return (max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1),
            max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1))


class PairPanel:
    """Frames 0 and 1 side by side in one axes (crop-pixel coordinates)."""

    def __init__(self, ax, rgb, box, gap=0.08):
        self.ax, self.box = ax, box
        x0, x1, y0, y1 = box
        self.W, self.H = x1 - x0, y1 - y0
        self.off = self.W * (1 + gap)
        for k in (0, 1):
            ax.imshow(rgb[k][y0:y1, x0:x1], extent=(self.off * k, self.off * k + self.W, self.H, 0),
                      interpolation="lanczos", zorder=0)
            ax.text(self.off * k + 0.02 * self.W, 0.03 * self.H, f"Frame {k}", color="white", fontsize=12,
                    weight="bold", va="top", zorder=20, path_effects=STROKE)
        ax.set_xlim(0, self.off + self.W)
        ax.set_ylim(self.H, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    def a(self, m, sel):
        return np.c_[m["ui"][sel] - self.box[0], m["vi"][sel] - self.box[2]]

    def b(self, m, sel):
        return np.c_[m["uj"][sel] - self.box[0] + self.off, m["vj"][sel] - self.box[2]]

    def keypoints(self, m, s=5, alpha=0.5):
        for k in (0, 1):
            p = m["kp"][k]
            u, v = p[:, 0] - self.box[0], p[:, 1] - self.box[2]
            ok = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
            self.ax.scatter(u[ok] + self.off * k, v[ok], s=s, c="white", alpha=alpha, linewidths=0, zorder=1)

    def matches(self, m, col, show_inliers=True, show_outliers=True, show_nodepth=True, scale=1.0, numbers=False):
        inl, use = m["inlier"], m["used"]
        out, nod = use & ~inl, ~use
        size = 14 + 30 * (1 - np.clip((np.nan_to_num(m["z"], nan=0.5) - 0.44) / 0.14, 0, 1))  # near = larger
        if show_nodepth and nod.any():
            self.ax.add_collection(LineCollection(np.stack([self.a(m, nod), self.b(m, nod)], 1), colors=[NODEPTH],
                                                  linewidths=0.5 * scale, alpha=0.3, linestyles=(0, (1, 2)), zorder=2))
            for pts in (self.a(m, nod), self.b(m, nod)):
                self.ax.scatter(pts[:, 0], pts[:, 1], s=10 * scale, marker="D", c=[NODEPTH], edgecolors="0.25",
                                linewidths=0.4, zorder=4)
        if show_inliers and inl.any():
            self.ax.add_collection(LineCollection(np.stack([self.a(m, inl), self.b(m, inl)], 1), colors=col[inl],
                                                  linewidths=0.85 * scale, alpha=0.6, zorder=3))
            for pts in (self.a(m, inl), self.b(m, inl)):
                self.ax.scatter(pts[:, 0], pts[:, 1], s=size[inl] * scale, c=col[inl], edgecolors="black",
                                linewidths=0.5, zorder=6)
        if show_outliers and out.any():
            self.ax.add_collection(LineCollection(np.stack([self.a(m, out), self.b(m, out)], 1),
                                                  colors=col[out] if numbers else [OUTLIER],
                                                  linewidths=(1.5 if numbers else 1.0) * scale, alpha=0.9,
                                                  linestyles="-" if numbers else (0, (4, 2.5)), zorder=4))
            for pts in (self.a(m, out), self.b(m, out)):
                self.ax.scatter(pts[:, 0], pts[:, 1], s=size[out] * scale, c=col[out], edgecolors=OUTLIER,
                                linewidths=1.6, zorder=7)
            if numbers:
                for n, idx in enumerate(np.nonzero(out)[0], 1):
                    for p in (self.a(m, [idx])[0], self.b(m, [idx])[0]):
                        self.ax.annotate(str(n), p, xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                                         weight="bold", color=col[idx], zorder=9, path_effects=STROKE)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220 if ext == "png" else None, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] pnp_from_scratch/results/figures/{name}.png|pdf")


def header(fig, title, line):
    fig.text(0.01, 0.995, title, fontsize=14, weight="bold", va="top", color=INK)
    fig.text(0.01, 0.955, line, fontsize=9.5, va="top", color="0.35")


# --------------------------------------------------------------- figures ---

def correspondence_maps(rgb, box, s, pair):
    col = id_colours(s)
    n_use, n_inl = int(s["used"].sum()), int(s["inlier"].sum())
    counts = (f"{pair['keypoints_i']} / {pair['keypoints_j']} keypoints  ·  {pair['filtered_matches']} filtered matches  ·  "
              f"{n_use} valid 3D→2D  ·  {n_inl} RANSAC inliers")
    key = ("colour = correspondence (same colour in both frames)  ·  marker size = depth, near = larger  ·  "
           "solid = RANSAC inlier  ·  magenta dashed + ring = outlier  ·  grey ◇ = no depth  ·  white = other keypoints")
    for name, kw, title, line in (
            ("bunny_correspondences_0_1", {}, "Scratch correspondences  ·  Frame 0 → Frame 1", counts),
            ("bunny_correspondences_0_1_inliers", dict(show_outliers=False, show_nodepth=False),
             "Scratch correspondences  ·  RANSAC inliers only", f"{n_inl} of {n_use} valid 3D→2D correspondences"),
            ("bunny_correspondences_0_1_outliers", dict(show_inliers=False, show_nodepth=False, numbers=True),
             "Scratch correspondences  ·  RANSAC outliers only",
             f"{n_use - n_inl} of {n_use} rejected (reprojection error ≥ 8 px under the RANSAC pose)")):
        W, H = box[1] - box[0], box[3] - box[2]
        fig = plt.figure(figsize=(13, 13 * H / (2.08 * W) + 0.8))
        ax = fig.add_axes([0.01, 0.03, 0.98, 0.86])
        P = PairPanel(ax, rgb, box)
        if name.endswith("outliers"):
            for pts in (P.a(s, s["inlier"]), P.b(s, s["inlier"])):  # inliers as faint context
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="white", edgecolors="0.3", linewidths=0.3, zorder=2)
        else:
            P.keypoints(s)
        P.matches(s, col, **kw)
        header(fig, title, line)
        fig.text(0.99, 0.005, key if name == "bunny_correspondences_0_1" else
                 "colour = correspondence (same colour in both frames)", fontsize=7, color="0.45", ha="right")
        save(fig, name)


def orb_keypoints(rgb, box, r):
    W, H = box[1] - box[0], box[3] - box[2]
    fig = plt.figure(figsize=(13, 13 * H / (2.08 * W) + 0.8))
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.86])
    P = PairPanel(ax, rgb, box)
    for k in (0, 1):
        p = r["kp"][k]
        u, v, ang = p[:, 0] - box[0], p[:, 1] - box[2], np.radians(p[:, 2])
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ax.scatter(u[ok] + P.off * k, v[ok], s=16, facecolors="none", edgecolors=REF_PNP, linewidths=0.9, zorder=3)
        L = 6.0
        segs = [[(x + P.off * k, y), (x + P.off * k + L * math.cos(a), y + L * math.sin(a))]
                for x, y, a in zip(u[ok], v[ok], ang[ok])]
        ax.add_collection(LineCollection(segs, colors=[REF_PNP], linewidths=0.8, zorder=3))
    header(fig, "Reference ORB keypoints  ·  Frame 0 and Frame 1",
           f"{len(r['kp'][0])} / {len(r['kp'][1])} keypoints (baseline pipeline: ORB, 8-level pyramid; ticks = "
           f"orientation)  ·  frozen data: docs/migration/baseline/reference_features")
    save(fig, "fig03_orb_keypoints")


def feature_comparison(rgb, box, s, r, pair):
    W, H = box[1] - box[0], box[3] - box[2]
    # layout in inches: [keypoint panel W] gap [frame 0 | frame 1 panel 2.08 W], two rows with headers
    fig_w, margin, gap, head = 15.0, 0.15, 0.25, 0.75
    unit = (fig_w - 2 * margin - gap) / (1 + 2.08)  # inches per image width
    row_h = unit * H / W
    fig_h = 2 * (row_h + head) + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    rows_ = ((s, SCRATCH, "Scratch", "FAST (threshold 20, single scale) → intensity-centroid orientation → "
              "rotated BRIEF (own Gaussian pattern) → Hamming → scratch RANSAC"),
             (r, REF_PNP, "Reference (baseline)", "ORB detector (pyramid) → ORB descriptor → Hamming → "
              "library RANSAC (frozen baseline)"))
    for row, (m, colr, name, chain) in enumerate(rows_):
        y_top = fig_h - row * (row_h + head)  # inches from the bottom
        y0 = (y_top - head - row_h) / fig_h
        axk = fig.add_axes([margin / fig_w, y0, unit / fig_w, row_h / fig_h])
        axk.imshow(rgb[0][box[2]:box[3], box[0]:box[1]], extent=(0, W, H, 0), interpolation="lanczos")
        p = m["kp"][0]
        u, v, ang = p[:, 0] - box[0], p[:, 1] - box[2], np.radians(p[:, 2])
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        axk.scatter(u[ok], v[ok], s=14, facecolors="none", edgecolors=colr, linewidths=0.9)
        axk.add_collection(LineCollection([[(x, y), (x + 6 * math.cos(a), y + 6 * math.sin(a))]
                                           for x, y, a in zip(u[ok], v[ok], ang[ok])], colors=[colr], linewidths=0.8))
        axk.set_xlim(0, W)
        axk.set_ylim(H, 0)
        axk.axis("off")
        axk.text(0.03 * W, 0.04 * H, f"Frame 0 keypoints: {len(p)}", color="white", fontsize=10, weight="bold",
                 va="top", path_effects=STROKE)
        axm = fig.add_axes([(margin + unit + gap) / fig_w, y0, 2.08 * unit / fig_w, row_h / fig_h])
        P = PairPanel(axm, rgb, box)
        P.keypoints(m, s=4, alpha=0.45)
        P.matches(m, id_colours(m), scale=0.85)
        n_f, n_u, n_i = len(m["ui"]), int(m["used"].sum()), int(m["inlier"].sum())
        fig.text(margin / fig_w, (y_top - 0.12) / fig_h, name, fontsize=13, weight="bold", color=colr, va="top",
                 path_effects=[pe.withStroke(linewidth=0.6, foreground="0.3")])
        fig.text(margin / fig_w, (y_top - 0.45) / fig_h, f"{chain}   ·   {len(m['kp'][0])} / {len(m['kp'][1])} "
                 f"keypoints, {n_f} filtered matches, {n_u} valid 3D→2D, {n_i} RANSAC inliers", fontsize=8.6,
                 color="0.3", va="top")
    fig.text(0.99, 0.002, "different detectors → different keypoints and matches; colour = correspondence identity "
             "within each row  ·  ticks = keypoint orientation  ·  solid = RANSAC inlier, magenta dashed = outlier, "
             "grey ◇ = no depth", fontsize=7, color="0.45", ha="right")
    save(fig, "feature_comparison_0_1")


def pnp_comparison(pairs, traj, base, ref_traj):
    P = len(pairs)
    x = np.arange(P)
    bp = base["pairs"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.2))
    ax = axs[0, 0]
    for key_s, key_r, lab, ls in (("keypoints_i", "keypoints_i", "keypoints (frame i)", "-"),
                                  ("filtered_matches", "filtered_matches", "filtered matches", "--"),
                                  ("correspondences_3d2d", "pnp_correspondences", "valid 3D→2D", ":"),
                                  ("pnp_inliers", "pnp_inliers", "RANSAC inliers", "-.")):
        ax.plot(x, [fl(p[key_s]) for p in pairs], ls, color=SCRATCH, lw=1.6, label=f"scratch: {lab}")
        ax.plot(x, [p[key_r] for p in bp], ls, color=REF_PNP, lw=1.6, label=f"reference: {lab}")
    ax.set_title("Per pair: keypoints, matches, correspondences, inliers", fontsize=10)
    ax.set_xlabel("pair i → i+1 (i)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax = axs[0, 1]
    w = 0.4
    ax.bar(x - w / 2, [fl(p["rot_err_deg"]) for p in pairs], w, color=SCRATCH, label="scratch PnP")
    ax.bar(x + w / 2, [p["pnp_rot_err_deg"] for p in bp], w, color=REF_PNP, label="reference PnP")
    ax.set_title("Per-pair relative rotation error (true step 10°)", fontsize=10)
    ax.set_ylabel("degrees")
    ax.set_xlabel("pair i → i+1 (i)")
    ax.legend(fontsize=8, frameon=False)
    for ax, key_s, key_r, lab in ((axs[1, 0], "translation_error_m", "translation_error", "position error (m)"),
                                  (axs[1, 1], "rotation_error_deg", "rotation_error", "rotation error (°)")):
        f = np.arange(len(traj))
        ax.plot(f, [fl(t[key_s]) for t in traj], "-o", ms=3, color=SCRATCH, label="scratch PnP")
        ax.plot(f, [fl(t[f"pnp_{key_r}"]) for t in ref_traj], "-o", ms=3, color=REF_PNP, label="reference PnP")
        ax.plot(f, [fl(t[f"icp_{key_r}"]) for t in ref_traj], "-o", ms=3, color=ICP, label="reference ICP")
        ax.set_title(f"Accumulated trajectory {lab} vs ground truth", fontsize=10)
        ax.set_xlabel("frame")
        ax.legend(fontsize=8, frameon=False)
    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Scratch PnP (own features + RANSAC + linear PnP) vs reference PnP (ORB + library RANSAC PnP)",
                 fontsize=12, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "pnp_comparison")


def comparison_table(pairs, traj, base, ref_traj, s, r):
    bs = base["summary"]
    sc = lambda k: [fl(p[k]) for p in pairs]
    mean = lambda v: float(np.nanmean(v))
    med = lambda v: float(np.nanmedian(v))
    kp_s = [fl(pairs[0]["keypoints_i"])] + sc("keypoints_j")
    p0 = pairs[0]
    b0 = base["pairs"][0]
    lines = [
        "# Scratch PnP vs reference PnP (synthetic Bunny, 36 frames)", "",
        "Generated by `pnp_from_scratch/tools/make_figures.py`.",
        "",
        "- **Scratch:** project-owned FAST (threshold 20) + rotated BRIEF + Hamming; RANSAC (300 iterations, 8 px) around the linear PnP.",
        "- **Reference:** the baseline pipeline (ORB + ORB descriptor + Hamming; library RANSAC PnP), frozen in `docs/migration/baseline/`.",
        "- **Different features:** the two pipelines use different keypoints and correspondences.", "",
        "| | Scratch | Reference |", "|---|---|---|",
        f"| keypoints per frame (mean) | {mean(kp_s):.1f} | {bs['keypoints_per_frame']['mean']:.1f} |",
        f"| filtered matches per pair (mean) | {mean(sc('filtered_matches')):.1f} | {bs['filtered_matches']['mean']:.1f} |",
        f"| valid 3D→2D correspondences (mean) | {mean(sc('correspondences_3d2d')):.1f} | {bs['pnp_correspondences']['mean']:.1f} |",
        f"| RANSAC inliers (mean) | {mean(sc('pnp_inliers')):.1f} | {bs['pnp_inliers']['mean']:.1f} |",
        f"| inlier reprojection error (mean over pairs) | {mean(sc('reproj_inlier_mean_px')):.2f} px | {bs['pnp_reproj_inlier_mean_px']['mean']:.2f} px |",
        f"| per-pair rotation error, mean / median / max | {mean(sc('rot_err_deg')):.2f}° / {med(sc('rot_err_deg')):.2f}° / {np.nanmax(sc('rot_err_deg')):.2f}° | "
        f"{bs['pnp_rel_rot_err_deg']['mean']:.2f}° / {bs['pnp_rel_rot_err_deg']['median']:.2f}° / {bs['pnp_rel_rot_err_deg']['max']:.2f}° |",
        f"| per-pair translation error, mean / median | {mean(sc('trans_err_m')):.4f} m / {med(sc('trans_err_m')):.4f} m | "
        f"{bs['pnp_rel_trans_err_m']['mean']:.4f} m / {bs['pnp_rel_trans_err_m']['median']:.4f} m |",
        f"| failed pairs | {sum(p['pnp_ok'] != '1' for p in pairs)} | {bs['pnp_failed_pairs']} |",
        f"| trajectory error, mean over all 36 frames | {mean([fl(t['translation_error_m']) for t in traj]):.4f} m / "
        f"{mean([fl(t['rotation_error_deg']) for t in traj]):.2f}° | {bs['pnp_mean_trans_err_m']:.4f} m / "
        f"{mean([fl(t['pnp_rotation_error']) for t in ref_traj]):.2f}° |",
        f"| frames where scratch is closer to GT (position) | "
        f"{sum(fl(t['translation_error_m']) < fl(rt['pnp_translation_error']) for t, rt in zip(traj[1:], ref_traj[1:]))} of 35 | |",
        f"| **final trajectory error (frame 35)** | **{fl(traj[-1]['translation_error_m']):.4f} m / {fl(traj[-1]['rotation_error_deg']):.2f}°** | "
        f"**{bs['pnp_final_trans_err_m']:.4f} m / {bs['pnp_final_rot_err_deg']:.2f}°** (reference ICP: {bs['icp_final_trans_err_m']:.4f} m / {bs['icp_final_rot_err_deg']:.2f}°) |",
        "",
        "**Reading the final-frame numbers.**",
        "- Scratch is less accurate per pair. One bad pair (10→11, about 30°) lifts its trajectory above the reference for frames 11–28; later errors partly cancel it.",
        "- Its lower frame-35 error is therefore not uniform accuracy. The per-frame curves are in `pnp_comparison.png`.",
        "- The reference's per-pair errors are small but systematically one-signed (under-rotation), so they accumulate steadily.",        "", "## Pair 0 → 1", "", "| | Scratch | Reference |", "|---|---|---|",
        f"| keypoints frame 0 / 1 | {p0['keypoints_i']} / {p0['keypoints_j']} | {b0['keypoints_i']} / {b0['keypoints_j']} |",
        f"| raw / filtered matches | {p0['raw_matches']} / {p0['filtered_matches']} | {b0['raw_matches']} / {b0['filtered_matches']} |",
        f"| valid 3D→2D / RANSAC inliers | {p0['correspondences_3d2d']} / {p0['pnp_inliers']} | {b0['pnp_correspondences']} / {b0['pnp_inliers']} |",
        f"| rotation / translation error | {fl(p0['rot_err_deg']):.3f}° / {fl(p0['trans_err_m']):.4f} m | {b0['pnp_rot_err_deg']:.3f}° / {b0['pnp_trans_err_m']:.4f} m |",
        f"| inlier reprojection error (mean) | {fl(p0['reproj_inlier_mean_px']):.2f} px | {b0['reproj_inlier_mean_px']:.2f} px |",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pnp_comparison.md").write_text("\n".join(lines) + "\n")
    print("[table]  pnp_from_scratch/results/figures/pnp_comparison.md")
    print("\n".join(lines))


def main():
    for p in (PIPE / "pairs.csv", PIPE / "keypoints" / "frame_000000.csv"):
        if not p.exists():
            sys.exit(f"missing {p}: run pnp_from_scratch/build/scratch_pipeline first")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    rgb = [read_png(DS / f"{k:06d}.png") for k in (0, 1)]
    depth = [read_png(DS / f"{k:06d}_depth.png") for k in (0, 1)]
    box = bunny_box(depth)
    s, r = load_scratch_pair(), load_reference_pair()
    pairs, traj = rows(PIPE / "pairs.csv"), rows(PIPE / "trajectory.csv")
    base = json.loads((BASE / "metrics.json").read_text())
    ref_traj = rows(BASE / "slam_trajectory.csv")
    correspondence_maps(rgb, box, s, pairs[0])
    orb_keypoints(rgb, box, r)
    feature_comparison(rgb, box, s, r, pairs[0])
    pnp_comparison(pairs, traj, base, ref_traj)
    comparison_table(pairs, traj, base, ref_traj, s, r)


if __name__ == "__main__":
    main()
