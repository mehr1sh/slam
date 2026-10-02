#!/usr/bin/env python3
"""Figures of the scratch pipeline and its comparison with the reference
(baseline ORB + RANSAC PnP) pipeline. NumPy + Matplotlib + the pure-Python
PNG reader in this directory; no computer-vision or imaging library, and no
reference program is run.

Inputs
  scratch    a scratch_pipeline output directory (--pipeline; default
             pnp_from_scratch/results/pipeline, the single-scale default run):
             keypoints/frame_*.csv, correspondences/pair_0_1.csv, pairs.csv, trajectory.csv
  reference  docs/migration/baseline/  (frozen, tracked):
             reference_features/pair_0_1_{keypoints,raw_matches}.csv,
             correspondences/pair_0_1_correspondences.csv, metrics.json, slam_trajectory.csv
  images     data/synthetic_bunny/00000{0,1}.png, _depth.png

Outputs (generated, gitignored): --out (default pnp_from_scratch/results/figures/)
  bunny_correspondences_0_1{,_inliers,_outliers}.png   scratch correspondences, RANSAC inliers/outliers
  fig03_orb_keypoints.png                              reference ORB keypoints, frames 0 and 1
                                                       (colour = pyramid level, size = scale, tick = orientation)
  feature_comparison_0_1.png                           scratch vs reference: keypoints (level/scale/orientation)
                                                       and correspondences
  pyramid_levels_0.png                                 keypoints per pyramid level, scratch vs reference, frame 0
  pnp_comparison.png, pnp_comparison.md                scratch single scale / scratch pyramid (whichever runs
                                                       exist: results/pipeline, results/pipeline_pyramid) vs
                                                       reference PnP and ICP, all pairs
(+ .pdf of every figure)

Usage (from the repository root):
  python3 pnp_from_scratch/tools/make_figures.py                                    # single-scale run
  python3 pnp_from_scratch/tools/make_figures.py --pipeline pnp_from_scratch/results/pipeline_pyramid \
                                                 --out pnp_from_scratch/results/figures_pyramid
"""

import argparse
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
SCRATCH_PYR = (0.12, 0.38, 0.95)
LEVEL_COLOURS = [plt.get_cmap("plasma")(0.08 + 0.12 * l) for l in range(8)]  # pyramid level 0 .. 7
RUNS = {"single": ROOT / "pnp_from_scratch" / "results" / "pipeline",
        "pyramid": ROOT / "pnp_from_scratch" / "results" / "pipeline_pyramid"}


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
    """kp arrays: x, y (original image), angle_deg, scale, level."""
    kp = {k: rows(PIPE / "keypoints" / f"frame_{k:06d}.csv") for k in (0, 1)}
    c = rows(PIPE / "correspondences" / "pair_0_1.csv")
    m = dict(ui=np.array([fl(r["u_i"]) for r in c]), vi=np.array([fl(r["v_i"]) for r in c]),
             uj=np.array([fl(r["u_j"]) for r in c]), vj=np.array([fl(r["v_j"]) for r in c]),
             z=np.array([fl(r["Z_i"]) for r in c]),
             used=np.array([r["pnp_used"] == "1" for r in c]), inlier=np.array([r["pnp_inlier"] == "1" for r in c]))
    m["kp"] = {k: np.array([[fl(r["x"]), fl(r["y"]), fl(r["angle_deg"]), fl(r.get("scale", 1)), fl(r.get("level", 0))]
                            for r in kp[k]]) for k in (0, 1)}
    return m


def load_reference_pair():
    kp = rows(BASE / "reference_features" / "pair_0_1_keypoints.csv")
    c = rows(BASE / "correspondences" / "pair_0_1_correspondences.csv")
    m = dict(ui=np.array([fl(r["u_i"]) for r in c]), vi=np.array([fl(r["v_i"]) for r in c]),
             uj=np.array([fl(r["u_j"]) for r in c]), vj=np.array([fl(r["v_j"]) for r in c]),
             z=np.array([fl(r["Z_i"]) for r in c]),
             used=np.array([r["pnp_used"] == "1" for r in c]), inlier=np.array([r["pnp_inlier"] == "1" for r in c]))
    m["kp"] = {k: np.array([[fl(r["u"]), fl(r["v"]), fl(r["angle"]), 1.2 ** int(r["octave"]), int(r["octave"])]
                            for r in kp if r["frame"] == f]) for k, f in ((0, "i"), (1, "j"))}
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
    print(f"[figure] {OUT.relative_to(ROOT)}/{name}.png|pdf")


def header(fig, title, line):
    fig.text(0.01, 0.995, title, fontsize=14, weight="bold", va="top", color=INK)
    fig.text(0.01, 0.955, line, fontsize=9.5, va="top", color="0.35")


def draw_keypoints(ax, kp, box, xoff=0.0):
    """Keypoints with their scale and orientation: circle radius 3 * scale crop
    pixels, tick of length 5 * scale along the orientation, colour = pyramid level."""
    W, H = box[1] - box[0], box[3] - box[2]
    u, v = kp[:, 0] - box[0], kp[:, 1] - box[2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for idx in np.argsort(-kp[:, 4])[::1]:  # coarse levels first, fine levels on top
        if not ok[idx]:
            continue
        x, y, a, sc, lv = u[idx] + xoff, v[idx], math.radians(kp[idx, 2]), kp[idx, 3], int(kp[idx, 4])
        c = LEVEL_COLOURS[lv]
        ax.add_patch(plt.Circle((x, y), 3.0 * sc, fill=False, ec=c, lw=0.8, zorder=3 + (8 - lv) * 0.01))
        ax.plot([x, x + 5.0 * sc * math.cos(a)], [y, y + 5.0 * sc * math.sin(a)], color=c, lw=0.8, zorder=3)


def level_legend(fig, x, y, levels):
    handles = [Line2D([], [], ls="none", marker="o", mfc="none", mec=LEVEL_COLOURS[l], ms=4 + 1.6 * l,
                      label=f"{l}") for l in range(levels)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=levels, frameon=False, fontsize=7.5,
               title="pyramid level (circle size = scale)", title_fontsize=7.5, handletextpad=0.2, columnspacing=0.8)


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
    fig = plt.figure(figsize=(13, 13 * H / (2.08 * W) + 1.0))
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.84])
    P = PairPanel(ax, rgb, box)
    for k in (0, 1):
        draw_keypoints(ax, r["kp"][k], box, P.off * k)
    header(fig, "Reference ORB keypoints  ·  Frame 0 and Frame 1",
           f"{len(r['kp'][0])} / {len(r['kp'][1])} keypoints (baseline pipeline: ORB on an 8-level pyramid, scale 1.2)  ·  "
           f"frozen data: docs/migration/baseline/reference_features")
    level_legend(fig, 0.60, 0.985, 8)
    save(fig, "fig03_orb_keypoints")


def feature_comparison(rgb, box, s, r, pair):
    W, H = box[1] - box[0], box[3] - box[2]
    # layout in inches: [keypoint panel W] gap [frame 0 | frame 1 panel 2.08 W], two rows with headers
    fig_w, margin, gap, head = 15.0, 0.15, 0.25, 0.75
    unit = (fig_w - 2 * margin - gap) / (1 + 2.08)  # inches per image width
    row_h = unit * H / W
    fig_h = 2 * (row_h + head) + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    nlev = int(s["kp"][0][:, 4].max()) + 1
    scale_txt = f"{nlev}-level pyramid (scale 1.2)" if nlev > 1 else "single scale"
    rows_ = ((s, SCRATCH_PYR if nlev > 1 else SCRATCH, "Scratch", f"own {scale_txt} → FAST (threshold 20) → "
              "intensity-centroid orientation → rotated BRIEF (own Gaussian pattern) → Hamming → scratch RANSAC"),
             (r, REF_PNP, "Reference (baseline)", "ORB detector (pyramid) → ORB descriptor → Hamming → "
              "library RANSAC (frozen baseline)"))
    for row, (m, colr, name, chain) in enumerate(rows_):
        y_top = fig_h - row * (row_h + head)  # inches from the bottom
        y0 = (y_top - head - row_h) / fig_h
        axk = fig.add_axes([margin / fig_w, y0, unit / fig_w, row_h / fig_h])
        axk.imshow(rgb[0][box[2]:box[3], box[0]:box[1]], extent=(0, W, H, 0), interpolation="lanczos")
        p = m["kp"][0]
        draw_keypoints(axk, p, box)
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
    level_legend(fig, 0.60, 1.0, 8)
    fig.text(0.99, 0.002, "left: keypoints (colour = pyramid level, circle = scale, tick = orientation)  ·  right: "
             "correspondences (colour = correspondence identity within each row; solid = RANSAC inlier, magenta dashed "
             "= outlier, grey ◇ = no depth)  ·  different detectors → different keypoints and matches",
             fontsize=7, color="0.45", ha="right")
    save(fig, "feature_comparison_0_1")


def pyramid_levels(s, r):
    lv_s = np.bincount(s["kp"][0][:, 4].astype(int), minlength=8)
    lv_r = np.bincount(r["kp"][0][:, 4].astype(int), minlength=8)
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.4
    ax.bar(x - w / 2, lv_s[:8], w, color=SCRATCH_PYR, label=f"scratch (total {int(lv_s.sum())})")
    ax.bar(x + w / 2, lv_r[:8], w, color=REF_PNP, label=f"reference ORB (total {int(lv_r.sum())})")
    for xi, (a, b) in enumerate(zip(lv_s[:8], lv_r[:8])):
        ax.text(xi - w / 2, a + 1, str(a), ha="center", fontsize=7.5)
        ax.text(xi + w / 2, b + 1, str(b), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{round(640 / 1.2 ** l)}x{round(480 / 1.2 ** l)}" for l in range(8)], fontsize=7.5)
    ax.set_xlabel("pyramid level (image size)")
    ax.set_ylabel("keypoints")
    ax.set_title("Frame 0: keypoints per pyramid level (FAST threshold 20, scale 1.2)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "pyramid_levels_0")


def pnp_comparison(runs, base, ref_traj):
    P = len(next(iter(runs.values()))[1])
    x = np.arange(P)
    bp = base["pairs"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.2))
    ax = axs[0, 0]
    ax.plot(x, [p["pnp_inliers"] for p in bp], "-", color=REF_PNP, lw=1.8, label="reference: RANSAC inliers")
    ax.plot(x, [p["pnp_correspondences"] for p in bp], ":", color=REF_PNP, lw=1.6, label="reference: valid 3D→2D")
    for name, (colr, pairs, traj) in runs.items():
        ax.plot(x, [fl(p["pnp_inliers"]) for p in pairs], "-", color=colr, lw=1.8, label=f"scratch {name}: RANSAC inliers")
        ax.plot(x, [fl(p["correspondences_3d2d"]) for p in pairs], ":", color=colr, lw=1.6,
                label=f"scratch {name}: valid 3D→2D")
    ax.set_title("Per pair: valid 3D→2D correspondences and RANSAC inliers", fontsize=10)
    ax.set_xlabel("pair i → i+1 (i)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax = axs[0, 1]
    n = len(runs) + 1
    w = 0.8 / n
    for k, (name, (colr, pairs, traj)) in enumerate(runs.items()):
        ax.bar(x - 0.4 + w * (k + 0.5), [fl(p["rot_err_deg"]) for p in pairs], w, color=colr, label=f"scratch {name}")
    ax.bar(x - 0.4 + w * (n - 0.5), [p["pnp_rot_err_deg"] for p in bp], w, color=REF_PNP, label="reference PnP")
    ax.set_title("Per-pair relative rotation error (true step 10°)", fontsize=10)
    ax.set_ylabel("degrees")
    ax.set_xlabel("pair i → i+1 (i)")
    ax.legend(fontsize=8, frameon=False)
    for ax, key_s, key_r, lab in ((axs[1, 0], "translation_error_m", "translation_error", "position error (m)"),
                                  (axs[1, 1], "rotation_error_deg", "rotation_error", "rotation error (°)")):
        f = np.arange(len(ref_traj))
        for name, (colr, pairs, traj) in runs.items():
            ax.plot(f, [fl(t[key_s]) for t in traj], "-o", ms=3, color=colr, label=f"scratch {name}")
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


def comparison_table(runs, base, ref_traj):
    bs = base["summary"]
    b0 = base["pairs"][0]
    mean = lambda v: float(np.nanmean(v))
    med = lambda v: float(np.nanmedian(v))
    names = list(runs)
    head = "| | " + " | ".join(f"Scratch {n}" for n in names) + " | Reference |"
    sep = "|---" * (len(names) + 2) + "|"

    def row(label, fn, ref):
        return f"| {label} | " + " | ".join(fn(runs[n][1], runs[n][2]) for n in names) + f" | {ref} |"

    col = lambda p, k: [fl(r[k]) for r in p]
    lines = [
        "# Scratch PnP vs reference PnP (synthetic Bunny, 36 frames)", "",
        "Generated by `pnp_from_scratch/tools/make_figures.py`.",
        "",
        "- **Scratch single:** project-owned FAST (threshold 20) at one scale.",
        "- **Scratch pyramid:** the same FAST on an own 8-level pyramid (scale 1.2).",
        "- Both scratch runs then use rotated BRIEF, Hamming matching and RANSAC (300 iterations, 8 px) around the linear PnP.",
        "- **Reference:** the baseline pipeline (ORB on an 8-level pyramid + ORB descriptor + Hamming; library RANSAC PnP), frozen in `docs/migration/baseline/`.",
        "- **Different features:** the pipelines use different keypoints and correspondences.", "",
        head, sep,
        row("keypoints per frame (mean)", lambda p, t: f"{mean([fl(p[0]['keypoints_i'])] + col(p, 'keypoints_j')):.1f}",
            f"{bs['keypoints_per_frame']['mean']:.1f}"),
        row("filtered matches per pair (mean)", lambda p, t: f"{mean(col(p, 'filtered_matches')):.1f}",
            f"{bs['filtered_matches']['mean']:.1f}"),
        row("valid 3D→2D correspondences (mean)", lambda p, t: f"{mean(col(p, 'correspondences_3d2d')):.1f}",
            f"{bs['pnp_correspondences']['mean']:.1f}"),
        row("RANSAC inliers (mean / min)", lambda p, t: f"{mean(col(p, 'pnp_inliers')):.1f} / {min(col(p, 'pnp_inliers')):.0f}",
            f"{bs['pnp_inliers']['mean']:.1f} / {min(q['pnp_inliers'] for q in base['pairs'])}"),
        row("inlier reprojection error (mean over pairs)", lambda p, t: f"{mean(col(p, 'reproj_inlier_mean_px')):.2f} px",
            f"{bs['pnp_reproj_inlier_mean_px']['mean']:.2f} px"),
        row("per-pair rotation error, mean / median / max",
            lambda p, t: f"{mean(col(p, 'rot_err_deg')):.2f}° / {med(col(p, 'rot_err_deg')):.2f}° / {np.nanmax(col(p, 'rot_err_deg')):.2f}°",
            f"{bs['pnp_rel_rot_err_deg']['mean']:.2f}° / {bs['pnp_rel_rot_err_deg']['median']:.2f}° / {bs['pnp_rel_rot_err_deg']['max']:.2f}°"),
        row("per-pair translation error, mean / median",
            lambda p, t: f"{mean(col(p, 'trans_err_m')):.4f} m / {med(col(p, 'trans_err_m')):.4f} m",
            f"{bs['pnp_rel_trans_err_m']['mean']:.4f} m / {bs['pnp_rel_trans_err_m']['median']:.4f} m"),
        row("sum of (estimated − true) rotation per pair",
            lambda p, t: f"{sum(fl(r['est_rotation_deg']) - 10 for r in p):+.1f}°", "−30.4° (baseline diagnostics)"),
        row("failed pairs", lambda p, t: f"{sum(r['pnp_ok'] != '1' for r in p)}", f"{bs['pnp_failed_pairs']}"),
        row("trajectory error, mean over all 36 frames",
            lambda p, t: f"{mean(col(t, 'translation_error_m')):.4f} m / {mean(col(t, 'rotation_error_deg')):.2f}°",
            f"{bs['pnp_mean_trans_err_m']:.4f} m / {mean([fl(r['pnp_rotation_error']) for r in ref_traj]):.2f}°"),
        row("trajectory error, worst frame (position)", lambda p, t: f"{max(col(t, 'translation_error_m')):.4f} m",
            f"{max(fl(r['pnp_translation_error']) for r in ref_traj):.4f} m"),
        row("final trajectory error (frame 35)",
            lambda p, t: f"{fl(t[-1]['translation_error_m']):.4f} m / {fl(t[-1]['rotation_error_deg']):.2f}°",
            f"{bs['pnp_final_trans_err_m']:.4f} m / {bs['pnp_final_rot_err_deg']:.2f}° (ICP {bs['icp_final_trans_err_m']:.4f} m / {bs['icp_final_rot_err_deg']:.2f}°)"),
        "",
        "Read the final-frame row together with the mean over frames: the single-scale run's small frame-35 error "
        "comes partly from errors of mixed sign cancelling, after a large 29.7° error at pair 10→11 (see `pnp_comparison.png`).",
        "", "## Pair 0 → 1", "", head, sep,
        row("keypoints frame 0 / 1", lambda p, t: f"{p[0]['keypoints_i']} / {p[0]['keypoints_j']}",
            f"{b0['keypoints_i']} / {b0['keypoints_j']}"),
        row("raw / filtered matches", lambda p, t: f"{p[0]['raw_matches']} / {p[0]['filtered_matches']}",
            f"{b0['raw_matches']} / {b0['filtered_matches']}"),
        row("valid 3D→2D / RANSAC inliers", lambda p, t: f"{p[0]['correspondences_3d2d']} / {p[0]['pnp_inliers']}",
            f"{b0['pnp_correspondences']} / {b0['pnp_inliers']}"),
        row("rotation / translation error", lambda p, t: f"{fl(p[0]['rot_err_deg']):.3f}° / {fl(p[0]['trans_err_m']):.4f} m",
            f"{b0['pnp_rot_err_deg']:.3f}° / {b0['pnp_trans_err_m']:.4f} m"),
        row("inlier reprojection error (mean)", lambda p, t: f"{fl(p[0]['reproj_inlier_mean_px']):.2f} px",
            f"{b0['reproj_inlier_mean_px']:.2f} px"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pnp_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"[table]  {OUT.relative_to(ROOT)}/pnp_comparison.md")
    print("\n".join(lines))


def main():
    global PIPE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=str(PIPE.relative_to(ROOT)), help="scratch_pipeline output for the "
                    "correspondence and keypoint figures (relative to the repository root)")
    ap.add_argument("--out", default=str(OUT.relative_to(ROOT)), help="figure directory")
    a = ap.parse_args()
    PIPE, OUT = (ROOT / a.pipeline).resolve(), (ROOT / a.out).resolve()
    for p in (PIPE / "pairs.csv", PIPE / "keypoints" / "frame_000000.csv"):
        if not p.exists():
            sys.exit(f"missing {p}: run pnp_from_scratch/build/scratch_pipeline first")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    rgb = [read_png(DS / f"{k:06d}.png") for k in (0, 1)]
    depth = [read_png(DS / f"{k:06d}_depth.png") for k in (0, 1)]
    box = bunny_box(depth)
    s, r = load_scratch_pair(), load_reference_pair()
    pairs = rows(PIPE / "pairs.csv")
    base = json.loads((BASE / "metrics.json").read_text())
    ref_traj = rows(BASE / "slam_trajectory.csv")
    runs = {name: (SCRATCH if name == "single" else SCRATCH_PYR, rows(d / "pairs.csv"), rows(d / "trajectory.csv"))
            for name, d in RUNS.items() if (d / "pairs.csv").exists()}
    correspondence_maps(rgb, box, s, pairs[0])
    orb_keypoints(rgb, box, r)
    feature_comparison(rgb, box, s, r, pairs[0])
    if int(s["kp"][0][:, 4].max()) > 0:
        pyramid_levels(s, r)
    pnp_comparison(runs, base, ref_traj)
    comparison_table(runs, base, ref_traj)


if __name__ == "__main__":
    main()
