#!/usr/bin/env python3
"""Correspondence maps of the ACTUAL sparse ORB matches of one frame pair of
the synthetic Stanford Bunny sequence, drawn for visual auditing (MASt3R-style
layout: two large frames, colour = correspondence identity).

Run (after `pixi run build`):
    pixi run -e results correspondence-map               # pair 0 -> 1
    pixi run -e results correspondence-map --pair 17     # pair 17 -> 18
    pixi run -e results correspondence-map --skip-cpp    # reuse the last export

Writes results/figures/bunny_correspondences_<i>_<j>{,_inliers,_outliers}.{png,pdf,svg}

Colour = correspondence ID. Each match with valid depth gets one deterministic
colour from its keypoint position in frame i (hue = angle around the bunny's
feature centroid, saturation = distance from it), and that same colour marks
its other end in frame i+1. A wrong match therefore shows up as a colour in
the wrong place in frame i+1, and as a long or crossing line.
Marker size encodes the depth of the frame-i point (near = larger).

Data: the export of build/slam_trajectory_test (--export-dir/--export-pair),
loaded and cross-checked by generate_correspondence_figure.load(): every
keypoint, match, depth value and RANSAC flag is the pipeline's own; nothing is
generated. Visualization only.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_correspondence_figure as base  # noqa: E402  (export, loading, validation)

FIG = base.FIG
OUTLIER = (1.00, 0.10, 0.55)
NODEPTH = (0.70, 0.70, 0.70)
STROKE = [pe.withStroke(linewidth=2.4, foreground="black")]


def id_colours(c):
    """Deterministic correspondence colours from the frame-i position."""
    m = c["pnp_used"]
    u, v = c["u_i"], c["v_i"]
    cu, cv = np.median(u[m]), np.median(v[m])
    ang = np.arctan2(v - cv, u - cu)
    r = np.hypot(u - cu, v - cv)
    r = np.clip(r / np.percentile(r[m], 95), 0, 1)
    # hue range 0 (red) .. 0.80 (violet): skips the pink/magenta band reserved for outliers
    hsv = np.c_[0.80 * (ang + np.pi) / (2 * np.pi), 0.50 + 0.50 * r, np.full_like(r, 1.0)]
    return hsv_to_rgb(hsv)


def depth_sizes(c, lo=14, hi=46):
    z = c["Z_i"]
    m = c["pnp_used"]
    zn = (z - np.nanmin(z[m])) / max(np.nanmax(z[m]) - np.nanmin(z[m]), 1e-9)
    return np.where(m, hi - (hi - lo) * np.nan_to_num(zn), lo)  # near = larger


class Panel:
    """Both frames side by side in one coordinate system (crop pixels)."""

    def __init__(self, ax, D, box, gap_frac=0.10):
        self.ax, self.D = ax, D
        self.x0, self.x1, self.y0, self.y1 = box
        self.W, self.H = self.x1 - self.x0, self.y1 - self.y0
        self.off = self.W * (1 + gap_frac)
        ax.set_xlim(0, self.off + self.W)
        ax.set_ylim(self.H, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    def image(self, interp="lanczos", labels=True, size=15):
        for k in (0, 1):
            xo = self.off * k
            self.ax.imshow(self.D["rgb"][k][self.y0:self.y1, self.x0:self.x1], extent=(xo, xo + self.W, self.H, 0),
                           interpolation=interp, zorder=0)
            if labels:
                self.ax.text(xo + 0.02 * self.W, 0.025 * self.H, f"Frame {self.D['i'] if k == 0 else self.D['j']}",
                             color="white", fontsize=size, weight="bold", va="top", zorder=20)

    def a(self, sel=slice(None)):
        c = self.D["c"]
        return np.c_[c["u_i"][sel] - self.x0, c["v_i"][sel] - self.y0]

    def b(self, sel=slice(None)):
        c = self.D["c"]
        return np.c_[c["u_j"][sel] - self.x0 + self.off, c["v_j"][sel] - self.y0]

    def keypoints(self, alpha=0.45, s=6):
        for k, f in ((0, "i"), (1, "j")):
            u, v = self.D["kp"][f][:, 0] - self.x0, self.D["kp"][f][:, 1] - self.y0
            ok = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
            self.ax.scatter(u[ok] + self.off * k, v[ok], s=s, c="white", alpha=alpha, linewidths=0, zorder=1)

    def lines(self, sel, colours, lw, alpha, ls="-", z=3):
        if not np.any(sel):
            return
        segs = np.stack([self.a(sel), self.b(sel)], 1)
        self.ax.add_collection(LineCollection(segs, colors=colours, linewidths=lw, alpha=alpha, linestyles=ls,
                                              zorder=z, capstyle="round"))

    def points(self, sel, colours, sizes, edge="black", lw=0.5, marker="o", z=6):
        if not np.any(sel):
            return
        for pts in (self.a(sel), self.b(sel)):
            self.ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=colours, edgecolors=edge, linewidths=lw,
                            marker=marker, zorder=z)


def draw_matches(P, col, size, scale=1.0, show_nodepth=True, show_outliers=True, show_inliers=True):
    c = P.D["c"]
    inl, pnp = c["pnp_inlier"], c["pnp_used"]
    out = pnp & ~inl
    nod = ~pnp
    if show_nodepth:
        P.lines(nod, [NODEPTH], 0.5 * scale, 0.22, ls=(0, (1, 2)), z=2)
        P.points(nod, [NODEPTH], 10 * scale ** 2, edge="0.25", lw=0.4, marker="D", z=4)
    if show_inliers:
        P.lines(inl, col[inl], 0.85 * scale, 0.55)
        P.points(inl, col[inl], size[inl] * scale ** 2, edge="black", lw=0.5 * scale)
    if show_outliers:
        P.lines(out, [OUTLIER], 1.0 * scale, 0.85, ls=(0, (4, 2.5)), z=4)
        P.points(out, col[out], size[out] * scale ** 2, edge=OUTLIER, lw=1.6 * scale, z=7)


def header(fig, D, line, top=0.985):
    fig.text(0.012, top, f"ORB Correspondences   ·   Frame {D['i']} → Frame {D['j']}", fontsize=16, weight="bold",
             va="top", color="0.1")
    fig.text(0.012, top - 0.032, line, fontsize=10.5, va="top", color="0.35")


def footer(fig, text, y=0.006):
    fig.text(0.988, y, text, fontsize=7.5, color="0.45", ha="right", va="bottom")


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=240 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"[figure] results/figures/{stem}.png|pdf|svg")


KEY = ("colour = correspondence (same colour in both frames)  ·  marker size = depth, near = larger  ·  "
       "solid = RANSAC inlier  ·  magenta dashed + ring = RANSAC outlier  ·  grey ◇ = match without depth  ·  "
       "white dots = other ORB keypoints")


def main_map(D, col, size, stem):
    n = D["n"]
    box = base.bunny_box(D, 12)
    W, H = box[1] - box[0], box[3] - box[2]
    fw = 16.0
    main_h = fw * 0.976 / (2.10 * W) * H
    zoom_h = 2.55
    fh = 0.85 + main_h + 0.25 + zoom_h + 0.30
    fig = plt.figure(figsize=(fw, fh), facecolor="white")
    header(fig, D, f"{n['filtered']} filtered matches  ·  {n['pnp']} valid depth pairs  ·  {n['inliers']} RANSAC inliers")
    ax = fig.add_axes([0.012, (0.30 + zoom_h + 0.25) / fh, 0.976, main_h / fh])
    P = Panel(ax, D, box)
    P.image()
    P.keypoints()
    draw_matches(P, col, size)

    zooms = base.pick_zooms(D, size=round(0.22 * min(W, H)))
    zw = 0.976 / len(zooms)
    for zi, z in enumerate(zooms):
        lab = "ABC"[zi]
        s = z["size"]
        sx, sy = np.round(z["shift"]).astype(int)
        # keep both zoom windows inside the displayed crop
        x0z = int(np.clip(z["x0"], box[0] - min(0, sx), box[1] - s - max(0, sx)))
        y0z = int(np.clip(z["y0"], box[2] - min(0, sy), box[3] - s - max(0, sy)))
        bxs = [(x0z, y0z), (x0z + sx, y0z + sy)]
        for k, (bx, by) in enumerate(bxs):
            px, py = bx - box[0] + P.off * k, by - box[2]
            ax.add_patch(patches.Rectangle((px, py), s, s, fill=False, lw=1.3, ec="white", ls=(0, (3, 2)), zorder=15))
            ax.text(px + 1.5, py + 1.5, lab, color="white", fontsize=12, weight="bold", va="top", zorder=15,
                    path_effects=STROKE)
        zax = fig.add_axes([0.012 + zi * zw + 0.006, 0.30 / fh, zw - 0.012, zoom_h / fh])
        # zoom = same two-frame panel, restricted to the two boxes (frame i+1 box shifted by the local motion)
        Z = ZoomPanel(zax, D, bxs, s)
        Z.draw(col, size, lab)
    footer(fig, KEY)
    save(fig, stem)
    return zooms


class ZoomPanel(Panel):
    def __init__(self, ax, D, boxes, s, gap_frac=0.06):
        self.ax, self.D, self.boxes, self.s = ax, D, boxes, s
        self.off = s * (1 + gap_frac)
        self.W = self.H = s
        ax.set_xlim(0, self.off + s)
        ax.set_ylim(s, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    def a(self, sel=slice(None)):
        c = self.D["c"]
        return np.c_[c["u_i"][sel] - self.boxes[0][0], c["v_i"][sel] - self.boxes[0][1]]

    def b(self, sel=slice(None)):
        c = self.D["c"]
        return np.c_[c["u_j"][sel] - self.boxes[1][0] + self.off, c["v_j"][sel] - self.boxes[1][1]]

    def draw(self, col, size, lab):
        s = self.s
        for k, (bx, by) in enumerate(self.boxes):
            xo = self.off * k
            self.ax.imshow(self.D["rgb"][k][by:by + s, bx:bx + s], extent=(xo, xo + s, s, 0),
                           interpolation="bicubic", zorder=0)
            f = "ij"[k]
            u, v = self.D["kp"][f][:, 0] - bx, self.D["kp"][f][:, 1] - by
            ok = (u >= 0) & (u < s) & (v >= 0) & (v < s)
            self.ax.scatter(u[ok] + xo, v[ok], s=14, c="white", alpha=0.5, linewidths=0, zorder=1)
        c = self.D["c"]
        A, B = self.a(), self.b()
        inside = ((A[:, 0] >= 0) & (A[:, 0] < s) & (A[:, 1] >= 0) & (A[:, 1] < s)) | \
                 ((B[:, 0] >= self.off) & (B[:, 0] < self.off + s) & (B[:, 1] >= 0) & (B[:, 1] < s))
        keep = {k: c[k].copy() for k in ("pnp_used", "pnp_inlier")}
        c["pnp_used"], c["pnp_inlier"] = keep["pnp_used"] & inside, keep["pnp_inlier"] & inside
        nod_mask = ~keep["pnp_used"] & inside
        try:
            P = self
            inl, pnp = c["pnp_inlier"], c["pnp_used"]
            out = pnp & ~inl
            P.lines(nod_mask, [NODEPTH], 1.0, 0.45, ls=(0, (1, 2)), z=2)
            P.points(nod_mask, [NODEPTH], 34, edge="0.25", lw=0.6, marker="D", z=4)
            P.lines(inl, col[inl], 1.8, 0.85)
            P.points(inl, col[inl], size[inl] * 2.4, edge="black", lw=0.8)
            P.lines(out, [OUTLIER], 1.8, 0.95, ls=(0, (4, 2.5)), z=4)
            P.points(out, col[out], size[out] * 2.4, edge=OUTLIER, lw=2.2, z=7)
        finally:
            c["pnp_used"], c["pnp_inlier"] = keep["pnp_used"], keep["pnp_inlier"]
        for k in (0, 1):
            self.ax.add_patch(patches.Rectangle((self.off * k, 0), s, s, fill=False, lw=1.2, ec="0.15", zorder=10,
                                                clip_on=False))
        self.ax.text(1.2, 1.2, lab, color="white", fontsize=13, weight="bold", va="top", zorder=12,
                     path_effects=STROKE)


def subset_map(D, col, size, stem, which):
    n, c = D["n"], D["c"]
    box = base.bunny_box(D, 12)
    W, H = box[1] - box[0], box[3] - box[2]
    fw = 16.0
    main_h = fw * 0.976 / (2.10 * W) * H
    fh = 0.85 + main_h + 0.25
    fig = plt.figure(figsize=(fw, fh), facecolor="white")
    ax = fig.add_axes([0.012, 0.25 / fh, 0.976, main_h / fh])
    P = Panel(ax, D, box)
    P.image()
    if which == "inliers":
        header(fig, D, f"RANSAC inliers only  ·  {n['inliers']} of {n['pnp']} valid depth pairs")
        draw_matches(P, col, size, show_nodepth=False, show_outliers=False)
        footer(fig, "colour = correspondence (same colour in both frames)  ·  marker size = depth, near = larger")
    else:
        out = c["pnp_used"] & ~c["pnp_inlier"]
        header(fig, D, f"RANSAC outliers only  ·  {int(out.sum())} of {n['pnp']} valid depth pairs rejected "
                       f"(8 px reprojection threshold)")
        inl = c["pnp_inlier"]
        P.points(inl, [(1, 1, 1)], 10, edge="0.3", lw=0.3, z=2)  # inliers as faint context
        P.lines(out, col[out], 1.6, 0.9, z=4)
        P.points(out, col[out], size[out] * 1.3, edge=OUTLIER, lw=1.8, z=7)
        idx = np.nonzero(out)[0]
        for num, e in enumerate(idx, 1):
            for p in (P.a([e])[0], P.b([e])[0]):
                ax.annotate(str(num), p, xytext=(5, 4), textcoords="offset points", fontsize=9, weight="bold",
                            color=col[e], zorder=9, path_effects=STROKE)
        footer(fig, "each rejected match: same colour and number at both ends  ·  white dots = RANSAC inliers (context)")
    save(fig, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--skip-cpp", action="store_true")
    args = ap.parse_args()
    if not args.skip_cpp:
        base.run_export(args.pair)
    D = base.load(args.pair)
    c = D["c"]
    col = id_colours(c)
    size = depth_sizes(c)
    i, j = D["i"], D["j"]
    stem = f"bunny_correspondences_{i}_{j}"
    zooms = main_map(D, col, size, stem)
    subset_map(D, col, size, stem + "_inliers", "inliers")
    subset_map(D, col, size, stem + "_outliers", "outliers")
    for lab, z in zip("ABC", zooms):
        print(f"[zoom] {lab}: {z['name']} (frame-{i} box x={z['x0']} y={z['y0']} size {z['size']})")
    n = D["n"]
    print(f"[counts] pair {i}->{j}: ORB {n['kp_i']}/{n['kp_j']}, raw {n['raw']}, filtered {n['filtered']}, "
          f"valid depth pairs {n['pnp']}, inliers {n['inliers']}, outliers {n['pnp'] - n['inliers']}, ICP {n['icp']}")


if __name__ == "__main__":
    main()
