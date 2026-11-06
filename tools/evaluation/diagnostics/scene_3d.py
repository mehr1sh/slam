#!/usr/bin/env python3
"""Static render of the reconstructed RGB-D scene with GT, PnP and ICP camera motion (real frames and depth).

Usage (repository root):
  python3 tools/evaluation/diagnostics/scene_3d.py --run <evaluator dir> --dataset data/rgbd_dataset_freiburg1_xyz \
          [--every 10] [--frames 73,74] [--output DIR]
Output (default <run>/../scene/): scene.png and scene.json. The interactive version is the viewer
(build_viewer.py); this is the reproducible static export.

Points: frames 0, every --every-th, the last and --frames; pixels (4 + 6a, 4 + 6b) with raw depth > 0,
back-projected with the official intrinsics (depth = raw / 5000 m, no undistortion, as the evaluator),
placed in the TUM world frame with the GROUND-TRUTH pose of their frame. Trajectories are the evaluator's
T_wc (raw: frame 0 at ground truth, nothing aligned or smoothed). Frustums with camera axes (x red,
y cyan, z = optical axis blue) at the selected frames. Left: whole extent; right: close-up of the scene.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import render3d as R3  # noqa: E402

PW, PH = 1000, 760
AXIS_COLORS = ("#d62728", "#17becf", "#1f77b4")


def sample_frames(n, every, extra):
    return sorted(set(range(0, n, every)) | {n - 1} | {k for k in extra if 0 <= k < n})


def gt_cloud(run, dataset, frames, K, step=6):
    P, col = [], []
    for k in frames:
        xyz, px = C.backproject(C.read_depth_raw(dataset, run["depth"][k]), K, step=step, offset=4)
        P.append(xyz @ run["gt_R"][k].T + run["gt_p"][k])
        col.append(C.read_rgb(dataset, run["rgb"][k])[px[:, 1], px[:, 0]] / 255.0)
    return np.concatenate(P), np.concatenate(col)


def scene_view(run, P, size, close=True, extent_methods=("gt",)):
    """Viewpoint behind and above the mean GT camera, along its mean optical axis; frames the cloud
    (close=True), everything incl. the estimated trajectories (False), or the cameras ("cameras")."""
    axis = run["gt_R"][:, :, 2].mean(0)
    axis /= np.linalg.norm(axis)
    if close == "cameras":  # the GT camera region and the first ~0.5 m in front of it
        target = run["gt_p"].mean(0) + 0.35 * axis
        rad = np.linalg.norm(np.ptp(run["gt_p"], axis=0)) / 2 + 0.35
        direction = -axis + np.array([0, 0, 0.9])
        direction /= np.linalg.norm(direction)
        return R3.View(target + direction * rad / np.tan(np.radians(25)) * 1.05, target, up=(0, 0, 1), fov_deg=50,
                       size=size)
    target = (np.median(P, axis=0) + run["gt_p"].mean(0)) / 2
    pts = np.r_[np.percentile(P, [5, 95], axis=0), *[run[f"{m}_p"] for m in extent_methods]]
    rad = np.linalg.norm(pts.max(0) - pts.min(0)) / 2
    if not close:
        target = (pts.max(0) + pts.min(0)) / 2
    direction = -axis + np.array([0, 0, 0.55])
    direction /= np.linalg.norm(direction)
    dist = rad / np.tan(np.radians(25)) * (0.9 if close else 1.1)
    return R3.View(target + direction * dist, target, up=(0, 0, 1), fov_deg=50, size=size)


def draw_panel(fig, rect, total, view, P, col, run, K, frustum_frames, title, connectors=None, cmap=None, emax=None,
               fade=0.0, scale_m=0.5, methods=("gt", "pnp", "icp"), cnorm=None):
    """One clipped render: points (z-buffered), trajectories, frustums, start/end markers, optional connectors."""
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 / total[0], 1 - (y0 + h) / total[1], w / total[0], h / total[1]])
    img, _ = view.splat(P, col * (1 - fade) + fade, radius=1)
    ax.imshow(img, extent=(0, w, h, 0))
    ax.set_xlim(0, w), ax.set_ylim(h, 0)
    ax.set_axis_off()
    if connectors is not None:
        for m in ("pnp", "icp"):
            for k in connectors:
                e = np.linalg.norm(run[f"{m}_p"][k] - run["gt_p"][k])
                t = cnorm(e) if cnorm else min(e / emax, 1)
                R3.draw_segments(ax, view, [[run["gt_p"][k], run[f"{m}_p"][k]]], color=cmap(t), lw=1.8)
    for m in methods:
        R3.draw_polyline(ax, view, run[f"{m}_p"], color=C.METHOD_COLOR[m], lw=2.6 if m == "gt" else 2.0)
        for k in frustum_frames:
            T = C.se3(run[f"{m}_R"][k], run[f"{m}_p"][k])
            R3.draw_segments(ax, view, C.frustum_lines(T, K, 0.07), color=C.METHOD_COLOR[m], lw=1.5)
            for a, c in enumerate(AXIS_COLORS):
                R3.draw_segments(ax, view, [[T[:3, 3], T[:3, 3] + 0.05 * T[:3, a]]], color=c, lw=1.5)
        for k, mk in ((0, "o"), (run["n"] - 1, "s")):
            uv, z = view.project(run[f"{m}_p"][k][None])
            if z[0] > view.near:
                ax.scatter(*uv[0], marker=mk, s=70, color=C.METHOD_COLOR[m], edgecolors="black", zorder=6)
    R3.draw_axes_gizmo(ax, view, labels=("x", "y", "z (up)"), at=(60, h - 60))
    R3.draw_scale_bar(ax, view, view.eye + (view.R[2] * np.linalg.norm(view.eye - P.mean(0))), scale_m,
                      at=(w - 260, h - 30))
    ax.text(8, 8, title, fontsize=12, weight="bold", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    return ax


def legend(ax, x, y):
    for n, m in enumerate(("gt", "pnp", "icp")):
        ax.plot([x, x + 30], [y + 22 * n] * 2, color=C.METHOD_COLOR[m], lw=3)
        ax.text(x + 38, y + 22 * n, C.METHOD_LABEL[m], fontsize=10.5, va="center")
    ax.scatter([x + 15], [y + 66], marker="o", s=60, color="0.7", edgecolors="black")
    ax.text(x + 38, y + 66, "start (frame 0)", fontsize=10.5, va="center")
    ax.scatter([x + 15], [y + 88], marker="s", s=60, color="0.7", edgecolors="black")
    ax.text(x + 38, y + 88, "end (last frame)", fontsize=10.5, va="center")
    for n, (c, t) in enumerate(zip(AXIS_COLORS, ("camera x", "camera y", "camera z (optical axis)"))):
        ax.plot([x + 200, x + 230], [y + 22 * n] * 2, color=c, lw=2)
        ax.text(x + 238, y + 22 * n, t, fontsize=10.5, va="center")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--frames", default="", help="extra frames (points and frustums), e.g. 73,74")
    ap.add_argument("--output")
    a = ap.parse_args()
    run_dir = Path(a.run)
    out = Path(a.output) if a.output else run_dir.parent / "scene"
    try:
        run = C.tm.load_run(run_dir)
    except C.tm.SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    K = C.run_intrinsics(run_dir, a.dataset)
    extra = [int(x) for x in a.frames.split(",") if x]
    frames = sample_frames(run["n"], a.every, extra)
    P, col = gt_cloud(run, a.dataset, frames, K)
    fr_frames = sorted({0, run["n"] - 1, *[k for k in extra if 0 <= k < run["n"]]})
    HEAD = 70
    total = (2 * PW + 12, HEAD + PH + 130)
    fig, ax = C.pixel_figure(*total)
    ov = scene_view(run, P, (PW, PH), close=False, extent_methods=("gt", "pnp", "icp"))
    cl = scene_view(run, P, (PW, PH), close="cameras")
    draw_panel(fig, (0, HEAD, PW, PH), total, ov, P, col, run, K, fr_frames, "whole extent (all trajectories)",
               scale_m=1.0)
    draw_panel(fig, (PW + 12, HEAD, PW, PH), total, cl, P, col, run, K, fr_frames, "close-up of the camera motion",
               scale_m=0.2)
    ax.text(6, 6, f"{Path(a.dataset).name} / {run_dir.parent.name}: reconstructed RGB-D scene and camera motion",
            fontsize=15, weight="bold", va="top")
    ax.text(6, 34, f"{len(P)} depth samples from {len(frames)} frames ({frames[0]}..{frames[-1]}, every {a.every} + "
                   f"{extra}) placed with their ground-truth poses; TUM world frame, z up, metres; {K['sensor']} "
                   f"intrinsics, depth = raw / {K['depth_factor']:g}. Trajectories raw (frame 0 at ground truth). "
                   f"Frustums at frames {fr_frames}.", fontsize=10.5, va="top")
    legend(ax, 20, HEAD + PH + 20)
    made = []
    C.save_png(fig, out / "scene.png", made)
    C.write_json(out / "scene.json", {"run": str(run_dir), "dataset": a.dataset, "frames": frames, "grid_step_px": 6,
                                      "points": int(len(P)), "frustum_frames": fr_frames, "intrinsics": K,
                                      "figures": made})
    print("\n".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
