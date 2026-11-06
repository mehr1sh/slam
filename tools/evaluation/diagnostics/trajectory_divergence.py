#!/usr/bin/env python3
"""Trajectory divergence inside the reconstructed scene: GT, PnP and ICP cameras with error connectors.

Usage (repository root):
  python3 tools/evaluation/diagnostics/trajectory_divergence.py --run <evaluator dir> \
          --dataset data/rgbd_dataset_freiburg1_xyz [--every 5] [--select 0,70,73,74,99] [--output DIR]
Output (default <run>/../divergence/): divergence.png, divergence.csv (every frame), divergence.json.

Poses are matched by trajectory.csv row: estimate and ground truth of a row belong to the same frame
timestamp (checked against frames.csv by tum_metrics.load_run). Raw estimates only (frame 0 at the
ground-truth pose; nothing aligned, smoothed or interpolated; held poses after failed pairs are flagged).
A connector joins the GT camera centre and the estimated centre of the same frame, every --every frames
and at --select, coloured by the translation error. Frustums with camera axes show orientation at the
selected frames. Bottom: the RGB image of each selected frame with its timestamp and pose errors.
The interactive version (any frame, raw or one-global-alignment) is the viewer (build_viewer.py).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import scene_3d as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--select", default="", help="frames with RGB, frustums and connectors, e.g. 0,70,73,74,99")
    ap.add_argument("--output")
    a = ap.parse_args()
    run_dir = Path(a.run)
    out = Path(a.output) if a.output else run_dir.parent / "divergence"
    try:
        run = C.tm.load_run(run_dir)
    except C.tm.SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    n = run["n"]
    K = C.run_intrinsics(run_dir, a.dataset)
    sel = sorted({k for k in (int(x) for x in a.select.split(",") if x) if 0 <= k < n} or {0, n - 1})
    conn = sorted(set(range(0, n, a.every)) | set(sel))
    err = {m: np.linalg.norm(run[f"{m}_p"] - run["gt_p"], axis=1) for m in ("pnp", "icp")}
    rot = {m: np.array([C.rot_deg(g.T @ e) for g, e in zip(run["gt_R"], run[f"{m}_R"])]) for m in ("pnp", "icp")}
    emax = float(max(err["pnp"].max(), err["icp"].max()))
    cmap = matplotlib.colormaps["viridis"]
    E0 = 0.01  # log colour scale from 1 cm: ICP's cm-level and PnP's m-level errors are both distinguishable
    cnorm = lambda e: float(np.clip(np.log(max(e, E0) / E0) / np.log(max(emax, 2 * E0) / E0), 0, 1))
    P, col = SC.gt_cloud(run, a.dataset, SC.sample_frames(n, 10, sel), K)
    PW, PH, HEAD = 1000, 700, 70
    TW = 2 * PW + 12
    th_w = (TW - (len(sel) - 1) * 10) // len(sel)
    th_h = int(th_w * 0.75)
    total = (TW, HEAD + PH + 40 + th_h + 80)
    fig, ax = C.pixel_figure(*total)
    ov = SC.scene_view(run, P, (PW, PH), close=False, extent_methods=("gt", "pnp", "icp"))
    cl = SC.scene_view(run, P, (PW, PH), close="cameras")
    for x0, view, title, sm in ((0, ov, "whole extent", 1.0), (PW + 12, cl, "close-up of the camera motion", 0.2)):
        SC.draw_panel(fig, (x0, HEAD, PW, PH), total, view, P, col, run, K, sel, title, connectors=conn, cmap=cmap,
                      emax=emax, fade=0.55, scale_m=sm, cnorm=cnorm)
    ax.text(6, 6, f"{Path(a.dataset).name} / {run_dir.parent.name}: where the estimates leave the true camera path",
            fontsize=15, weight="bold", va="top")
    ax.text(6, 34, f"Raw estimates (frame 0 at ground truth). Connector = GT centre -> estimated centre of the same frame "
                   f"(same timestamp), every {a.every} frames and at {sel}; frustums with camera axes at {sel}; "
                   f"depth points placed with GT poses (faded).",
            fontsize=10.5, va="top")
    # colour bar
    by = HEAD + PH + 8
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(TW - 420, TW - 120, by + 14, by + 2), cmap=cmap, aspect="auto")
    ax.text(TW - 430, by + 8, "connector |error| (log)  <= 1 cm", fontsize=10, ha="right", va="center")
    ax.text(TW - 112, by + 8, f"{emax:.2f} m", fontsize=10, va="center")
    for k_, m in enumerate(("gt", "pnp", "icp")):
        ax.plot([10 + 170 * k_, 40 + 170 * k_], [by + 8] * 2, color=C.METHOD_COLOR[m], lw=3)
        ax.text(46 + 170 * k_, by + 8, C.METHOD_LABEL[m], fontsize=10.5, va="center")
    # RGB of the selected frames
    ty = HEAD + PH + 40
    for idx, k in enumerate(sel):
        x0 = idx * (th_w + 10)
        ax.imshow(C.read_rgb(a.dataset, run["rgb"][k]), extent=(x0, x0 + th_w, ty + th_h, ty))
        ax.text(x0 + 4, ty + th_h + 6, f"frame {k}   t = {run['t'][k]:.6f} s", fontsize=10.5, weight="bold", va="top")
        held = [m.upper() for m in ("pnp", "icp") if run[f"{m}_held"][k]]
        ax.text(x0 + 4, ty + th_h + 26, f"PnP {100 * err['pnp'][k]:.1f} cm / {rot['pnp'][k]:.1f} deg   "
                                        f"ICP {100 * err['icp'][k]:.1f} cm / {rot['icp'][k]:.1f} deg"
                                        + (f"   held: {', '.join(held)}" if held else ""), fontsize=10, va="top")
        ax.text(x0 + 4, ty + th_h + 46, run["rgb"][k], fontsize=9, va="top", color="#555")
    made = []
    C.save_png(fig, out / "divergence.png", made)
    with open(out / "divergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp", "pnp_err_m", "pnp_rot_err_deg", "icp_err_m", "icp_rot_err_deg", "pnp_held",
                    "icp_held", "connector", "selected"])
        for k in range(n):
            w.writerow([k, f"{run['t'][k]:.6f}", err["pnp"][k], rot["pnp"][k], err["icp"][k], rot["icp"][k],
                        int(run["pnp_held"][k]), int(run["icp_held"][k]), int(k in conn), int(k in sel)])
    C.write_json(out / "divergence.json", {"run": str(run_dir), "frames": n, "selected": sel, "connectors": conn,
                                           "max_error_m": {m: float(err[m].max()) for m in err},
                                           "final_error_m": {m: float(err[m][-1]) for m in err}, "figures": made})
    print("\n".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
