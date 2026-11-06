#!/usr/bin/env python3
"""ICP registration before / after, rendered as 3D point clouds (from a --export-pairs export).

Usage (repository root):
  python3 tools/evaluation/diagnostics/icp_registration.py --run <evaluator dir> --pair 73 \
          --dataset data/rgbd_dataset_freiburg1_xyz [--output DIR] [--step 4]
Output (default <run>/../icp/pair_<i>_<j>/): registration.png, registration.json.

Top row, one virtual camera for all three panels (camera i's frame: x right, y down, z forward, metres):
  target = frame i depth (blue tint), source = frame j depth (orange tint), both back-projected on a
  regular pixel grid (--step) with the official intrinsics; the source is drawn under
    (1) the identity (before alignment), (2) ICP's estimated T_i<-j, (3) the ground-truth T_i<-j.
  Vectors: the ACTUAL ICP correspondences (feature matches with depth in both frames), from the
  transformed source point to its target point, coloured by that residual; solid = EstimateRigidRansac
  inlier, dashed = rejected; all inliers and every k-th rejected one (at most 20) are drawn.
Bottom row (NOT used by ICP): every source grid point moved into frame i, projected, and compared with
frame i's measured depth: |z_measured - z_predicted| per pixel, under ICP's and the ground-truth transform.
This shows whether the transform that fits the sparse inliers also aligns the rest of the scene. It measures
agreement with frame i's depth (which ICP also fits); ground truth is interpolated at the RGB time, so it can
score lower there: a consistency check, not ground truth.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import render3d as R3  # noqa: E402

PW, PH = 900, 640           # one 3D panel
RES_MAX = 0.05              # residual colour scale (m)
TGT_TINT = np.array([0.20, 0.45, 0.90])
SRC_TINT = np.array([0.98, 0.55, 0.10])


def tint(rgb, colour):
    lum = rgb.astype(float).mean(axis=1, keepdims=True) / 255.0
    return np.clip(colour * (0.35 + 0.75 * lum), 0, 1)


def dense_agreement(src_cam_j, T_i_from_j, depth_i, K):
    P = src_cam_j @ T_i_from_j[:3, :3].T + T_i_from_j[:3, 3]
    ok = P[:, 2] > 0
    u = np.full(len(P), -1)
    v = np.full(len(P), -1)
    u[ok] = np.floor(K["fx"] * P[ok, 0] / P[ok, 2] + K["cx"]).astype(int)
    v[ok] = np.floor(K["fy"] * P[ok, 1] / P[ok, 2] + K["cy"]).astype(int)
    H, W = depth_i.shape
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z_obs = np.zeros(len(P))
    z_obs[ok] = depth_i[v[ok], u[ok]] / K["depth_factor"]
    ok &= z_obs > 0
    return np.abs(z_obs[ok] - P[ok, 2]), u[ok], v[ok], int(ok.sum()), len(P)


def colour_bar(ax, x, y, w, h, cmap, label, vmax_text):
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(x, x + w, y + h, y), cmap=cmap, aspect="auto")
    ax.text(x, y + h + 3, "0", fontsize=9, va="top")
    ax.text(x + w, y + h + 3, vmax_text, fontsize=9, va="top", ha="right")
    ax.text(x + w / 2, y - 4, label, fontsize=9.5, va="bottom", ha="center")


def run(run_dir, pair, dataset, out=None, step=4):
    run_dir = Path(run_dir)
    p = C.load_pair(run_dir / "diagnostics" / f"pair_{pair}_{pair + 1}")
    problems = C.validate_pair(p, C.pairs_rows(run_dir).get(pair))
    out = Path(out) if out else run_dir.parent / "icp" / f"pair_{pair}_{pair + 1}"
    m, K = p["meta"], C.intrinsics(p)
    dep_i, dep_j = C.read_depth_raw(dataset, m["depth_i"]), C.read_depth_raw(dataset, m["depth_j"])
    rgb_i, rgb_j = C.read_rgb(dataset, m["rgb_i"]), C.read_rgb(dataset, m["rgb_j"])
    tgt, tpx = C.backproject(dep_i, K, step)
    src, spx = C.backproject(dep_j, K, step)
    tcol, scol = tint(rgb_i[tpx[:, 1], tpx[:, 0]], TGT_TINT), tint(rgb_j[spx[:, 1], spx[:, 0]], SRC_TINT)
    Tn = p["T_i_from_j_icp_native"]
    G = np.linalg.inv(p["T_j_from_i_gt"])  # ground-truth T_i<-j
    panels = [("before alignment (identity)", np.eye(4)), ("after ICP  T_i<-j", Tn), ("after ground truth  T_i<-j", G)]
    q = p["icp_corr"] >= 0
    Pi = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][q]
    Pj = np.c_[p["Xj_x"], p["Xj_y"], p["Xj_z"]][q]
    lab = p["icp_class"][q]
    rej = np.nonzero(lab == "rejected")[0]
    rej_step = max(1, int(np.ceil(len(rej) / 20)))
    sample = np.r_[np.nonzero(lab == "inlier")[0], rej[::rej_step]].astype(int)
    c = np.median(tgt, axis=0)
    # virtual camera just behind and above camera i, looking at the target's median point
    view = R3.View(eye=np.array([-0.15, -0.35, -0.25]), target=c, up=(0, -1, 0), fov_deg=50, size=(PW, PH))
    thr = m["config"]["icp"]["threshold_m"]
    cmap = matplotlib.colormaps["viridis"]
    HEAD, MID = 96, 70
    W_all = 3 * PW + 2 * 12
    BH = 480
    H_all = HEAD + PH + MID + BH + 40
    fig, ax = C.pixel_figure(W_all, H_all)
    res = {"pair": pair, "frame_i": m["frame_i"], "frame_j": m["frame_j"], "t_rgb_i": m["t_rgb_i"],
           "t_rgb_j": m["t_rgb_j"], "grid_step_px": step, "icp_threshold_m": thr,
           "icp_correspondences": int(q.sum()), "icp_inliers": int((lab == "inlier").sum()),
           "vectors_drawn": {"inlier": int((lab == "inlier").sum()), "rejected": int(len(rej[::rej_step])),
                             "rejected_every": rej_step}, "export_problems": problems, "panels": {}}
    for n, (title, T) in enumerate(panels):
        x0 = n * (PW + 12)
        if T is None:
            ax.text(x0 + PW / 2, HEAD + PH / 2, "ICP returned no transform", ha="center", fontsize=14)
            continue
        S = src @ T[:3, :3].T + T[:3, 3]
        img, zb = view.splat(tgt, tcol, radius=1)
        img, zb = view.splat(S, scol, radius=1, image=img, depth=zb)
        pax = fig.add_axes([x0 / W_all, 1 - (HEAD + PH) / H_all, PW / W_all, PH / H_all])  # clips its vectors
        pax.imshow(img, extent=(0, PW, PH, 0))
        pax.set_xlim(0, PW), pax.set_ylim(PH, 0)
        pax.set_axis_off()
        Pm = Pj @ T[:3, :3].T + T[:3, 3]
        r = np.linalg.norm(Pi - Pm, axis=1)
        for k in sample:
            seg = np.array([[Pm[k], Pi[k]]])
            uv, keep = view.segments_2d(seg)
            if not keep[0]:
                continue
            (a, b), = uv
            inl = lab[k] == "inlier"
            pax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(min(r[k] / RES_MAX, 1)),
                     lw=2.2 if inl else 1.1, ls="-" if inl else "--", alpha=1 if inl else 0.85)
            pax.scatter([b[0]], [b[1]], s=16 if inl else 6, color="white", edgecolors="black", lw=0.6, zorder=5)
        d, du, dv, n_ok, n_all = dense_agreement(src, T, dep_i, K)
        inl_r = r[lab == "inlier"]
        info = {"sparse_inlier_residual_median_m": float(np.median(inl_r)) if len(inl_r) else None,
                "sparse_all_residual_median_m": float(np.median(r)) if len(r) else None,
                "correspondences_within_threshold": int((r < thr).sum()),
                "dense_compared_points": n_ok, "dense_source_points": n_all,
                "dense_abs_depth_diff_median_m": float(np.median(d)) if n_ok else None,
                "dense_fraction_within_1cm": float((d < 0.01).mean()) if n_ok else None}
        res["panels"][title] = info
        pax.text(6, 6, f"({n + 1}) {title}", fontsize=13, weight="bold", va="top",
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        pax.text(6, PH - 6,
                f"ICP inliers: median residual {100 * (info['sparse_inlier_residual_median_m'] or np.nan):.2f} cm\n"
                f"all {int(q.sum())} correspondences: median {100 * info['sparse_all_residual_median_m']:.1f} cm, "
                f"{info['correspondences_within_threshold']} within {100 * thr:g} cm",
                fontsize=10, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=3))
        if n > 0:  # dense agreement image under this transform
            bx = (n - 1) * (640 + 40) + 330
            by = HEAD + PH + MID
            grid = np.full((dep_i.shape[0] // step + 1, dep_i.shape[1] // step + 1), np.nan)
            grid[dv // step, du // step] = d
            ax.imshow(rgb_i, extent=(bx, bx + 640, by + BH, by), alpha=0.35)
            ax.imshow(np.ma.masked_invalid(grid), extent=(bx, bx + grid.shape[1] * step, by + grid.shape[0] * step, by),
                      cmap="magma_r", vmin=0, vmax=0.05, interpolation="nearest")
            ax.text(bx, by - 6, f"dense check under {title.split('  ')[0][6:]}: |z_measured(i) - z_predicted|, "
                                f"median {100 * (info['dense_abs_depth_diff_median_m'] or np.nan):.2f} cm, "
                                f"{100 * (info['dense_fraction_within_1cm'] or 0):.0f} % < 1 cm", fontsize=10, va="bottom")
    ax.text(6, 6, f"ICP registration, pair {m['frame_i']} -> {m['frame_j']}  (t = {m['t_rgb_i']:.6f} -> {m['t_rgb_j']:.6f}"
                  f" s; {run_dir.parent.name})", fontsize=15, weight="bold", va="top")
    ax.text(6, 34, f"target: frame {m['frame_i']} depth (blue)   source: frame {m['frame_j']} depth (orange), every "
                   f"{step} px, {K['sensor']} intrinsics, metres, viewed in camera {m['frame_i']}'s frame.  "
                   f"ICP correspondences {int(q.sum())}, inliers {int((lab == 'inlier').sum())} (EstimateRigidRansac, "
                   f"{100 * thr:g} cm threshold); vectors: all inliers (solid) + every {rej_step}th rejected (dashed)",
            fontsize=10.5, va="top")
    colour_bar(ax, W_all - 330, 62, 300, 12, cmap, "correspondence residual |p_i - T p_j|", f">= {100 * RES_MAX:g} cm")
    colour_bar(ax, 20, HEAD + PH + MID + 40, 280, 12, "magma_r", "dense |dz| (not used by ICP)", ">= 5 cm")
    ax.text(20, HEAD + PH + MID + 90, "Bottom row: frame j's depth points moved\ninto frame i and compared with frame\n"
                                      "i's measured depth per pixel. It checks\nconsistency with frame i's depth (which\n"
                                      "ICP also fits), not ground truth: the GT\npose is interpolated at the RGB time.",
            fontsize=10, va="top")
    made = []
    C.save_png(fig, out / "registration.png", made)
    C.write_json(out / "registration.json", res)
    return made, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--step", type=int, default=4)
    a = ap.parse_args()
    try:
        made, problems = run(a.run, a.pair, a.dataset, a.output, a.step)
    except (C.ExportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(made))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
