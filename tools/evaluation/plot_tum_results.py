#!/usr/bin/env python3
"""Figures, metrics and a report for one tum_trajectory_eval output directory (headless).

Usage (repository root):
  python3 tools/evaluation/plot_tum_results.py --input results/runs/stress/xyz_300 [--output DIR]
         [--dataset data/rgbd_dataset_freiburg1_xyz] [--max-dt 0.02] [--max-gt-gap 0.05]
Default output: <input>/figures/. The sequence directory (for the RGB-D examples and the
association / ground-truth re-check) is taken from --dataset, else from <input>/config.json
(written by run_tum_stress.py); without it those two items are reported as not available.

Writes PNG + PDF figures, metrics.json and report.md. Exit status: 0 ok, 1 a sanity check
failed, 2 unsupported or invalid input (nothing fabricated).
Metric definitions: see tum_metrics.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "pipeline_analysis"))
import tum_metrics as tm  # noqa: E402

STYLE = {"gt": dict(color="#2ca02c", label="ground truth", lw=2.0),
         "pnp": dict(color="#ff7f0e", label="PnP (RANSAC)", lw=1.4),
         "icp": dict(color="#9467bd", label="ICP (robust)", lw=1.4)}
DEPTH_FACTOR = 5000.0  # TUM: depth PNG value / 5000 = metres (include/dataset/tum.hpp)
FEATURE_MATCHES_BLOCKED = (
    "feature_matches: NOT GENERATED. The evaluator does not export keypoints, matches or inlier "
    "flags (frames.csv / pairs.csv hold counts only), so real correspondences cannot be drawn. "
    "Smallest addition: an opt-in `--export-matches K` in tests/tum/tum_trajectory_eval.cpp that "
    "writes, for pair K -> K+1 only, one row per filtered match (u_i, v_i, u_j, v_j, depth_raw_i, "
    "depth_raw_j, pnp_used, pnp_inlier, icp_used, icp_inlier) from the values RunPair already "
    "computes; no estimator change.")


def save(fig, out, name, made):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    made.append(name)


def trajectory_3d(run, out, made, checks):
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    allp = np.concatenate([run[f"{m}_p"] for m in STYLE])
    for m, st in STYLE.items():
        p = run[f"{m}_p"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], **st)
        line, = ax2.plot(p[:, 0], p[:, 1], **st)
        checks.append(("trajectory_3d", np.allclose(line.get_xdata(), p[:, 0]) and np.allclose(line.get_ydata(), p[:, 1])))
        for a in (ax, ax2):
            kw = dict(color=st["color"], edgecolors="black", zorder=5)
            xyz = (p[0, 0], p[0, 1]) + ((p[0, 2],) if a is ax else ())
            a.scatter(*xyz, marker="o", s=45, **kw)
            xyz = (p[-1, 0], p[-1, 1]) + ((p[-1, 2],) if a is ax else ())
            a.scatter(*xyz, marker="s", s=45, **kw)
        if m != "gt" and run[f"{m}_held"].any():
            h = p[run[f"{m}_held"]]
            ax2.scatter(h[:, 0], h[:, 1], marker="x", s=60, color="red", zorder=6)
    mid, half = (allp.max(0) + allp.min(0)) / 2, max((allp.max(0) - allp.min(0)).max() / 2, 1e-3) * 1.1
    for setl, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setl(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.set_title("3D, equal axis scale", fontsize=10)
    ax2.set_xlim(mid[0] - half, mid[0] + half), ax2.set_ylim(mid[1] - half, mid[1] + half)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)"), ax2.set_ylabel("y (m)")
    ax2.set_title("projection on the world x-y plane (equal scale)", fontsize=10)
    ax2.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], **st) for st in STYLE.values()] + [
        Line2D([], [], ls="none", marker="o", mfc="0.7", mec="black", label="start (frame 0)"),
        Line2D([], [], ls="none", marker="s", mfc="0.7", mec="black", label=f"end (frame {run['n'] - 1})")]
    if any(run[f"{m}_held"].any() for m in tm.METHODS):
        h.append(Line2D([], [], ls="none", marker="x", color="red", label="held pose (failed pair)"))
    ax2.legend(handles=h, fontsize=8, loc="best")
    fig.suptitle(f"Camera trajectories (T_wc positions, TUM world frame), {run['n']} frames; estimates start at the "
                 "ground-truth pose", fontsize=11)
    save(fig, out, "trajectory_3d", made)


def trajectory_errors(run, out, made, checks):
    t = run["t"] - run["t"][0]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m in tm.METHODS:
        e = np.linalg.norm(run[f"{m}_p"] - run["gt_p"], axis=1)
        r = np.array([tm.rot_angle_deg(a.T @ b) for a, b in zip(run["gt_R"], run[f"{m}_R"])])
        l1, = axs[0].plot(t, e * 100, "-o", ms=2.5, **STYLE[m])
        axs[1].plot(t, r, "-o", ms=2.5, **STYLE[m])
        # CSV poses and errors carry 6 decimals: same tolerance as tm.check_consistency
        checks.append(("trajectory_errors", np.allclose(l1.get_ydata() / 100, run[f"{m}_file_terr"], rtol=0, atol=1e-5)))
        for k in np.nonzero(~run[f"{m}_ok"])[0]:
            for a in axs:
                a.axvline(t[k + 1], color=STYLE[m]["color"], ls=":", lw=1)
    axs[0].set_ylabel("position error (cm)")
    axs[1].set_ylabel("orientation error (deg)")
    axs[1].set_xlabel(f"time since the first frame (s); frame timestamps from trajectory.csv")
    axs[0].set_title("Accumulated error vs ground truth per frame (raw, no alignment); dotted line = pose held "
                     "after a failed pair", fontsize=10)
    for a in axs:
        a.legend(fontsize=8), a.grid(alpha=0.3)
    save(fig, out, "trajectory_errors", made)


def relative_pose_errors(run, out, made, checks):
    k = np.arange(run["n"] - 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    w = 0.4
    long_run = len(k) > 60  # bars become unreadable: lines on a log axis (NaN = failed pair = gap)
    for s, m in zip((-1, 1), tm.METHODS):
        ok = run[f"{m}_ok"]
        if long_run:
            l0, = axs[0].plot(k, run[f"{m}_trans_err_m"] * 1000, lw=0.8, color=STYLE[m]["color"], label=STYLE[m]["label"])
            axs[1].plot(k, run[f"{m}_rot_err_deg"], lw=0.8, color=STYLE[m]["color"], label=STYLE[m]["label"])
            h = np.asarray(l0.get_ydata())
            for a in axs:
                a.plot(k[~ok], np.zeros((~ok).sum()), "x", color="red", ms=6, clip_on=False,
                       transform=a.get_xaxis_transform())  # y in axes fraction: bottom edge
        else:
            bars = axs[0].bar(k + s * w / 2, run[f"{m}_trans_err_m"] * 1000, w, color=STYLE[m]["color"], label=STYLE[m]["label"])
            axs[1].bar(k + s * w / 2, run[f"{m}_rot_err_deg"], w, color=STYLE[m]["color"], label=STYLE[m]["label"])
            h = np.array([b.get_height() for b in bars])
            for j in np.nonzero(~ok)[0]:
                for a in axs:
                    a.annotate("failed", (j + s * w / 2, 0), xytext=(0, 3), textcoords="offset points", rotation=90,
                               color="red", fontsize=7, ha="center")
        checks.append(("relative_pose_errors", np.allclose(h[ok], run[f"{m}_trans_err_m"][ok] * 1000)))
    if long_run:
        axs[0].plot(k, run["gt_motion_m"] * 1000, color="black", lw=0.6, alpha=0.6, label="true motion per pair")
        axs[1].plot(k, run["gt_motion_deg"], color="black", lw=0.6, alpha=0.6, label="true motion per pair")
        for a in axs:
            a.set_yscale("log")
        if (~run["pnp_ok"]).any() or (~run["icp_ok"]).any():
            axs[0].plot([], [], "x", color="red", label="failed pair (no estimate)")
    else:
        axs[0].plot(k, run["gt_motion_m"] * 1000, "k_", ms=8, label="true motion per pair")
        axs[1].plot(k, run["gt_motion_deg"], "k_", ms=8, label="true motion per pair")
    axs[0].set_ylabel("translation error (mm)")
    axs[1].set_ylabel("rotation error (deg)")
    axs[1].set_xlabel("pair i -> i+1 (i)")
    axs[0].set_title("Per-pair relative pose error of T_{i+1<-i} vs ground truth (pairs.csv)", fontsize=10)
    for a in axs:
        a.legend(fontsize=8), a.grid(alpha=0.3, axis="y")
    save(fig, out, "relative_pose_errors", made)


def rgb_depth_examples(run, dataset, out, made):
    from png_reader import read_png
    idx = sorted({0, run["n"] // 2, run["n"] - 1})
    fig, axs = plt.subplots(len(idx), 3, figsize=(12, 3.3 * len(idx)), squeeze=False)
    for r, k in enumerate(idx):
        rgb = read_png(Path(dataset) / run["rgb"][k])
        raw = read_png(Path(dataset) / run["depth"][k])
        if raw.ndim != 2:
            raise tm.SchemaError(f"{run['depth'][k]}: depth image is not single-channel")
        z = raw.astype(float) / DEPTH_FACTOR
        valid = raw > 0
        axs[r, 0].imshow(rgb)
        axs[r, 0].set_title(f"frame {k}: RGB  t = {run['t_rgb'][k]:.6f}", fontsize=9)
        im = axs[r, 1].imshow(np.where(valid, z, np.nan), cmap="viridis")
        fig.colorbar(im, ax=axs[r, 1], fraction=0.04, label="depth (m)")
        axs[r, 1].set_title(f"depth (raw / {DEPTH_FACTOR:g})  t = {run['t_depth'][k]:.6f} ({run['dt_ms'][k]:+.1f} ms)",
                            fontsize=9)
        axs[r, 2].imshow(valid, cmap="gray", vmin=0, vmax=1)
        axs[r, 2].set_title(f"valid depth (white): {100 * valid.mean():.1f} % of pixels", fontsize=9)
        for a in axs[r]:
            a.axis("off")
    fig.suptitle(f"RGB, depth and valid-depth mask of frames {idx} (source images from {Path(dataset).name})", fontsize=11)
    fig.tight_layout()
    save(fig, out, "rgb_depth_examples", made)


def run_metrics(run, met, out, made):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    keys = [("ate_raw_rmse_m", "ATE raw"), ("ate_aligned_rmse_m", "ATE aligned"), ("final_position_error_m", "final pos.")]
    x = np.arange(len(keys))
    for s, m in zip((-1, 1), tm.METHODS):
        v = [met[m][k] * 100 for k, _ in keys] + [met[m]["rpe_1frame"]["trans_rmse_m"] * 100]
        axs[0].bar(np.arange(4) + s * 0.2, v, 0.4, color=STYLE[m]["color"], label=STYLE[m]["label"])
        for xi, vi in zip(np.arange(4) + s * 0.2, v):
            axs[0].text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=7)
        rv = [met[m]["final_orientation_error_deg"], met[m]["rpe_1frame"]["rot_rmse_deg"]]
        axs[1].bar(np.arange(2) + s * 0.2, rv, 0.4, color=STYLE[m]["color"], label=STYLE[m]["label"])
        for xi, vi in zip(np.arange(2) + s * 0.2, rv):
            axs[1].text(xi, vi, f"{vi:.3f}", ha="center", va="bottom", fontsize=7)
    axs[0].set_xticks(np.arange(4))
    axs[0].set_xticklabels([l for _, l in keys] + ["RPE trans\n(1 frame, RMSE)"], fontsize=8)
    axs[0].set_ylabel("cm")
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["final orientation", "RPE rot\n(1 frame, RMSE)"], fontsize=8)
    axs[1].set_ylabel("deg")
    for a in axs[:2]:
        a.legend(fontsize=8), a.grid(alpha=0.3, axis="y")
    lines = [f"frames {met['frames']}, attempted pairs {met['attempted_pairs']}, span {met['time_span_s']:.2f} s", ""]
    for m in tm.METHODS:
        mm = met[m]
        lines += [f"{STYLE[m]['label']}:",
                  f"  successful {mm['successful_pairs']} / failed {mm['failed_pairs']}  ({100 * mm['success_rate']:.1f} %)",
                  f"  correspondences mean {mm['correspondences_mean']:.1f}",
                  f"  inliers mean {mm['inliers_mean_successful']:.1f}, min {mm['inliers_min_successful']}", ""]
    lines += [f"matches per pair: mean {met['matches_mean']:.1f}, min {met['matches_min']}",
              f"RGB-depth offset max {met['rgb_depth_dt_ms_max_abs']:.1f} ms"]
    axs[2].axis("off")
    axs[2].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle("Run metrics (frame 0 anchored to ground truth; 'aligned' = one global SE(3), no scale)", fontsize=11)
    fig.tight_layout()
    save(fig, out, "run_metrics", made)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--dataset")
    ap.add_argument("--max-dt", type=float)
    ap.add_argument("--max-gt-gap", type=float)
    a = ap.parse_args()
    inp = Path(a.input)
    out = Path(a.output) if a.output else inp / "figures"
    cfg = json.loads((inp / "config.json").read_text()) if (inp / "config.json").exists() else {}
    dataset = a.dataset or cfg.get("dataset")
    max_dt = a.max_dt if a.max_dt is not None else cfg.get("max_dt", 0.02)
    max_gap = a.max_gt_gap if a.max_gt_gap is not None else cfg.get("max_gt_gap", 0.05)
    try:
        run = tm.load_run(inp)
    except tm.SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    problems = tm.check_consistency(run)
    unavailable = []
    if dataset and Path(dataset).is_dir():
        problems += tm.ground_truth_checks(run, dataset, max_dt, max_gap)
    else:
        unavailable.append("association / ground-truth re-check and rgb_depth_examples: no sequence directory "
                           "(pass --dataset or run through run_tum_stress.py)")
    met = tm.compute_metrics(run)
    made, checks = [], []
    trajectory_3d(run, out, made, checks)
    trajectory_errors(run, out, made, checks)
    relative_pose_errors(run, out, made, checks)
    if dataset and Path(dataset).is_dir():
        rgb_depth_examples(run, dataset, out, made)
    run_metrics(run, met, out, made)
    mismatch = sorted({name for name, ok in checks if not ok})
    if mismatch:
        problems.append(f"plotted values differ from the CSV values in {mismatch}")
    result = tm.to_json_safe({"input": str(inp), "dataset": dataset, "metrics": met, "sanity_problems": problems,
                              "plotted_values_match_csv": not mismatch, "figures": made,
                              "not_generated": [FEATURE_MATCHES_BLOCKED] + unavailable})
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    rep = [f"# TUM evaluation report: {inp}", "", f"- sequence directory: {dataset or 'not given'}",
           f"- frames {met['frames']}, attempted pairs {met['attempted_pairs']}, time span {met['time_span_s']:.3f} s",
           "- frame 0 of both estimates is the ground-truth pose; errors are raw unless marked aligned", "",
           "| metric | PnP | ICP |", "|---|---|---|"]
    fmt = lambda v, f: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
    for key, label, f in (("successful_pairs", "successful pairs", "{}"), ("failed_pairs", "failed pairs", "{}"),
                          ("success_rate", "success rate", "{:.3f}"), ("ate_raw_rmse_m", "ATE raw RMSE (m)", "{:.4f}"),
                          ("ate_aligned_rmse_m", "ATE RMSE after one global SE(3) (m)", "{:.4f}"),
                          ("final_position_error_m", "final position error (m)", "{:.4f}"),
                          ("final_orientation_error_deg", "final orientation error (deg)", "{:.3f}"),
                          ("correspondences_mean", "correspondences per pair (mean)", "{:.1f}"),
                          ("inliers_mean_successful", "inliers per successful pair (mean)", "{:.1f}")):
        rep.append(f"| {label} | {fmt(met['pnp'][key], f)} | {fmt(met['icp'][key], f)} |")
    for rk in ("rpe_1frame", "rpe_1s"):
        for sub, label, f in (("trans_rmse_m", "translation RMSE (m)", "{:.4f}"), ("rot_rmse_deg", "rotation RMSE (deg)", "{:.3f}")):
            g = lambda m: (met[m].get(rk) or {}).get(sub)
            d = (met["pnp"].get(rk) or {}).get("delta_frames", met["rpe_interval_frames_for_seconds"])
            rep.append(f"| RPE {rk[4:]} ({d} frames) {label} | {fmt(g('pnp'), f)} | {fmt(g('icp'), f)} |")
    rep += ["", f"- matches per pair: mean {met['matches_mean']:.1f}, min {met['matches_min']}",
            f"- max RGB-depth offset {met['rgb_depth_dt_ms_max_abs']:.2f} ms",
            f"- sanity problems: {problems or 'none'}", f"- plotted values match the CSV: {not mismatch}",
            f"- figures: {', '.join(made)}", "", "## Not generated", ""] + [f"- {u}" for u in [FEATURE_MATCHES_BLOCKED] + unavailable]
    if met["notes"]:
        rep += ["", "## Notes", ""] + [f"- {n}" for n in met["notes"]]
    (out / "report.md").write_text("\n".join(rep) + "\n")
    print(f"wrote {len(made)} figures (PNG + PDF), metrics.json and report.md to {out}")
    if problems:
        print("SANITY PROBLEMS:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
