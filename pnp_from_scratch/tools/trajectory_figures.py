#!/usr/bin/env python3
"""Trajectory figures of the scratch pipeline against GT, the reference PnP and
the reference ICP. NumPy + Matplotlib only (no vision library); the data are
the exported CSVs:

  GT, reference PnP, ICP   data/synthetic_bunny/slam_trajectory.csv (frozen baseline, the Blender scene's file)
                           docs/migration/baseline/metrics.json      (reference per-pair errors)
  scratch refined          pnp_from_scratch/results/pipeline/trajectory.csv, pairs.csv
                           (the linear pose's per-pair errors are the pairs.csv linear_* columns)
  bunny                    data/meshes/bunny/bun_zipper.ply (world = mesh frame, Y up)

Outputs (generated, gitignored), --out (default pnp_from_scratch/results/figures/), PNG + PDF:
  trajectory_3d_comparison     camera centres in 3D, equal axes, start and end marked
  trajectory_position_error    camera-centre error vs GT per frame
  trajectory_rotation_error    rotation error vs GT per frame
  relative_pose_error          per-pair rotation and translation error (scratch linear / refined, reference PnP)
  final_trajectory_comparison  measured final and mean errors (no ranking)

Usage (repository root): python3 pnp_from_scratch/tools/trajectory_figures.py [--out DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_figures as mf  # noqa: E402  (shared loaders, colours, save())

ROOT = mf.ROOT
PIPE = ROOT / "pnp_from_scratch" / "results" / "pipeline"
COLOURS = {"GT": mf.GT, "Scratch PnP (refined)": mf.SCRATCH_REF, "Reference PnP": mf.REF_PNP, "Reference ICP": mf.ICP}


def load():
    ref = mf.rows(ROOT / "data" / "synthetic_bunny" / "slam_trajectory.csv")
    scr = mf.rows(PIPE / "trajectory.csv")
    pos = {"GT": np.array([[float(r[f"gt_{a}"]) for a in "xyz"] for r in ref]),
           "Scratch PnP (refined)": np.array([[float(r[f"t{a}"]) for a in "xyz"] for r in scr]),
           "Reference PnP": np.array([[float(r[f"pnp_{a}"]) for a in "xyz"] for r in ref]),
           "Reference ICP": np.array([[float(r[f"icp_{a}"]) for a in "xyz"] for r in ref])}
    err_t = {"Scratch PnP (refined)": np.array([float(r["translation_error_m"]) for r in scr]),
             "Reference PnP": np.array([float(r["pnp_translation_error"]) for r in ref]),
             "Reference ICP": np.array([float(r["icp_translation_error"]) for r in ref])}
    err_r = {"Scratch PnP (refined)": np.array([float(r["rotation_error_deg"]) for r in scr]),
             "Reference PnP": np.array([float(r["pnp_rotation_error"]) for r in ref]),
             "Reference ICP": np.array([float(r["icp_rotation_error"]) for r in ref])}
    # sanity: the scratch errors are those of the stored poses (position part)
    assert np.allclose(np.linalg.norm(pos["Scratch PnP (refined)"] - pos["GT"], axis=1), err_t["Scratch PnP (refined)"],
                       atol=1e-5)
    pairs = mf.rows(PIPE / "pairs.csv")
    base = json.loads((mf.BASE / "metrics.json").read_text())["pairs"]
    rel = {"Scratch PnP (linear)": (np.array([float(p["linear_rot_err_deg"]) for p in pairs]),
                                    np.array([float(p["linear_trans_err_m"]) for p in pairs])),
           "Scratch PnP (refined)": (np.array([float(p["rot_err_deg"]) for p in pairs]),
                                     np.array([float(p["trans_err_m"]) for p in pairs])),
           "Reference PnP": (np.array([b["pnp_rot_err_deg"] for b in base]),
                             np.array([b["pnp_trans_err_m"] for b in base]))}
    return pos, err_t, err_r, rel


def bunny_points(step=12):
    lines = (ROOT / "data" / "meshes" / "bunny" / "bun_zipper.ply").read_text().splitlines()
    n = next(int(l.split()[2]) for l in lines if l.startswith("element vertex"))
    start = lines.index("end_header") + 1
    return np.array([[float(v) for v in l.split()[:3]] for l in lines[start:start + n:step]])


def to_plot(p):  # world (Y up) -> plot axes (x, -z, y) so that up is vertical
    return p[..., 0], -p[..., 2], p[..., 1]


def trajectory_3d(pos):
    fig = plt.figure(figsize=(8.6, 7.4))
    ax = fig.add_subplot(projection="3d")
    b = bunny_points()
    ax.scatter(*to_plot(b), s=0.4, c="0.55", alpha=0.35, depthshade=False, label="Stanford Bunny (mesh vertices)")
    for name, p in pos.items():
        lw, ls = (2.4, "-") if name == "GT" else (1.6, "-")
        ax.plot(*to_plot(p), ls, color=COLOURS[name], lw=lw, label=name, zorder=3)
        ax.scatter(*to_plot(p[-1:]), s=46, marker="s", color=COLOURS[name], edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter(*to_plot(pos["GT"][:1]), s=90, marker="o", facecolors="white", edgecolors="black", linewidths=1.2,
               zorder=5)
    allp = np.concatenate([np.stack(to_plot(p), 1) for p in pos.values()] + [np.stack(to_plot(b), 1)])
    mid, half = (allp.max(0) + allp.min(0)) / 2, (allp.max(0) - allp.min(0)).max() / 2
    for setlim, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setlim(m - half, m + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("−z (m)")
    ax.set_zlabel("y, up (m)")
    ax.view_init(elev=28, azim=-60)
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], ls="none", marker="o", mfc="white", mec="black", ms=8),
          Line2D([], [], ls="none", marker="s", mfc="0.7", mec="black", ms=6)]
    l += ["start (frame 0, all methods)", "end (frame 35)"]
    ax.legend(h, l, loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Camera trajectories, frames 0–35 (equal axis scaling)", fontsize=11)
    fig.tight_layout()
    mf.save(fig, "trajectory_3d_comparison")


def per_frame(err, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    f = np.arange(len(next(iter(err.values()))))
    for k, v in err.items():
        ax.plot(f, v, "-o", ms=3, lw=1.6, color=COLOURS[k], label=k)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, f[-1])
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    mf.save(fig, name)


def relative_pose(rel):
    cols = {"Scratch PnP (linear)": mf.SCRATCH_PYR, "Scratch PnP (refined)": mf.SCRATCH_REF, "Reference PnP": mf.REF_PNP}
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    x = np.arange(35)
    w = 0.27
    for ax, k, unit, lab in ((axs[0], 0, 1, "rotation error (°)"), (axs[1], 1, 100, "translation error (cm)")):
        for i, (name, v) in enumerate(rel.items()):
            ax.bar(x + (i - 1) * w, v[k] * unit, w, color=cols[name],
                   label=f"{name}: mean {v[k].mean() * unit:.2f}, median {np.median(v[k]) * unit:.2f}")
        ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axs[1].set_xlabel("pair i → i+1 (i)")
    axs[0].set_title("Per-pair relative pose error vs ground truth (true motion: 10°, 8.7 cm per pair)", fontsize=10)
    fig.tight_layout()
    mf.save(fig, "relative_pose_error")


def final_summary(err_t, err_r):
    names = list(err_t)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, err, unit, lab in ((axs[0], err_t, "m", "position error (m)"), (axs[1], err_r, "°", "rotation error (°)")):
        x = np.arange(len(names))
        fin = [err[n][-1] for n in names]
        mean = [err[n].mean() for n in names]
        ax.bar(x - 0.18, fin, 0.36, color=[COLOURS[n] for n in names], edgecolor="black", lw=0.6, label="frame 35")
        ax.bar(x + 0.18, mean, 0.36, color=[COLOURS[n] for n in names], alpha=0.45, edgecolor="black", lw=0.6,
               hatch="//", label="mean over frames 0–35")
        fmt = "{:.3f}" if unit == "m" else "{:.1f}"
        for xi, a, b in zip(x, fin, mean):
            ax.text(xi - 0.18, a, fmt.format(a), ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.18, b, fmt.format(b), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Accumulated trajectory error vs ground truth (measured values)", fontsize=11)
    fig.tight_layout()
    mf.save(fig, "final_trajectory_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(mf.OUT.relative_to(ROOT)))
    mf.OUT = (ROOT / ap.parse_args().out).resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    pos, err_t, err_r, rel = load()
    trajectory_3d(pos)
    per_frame(err_t, "camera-centre error (m)", "Camera-centre position error vs ground truth", "trajectory_position_error")
    per_frame(err_r, "rotation error (°)", "Camera rotation error vs ground truth", "trajectory_rotation_error")
    relative_pose(rel)
    final_summary(err_t, err_r)
    for n in err_t:
        print(f"{n:22s} mean {err_t[n].mean():.4f} m / {err_r[n].mean():.2f} deg   final {err_t[n][-1]:.4f} m / "
              f"{err_r[n][-1]:.2f} deg")


if __name__ == "__main__":
    main()
