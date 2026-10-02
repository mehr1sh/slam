#!/usr/bin/env python3
"""Machine-readable Bunny-experiment metrics for the vision-library migration.

Runs build/slam_trajectory_test on a TEMPORARY COPY of data/synthetic_bunny/
with its export-only flags (--export-dir, --export-pair all), so committed
data is never modified, and writes to OUT_DIR:

  metrics.json                       summary + per-pair metrics (see below)
  pair_metrics.csv                   the program's per-pair export (copied)
  slam_trajectory.csv                the trajectories this run produced
  correspondences/pair_<i>_<j>_correspondences.csv   all per-pair correspondences
  comparison.md                      only with --compare BASE_DIR

Tracked metrics: keypoints per frame, raw / filtered matches, 3D->2D and 3D->3D
correspondences, RANSAC inliers, PnP reprojection error (estimated pose, over
inliers and over all correspondences), per-pair relative rotation/translation
errors (PnP, ICP), failed pairs, final and mean PnP/ICP trajectory errors,
and whether the trajectory files equal the committed ones byte for byte.

Usage (results environment, after `pixi run build`):
  pixi run -e results python scripts/migration/bunny_metrics.py OUT_DIR [--compare BASE_DIR]
"""

import argparse
import csv
import filecmp
import json
import math
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "synthetic_bunny"
EXE = ROOT / "build" / "slam_trajectory_test"


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def stats(v):
    v = [x for x in v if x == x]
    if not v:
        return dict(mean=None, median=None, max=None)
    return dict(mean=st.mean(v), median=st.median(v), max=max(v))


def run(out):
    if not EXE.exists():
        sys.exit("build/slam_trajectory_test not found -- run `pixi run build` first")
    out.mkdir(parents=True, exist_ok=True)
    corr_dir = out / "correspondences"
    if corr_dir.exists():
        shutil.rmtree(corr_dir)
    corr_dir.mkdir()
    intr = [float(x) for x in (DATASET / "intrinsics.txt").read_text().split()]
    fx, fy, cx, cy = intr[2:6]
    with tempfile.TemporaryDirectory() as tmp:
        ds, exp = Path(tmp) / "ds", Path(tmp) / "export"
        shutil.copytree(DATASET, ds)
        exp.mkdir()
        r = subprocess.run([str(EXE), str(ds), "--export-dir", str(exp), "--export-pair", "all"],
                           cwd=EXE.parent, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"slam_trajectory_test failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        identical = {f: filecmp.cmp(ds / f, DATASET / f, shallow=False)
                     for f in ("pnp_trajectory.txt", "icp_trajectory.txt", "slam_trajectory.csv")}
        shutil.copy(exp / "pair_metrics.csv", out / "pair_metrics.csv")
        shutil.copy(ds / "slam_trajectory.csv", out / "slam_trajectory.csv")
        for p in sorted(exp.glob("pair_*_correspondences.csv")):
            shutil.copy(p, corr_dir / p.name)

    pm = read_csv(out / "pair_metrics.csv")
    traj = read_csv(out / "slam_trajectory.csv")
    pairs = []
    for r in pm:
        i, j = int(r["frame_i"]), int(r["frame_j"])
        rows = read_csv(corr_dir / f"pair_{i}_{j}_correspondences.csv")
        R = quat_to_R(*(float(r[f"pnp_rel_q{c}"]) for c in "xyzw"))
        t = np.array([float(r[f"pnp_rel_t{c}"]) for c in "xyz"])
        e_all, e_inl = [], []
        for c in rows:
            if c["pnp_used"] != "1":
                continue
            X = np.array([float(c["X_i"]), float(c["Y_i"]), float(c["Z_i"])])
            Xc = R @ X + t
            u, v = fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy
            e = math.hypot(u - float(c["u_j"]), v - float(c["v_j"]))
            e_all.append(e)
            if c["pnp_inlier"] == "1":
                e_inl.append(e)
        pairs.append(dict(
            pair=f"{i}->{j}", keypoints_i=int(r["orb_keypoints_i"]), keypoints_j=int(r["orb_keypoints_j"]),
            raw_matches=int(r["raw_matches"]), filtered_matches=int(r["filtered_matches"]),
            pnp_correspondences=int(r["pnp_correspondences"]), pnp_inliers=int(r["pnp_inliers"]),
            icp_correspondences=int(r["icp_correspondences"]), pnp_ok=int(r["pnp_ok"]), icp_ok=int(r["icp_ok"]),
            pnp_rot_err_deg=float(r["pnp_relative_rotation_error_deg"]),
            pnp_trans_err_m=float(r["pnp_relative_translation_error_m"]),
            icp_rot_err_deg=float(r["icp_relative_rotation_error_deg"]),
            icp_trans_err_m=float(r["icp_relative_translation_error_m"]),
            reproj_inlier_mean_px=stats(e_inl)["mean"], reproj_inlier_median_px=stats(e_inl)["median"],
            reproj_all_mean_px=stats(e_all)["mean"], reproj_all_median_px=stats(e_all)["median"]))

    def col(k):
        return [p[k] for p in pairs]

    kp = [pairs[0]["keypoints_i"]] + col("keypoints_j")
    summary = dict(
        n_frames=len(traj), n_pairs=len(pairs),
        keypoints_per_frame=stats(kp),
        raw_matches=stats(col("raw_matches")), filtered_matches=stats(col("filtered_matches")),
        pnp_correspondences=stats(col("pnp_correspondences")), pnp_inliers=stats(col("pnp_inliers")),
        icp_correspondences=stats(col("icp_correspondences")),
        pnp_failed_pairs=sum(1 - p["pnp_ok"] for p in pairs), icp_failed_pairs=sum(1 - p["icp_ok"] for p in pairs),
        pnp_rel_rot_err_deg=stats(col("pnp_rot_err_deg")), pnp_rel_trans_err_m=stats(col("pnp_trans_err_m")),
        icp_rel_rot_err_deg=stats(col("icp_rot_err_deg")), icp_rel_trans_err_m=stats(col("icp_trans_err_m")),
        pnp_reproj_inlier_mean_px=stats(col("reproj_inlier_mean_px")),
        pnp_final_trans_err_m=float(traj[-1]["pnp_translation_error"]),
        pnp_final_rot_err_deg=float(traj[-1]["pnp_rotation_error"]),
        icp_final_trans_err_m=float(traj[-1]["icp_translation_error"]),
        icp_final_rot_err_deg=float(traj[-1]["icp_rotation_error"]),
        pnp_mean_trans_err_m=st.mean(float(t["pnp_translation_error"]) for t in traj),
        icp_mean_trans_err_m=st.mean(float(t["icp_translation_error"]) for t in traj),
        trajectory_files_identical_to_committed=identical,
        keypoints_per_frame_list=kp)
    result = dict(summary=summary, pairs=pairs)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def compare(cur, base_dir, out):
    base = json.loads((base_dir / "metrics.json").read_text())
    bs, cs = base["summary"], cur["summary"]
    lines = [f"# Bunny metrics: {out.name} vs {base_dir.name}\n",
             "| metric | " + base_dir.name + " | " + out.name + " | change |", "|---|---|---|---|"]
    keys = ["keypoints_per_frame", "raw_matches", "filtered_matches", "pnp_correspondences", "pnp_inliers",
            "icp_correspondences", "pnp_failed_pairs", "icp_failed_pairs", "pnp_rel_rot_err_deg",
            "pnp_rel_trans_err_m", "icp_rel_rot_err_deg", "icp_rel_trans_err_m", "pnp_reproj_inlier_mean_px",
            "pnp_final_trans_err_m", "pnp_final_rot_err_deg", "icp_final_trans_err_m", "icp_final_rot_err_deg",
            "pnp_mean_trans_err_m", "icp_mean_trans_err_m"]
    for k in keys:
        b, c = bs[k], cs[k]
        if isinstance(b, dict):
            for sub in ("mean", "median", "max"):
                bv, cv = b.get(sub), c.get(sub)
                d = (cv - bv) if (bv is not None and cv is not None) else None
                lines.append(f"| {k} ({sub}) | {fmt(bv)} | {fmt(cv)} | {fmt(d)} |")
        else:
            d = (c - b) if isinstance(b, (int, float)) else None
            lines.append(f"| {k} | {fmt(b)} | {fmt(c)} | {fmt(d)} |")
    lines += ["", "## Per pair", "",
              "| pair | kp i/j (base → now) | filtered | PnP corr | inliers | PnP rot err ° | ICP rot err ° |",
              "|---|---|---|---|---|---|---|"]
    for bp, cp in zip(base["pairs"], cur["pairs"]):
        lines.append(f"| {cp['pair']} | {bp['keypoints_i']}/{bp['keypoints_j']} → {cp['keypoints_i']}/{cp['keypoints_j']} "
                     f"| {bp['filtered_matches']} → {cp['filtered_matches']} "
                     f"| {bp['pnp_correspondences']} → {cp['pnp_correspondences']} "
                     f"| {bp['pnp_inliers']} → {cp['pnp_inliers']} "
                     f"| {bp['pnp_rot_err_deg']:.3f} → {cp['pnp_rot_err_deg']:.3f} "
                     f"| {bp['icp_rot_err_deg']:.3f} → {cp['icp_rot_err_deg']:.3f} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:len(keys) * 3 + 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--compare", help="baseline/previous checkpoint directory with metrics.json")
    a = ap.parse_args()
    out = Path(a.out_dir).resolve()
    res = run(out)
    s = res["summary"]
    print(f"[metrics] {out.name}: PnP final {s['pnp_final_trans_err_m']:.4f} m / "
          f"{s['pnp_final_rot_err_deg']:.2f} deg, ICP final {s['icp_final_trans_err_m']:.4f} m / "
          f"{s['icp_final_rot_err_deg']:.2f} deg, failed pairs PnP {s['pnp_failed_pairs']} ICP {s['icp_failed_pairs']}, "
          f"identical to committed: {s['trajectory_files_identical_to_committed']}")
    if a.compare:
        compare(res, Path(a.compare).resolve(), out)


if __name__ == "__main__":
    main()
