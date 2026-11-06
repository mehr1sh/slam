#!/usr/bin/env python3
"""Pose-convention tests with non-identity poses, end to end (synthetic + the real evaluator, if available).

Usage (repository root): python3 tools/evaluation/test_pose_conventions.py [--dataset DIR]     exit 1 on failure

Conventions under test (tests/tum/tum_trajectory_eval.cpp, include/dataset/tum.hpp):
  T_wc            X_world = R X_cam + t (TUM ground truth and trajectory.csv), quaternion order qx qy qz qw
  T_j<-i          X_j = R X_i + t; from poses: T_j<-i = T_wc[j]^-1 T_wc[i]; solvePnPRansac with object points in
                  camera i and image points in camera j returns exactly this (object -> camera)
  accumulation    T_wc[i+1] = T_wc[i] T_j<-i^-1   (failed or rejected pair: T_wc[i+1] = T_wc[i])
  viewer          pose rows [R row-major (9), t (3)]; a camera-frame point x is drawn at R x + t
"""

import argparse
import json
import os
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "diagnostics"))
sys.path.insert(0, str(HERE / "dashboard_app"))
import tum_metrics as tm  # noqa: E402

failures = 0


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def synthetic():
    print("=== synthetic, non-identity poses ===")
    rng = np.random.default_rng(7)
    Twc = [se3(rot(rng.normal(size=3), rng.uniform(5, 40)), rng.normal(size=3)) for _ in range(6)]
    rel = [np.linalg.inv(Twc[k + 1]) @ Twc[k] for k in range(5)]  # T_j<-i
    X_i = rng.normal(size=3) + [0, 0, 3]
    Xw = Twc[0][:3, :3] @ X_i + Twc[0][:3, 3]
    X_j = np.linalg.inv(Twc[1]) @ np.r_[Xw, 1]
    check(np.allclose(rel[0] @ np.r_[X_i, 1], X_j), "T_j<-i = T_wc[j]^-1 T_wc[i] maps a camera-i point to camera j")
    acc = [Twc[0]]
    for T in rel:
        acc.append(acc[-1] @ np.linalg.inv(T))
    check(all(np.allclose(a, b) for a, b in zip(acc, Twc)), "T_wc[i+1] = T_wc[i] T_j<-i^-1 reproduces every pose")
    wrong1 = [Twc[0]]
    for T in rel:
        wrong1.append(np.linalg.inv(T) @ wrong1[-1])
    wrong2 = [Twc[0]]
    for T in rel:
        wrong2.append(wrong2[-1] @ T)
    check(not np.allclose(wrong1[-1], Twc[-1]) and not np.allclose(wrong2[-1], Twc[-1]),
          "controls: left multiplication or a missing inverse do NOT reproduce the poses")
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    R = tm.quat_to_R(np.array([0.0, 0.0, s, c]))  # 90 deg about z, order qx qy qz qw
    check(np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12), "quaternion (qx, qy, qz, qw) = 90 deg about z")
    R = tm.quat_to_R(np.array([s, 0.0, 0.0, c]))
    check(np.allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12), "quaternion 90 deg about x maps y to z")
    # viewer row layout: [R row-major, t]; a point x is drawn at R x + t
    import data as dash
    row = dash.pose_list(np.array([Twc[2][:3, :3]]), np.array([Twc[2][:3, 3]]))[0]
    xf = lambda T, x: np.array([T[0] * x[0] + T[1] * x[1] + T[2] * x[2] + T[9], T[3] * x[0] + T[4] * x[1] + T[5] * x[2] + T[10],
                                T[6] * x[0] + T[7] * x[1] + T[8] * x[2] + T[11]])  # = view3d.js xf
    check(np.allclose(xf(row, X_i), Twc[2][:3, :3] @ X_i + Twc[2][:3, 3], atol=1e-6),
          "viewer pose rows: camera-frame point drawn at R x + t (T_wc, not its inverse)")
    check(np.allclose(xf(row, [0, 0, 1]) - xf(row, [0, 0, 0]), Twc[2][:3, 2], atol=1e-6),
          "viewer: frustum optical axis = third column of R_wc")


def real(dataset):
    print("=== real evaluator (tum_trajectory_eval) ===")
    binary = ROOT / "build" / "tum_trajectory_eval"
    if not binary.is_file() or not (Path(dataset) / "rgb.txt").is_file():
        print(f"  [SKIPPED] needs {binary} and {dataset}")
        return
    tmp = Path(tempfile.mkdtemp(prefix="pose_conv_"))
    try:
        r = subprocess.run([str(binary), dataset, "--frames", "12", "--stride", "4", "--out-dir", str(tmp / "a"),
                            "--export-pairs", "2,7"], capture_output=True, text=True)
        check(r.returncode == 0, "evaluator runs")
        run = tm.load_run(tmp / "a")
        check(tm.check_consistency(run) == [], "trajectory relative poses equal pairs.csv errors; frame 0 at GT")
        summ = (tmp / "a" / "summary.txt").read_text()
        import re
        g, w = (float(x) for x in re.search(r"median of pair medians ([\d.]+) px \(control, poses read as T_cw instead: "
                                            r"([\d.]+) px", summ).groups())
        check(g < 3 and w > 2 * g, f"solvePnP convention: PnP inliers reproject {g:.2f} px with the GT T_j<-i "
                                   f"(T_wc as T_cw: {w:.2f} px)")
        for k in (2, 7):
            p = json.loads((tmp / "a" / "diagnostics" / f"pair_{k}_{k + 1}" / "pair.json").read_text())
            Ti = se3(run["gt_R"][k], run["gt_p"][k])
            Tj = se3(run["gt_R"][k + 1], run["gt_p"][k + 1])
            check(np.allclose(np.linalg.inv(Tj) @ Ti, p["T_j_from_i_gt"], atol=2e-5),
                  f"pair {k}: exported GT T_j<-i = T_wc[j]^-1 T_wc[i] from trajectory.csv")
            Pi = se3(run["pnp_R"][k], run["pnp_p"][k])
            Pj = se3(run["pnp_R"][k + 1], run["pnp_p"][k + 1])
            if p["status"]["pnp_accepted"]:
                check(np.allclose(np.linalg.inv(Pj) @ Pi, p["T_j_from_i_pnp"], atol=2e-5),
                      f"pair {k}: trajectory PnP poses reproduce the solver's T_j<-i (accumulation order and inverse)")
            check(abs(tm.rot_angle_deg(np.array(p["T_j_from_i_gt"])[:3, :3])) > 0.1,
                  f"pair {k}: the test motion is not the identity ({tm.rot_angle_deg(np.array(p['T_j_from_i_gt'])[:3, :3]):.2f} deg)")
        # dashboard scene loader returns exactly the CSV poses
        import data as dash
        dash.OUTPUT = tmp
        sc_rows = dash.pose_list(run["pnp_R"], run["pnp_p"])
        check(all(np.allclose(np.array(sc_rows[k][9:]), run["pnp_p"][k], atol=1e-6) and
                  np.allclose(np.array(sc_rows[k][:9]).reshape(3, 3), run["pnp_R"][k], atol=1e-6) for k in range(run["n"])),
              "viewer pose rows equal trajectory.csv (position and quaternion -> R)")
        # acceptance: rejected poses are held and kept apart from the solver status
        r = subprocess.run([str(binary), dataset, "--frames", "5", "--out-dir", str(tmp / "b"),
                            "--pnp-min-distinct-inliers", "100000"], capture_output=True, text=True)
        b = tm.load_run(tmp / "b")
        check(r.returncode == 0 and b["pnp_solver_returned"].all() and not b["pnp_accepted"].any(),
              "acceptance: every returned pose rejected under an impossible threshold; solver status kept")
        check(np.allclose(b["pnp_p"], b["pnp_p"][0]) and np.isfinite(b["pnp_returned_rot_err_deg"]).all(),
              "acceptance: rejected poses are held (not used), their errors kept in pnp_returned_*")
        met = tm.compute_metrics(b)["pnp"]
        check(met["rejected_pairs"] == 4 and met["solver_returned_pairs"] == 4 and met["failed_pairs"] == 4,
              "metrics: rejected pairs counted as not accepted, solver-returned counted separately")
        r = subprocess.run([str(binary), dataset, "--frames", "5", "--out-dir", str(tmp / "c"),
                            "--pnp-min-distinct-inliers", "0"], capture_output=True, text=True)
        c = tm.load_run(tmp / "c")
        check(c["pnp_accepted"].all() and tm.compute_metrics(c)["pnp"]["rejected_pairs"] == 0,
              "acceptance off (0): every returned pose accepted (previous behaviour)")
    finally:
        shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    # default: $SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz if SLAM_TUM_ROOT is set, else data/ (docs/DATA_AND_ARCHIVE.md)
    tum_root = Path(os.environ["SLAM_TUM_ROOT"]) if os.environ.get("SLAM_TUM_ROOT") else ROOT / "data"
    ap.add_argument("--dataset", default=str(tum_root / "rgbd_dataset_freiburg1_xyz"))
    a = ap.parse_args()
    synthetic()
    real(a.dataset)
    print(f"\ntest_pose_conventions: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
