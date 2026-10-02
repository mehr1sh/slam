#!/usr/bin/env python3
"""Trajectory-error diagnostics: CSVs and figures in diagnostics/.

Run (after `pixi run build`):
    pixi run -e results diagnostics            # = python scripts/run_diagnostics.py
    pixi run -e results diagnostics --skip-cpp # reuse diagnostics/cpp/

What it does:
  1. Copies data/synthetic_bunny/ to a temporary directory and runs
       build/slam_trajectory_test <copy> --export-dir diagnostics/cpp/export --export-pair all
     (export-only flags), checking that pnp_trajectory.txt, icp_trajectory.txt and
     slam_trajectory.csv are reproduced byte for byte. Then runs
       build/trajectory_diagnostics_test <copy> diagnostics/cpp
     which repeats the unchanged solver calls and measures them against ground truth.
  2. Stops if the GT replay of the accumulation formula fails.
  3. Writes the diagnostics/*.csv tables and diagnostics/figures/*.png listed in
     diagnostics/REPORT.md.

Nothing here changes the PnP/ICP algorithms, their parameters, the dataset or the
accumulation. Subset re-solves reported by trajectory_diagnostics_test are
diagnostic measurements only.
"""

import argparse
import csv
import filecmp
import math
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "synthetic_bunny"
BUILD = ROOT / "build"
OUT = ROOT / "diagnostics"
FIG = OUT / "figures"
CPP = OUT / "cpp"
EXPORT = CPP / "export"
N_WORST = 5
C = {"GT": (0.05, 0.55, 0.28), "PnP": (0.85, 0.45, 0.00), "ICP": (0.62, 0.22, 0.72)}


# ---------------------------------------------------------------- helpers ---

def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def se3(t, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*q)
    T[:3, 3] = t
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_err_deg(Ra, Rb):
    """Same formula as report_utils.hpp RotationAngleDeg."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def angle_deg(T):
    return rot_err_deg(np.eye(3), T[:3, :3])


def axis(T):
    R = T[:3, :3]
    a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_tum(path):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        out.append(se3(v[1:4], v[4:8]))
    return out


def pose_cols(r, p):
    return se3([fnum(r[f"{p}_tx"]), fnum(r[f"{p}_ty"]), fnum(r[f"{p}_tz"])],
               [fnum(r[f"{p}_qx"]), fnum(r[f"{p}_qy"]), fnum(r[f"{p}_qz"]), fnum(r[f"{p}_qw"])])


def stats(v):
    v = [x for x in v if x == x]
    return dict(mean=st.mean(v), median=st.median(v), min=min(v), max=max(v), std=st.pstdev(v))


def write_csv(path, header, rows, fmt="{:.6f}"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt.format(x) if isinstance(x, float) else x for x in r])


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label} {detail}")
    if not ok:
        sys.exit(f"diagnostic precondition failed: {label}")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ C++ runs ------

def run_cpp():
    for exe in ("slam_trajectory_test", "trajectory_diagnostics_test"):
        if not (BUILD / exe).exists():
            sys.exit(f"missing build/{exe}; run `pixi run build` first")
    if CPP.exists():
        shutil.rmtree(CPP)
    EXPORT.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "synthetic_bunny"
        shutil.copytree(DATASET, ds)
        with open(EXPORT / "slam_trajectory_test.log", "w") as log:
            subprocess.run([str(BUILD / "slam_trajectory_test"), str(ds), "--export-dir", str(EXPORT),
                            "--export-pair", "all"], cwd=BUILD, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        for name in ("pnp_trajectory.txt", "icp_trajectory.txt", "slam_trajectory.csv"):
            check(f"export run reproduces {name} byte for byte",
                  filecmp.cmp(ds / name, DATASET / name, shallow=False))
        with open(CPP / "trajectory_diagnostics_test.log", "w") as log:
            subprocess.run([str(BUILD / "trajectory_diagnostics_test"), str(ds), str(CPP)], cwd=BUILD,
                           stdout=log, stderr=subprocess.DEVNULL, check=True)


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cpp", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if not args.skip_cpp:
        run_cpp()

    # ---- 1. GT replay (highest priority: stop if it fails) ----
    replay = read_csv(CPP / "gt_replay.csv")
    shutil.copy(CPP / "gt_replay.csv", OUT / "gt_replay.csv")
    max_t = max(fnum(r["translation_error"]) for r in replay)
    max_r = max(fnum(r["rotation_error_deg"]) for r in replay)
    print(f"GT replay: max translation error {max_t:.3e} m, max rotation error {max_r:.3e} deg")
    if not (max_t < 1e-9 and max_r < 1e-6):
        sys.exit("STOP: GT replay FAILED -- the pose accumulation / transform convention / "
                 "error calculation is wrong. Investigation halted.")
    controls = read_csv(CPP / "gt_replay_controls.csv")
    shutil.copy(CPP / "gt_replay_controls.csv", OUT / "gt_replay_controls.csv")
    reacc = read_csv(CPP / "reaccumulation_check.csv")
    reacc_max = max(max(fnum(r["pnp_vs_file_trans_m"]), fnum(r["icp_vs_file_trans_m"])) for r in reacc)
    check("re-run solvers + accumulation reproduce the trajectory files (6-decimal rounding)",
          reacc_max < 2e-6, f"max {reacc_max:.2e} m")

    # ---- inputs ----
    T_gt = read_tum(DATASET / "groundtruth.txt")
    N = len(T_gt)
    traj = read_csv(DATASET / "slam_trajectory.csv")
    pm = read_csv(EXPORT / "pair_metrics.csv")
    sc = read_csv(CPP / "pair_solver_checks.csv")
    cd = read_csv(CPP / "correspondence_diagnostics.csv")
    P = N - 1
    check("pair counts", len(pm) == P and len(sc) == P)
    for a, b in zip(pm, sc):
        for k_pm, k_sc in (("pnp_correspondences", "pnp_correspondences"), ("pnp_inliers", "pnp_inliers"),
                           ("icp_correspondences", "icp_correspondences"), ("filtered_matches", "filtered_matches")):
            if int(a[k_pm]) != int(b[k_sc]):
                check(f"diagnostic re-run matches export ({k_pm}, pair {a['pair']})", False)
        for k_pm, k_sc in (("pnp_relative_rotation_error_deg", "pnp_rot_err_deg"),
                           ("icp_relative_rotation_error_deg", "icp_g2o_rot_err_deg"),
                           ("pnp_relative_translation_error_m", "pnp_trans_err_m"),
                           ("icp_relative_translation_error_m", "icp_g2o_trans_err_m")):
            if abs(fnum(a[k_pm]) - fnum(b[k_sc])) > 1e-6:
                check(f"diagnostic re-run matches export ({k_pm}, pair {a['pair']})", False)
    check("diagnostic re-run reproduces every exported count and relative error", True)

    # ---- 3. GT relative motion ----
    G = [inv(T_gt[i + 1]) @ T_gt[i] for i in range(P)]
    rows = []
    for i, g in enumerate(G):
        ax = axis(g)
        rows.append([f"{i}->{i+1}", angle_deg(g), float(np.linalg.norm(g[:3, 3])),
                     float(np.linalg.norm(T_gt[i + 1][:3, 3] - T_gt[i][:3, 3])),
                     *[float(x) for x in ax], *[float(x) for x in g[:3, 3]]])
    write_csv(OUT / "gt_relative_motion.csv",
              ["pair", "rotation_deg", "translation_m", "camera_centre_distance_m", "axis_x_cam", "axis_y_cam",
               "axis_z_cam", "t_x_cam", "t_y_cam", "t_z_cam"], rows)

    # ---- 4/5. relative errors ----
    rel = {}
    for meth, pre in (("pnp", "pnp_rel"), ("icp", "icp_rel")):
        out = []
        for i, r in enumerate(pm):
            E = pose_cols(r, pre)
            g = G[i]
            re = rot_err_deg(g[:3, :3], E[:3, :3])
            te = float(np.linalg.norm(g[:3, 3] - E[:3, 3]))
            cpp_re = fnum(r[f"{meth}_relative_rotation_error_deg"])
            cpp_te = fnum(r[f"{meth}_relative_translation_error_m"])
            if abs(re - cpp_re) > 1e-4 or abs(te - cpp_te) > 1e-7:
                check(f"{meth} relative error recomputation, pair {i}", False, f"{re} vs {cpp_re}")
            out.append(dict(pair=i, gt_rot=angle_deg(g), est_rot=angle_deg(E), rot_err=cpp_re,
                            gt_t=float(np.linalg.norm(g[:3, 3])), est_t=float(np.linalg.norm(E[:3, 3])),
                            t_err=cpp_te, E=E))
        rel[meth] = out
        write_csv(OUT / f"{meth}_relative_errors.csv",
                  ["pair", "gt_rotation_deg", f"{meth}_rotation_deg", "rotation_error_deg", "gt_translation_m",
                   f"{meth}_translation_m", "translation_error_m"],
                  [[f"{o['pair']}->{o['pair']+1}", o["gt_rot"], o["est_rot"], o["rot_err"], o["gt_t"], o["est_t"],
                    o["t_err"]] for o in out])
    check("relative errors recomputed independently in Python match the C++ values", True)

    # absolute errors per frame (C++ values) and an independent recomputation
    abs_err = {m: [(fnum(r[f"{m}_translation_error"]), fnum(r[f"{m}_rotation_error"])) for r in traj]
               for m in ("pnp", "icp")}
    T_est = {m: [se3([fnum(r[f"{m}_x"]), fnum(r[f"{m}_y"]), fnum(r[f"{m}_z"])],
                     [fnum(r[f"{m}_qx"]), fnum(r[f"{m}_qy"]), fnum(r[f"{m}_qz"]), fnum(r[f"{m}_qw"])])
                 for r in traj] for m in ("pnp", "icp")}
    metric_rows = []
    for m in ("pnp", "icp"):
        for k in range(N):
            te = float(np.linalg.norm(T_gt[k][:3, 3] - T_est[m][k][:3, 3]))
            re = rot_err_deg(T_gt[k][:3, :3], T_est[m][k][:3, :3])
            # alternative (wrong) conventions, for the record
            te_cw = float(np.linalg.norm(inv(T_gt[k])[:3, 3] - inv(T_est[m][k])[:3, 3]))
            metric_rows.append([m, k, abs_err[m][k][0], te, abs_err[m][k][1], re, te_cw])
    write_csv(OUT / "error_metric_recomputation.csv",
              ["method", "frame", "cpp_translation_error_m", "python_translation_error_m",
               "cpp_rotation_error_deg", "python_rotation_error_deg", "translation_of_T_cw_diff_m"], metric_rows)
    met_t = max(abs(r[2] - r[3]) for r in metric_rows)
    met_r = max(abs(r[4] - r[5]) for r in metric_rows)

    # known-answer tests of the rotation metric
    def Rot(ax_, deg):
        ax_ = np.asarray(ax_, float) / np.linalg.norm(ax_)
        K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
        th = math.radians(deg)
        return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    rng = np.random.default_rng(0)
    kat = []
    for deg in (0.0, 0.5, 1.0, 10.0, 45.0, 90.0, 135.0, 179.0):
        A = Rot(rng.normal(size=3), rng.uniform(0, 180))
        B = A @ Rot(rng.normal(size=3), deg)
        Bl = Rot(rng.normal(size=3), deg) @ A
        kat.append((deg, rot_err_deg(A, B), rot_err_deg(A, Bl)))
    kat_max = max(max(abs(a - b), abs(a - c)) for a, b, c in kat)
    # quaternion sign invariance
    q = rng.normal(size=4)
    qsign = rot_err_deg(quat_to_R(*q), quat_to_R(*(-q)))

    # ---- 6. correspondence metrics ----
    by_pair = {i: [r for r in cd if int(r["pair"]) == i] for i in range(P)}

    def flow_ratio(r):
        fu = fnum(r["gt_pred_u_j"]) - fnum(r["u_i"])
        fv = fnum(r["gt_pred_v_j"]) - fnum(r["v_i"])
        ou = fnum(r["u_j"]) - fnum(r["u_i"])
        ov = fnum(r["v_j"]) - fnum(r["v_i"])
        n2 = fu * fu + fv * fv
        return (ou * fu + ov * fv) / n2 if n2 > 9 else None

    def category(r):
        s = min(int(r["sil_dist_i"]), int(r["sil_dist_j"]))
        dj = max(fnum(r["depth_jump_i"]) if r["depth_jump_i"] not in ("", "nan") else 0.0,
                 fnum(r["depth_jump_j"]) if r["depth_jump_j"] not in ("", "nan") else 0.0)
        where = "silhouette (<=1 px)" if s <= 1 else ("near silhouette (2-4 px)" if s <= 4 else "interior (>=5 px)")
        return where, dj > 0.005

    cm_rows = []
    for i in range(P):
        a, b = pm[i], sc[i]
        rs = by_pair[i]
        pnp = [r for r in rs if r["pnp_used"] == "1"]
        inl = [r for r in pnp if r["pnp_inlier"] == "1"]
        icp = [r for r in rs if r["icp_used"] == "1"]
        fr = [x for x in (flow_ratio(r) for r in inl) if x is not None]
        near_sil = sum(min(int(r["sil_dist_i"]), int(r["sil_dist_j"])) <= 4 for r in pnp) / max(1, len(pnp))
        gross_pnp = sum(fnum(r["gt_reproj_err_px"]) > 8 for r in pnp)
        gross_icp = sum(fnum(r["icp_gt_residual_m"]) > 0.02 for r in icp)
        cm_rows.append([f"{i}->{i+1}", int(a["orb_keypoints_i"]), int(a["orb_keypoints_j"]), int(a["raw_matches"]),
                        int(a["filtered_matches"]), int(a["pnp_correspondences"]), int(a["pnp_inliers"]),
                        int(a["pnp_inliers"]) / int(a["pnp_correspondences"]), int(a["icp_correspondences"]),
                        near_sil, gross_pnp, gross_icp, st.median(fr), st.median(fnum(r["gt_reproj_err_px"]) for r in inl)])
    write_csv(OUT / "correspondence_metrics.csv",
              ["pair", "orb_keypoints_i", "orb_keypoints_j", "raw_matches", "filtered_matches", "pnp_correspondences",
               "pnp_inliers", "pnp_inlier_ratio", "icp_correspondences", "pnp_fraction_within_4px_of_silhouette",
               "pnp_gross_mismatches_gt_reproj_gt_8px", "icp_gross_mismatches_gt_residual_gt_2cm",
               "inlier_median_flow_ratio_observed_over_gt", "inlier_median_gt_reprojection_px"], cm_rows)

    # flow bias by category (inliers)
    cat = {}
    for r in cd:
        if r["pnp_inlier"] != "1":
            continue
        q_ = flow_ratio(r)
        if q_ is None:
            continue
        where, edge = category(r)
        cat.setdefault((where, edge), []).append(q_)
    order = ["interior (>=5 px)", "near silhouette (2-4 px)", "silhouette (<=1 px)"]
    fb_rows = []
    for where in order:
        for edge in (False, True):
            v = cat.get((where, edge), [])
            if v:
                fb_rows.append([where, "depth edge (>5 mm)" if edge else "smooth", len(v), st.median(v), st.mean(v)])
    write_csv(OUT / "flow_bias_by_feature_location.csv",
              ["location", "depth_neighbourhood", "n_inliers", "median_flow_ratio", "mean_flow_ratio"], fb_rows)
    all_fr = [x for x in (flow_ratio(r) for r in cd if r["pnp_inlier"] == "1") if x is not None]
    per_pair_fr = np.array([row[12] for row in cm_rows])
    pnp_rot_ratio = np.array([o["est_rot"] / o["gt_rot"] for o in rel["pnp"]])
    icp_rot_ratio = np.array([o["est_rot"] / o["gt_rot"] for o in rel["icp"]])
    corr_fr_pnp = float(np.corrcoef(per_pair_fr, pnp_rot_ratio)[0, 1])
    corr_fr_icp = float(np.corrcoef(per_pair_fr, icp_rot_ratio)[0, 1])

    # ---- 7. PnP reprojection ----
    rp_rows = []
    for i, b in enumerate(sc):
        rp_rows.append([f"{i}->{i+1}", fnum(b["pnp_reproj_inl_mean"]), fnum(b["pnp_reproj_inl_median"]),
                        fnum(b["pnp_reproj_inl_p90"]), fnum(b["pnp_reproj_inl_max"]), fnum(b["pnp_reproj_all_mean"]),
                        fnum(b["pnp_reproj_all_median"]), fnum(b["pnp_reproj_all_p90"]), fnum(b["pnp_reproj_all_max"]),
                        fnum(b["pnp_gt_reproj_inl_mean"]), fnum(b["pnp_gt_reproj_inl_median"])])
    write_csv(OUT / "pnp_reprojection.csv",
              ["pair", "inlier_mean_px", "inlier_median_px", "inlier_p90_px", "inlier_max_px", "all_mean_px",
               "all_median_px", "all_p90_px", "all_max_px", "inlier_mean_at_gt_pose_px", "inlier_median_at_gt_pose_px"],
              rp_rows)

    # ---- 8/9. ICP residuals, SVD vs g2o ----
    ir_rows, sg_rows = [], []
    for i, b in enumerate(sc):
        ir_rows.append([f"{i}->{i+1}", int(b["icp_correspondences"]), fnum(b["icp_res_identity_mean"]),
                        fnum(b["icp_res_identity_median"]), fnum(b["icp_res_svd_mean"]), fnum(b["icp_res_svd_median"]),
                        fnum(b["icp_res_g2o_mean"]), fnum(b["icp_res_g2o_median"]), fnum(b["icp_res_g2o_rms"]),
                        fnum(b["icp_res_gt_mean"]), fnum(b["icp_res_gt_median"]), fnum(b["icp_res_gt_rms"]),
                        int(fnum(b["icp_res_g2o_rms"]) <= fnum(b["icp_res_gt_rms"]) + 1e-12)])
        sg_rows.append([f"{i}->{i+1}", fnum(b["icp_svd_rot_deg"]), fnum(b["icp_g2o_rot_deg"]),
                        fnum(b["icp_svd_rot_err_deg"]), fnum(b["icp_g2o_rot_err_deg"]),
                        fnum(b["icp_svd_trans_err_m"]), fnum(b["icp_g2o_trans_err_m"]),
                        fnum(b["svd_vs_g2o_rot_deg"]), fnum(b["svd_vs_g2o_trans_m"])])
    write_csv(OUT / "icp_residuals.csv",
              ["pair", "icp_correspondences", "before_identity_mean_m", "before_identity_median_m", "after_svd_mean_m",
               "after_svd_median_m", "after_g2o_mean_m", "after_g2o_median_m", "after_g2o_rms_m", "at_gt_motion_mean_m",
               "at_gt_motion_median_m", "at_gt_motion_rms_m", "g2o_rms_le_gt_rms"], ir_rows, fmt="{:.7f}")
    write_csv(OUT / "svd_vs_g2o.csv",
              ["pair", "svd_rotation_deg", "g2o_rotation_deg", "svd_rotation_error_deg", "g2o_rotation_error_deg",
               "svd_translation_error_m", "g2o_translation_error_m", "svd_vs_g2o_rotation_deg",
               "svd_vs_g2o_translation_m"], sg_rows, fmt="{:.7f}")

    # ---- perfect correspondences + solver sensitivity ----
    pc_rows = [[f"{i}->{i+1}", int(b["pnp_correspondences"]), fnum(b["perfect_pnp_rot_err_deg"]),
                fnum(b["perfect_pnp_trans_err_m"]), int(b["icp_correspondences"]),
                fnum(b["perfect_icp_svd_rot_err_deg"]), fnum(b["perfect_icp_svd_trans_err_m"]),
                fnum(b["perfect_icp_rot_err_deg"]), fnum(b["perfect_icp_trans_err_m"])] for i, b in enumerate(sc)]
    write_csv(OUT / "perfect_correspondence_test.csv",
              ["pair", "pnp_points", "perfect_pnp_rotation_error_deg", "perfect_pnp_translation_error_m", "icp_points",
               "perfect_icp_svd_rotation_error_deg", "perfect_icp_svd_translation_error_m",
               "perfect_icp_g2o_rotation_error_deg", "perfect_icp_g2o_translation_error_m"], pc_rows, fmt="{:.3e}")
    ss_cols = ["pnp_rot_deg", "pnp_rot_err_deg", "half_px_pnp_rot_deg", "half_px_pnp_rot_err_deg", "n_pnp_interior",
               "interior_pnp_rot_deg", "interior_pnp_rot_err_deg", "n_pnp_gtconsistent", "gtcons_pnp_rot_deg",
               "gtcons_pnp_rot_err_deg", "icp_g2o_rot_deg", "icp_g2o_rot_err_deg", "n_icp_interior",
               "interior_icp_rot_deg", "interior_icp_rot_err_deg", "n_icp_gtconsistent", "gtcons_icp_rot_deg",
               "gtcons_icp_rot_err_deg"]
    write_csv(OUT / "solver_sensitivity.csv", ["pair"] + ss_cols,
              [[f"{i}->{i+1}"] + [fnum(b[c]) for c in ss_cols] for i, b in enumerate(sc)])

    # ---- 10. depth metrics ----
    dm_rows = []
    for i in range(P):
        rs = by_pair[i]
        n = len(rs)
        no_i = sum(r["depth_raw_i"] == "0" for r in rs)
        no_j = sum(r["depth_raw_j"] == "0" for r in rs)
        both = [r for r in rs if r["icp_used"] == "1"]
        dc = [abs(fnum(r["depth_consistency_m"])) for r in both if r["depth_consistency_m"] not in ("", "nan")]
        edge = sum(category(r)[1] for r in both)
        dm_rows.append([f"{i}->{i+1}", n, no_i / n, no_j / n, len(both), st.median(dc), st.mean(dc),
                        float(np.percentile(dc, 90)), max(dc), edge / max(1, len(both)),
                        sum(x > 0.01 for x in dc) / max(1, len(dc))])
    write_csv(OUT / "depth_metrics.csv",
              ["pair", "filtered_matches", "no_depth_fraction_i", "no_depth_fraction_j", "matches_with_both_depths",
               "abs_depth_consistency_median_m", "abs_depth_consistency_mean_m", "abs_depth_consistency_p90_m",
               "abs_depth_consistency_max_m", "fraction_at_depth_edge", "fraction_depth_inconsistent_gt_1cm"], dm_rows)

    # ---- 13. leave-one-out (replace one pair's estimate by the GT motion) ----
    def accumulate(rels):
        T = T_gt[0].copy()
        out = [T]
        for E in rels:
            T = T @ inv(E)
            out.append(T)
        return out

    def final_err(Ts):
        return (float(np.linalg.norm(Ts[-1][:3, 3] - T_gt[-1][:3, 3])), rot_err_deg(T_gt[-1][:3, :3], Ts[-1][:3, :3]))

    loo_rows = []
    loo = {}
    for m in ("pnp", "icp"):
        rels = [o["E"] for o in rel[m]]
        base = final_err(accumulate(rels))
        check(f"{m} re-accumulation from exported motions reproduces final error",
              abs(base[0] - abs_err[m][-1][0]) < 1e-5 and abs(base[1] - abs_err[m][-1][1]) < 1e-3,
              f"{base} vs {abs_err[m][-1]}")
        loo[m] = []
        for k in range(P):
            mod = list(rels)
            mod[k] = G[k]
            fe = final_err(accumulate(mod))
            loo[m].append((k, base, fe))
            loo_rows.append([m.upper(), f"{k}->{k+1}", base[0], fe[0], base[0] - fe[0], base[1], fe[1], base[1] - fe[1]])
    write_csv(OUT / "leave_one_out.csv",
              ["method", "pair", "original_final_translation_error_m", "modified_final_translation_error_m",
               "translation_error_reduction_m", "original_final_rotation_error_deg", "modified_final_rotation_error_deg",
               "rotation_error_reduction_deg"], loo_rows)
    # all-pairs-replaced-by-GT-scaled check: with every rotation corrected the drift would vanish (sanity)

    # ---------------------------------------------------------- figures ---
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(P)
    labels = [f"{i}" for i in range(P)]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, [r[4] for r in cm_rows], "-o", ms=3, color="0.4", label="filtered matches")
    ax.plot(x, [r[5] for r in cm_rows], "-o", ms=3, color=C["PnP"], label="PnP 3D-2D correspondences")
    ax.plot(x, [r[6] for r in cm_rows], "--", color=C["PnP"], label="PnP RANSAC inliers")
    ax.plot(x, [r[8] for r in cm_rows], "-o", ms=3, color=C["ICP"], label="ICP 3D-3D correspondences")
    ax.set_xlabel("pair i -> i+1 (i)")
    ax.set_ylabel("count")
    ax.set_title("Correspondences per frame pair")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "correspondences_vs_pair")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(x, [r[7] for r in cm_rows], color=C["PnP"], alpha=0.8)
    ax.set_ylim(0.8, 1.0)
    ax.set_xlabel("pair i -> i+1 (i)")
    ax.set_ylabel("inliers / correspondences")
    ax.set_title("PnP RANSAC inlier ratio (8 px threshold)")
    save(fig, "pnp_inlier_ratio")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, m, ncol in ((axs[0], "pnp", 5), (axs[1], "icp", 8)):
        n_ = [r[ncol] for r in cm_rows]
        e_ = [o["rot_err"] for o in rel[m]]
        ax.scatter(n_, e_, color=C[m.upper() if m == "icp" else "PnP"], s=18)
        for k in sorted(range(P), key=lambda k: -e_[k])[:3]:
            ax.annotate(f"{k}->{k+1}", (n_[k], e_[k]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rr = float(np.corrcoef(n_, e_)[0, 1])
        ax.set_title(f"{m.upper()}: relative rotation error vs correspondences (r = {rr:+.2f})", fontsize=9)
        ax.set_xlabel("correspondences")
        ax.set_ylabel("relative rotation error (deg)")
    fig.tight_layout()
    save(fig, "relative_error_vs_correspondence_count")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, [r[2] for r in rp_rows], "-o", ms=3, color=C["PnP"], label="inliers, median, at PnP pose")
    ax.fill_between(x, [r[2] for r in rp_rows], [r[3] for r in rp_rows], color=C["PnP"], alpha=0.15,
                    label="inliers, median..p90")
    ax.plot(x, [r[10] for r in rp_rows], "-s", ms=3, color=C["GT"], label="inliers, median, at GT motion")
    ax.plot(x, [r[6] for r in rp_rows], ":", color="0.4", label="all correspondences, median, at PnP pose")
    ax.set_xlabel("pair i -> i+1 (i)")
    ax.set_ylabel("reprojection error (px)")
    ax.set_title("PnP reprojection error per pair")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "pnp_reprojection_error")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, [r[3] * 1000 for r in ir_rows], "-o", ms=3, color="0.5", label="before (identity), median")
    ax.plot(x, [r[7] * 1000 for r in ir_rows], "-o", ms=3, color=C["ICP"], label="after SVD + g2o, median")
    ax.plot(x, [r[10] * 1000 for r in ir_rows], "-s", ms=3, color=C["GT"], label="at the GT motion, median")
    ax.plot(x, [r[8] * 1000 for r in ir_rows], "--", color=C["ICP"], label="after SVD + g2o, RMS")
    ax.plot(x, [r[11] * 1000 for r in ir_rows], "--", color=C["GT"], label="at the GT motion, RMS")
    ax.set_xlabel("pair i -> i+1 (i)")
    ax.set_ylabel("3D residual |p_i - (R p_j + t)| (mm)")
    ax.set_title("ICP 3D residuals before and after alignment")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "icp_residual_before_after")

    for m in ("pnp", "icp"):
        lab = m.upper() if m == "icp" else "PnP"
        fig, axs = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
        axs[0].bar(x + 1, [o["rot_err"] for o in rel[m]], color=C[lab], alpha=0.55, label="relative error of pair (k-1)->k")
        axs[0].plot(range(N), [e[1] for e in abs_err[m]], "-o", ms=3, color=C[lab], label="accumulated error at frame k")
        axs[0].plot(x + 1, np.cumsum([o["rot_err"] for o in rel[m]]), ":", color="0.4",
                    label="running sum of relative errors (upper bound)")
        axs[0].set_ylabel("rotation error (deg)")
        axs[0].legend(frameon=False, fontsize=8)
        axs[0].set_title(f"{lab}: relative (per pair) vs accumulated error")
        axs[1].bar(x + 1, [o["t_err"] for o in rel[m]], color=C[lab], alpha=0.55, label="relative error of pair (k-1)->k")
        axs[1].plot(range(N), [e[0] for e in abs_err[m]], "-o", ms=3, color=C[lab], label="accumulated error at frame k")
        axs[1].set_ylabel("translation error (m)")
        axs[1].set_xlabel("frame k")
        axs[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save(fig, f"{m}_relative_vs_cumulative")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, [o["gt_rot"] for o in rel["pnp"]], "-", color=C["GT"], label="GT (10 deg)")
    ax.plot(x, [o["est_rot"] for o in rel["pnp"]], "-o", ms=3, color=C["PnP"], label="PnP")
    ax.plot(x, [o["est_rot"] for o in rel["icp"]], "-o", ms=3, color=C["ICP"], label="ICP")
    ax.plot(x, [fnum(b["gtcons_pnp_rot_deg"]) for b in sc], ":", color=C["PnP"],
            label="PnP re-solved on GT-consistent subset (diagnostic)")
    ax.set_xlabel("pair i -> i+1 (i)")
    ax.set_ylabel("rotation magnitude (deg)")
    ax.set_title("Estimated vs true rotation per pair")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "rotation_magnitude_per_pair")

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    names = [f"{r[0].split(' ')[0]}\n{'edge' if 'edge' in r[1] else 'smooth'}" for r in fb_rows]
    axs[0].bar(range(len(fb_rows)), [r[3] for r in fb_rows], color=["0.55" if "smooth" in r[1] else C["ICP"] for r in fb_rows])
    for k, r in enumerate(fb_rows):
        axs[0].text(k, r[3] + 0.005, f"{r[3]:.3f}\nn={r[2]}", ha="center", fontsize=7)
    axs[0].axhline(1.0, color=C["GT"], lw=1)
    axs[0].set_xticks(range(len(fb_rows)))
    axs[0].set_xticklabels(names, fontsize=7)
    axs[0].set_ylim(0.6, 1.1)
    axs[0].set_ylabel("observed / GT image flow (median)")
    axs[0].set_title("PnP inliers: flow ratio by feature location", fontsize=9)
    axs[1].scatter(per_pair_fr, pnp_rot_ratio, color=C["PnP"], s=16, label=f"PnP (r = {corr_fr_pnp:+.2f})")
    axs[1].scatter(per_pair_fr, icp_rot_ratio, color=C["ICP"], s=16, label=f"ICP (r = {corr_fr_icp:+.2f})")
    axs[1].plot([0.85, 1.05], [0.85, 1.05], color="0.6", lw=0.8)
    axs[1].set_xlabel("per-pair median flow ratio (inliers)")
    axs[1].set_ylabel("estimated / GT rotation")
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("Feature lag vs rotation under-estimate", fontsize=9)
    fig.tight_layout()
    save(fig, "flow_bias")

    # worst-pair figures
    def img(i):
        return np.asarray(Image.open(DATASET / f"{i:06d}.png").convert("RGB"))

    def depth(i):
        return np.asarray(Image.open(DATASET / f"{i:06d}_depth.png"))

    def bbox(i, j, margin=30):
        m = (depth(i) > 0) | (depth(j) > 0)
        ys, xs = np.nonzero(m)
        return max(0, xs.min() - margin), min(m.shape[1], xs.max() + margin), max(0, ys.min() - margin), min(m.shape[0], ys.max() + margin)

    worst_pnp = sorted(range(P), key=lambda k: -rel["pnp"][k]["rot_err"])[:N_WORST]
    worst_icp = sorted(range(P), key=lambda k: -rel["icp"][k]["rot_err"])[:N_WORST]
    for k in worst_pnp:
        x0, x1, y0, y1 = bbox(k, k + 1)
        A, B = img(k)[y0:y1, x0:x1], img(k + 1)[y0:y1, x0:x1]
        gap = 20
        W_ = A.shape[1]
        canvas = np.full((A.shape[0], 2 * W_ + gap, 3), 255, np.uint8)
        canvas[:, :W_], canvas[:, W_ + gap:] = A, B
        fig, ax = plt.subplots(figsize=(9, 9 * canvas.shape[0] / canvas.shape[1]))
        ax.imshow(canvas)
        for r in by_pair[k]:
            if r["pnp_used"] != "1":
                continue
            ui, vi = fnum(r["u_i"]) - x0, fnum(r["v_i"]) - y0
            uj, vj = fnum(r["u_j"]) - x0 + W_ + gap, fnum(r["v_j"]) - y0
            good = r["pnp_inlier"] == "1"
            ax.plot([ui, uj], [vi, vj], "-", lw=0.5, color=(0.1, 0.7, 0.2) if good else (0.9, 0.1, 0.1),
                    alpha=0.6 if good else 0.9)
            gu, gv = fnum(r["gt_pred_u_j"]) - x0 + W_ + gap, fnum(r["gt_pred_v_j"]) - y0
            ax.plot([uj, gu], [vj, gv], "-", lw=1.0, color=(0.1, 0.3, 0.95))
        ax.set_axis_off()
        o = rel["pnp"][k]
        ax.set_title(f"PnP pair {k}->{k+1}: {pm[k]['pnp_inliers']}/{pm[k]['pnp_correspondences']} inliers (green), "
                     f"outliers red; blue = matched point -> GT-predicted point\n"
                     f"estimated {o['est_rot']:.2f} deg vs GT {o['gt_rot']:.2f} deg, rotation error {o['rot_err']:.2f} deg",
                     fontsize=8)
        save(fig, f"pnp_pair_{k}_{k+1}")

    spatial = sorted(set(worst_pnp[:3] + worst_icp[:3] + [13, 24, 28]))
    for k in spatial:
        x0, x1, y0, y1 = bbox(k, k + 1)
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4))
        for ax, m in zip(axs, ("pnp", "icp")):
            ax.imshow(img(k)[y0:y1, x0:x1], alpha=0.55)
            rs = [r for r in by_pair[k] if r[f"{m}_used"] == "1"]
            if m == "pnp":
                val = [min(fnum(r["gt_reproj_err_px"]), 10.0) for r in rs]
                lab = "GT reprojection error of the match (px, clipped at 10)"
                mk = ["o" if r["pnp_inlier"] == "1" else "x" for r in rs]
            else:
                val = [min(fnum(r["icp_gt_residual_m"]) * 1000, 20.0) for r in rs]
                lab = "3D residual at the GT motion (mm, clipped at 20)"
                mk = ["o"] * len(rs)
            for marker in ("o", "x"):
                sel = [n for n in range(len(rs)) if mk[n] == marker]
                if not sel:
                    continue
                sca = ax.scatter([fnum(rs[n]["u_i"]) - x0 for n in sel], [fnum(rs[n]["v_i"]) - y0 for n in sel],
                                 c=[val[n] for n in sel], cmap="viridis", vmin=0, vmax=10 if m == "pnp" else 20,
                                 s=22, marker=marker)
            fig.colorbar(sca, ax=ax, fraction=0.04, label=lab)
            o = rel[m][k]
            ax.set_title(f"{m.upper() if m == 'icp' else 'PnP'} {k}->{k+1}: {len(rs)} correspondences, "
                         f"rot. error {o['rot_err']:.2f} deg", fontsize=8)
            ax.set_axis_off()
        fig.tight_layout()
        save(fig, f"spatial_pair_{k}_{k+1}")

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, m in zip(axs, ("pnp", "icp")):
        lab = m.upper() if m == "icp" else "PnP"
        red = [r[1] for r in loo[m]]
        ax.bar(x, [b[1] - f[1] for _, b, f in loo[m]], color=C[lab], alpha=0.8)
        ax.set_title(f"{lab}: final rotation error reduction if pair k is replaced by GT", fontsize=8)
        ax.set_xlabel("pair k -> k+1 (k)")
        ax.set_ylabel("reduction (deg)")
    fig.tight_layout()
    save(fig, "leave_one_out")

    # ------------------------------------------------- summary for REPORT ---
    def worst(m, key, n=N_WORST):
        return sorted(rel[m], key=lambda o: -o[key])[:n]

    lines = []
    w = lines.append
    w("# Diagnostic summary (generated by scripts/run_diagnostics.py)\n")
    w(f"- GT replay: max translation error {max_t:.3e} m, max rotation error {max_r:.3e} deg over {N} frames")
    cmax = {k: max(fnum(r[k]) for r in controls) for k in controls[0] if k != "frame"}
    w(f"- GT replay negative controls, max over frames: no inverse {cmax['right_mul_no_inverse_trans_m']:.4f} m / "
      f"{cmax['right_mul_no_inverse_rot_deg']:.2f} deg; wrong side {cmax['left_mul_inverse_trans_m']:.4f} m / "
      f"{cmax['left_mul_inverse_rot_deg']:.2f} deg; at frame 1: {fnum(controls[1]['right_mul_no_inverse_trans_m']):.4f} m / "
      f"{fnum(controls[1]['right_mul_no_inverse_rot_deg']):.2f} deg and {fnum(controls[1]['left_mul_inverse_trans_m']):.4f} m / "
      f"{fnum(controls[1]['left_mul_inverse_rot_deg']):.2f} deg")
    w(f"- Re-run solvers + accumulation vs trajectory files: max {reacc_max:.2e} m")
    w(f"- Error metric: Python vs C++ absolute errors max diff {met_t:.2e} m / {met_r:.2e} deg; "
      f"known-angle tests max deviation {kat_max:.2e} deg; q vs -q {qsign:.2e} deg")
    for m in ("pnp", "icp"):
        lab = m.upper()
        for key, unit in (("rot_err", "deg"), ("t_err", "m")):
            s = stats([o[key] for o in rel[m]])
            w(f"- {lab} relative {key}: mean {s['mean']:.4f} median {s['median']:.4f} min {s['min']:.4f} "
              f"max {s['max']:.4f} std {s['std']:.4f} {unit}")
        s = stats([o["est_rot"] for o in rel[m]])
        w(f"- {lab} estimated rotation magnitude: mean {s['mean']:.3f} min {s['min']:.3f} max {s['max']:.3f} deg "
          f"(GT {stats([o['gt_rot'] for o in rel[m]])['mean']:.3f})")
        s = stats([o["est_t"] for o in rel[m]])
        w(f"- {lab} estimated translation magnitude: mean {s['mean']:.4f} m (GT {stats([o['gt_t'] for o in rel[m]])['mean']:.4f})")
        w(f"- {lab} worst 5 by rotation: " + ", ".join(f"{o['pair']}->{o['pair']+1} ({o['rot_err']:.2f} deg, {o['t_err']:.4f} m)" for o in worst(m, 'rot_err')))
        w(f"- {lab} worst 5 by translation: " + ", ".join(f"{o['pair']}->{o['pair']+1} ({o['t_err']:.4f} m, {o['rot_err']:.2f} deg)" for o in worst(m, 't_err')))
        lo = sorted(loo[m], key=lambda t: -(t[1][1] - t[2][1]))[:5]
        w(f"- {lab} leave-one-out top 5 (final rot. reduction): " + ", ".join(
            f"{k}->{k+1} ({b[1]-f[1]:+.2f} deg, {b[0]-f[0]:+.4f} m)" for k, b, f in lo))
        w(f"- {lab} final error: {abs_err[m][-1][0]:.4f} m / {abs_err[m][-1][1]:.2f} deg; signed sum of "
          f"(est - GT) rotation magnitudes {sum(o['est_rot'] - o['gt_rot'] for o in rel[m]):+.2f} deg")
    w(f"- Flow ratio (observed/GT image flow) over all PnP inliers: median {st.median(all_fr):.4f}, mean {st.mean(all_fr):.4f}, n={len(all_fr)}")
    for r in fb_rows:
        w(f"  - {r[0]}, {r[1]}: n={r[2]} median {r[3]:.4f} mean {r[4]:.4f}")
    w(f"- Correlation per-pair flow ratio vs estimated/GT rotation: PnP {corr_fr_pnp:+.3f}, ICP {corr_fr_icp:+.3f}")
    for col, name in ((2, "perfect PnP rot err"), (3, "perfect PnP trans err"), (5, "perfect ICP SVD rot err"),
                      (7, "perfect ICP g2o rot err"), (8, "perfect ICP g2o trans err")):
        w(f"- {name}: max {max(r[col] for r in pc_rows):.3e}")
    w(f"- SVD vs g2o: max rotation diff {max(r[7] for r in sg_rows):.2e} deg, max translation diff {max(r[8] for r in sg_rows):.2e} m")
    w(f"- g2o RMS residual <= GT-motion RMS residual in {sum(r[12] for r in ir_rows)}/{P} pairs")
    for c in ss_cols:
        v = [fnum(b[c]) for b in sc]
        w(f"- {c}: mean {np.nanmean(v):.4f} min {np.nanmin(v):.4f} max {np.nanmax(v):.4f}")
    for k in (13, 24, 28):
        rs = by_pair[k]
        icp = [r for r in rs if r["icp_used"] == "1"]
        res = sorted((fnum(r["icp_gt_residual_m"]) for r in icp), reverse=True)
        w(f"- pair {k}->{k+1}: PnP {rel['pnp'][k]['est_rot']:.2f} deg (err {rel['pnp'][k]['rot_err']:.2f}), "
          f"ICP {rel['icp'][k]['est_rot']:.2f} deg (err {rel['icp'][k]['rot_err']:.2f}, {rel['icp'][k]['t_err']:.4f} m); "
          f"ICP pts {len(icp)}, >2 cm GT residual {sum(x > 0.02 for x in res)}, top residuals "
          + ", ".join(f"{x*1000:.0f}" for x in res[:4]) + " mm; "
          f"ICP GT-consistent re-solve {fnum(sc[k]['gtcons_icp_rot_deg']):.2f} deg (err {fnum(sc[k]['gtcons_icp_rot_err_deg']):.2f}); "
          f"accumulated ICP error {abs_err['icp'][k][0]:.4f} -> {abs_err['icp'][k+1][0]:.4f} m, "
          f"{abs_err['icp'][k][1]:.2f} -> {abs_err['icp'][k+1][1]:.2f} deg; "
          f"PnP {abs_err['pnp'][k][0]:.4f} -> {abs_err['pnp'][k+1][0]:.4f} m")
    (OUT / "summary_generated.md").write_text("\n".join(lines) + "\n")

    # worst_pairs.md
    wp = ["# Worst frame pairs\n",
          "Generated by `scripts/run_diagnostics.py`. \"Accumulated error\" is the absolute error of that "
          "method's trajectory (from `slam_trajectory.csv`) at frame i (before the pair) and frame i+1 (after). "
          "\"Gross\" = PnP correspondence whose GT reprojection error exceeds 8 px, or ICP pair whose 3D residual "
          "at the GT motion exceeds 2 cm. Flow ratio = observed image motion of the match / GT-predicted motion "
          "(median over PnP inliers).\n"]
    for m, key, title in (("pnp", "rot_err", "PnP, by relative rotation error"),
                          ("pnp", "t_err", "PnP, by relative translation error"),
                          ("icp", "rot_err", "ICP, by relative rotation error"),
                          ("icp", "t_err", "ICP, by relative translation error")):
        wp.append(f"\n## {title}\n")
        wp.append("| pair | filtered matches | corr. | inliers | gross | flow ratio | est. rot (deg) | GT rot | "
                  "rot. err (deg) | trans. err (m) | acc. trans. err before → after (m) | acc. rot. err before → after (deg) |")
        wp.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for o in worst(m, key):
            k = o["pair"]
            cmr = cm_rows[k]
            corr = cmr[5] if m == "pnp" else cmr[8]
            inl = cmr[6] if m == "pnp" else "n/a"
            gross = cmr[10] if m == "pnp" else cmr[11]
            wp.append(f"| {k}→{k+1} | {cmr[4]} | {corr} | {inl} | {gross} | {cmr[12]:.3f} | {o['est_rot']:.2f} | "
                      f"{o['gt_rot']:.2f} | {o['rot_err']:.2f} | {o['t_err']:.4f} | "
                      f"{abs_err[m][k][0]:.4f} → {abs_err[m][k+1][0]:.4f} | {abs_err[m][k][1]:.2f} → {abs_err[m][k+1][1]:.2f} |")
    (OUT / "worst_pairs.md").write_text("\n".join(wp) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
