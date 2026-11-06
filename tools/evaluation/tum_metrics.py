"""Loading, validation and metrics for tum_trajectory_eval outputs (NumPy + standard library).

Input: one output directory of build/tum_trajectory_eval with
  frames.csv      frame, t_rgb, t_depth, dt_ms, rgb, depth, gt_x..gt_z, gt_qx..gt_qw
  pairs.csv       pair, frame_i, frame_j, matches, pnp_correspondences, pnp_inliers, pnp_rot_err_deg,
                  pnp_trans_err_m, icp_correspondences, icp_inliers, icp_rot_err_deg, icp_trans_err_m,
                  gt_reproj_median_px, gt_motion_deg, gt_motion_m
  trajectory.csv  frame, timestamp, {gt,pnp,icp}_{x,y,z}, {gt,pnp,icp}_q{x,y,z,w}, {pnp,icp}_{translation,rotation}_error
  summary.txt
Pose convention (from the evaluator and include/dataset/tum.hpp): every row stores T_wc (camera ->
world, X_world = R X_cam + t): the position is the camera centre, the quaternion (qx, qy, qz, qw)
is R_wc. Frame 0 of PnP and ICP is set to the ground-truth pose (anchor).

Failures: a pair whose estimator failed has NaN errors in pairs.csv; the evaluator then HOLDS the
previous pose (T_wc[j] = T_wc[i]). Such poses are not estimates. They are kept in the trajectory as
the evaluator wrote them (no interpolation, no removal) but flagged `held`, and every metric that
needs a valid relative estimate excludes windows containing a failed pair while still counting
the failure in the success rate.

Metrics (per method):
  ate_raw_rmse_m       sqrt(mean_k |p_est,k - p_gt,k|^2) over all frames (frame 0 included, error 0)
  ate_aligned_rmse_m   the same after ONE global rigid SE(3) alignment of the estimated positions
                       onto ground truth (least squares, Kabsch/Umeyama without scale)
  final_*              position / orientation error at the last frame (raw, no alignment)
  rpe_*                relative pose error over an interval of `delta` frames (Sturm et al. 2012):
                       E_k = (Q_k^-1 Q_{k+d})^-1 (P_k^-1 P_{k+d}), Q = ground-truth T_wc, P = estimate;
                       translation |t(E_k)| (m), rotation angle(R(E_k)) (deg); RMSE and mean over the
                       windows without a failed pair (the number used and excluded is reported)
  pairs_*              attempted (N - 1), successful (finite error), failed, success rate
  matches / inliers    from pairs.csv (mean, min), successful pairs only for inliers
"""

import csv
import math
from pathlib import Path

import numpy as np

FRAMES_COLS = ["frame", "t_rgb", "t_depth", "dt_ms", "rgb", "depth", "gt_x", "gt_y", "gt_z", "gt_qx", "gt_qy",
               "gt_qz", "gt_qw"]
PAIRS_COLS = ["pair", "frame_i", "frame_j", "matches", "pnp_correspondences", "pnp_inliers", "pnp_rot_err_deg",
              "pnp_trans_err_m", "icp_correspondences", "icp_inliers", "icp_rot_err_deg", "icp_trans_err_m",
              "gt_reproj_median_px", "gt_motion_deg", "gt_motion_m"]
TRAJ_COLS = ["frame", "timestamp"] + [f"{m}_{a}" for m in ("gt", "pnp", "icp") for a in "xyz"] + \
            [f"{m}_q{a}" for m in ("gt", "pnp", "icp") for a in "xyzw"] + \
            ["pnp_translation_error", "pnp_rotation_error", "icp_translation_error", "icp_rotation_error"]
METHODS = ("pnp", "icp")


class SchemaError(ValueError):
    """The input directory is not a supported tum_trajectory_eval output."""


def _read(path, required):
    if not path.exists():
        raise SchemaError(f"missing file {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SchemaError(f"{path}: no data rows")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise SchemaError(f"{path}: unsupported schema, missing columns {missing}")
    return rows


def _num(rows, col, path, allow_nan=False):
    out = []
    for k, r in enumerate(rows):
        v = r[col].strip()
        try:
            x = float(v)
        except ValueError:
            raise SchemaError(f"{path}: row {k + 1}, column {col}: not a number ({v!r})")
        if not allow_nan and not math.isfinite(x):
            raise SchemaError(f"{path}: row {k + 1}, column {col}: non-finite value {v!r}")
        out.append(x)
    return np.array(out)


def quat_to_R(q):
    """(qx, qy, qz, qw) -> rotation matrix (normalised)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def load_run(run_dir):
    """Reads and validates one run. Returns a dict of arrays; raises SchemaError on any problem."""
    d = Path(run_dir)
    fr = _read(d / "frames.csv", FRAMES_COLS)
    pr = _read(d / "pairs.csv", PAIRS_COLS)
    tr = _read(d / "trajectory.csv", TRAJ_COLS)
    n = len(tr)
    if len(fr) != n:
        raise SchemaError(f"frames.csv has {len(fr)} rows, trajectory.csv {n}")
    if len(pr) != n - 1:
        raise SchemaError(f"pairs.csv has {len(pr)} rows, expected {n - 1} for {n} frames")
    run = {"dir": d, "n": n}
    run["frame"] = _num(tr, "frame", d / "trajectory.csv").astype(int)
    run["t"] = _num(tr, "timestamp", d / "trajectory.csv")
    run["t_rgb"] = _num(fr, "t_rgb", d / "frames.csv")
    run["t_depth"] = _num(fr, "t_depth", d / "frames.csv")
    run["dt_ms"] = _num(fr, "dt_ms", d / "frames.csv")
    run["rgb"] = [r["rgb"] for r in fr]
    run["depth"] = [r["depth"] for r in fr]
    if not np.array_equal(run["frame"], np.arange(n)):
        raise SchemaError("trajectory.csv frames are not 0..N-1 in order")
    if np.any(np.diff(run["t"]) <= 0):
        raise SchemaError("trajectory.csv timestamps are not strictly increasing")
    if np.any(np.abs(run["t"] - run["t_rgb"]) > 5e-6):  # both written with 6 decimals
        raise SchemaError("trajectory.csv timestamps differ from frames.csv t_rgb (poses not aligned by timestamp)")
    for m in ("gt",) + METHODS:
        run[f"{m}_p"] = np.c_[[_num(tr, f"{m}_{a}", d / "trajectory.csv") for a in "xyz"]].T
        q = np.c_[[_num(tr, f"{m}_q{a}", d / "trajectory.csv") for a in "xyzw"]].T
        if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 1e-3):
            raise SchemaError(f"trajectory.csv: {m} quaternions are not unit")
        run[f"{m}_R"] = np.array([quat_to_R(x) for x in q])
    for m in METHODS:
        run[f"{m}_file_terr"] = _num(tr, f"{m}_translation_error", d / "trajectory.csv")
        run[f"{m}_file_rerr"] = _num(tr, f"{m}_rotation_error", d / "trajectory.csv")
    for c in ("pair", "frame_i", "frame_j", "matches", "pnp_correspondences", "pnp_inliers", "icp_correspondences",
              "icp_inliers"):
        run[c] = _num(pr, c, d / "pairs.csv").astype(int)
    for c in ("pnp_rot_err_deg", "pnp_trans_err_m", "icp_rot_err_deg", "icp_trans_err_m", "gt_reproj_median_px"):
        run[c] = _num(pr, c, d / "pairs.csv", allow_nan=True)
    for c in ("gt_motion_deg", "gt_motion_m"):
        run[c] = _num(pr, c, d / "pairs.csv")
    if not (np.array_equal(run["frame_i"], np.arange(n - 1)) and np.array_equal(run["frame_j"], np.arange(1, n))):
        raise SchemaError("pairs.csv is not the consecutive pairs 0->1, 1->2, ...")
    for m in METHODS:  # a pair fails consistently in both error columns
        a, b = np.isfinite(run[f"{m}_rot_err_deg"]), np.isfinite(run[f"{m}_trans_err_m"])
        if not np.array_equal(a, b):
            raise SchemaError(f"pairs.csv: {m} rotation and translation errors disagree on which pairs failed")
        run[f"{m}_ok"] = a
        run[f"{m}_held"] = np.r_[False, ~a]  # frame j of a failed pair holds the previous pose
    # optional (evaluator with the PnP validity check): solver status separate from acceptance
    if "pnp_solver_returned" in pr[0]:
        run["pnp_solver_returned"] = _num(pr, "pnp_solver_returned", d / "pairs.csv").astype(bool)
        run["pnp_accepted"] = _num(pr, "pnp_accepted", d / "pairs.csv").astype(bool)
        if not np.array_equal(run["pnp_accepted"], run["pnp_ok"]):
            raise SchemaError("pairs.csv: pnp_accepted disagrees with the finite PnP error columns")
        run["pnp_returned_rot_err_deg"] = _num(pr, "pnp_returned_rot_err_deg", d / "pairs.csv", allow_nan=True)
        run["pnp_returned_trans_err_m"] = _num(pr, "pnp_returned_trans_err_m", d / "pairs.csv", allow_nan=True)
    run["summary"] = (d / "summary.txt").read_text() if (d / "summary.txt").exists() else ""
    return run


def check_consistency(run, tol_m=1e-5, tol_deg=1e-3):
    """Cross-checks of the pose convention against the evaluator's own error columns.
    Returns a list of problems (empty = consistent)."""
    problems = []
    for m in METHODS:
        te = np.linalg.norm(run[f"{m}_p"] - run["gt_p"], axis=1)
        re = np.array([rot_angle_deg(Rg.T @ Re) for Rg, Re in zip(run["gt_R"], run[f"{m}_R"])])
        if np.max(np.abs(te - run[f"{m}_file_terr"])) > tol_m:
            problems.append(f"{m}: position errors recomputed from the poses differ from the file (max "
                            f"{np.max(np.abs(te - run[f'{m}_file_terr'])):.2e} m)")
        if np.max(np.abs(re - run[f"{m}_file_rerr"])) > 1e-2:
            problems.append(f"{m}: rotation errors recomputed from the poses differ from the file")
        if np.linalg.norm(run[f"{m}_p"][0] - run["gt_p"][0]) > tol_m:
            problems.append(f"{m}: frame 0 is not anchored to the ground truth")
        # per-pair errors: relative pose T_{j<-i} = T_wc[j]^-1 T_wc[i] from the trajectory vs pairs.csv
        for k in np.nonzero(run[f"{m}_ok"])[0]:
            Rg = run["gt_R"][k + 1].T @ run["gt_R"][k]
            tg = run["gt_R"][k + 1].T @ (run["gt_p"][k] - run["gt_p"][k + 1])
            Re = run[f"{m}_R"][k + 1].T @ run[f"{m}_R"][k]
            te_ = run[f"{m}_R"][k + 1].T @ (run[f"{m}_p"][k] - run[f"{m}_p"][k + 1])
            if (abs(rot_angle_deg(Rg.T @ Re) - run[f"{m}_rot_err_deg"][k]) > 5e-2 or
                    abs(np.linalg.norm(tg - te_) - run[f"{m}_trans_err_m"][k]) > 1e-4):
                problems.append(f"{m}: pair {k}->{k + 1} relative pose from the trajectory differs from pairs.csv")
                break
    return problems


def umeyama_rigid(src, dst):
    """R, t minimising sum |dst - (R src + t)|^2 (no scale)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - cd).T @ (src - cs))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, cd - R @ cs


def rpe(run, m, delta):
    """Relative pose errors over `delta` frames; windows containing a failed pair are excluded."""
    t_err, r_err, excluded = [], [], 0
    for k in range(run["n"] - delta):
        if not np.all(run[f"{m}_ok"][k:k + delta]):
            excluded += 1
            continue
        Qk, Qd = (run["gt_R"][k], run["gt_p"][k]), (run["gt_R"][k + delta], run["gt_p"][k + delta])
        Pk, Pd = (run[f"{m}_R"][k], run[f"{m}_p"][k]), (run[f"{m}_R"][k + delta], run[f"{m}_p"][k + delta])
        # A^-1 B for T = (R, p): (R_a^T R_b, R_a^T (p_b - p_a))
        Rq, tq = Qk[0].T @ Qd[0], Qk[0].T @ (Qd[1] - Qk[1])
        Rp, tp = Pk[0].T @ Pd[0], Pk[0].T @ (Pd[1] - Pk[1])
        RE, tE = Rq.T @ Rp, Rq.T @ (tp - tq)
        t_err.append(np.linalg.norm(tE))
        r_err.append(rot_angle_deg(RE))
    t_err, r_err = np.array(t_err), np.array(r_err)
    nan = float("nan")
    return {"delta_frames": delta, "windows_used": int(len(t_err)), "windows_excluded_failed": excluded,
            "trans_rmse_m": float(np.sqrt(np.mean(t_err ** 2))) if len(t_err) else nan,
            "trans_mean_m": float(np.mean(t_err)) if len(t_err) else nan,
            "rot_rmse_deg": float(np.sqrt(np.mean(r_err ** 2))) if len(r_err) else nan,
            "rot_mean_deg": float(np.mean(r_err)) if len(r_err) else nan}


def compute_metrics(run, rpe_seconds=1.0):
    """All metrics as a JSON-ready dict. NaN (-> null) where a metric is undefined, with a reason."""
    n = run["n"]
    span = float(run["t"][-1] - run["t"][0])
    # RPE interval of ~rpe_seconds: the frame offset whose median time gap is closest to it
    dt_med = float(np.median(np.diff(run["t"])))
    d_sec = max(1, int(round(rpe_seconds / dt_med))) if dt_med > 0 else 1
    out = {"frames": n, "attempted_pairs": n - 1, "time_span_s": span,
           "frame0_anchored_to_ground_truth": True,
           "notes": []}
    for m in run.get("methods", METHODS):  # other loaders (dashboard_app/methods.py) may supply other methods
        ok = run[f"{m}_ok"]
        e = np.linalg.norm(run[f"{m}_p"] - run["gt_p"], axis=1)
        R, t = umeyama_rigid(run[f"{m}_p"], run["gt_p"])
        ea = np.linalg.norm(run["gt_p"] - (run[f"{m}_p"] @ R.T + t), axis=1)
        mm = {
            "successful_pairs": int(ok.sum()), "failed_pairs": int((~ok).sum()),
            "success_rate": float(ok.mean()) if n > 1 else float("nan"),
            "held_poses": int(run[f"{m}_held"].sum()),
            "ate_raw_rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "ate_aligned_rmse_m": float(np.sqrt(np.mean(ea ** 2))),
            "alignment_rotation_deg": rot_angle_deg(R), "alignment_translation_m": float(np.linalg.norm(t)),
            "final_position_error_m": float(e[-1]),
            "final_orientation_error_deg": rot_angle_deg(run["gt_R"][-1].T @ run[f"{m}_R"][-1]),
            "pair_rot_err_mean_deg": float(np.nanmean(run[f"{m}_rot_err_deg"])) if ok.any() else float("nan"),
            "pair_trans_err_mean_m": float(np.nanmean(run[f"{m}_trans_err_m"])) if ok.any() else float("nan"),
            "rpe_1frame": rpe(run, m, 1),
            f"rpe_{rpe_seconds:g}s": rpe(run, m, d_sec) if d_sec < n else None,
            "correspondences_mean": float(np.mean(run[f"{m}_correspondences"])),
            "inliers_mean_successful": float(np.mean(run[f"{m}_inliers"][ok])) if ok.any() else float("nan"),
            # solver status vs acceptance: only the evaluator with the PnP validity check records both
            "solver_returned_pairs": (int(run[f"{m}_solver_returned"].sum()) if f"{m}_solver_returned" in run
                                      else None),
            "rejected_pairs": (int((run[f"{m}_solver_returned"] & ~run[f"{m}_accepted"]).sum())
                               if f"{m}_solver_returned" in run else None),
            "inliers_min_successful": int(np.min(run[f"{m}_inliers"][ok])) if ok.any() else None,
        }
        if d_sec >= n:
            out["notes"].append(f"{m}: rpe_{rpe_seconds:g}s undefined: the run spans fewer than {d_sec} frames")
        if mm["held_poses"]:
            out["notes"].append(f"{m}: {mm['held_poses']} pose(s) held after a failed pair are included in the ATE "
                                "as the evaluator wrote them (not estimates)")
        out[m] = mm
    out["matches_mean"] = float(np.mean(run["matches"]))
    out["matches_min"] = int(np.min(run["matches"]))
    out["rgb_depth_dt_ms_max_abs"] = float(np.max(np.abs(run["dt_ms"])))
    gr = run.get("gt_reproj_median_px", np.full(max(n - 1, 1), np.nan))
    out["gt_reproj_median_px_median"] = float(np.nanmedian(gr)) if np.isfinite(gr).any() else float("nan")
    out["rpe_interval_frames_for_seconds"] = d_sec
    return out


def to_json_safe(x):
    """NaN/inf -> None (JSON null)."""
    if isinstance(x, dict):
        return {k: to_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def ground_truth_checks(run, dataset_dir, max_dt_s, max_gt_gap_s):
    """Re-checks association and ground truth against the dataset files (read only).
    Returns a list of problems (empty = all within limits)."""
    d = Path(dataset_dir)
    problems = []
    if np.max(np.abs(run["dt_ms"])) > max_dt_s * 1000 + 1e-3:
        problems.append(f"RGB-depth offset {np.max(np.abs(run['dt_ms'])):.3f} ms exceeds {max_dt_s * 1000} ms")
    gt = []
    with open(d / "groundtruth.txt") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                gt.append([float(v) for v in line.split()])
    gt = np.array(sorted(gt, key=lambda r: r[0]))
    for k, t in enumerate(run["t"]):
        if not (d / run["rgb"][k]).exists() or not (d / run["depth"][k]).exists():
            problems.append(f"frame {k}: image file missing")
        j = np.searchsorted(gt[:, 0], t)
        if j == 0 or j >= len(gt):
            if not (j < len(gt) and gt[j, 0] == t):
                problems.append(f"frame {k}: timestamp outside the ground-truth span")
            continue
        gap = gt[j, 0] - gt[j - 1, 0]
        if gt[j, 0] != t and gap > max_gt_gap_s + 1e-9:
            problems.append(f"frame {k}: ground-truth samples {gap:.3f} s apart > {max_gt_gap_s} s")
        s = (t - gt[j - 1, 0]) / gap if gt[j, 0] != t else 1.0
        p = (1 - s) * gt[j - 1, 1:4] + s * gt[j, 1:4]
        if np.linalg.norm(p - run["gt_p"][k]) > 2e-6:
            problems.append(f"frame {k}: interpolated ground-truth position differs from the evaluator's "
                            f"({np.linalg.norm(p - run['gt_p'][k]):.2e} m)")
    return problems
