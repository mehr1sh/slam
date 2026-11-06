#!/usr/bin/env python3
"""Self-checking tests of tum_metrics.py on synthetic trajectories with known errors (NumPy only).

Usage (repository root): python3 tools/evaluation/test_tum_metrics.py      exit 1 on failure
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tum_metrics as tm  # noqa: E402
from synthetic_runs import R_to_q, Rz, write_run  # noqa: E402

failures = 0


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


print("=== tum_metrics ===")
tmp = Path(tempfile.mkdtemp(prefix="tum_metrics_test_"))
gt = [(Rz(10 * k), np.array([0.1 * k, 0.05 * k * k, 1.0])) for k in range(8)]

# --- perfect estimates ---
write_run(tmp / "perfect", gt, gt, gt)
r = tm.load_run(tmp / "perfect")
m = tm.compute_metrics(r)
check(tm.check_consistency(r) == [], "perfect: consistent with the error columns")
check(m["pnp"]["ate_raw_rmse_m"] < 1e-8 and m["pnp"]["ate_aligned_rmse_m"] < 1e-8 and
      m["pnp"]["rpe_1frame"]["trans_rmse_m"] < 1e-7 and m["pnp"]["rpe_1frame"]["rot_rmse_deg"] < 1e-4,
      "perfect: ATE raw / aligned and RPE are 0")
check(m["pnp"]["success_rate"] == 1.0 and m["pnp"]["failed_pairs"] == 0 and m["attempted_pairs"] == 7,
      "perfect: 7 attempted, 7 successful pairs")

# --- constant position offset after frame 0: raw ATE = offset, aligned ATE smaller ---
off = np.array([0.0, 0.1, 0.0])
est = [gt[0]] + [(R, p + off) for R, p in gt[1:]]
write_run(tmp / "offset", gt, est, gt)
m = tm.compute_metrics(tm.load_run(tmp / "offset"))
check(abs(m["pnp"]["ate_raw_rmse_m"] - 0.1 * math.sqrt(7 / 8)) < 1e-6,
      f"offset 0.1 m on frames 1..7: raw ATE {m['pnp']['ate_raw_rmse_m']:.4f} m = 0.1 sqrt(7/8)")
check(m["pnp"]["ate_aligned_rmse_m"] < m["pnp"]["ate_raw_rmse_m"], "the global alignment can only lower the ATE")
check(abs(m["pnp"]["final_position_error_m"] - 0.1) < 1e-6, "final position error 0.1 m (raw, not aligned)")

# --- the whole estimate rigidly moved (except frame 0, the anchor, moved too in a second run) ---
Rg, tg = Rz(5), np.array([0.2, -0.1, 0.05])
moved = [(Rg @ R, Rg @ p + tg) for R, p in gt]
write_run(tmp / "moved", gt, moved, gt)
r = tm.load_run(tmp / "moved")
m = tm.compute_metrics(r)
check(m["pnp"]["ate_raw_rmse_m"] > 0.1 and m["pnp"]["ate_aligned_rmse_m"] < 1e-6,
      f"rigidly moved trajectory: raw ATE {m['pnp']['ate_raw_rmse_m']:.3f} m, aligned ATE "
      f"{m['pnp']['ate_aligned_rmse_m']:.1e} m (one global SE(3), no scale)")
check(m["pnp"]["rpe_1frame"]["trans_rmse_m"] < 1e-6 and m["pnp"]["rpe_1frame"]["rot_rmse_deg"] < 1e-4,
      "a global rigid motion leaves the RPE at 0 (RPE is invariant to it)")
check(any("anchored" in p for p in tm.check_consistency(r)), "a trajectory not starting at ground truth is flagged")

# --- known RPE: every estimated step rotated 1 deg more about the camera z axis ---
est = [gt[0]]
for k in range(1, 8):
    Rrel = gt[k - 1][0].T @ gt[k][0]   # true T_{k-1 <- k}: R part
    trel = gt[k - 1][0].T @ (gt[k][1] - gt[k - 1][1])
    Rp, pp = est[-1]
    est.append((Rp @ Rrel @ Rz(1.0), Rp @ trel + pp))
write_run(tmp / "rpe", gt, est, gt)
m = tm.compute_metrics(tm.load_run(tmp / "rpe"))
check(abs(m["pnp"]["rpe_1frame"]["rot_mean_deg"] - 1.0) < 1e-3 and abs(m["pnp"]["rpe_1frame"]["rot_rmse_deg"] - 1.0) < 1e-3,
      f"known 1 deg per-step rotation error: RPE rotation {m['pnp']['rpe_1frame']['rot_mean_deg']:.4f} deg")
check(abs(m["pnp"]["pair_rot_err_mean_deg"] - 1.0) < 1e-3, "matches the per-pair error column (1 deg)")

# --- failed pair: counted, pose held, RPE window excluded ---
est = [g for g in gt]
est[4] = est[3]  # pair 3->4 failed: frame 4 holds frame 3's pose
est[5:] = [(est[3][0] @ gt[3][0].T @ R, est[3][0] @ gt[3][0].T @ (p - gt[3][1]) + est[3][1]) for R, p in gt[5:]]
est[5:] = [(R, p) for R, p in est[5:]]
write_run(tmp / "failed", gt, est, gt, failed=(3,))
r = tm.load_run(tmp / "failed")
m = tm.compute_metrics(r)
check(m["pnp"]["successful_pairs"] == 6 and m["pnp"]["failed_pairs"] == 1 and abs(m["pnp"]["success_rate"] - 6 / 7) < 1e-12,
      "failed pair counted: 6/7 successful (the denominator keeps the failure)")
check(r["pnp_held"].tolist() == [False, False, False, False, True, False, False, False] and m["pnp"]["held_poses"] == 1,
      "frame 4 flagged as a held pose (not an estimate)")
check(m["pnp"]["rpe_1frame"]["windows_used"] == 6 and m["pnp"]["rpe_1frame"]["windows_excluded_failed"] == 1,
      "the RPE window over the failed pair is excluded and counted")
check(any("held" in s for s in m["notes"]), "the note explains the held pose in the ATE")
check(tm.to_json_safe({"a": float("nan")}) == {"a": None}, "NaN metrics become null in JSON, never 0")

# --- schema validation ---
def expect_error(name, mutate, words):
    d = tmp / name
    shutil.copytree(tmp / "perfect", d)
    mutate(d)
    try:
        tm.load_run(d)
        check(False, f"{name}: rejected")
    except tm.SchemaError as e:
        check(any(w in str(e) for w in words), f"{name}: rejected ({e})")

def drop_col(d):
    lines = (d / "pairs.csv").read_text().splitlines()
    (d / "pairs.csv").write_text("\n".join(",".join(l.split(",")[:-1]) for l in lines) + "\n")

def swap_ts(d):
    lines = (d / "trajectory.csv").read_text().splitlines()
    a, b = lines[2].split(","), lines[3].split(",")
    a[1], b[1] = b[1], a[1]
    lines[2], lines[3] = ",".join(a), ",".join(b)
    (d / "trajectory.csv").write_text("\n".join(lines) + "\n")

def inf_pose(d):
    lines = (d / "trajectory.csv").read_text().splitlines()
    a = lines[3].split(",")
    a[5] = "inf"
    lines[3] = ",".join(a)
    (d / "trajectory.csv").write_text("\n".join(lines) + "\n")

def half_nan(d):
    lines = (d / "pairs.csv").read_text().splitlines()
    a = lines[2].split(",")
    a[6] = "nan"
    lines[2] = ",".join(a)
    (d / "pairs.csv").write_text("\n".join(lines) + "\n")

expect_error("missing_column", drop_col, ["missing columns"])
expect_error("unordered_timestamps", swap_ts, ["increasing", "differ"])
expect_error("non_finite_pose", inf_pose, ["non-finite"])
expect_error("inconsistent_failure", half_nan, ["disagree"])
expect_error("missing_file", lambda d: (d / "frames.csv").unlink(), ["missing file"])

# --- convention: an estimate written as T_cw instead of T_wc is caught ---
inv = [gt[0]] + [(R.T, -R.T @ p) for R, p in gt[1:]]
write_run(tmp / "conv", gt, gt, gt)
traj = (tmp / "conv" / "trajectory.csv").read_text().splitlines()
hdr = traj[0].split(",")
for k in range(1, 8):
    row = traj[k + 1].split(",")
    R, p = inv[k]
    for a, v in zip("xyz", p):
        row[hdr.index(f"pnp_{a}")] = f"{v:.9f}"
    for a, v in zip("xyzw", R_to_q(R)):
        row[hdr.index(f"pnp_q{a}")] = f"{v:.9f}"
    traj[k + 1] = ",".join(row)
(tmp / "conv" / "trajectory.csv").write_text("\n".join(traj) + "\n")
check(len(tm.check_consistency(tm.load_run(tmp / "conv"))) > 0, "poses written in the wrong convention (T_cw) are detected")

shutil.rmtree(tmp)
print(f"\ntest_tum_metrics: {'FAIL' if failures else 'PASS'} ({failures} failures)")
sys.exit(1 if failures else 0)
