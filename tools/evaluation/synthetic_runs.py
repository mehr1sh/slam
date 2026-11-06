"""Synthetic tum_trajectory_eval outputs with known poses, shared by the evaluation tests."""

import math

import numpy as np

import tum_metrics as tm


def Rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def R_to_q(R):  # rotation matrix -> (qx, qy, qz, qw), w >= 0
    w = math.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def write_run(d, gt, pnp, icp, failed=(), ts=None):
    """gt/pnp/icp: lists of (R_wc, p). Writes the evaluator's three CSVs with consistent error columns.
    failed: pair indices whose PnP and ICP errors are NaN (the caller holds the poses)."""
    d.mkdir(parents=True, exist_ok=True)
    n = len(gt)
    ts = np.arange(n) * 0.033 + 1000.0 if ts is None else ts
    with open(d / "frames.csv", "w") as f:
        f.write(",".join(tm.FRAMES_COLS) + "\n")
        for k in range(n):
            q = R_to_q(gt[k][0])
            f.write(f"{k},{ts[k]:.6f},{ts[k] + 0.001:.6f},1.000,rgb/{k}.png,depth/{k}.png," +
                    ",".join(f"{v:.9f}" for v in list(gt[k][1]) + list(q)) + "\n")
    with open(d / "pairs.csv", "w") as f:
        f.write(",".join(tm.PAIRS_COLS) + "\n")
        for k in range(n - 1):
            vals = [k, k, k + 1, 300, 250, 200]
            errs = []
            for est in (pnp, icp):
                if k in failed:
                    errs.append((float("nan"), float("nan")))
                    continue
                Rg = gt[k + 1][0].T @ gt[k][0]
                tg = gt[k + 1][0].T @ (gt[k][1] - gt[k + 1][1])
                Re = est[k + 1][0].T @ est[k][0]
                te = est[k + 1][0].T @ (est[k][1] - est[k + 1][1])
                errs.append((tm.rot_angle_deg(Rg.T @ Re), float(np.linalg.norm(tg - te))))
            vals += [errs[0][0], errs[0][1], 240, 150, errs[1][0], errs[1][1], 1.0, 10.0, 0.08]
            f.write(",".join(str(v) for v in vals) + "\n")
    with open(d / "trajectory.csv", "w") as f:
        f.write(",".join(tm.TRAJ_COLS) + "\n")
        for k in range(n):
            row = [k, f"{ts[k]:.6f}"]
            for T in (gt, pnp, icp):
                row += [f"{v:.9f}" for v in T[k][1]]
            for T in (gt, pnp, icp):
                row += [f"{v:.9f}" for v in R_to_q(T[k][0])]
            for T in (pnp, icp):
                row += [np.linalg.norm(T[k][1] - gt[k][1]), tm.rot_angle_deg(gt[k][0].T @ T[k][0])]
            f.write(",".join(str(v) for v in row) + "\n")
