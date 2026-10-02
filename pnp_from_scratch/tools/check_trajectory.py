#!/usr/bin/env python3
"""Numerical check of the scratch trajectory transform chain (NumPy only).

Compares, for every frame 0..35, the camera of
  GT                  data/synthetic_bunny/groundtruth.txt
  reference PnP, ICP  data/synthetic_bunny/slam_trajectory.csv (committed reference outputs)
  scratch M2          pnp_from_scratch/results/scratch_pnp_trajectory.csv (linear PnP, no RANSAC)
  scratch pipeline    pnp_from_scratch/results/pipeline/trajectory.csv     (RANSAC + linear PnP)
by camera centre, position error, rotation error and LOOK-AT error: the
angle between the camera's optical axis (third column of R_wc, the +Z axis of
the camera in world coordinates) and the direction from the camera centre to
the bunny centre. The bunny centre is estimated independently as the
least-squares intersection of the GT optical axes (every GT camera looks at it).

It also verifies the representation identities of the scratch pipeline:
  - trajectory frame 0 equals GT frame 0
  - quaternion columns equal the rotation-matrix columns
  - T_wc[i+1]^-1 T_wc[i] equals the exported relative pose T_{i+1<-i} (pairs.csv)
  - the camera centre equals -R_cw^T t_cw of the world-to-camera inverse
and prints the explicit 0 -> 1 chain:  T_rel, T_rel^-1, T_wc[1] = T_wc[0] T_rel^-1, C_w[1].

Usage (from the repository root, any Python 3 with NumPy):
    python3 pnp_from_scratch/tools/check_trajectory.py [--csv OUT.csv]
Exit status 1 if an identity check fails.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "synthetic_bunny"
RES = ROOT / "pnp_from_scratch" / "results"


def q2R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def load_scratch(path):
    out = []
    for r in rows(path):
        R = np.array([[float(r[f"r{a}{b}"]) for b in range(3)] for a in range(3)])
        out.append((T(R, [float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                    q2R(*(float(r[c]) for c in ("qx", "qy", "qz", "qw")))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="write the per-frame table to this CSV")
    a = ap.parse_args()
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")
        if not ok:
            fails.append(label)

    gt = [T(q2R(*v[4:8]), v[1:4]) for v in ([float(x) for x in l.split()] for l in open(DS / "groundtruth.txt")
                                             if l.strip() and not l.startswith("#"))]
    ref = rows(DS / "slam_trajectory.csv")
    pose = lambda r, p: T(q2R(*(float(r[f"{p}_q{c}"]) for c in "xyzw")), [float(r[f"{p}_{c}"]) for c in "xyz"])
    traj = {"GT": gt, "ref PnP": [pose(r, "pnp") for r in ref], "ref ICP": [pose(r, "icp") for r in ref]}
    m2_path, m3_path = RES / "scratch_pnp_trajectory.csv", RES / "pipeline" / "trajectory.csv"
    if m2_path.exists():
        traj["scratch M2"] = [m for m, _ in load_scratch(m2_path)]
    if not m3_path.exists():
        sys.exit(f"missing {m3_path}: run pnp_from_scratch/build/scratch_pipeline first")
    m3 = load_scratch(m3_path)
    traj["scratch pipeline"] = [m for m, _ in m3]
    N = len(gt)

    # bunny centre: least-squares intersection of the GT optical axes
    A, b = np.zeros((3, 3)), np.zeros(3)
    for M in gt:
        d = M[:3, 2] / np.linalg.norm(M[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ M[:3, 3]
    c = np.linalg.solve(A, b)
    print(f"bunny centre from the GT optical axes: ({c[0]:+.5f}, {c[1]:+.5f}, {c[2]:+.5f})")

    def lookat(M):
        return math.degrees(math.acos(max(-1.0, min(1.0, M[:3, 2] @ (c - M[:3, 3]) / np.linalg.norm(c - M[:3, 3])))))

    # ---------------------------------------------------------------- identities
    print("\nRepresentation identities of the scratch pipeline trajectory:")
    check("frame 0 equals GT frame 0", np.allclose(m3[0][0], gt[0], atol=1e-6))
    check("quaternion columns equal matrix columns", all(np.allclose(m[:3, :3], q, atol=1e-6) for m, q in m3))
    pr = rows(RES / "pipeline" / "pairs.csv")
    worst_r = worst_t = 0.0
    for i, p in enumerate(pr):
        Rrel = np.array([[float(p[f"r{a}{b}"]) for b in range(3)] for a in range(3)])
        trel = np.array([float(p[c_]) for c_ in ("tx", "ty", "tz")])
        rec = inv(traj["scratch pipeline"][i + 1]) @ traj["scratch pipeline"][i]
        worst_r = max(worst_r, np.abs(rec[:3, :3] - Rrel).max())  # matrix entries (acos near 1 has a ~1e-3 deg floor)
        worst_t = max(worst_t, np.linalg.norm(rec[:3, 3] - trel))
    check("T_wc[i+1]^-1 T_wc[i] == exported T_{i+1<-i} (all 35 pairs)", worst_r < 1e-7 and worst_t < 1e-7,
          f"(max |R entry diff| {worst_r:.1e}, |t diff| {worst_t:.1e} m; CSVs store 9 decimals)")
    cdiff = max(np.linalg.norm((-inv(M)[:3, :3].T @ inv(M)[:3, 3]) - M[:3, 3]) for M in traj["scratch pipeline"])
    check("camera centre == -R_cw^T t_cw of the world-to-camera inverse", cdiff < 1e-8, f"(max {cdiff:.1e} m)")

    # ---------------------------------------------------------------- 0 -> 1 chain
    p0 = pr[0]
    Rrel = np.array([[float(p0[f"r{a}{b}"]) for b in range(3)] for a in range(3)])
    trel = np.array([float(p0[c_]) for c_ in ("tx", "ty", "tz")])
    Trel = T(Rrel, trel)
    Tw1 = gt[0] @ inv(Trel)
    G = inv(gt[1]) @ gt[0]
    np.set_printoptions(precision=6, suppress=True)
    print("\nExplicit chain for pair 0 -> 1 (scratch pipeline):")
    print("T_rel = T_{1<-0} (x_1 = R x_0 + t), from pairs.csv:\n", Trel)
    print("ground truth T_{1<-0} = T_gt[1]^-1 T_gt[0]:\n", G)
    print("T_rel^-1 = T_{0<-1}:\n", inv(Trel))
    print("T_wc[1] = T_wc[0] T_rel^-1:\n", Tw1)
    print(f"C_w[1] = translation of T_wc[1] = {Tw1[:3, 3]}   (trajectory.csv frame 1: {m3[1][0][:3, 3]})")
    print(f"C_w[1] = C_w[0] + R_wc[0] (-R^T t)  = {gt[0][:3, 3] + gt[0][:3, :3] @ (-Rrel.T @ trel)}")
    print(f"GT C_w[1] = {gt[1][:3, 3]};  position error {np.linalg.norm(Tw1[:3, 3] - gt[1][:3, 3]):.4f} m, "
          f"rotation error {angle(gt[1][:3, :3].T @ Tw1[:3, :3]):.3f} deg, look-at error {lookat(Tw1):.3f} deg "
          f"(GT {lookat(gt[1]):.3f} deg)")
    check("independent recomputation of T_wc[1] equals trajectory.csv", np.allclose(Tw1, m3[1][0], atol=1e-6))

    # ---------------------------------------------------------------- per frame
    names = list(traj)
    table = []
    print("\nPer frame: position error m / rotation error deg / look-at error deg")
    print("frame " + "".join(f"| {n:^24}" for n in names))
    for k in range(N):
        line = f"{k:5d} "
        rec = {"frame": k}
        for n in names:
            M = traj[n][k]
            pe = np.linalg.norm(M[:3, 3] - gt[k][:3, 3])
            re = angle(gt[k][:3, :3].T @ M[:3, :3])
            la = lookat(M)
            line += f"| {pe:6.3f} {re:7.2f} {la:7.2f}  "
            rec.update({f"{n}_cx": M[0, 3], f"{n}_cy": M[1, 3], f"{n}_cz": M[2, 3], f"{n}_pos_err_m": pe,
                        f"{n}_rot_err_deg": re, f"{n}_lookat_deg": la})
        table.append(rec)
        print(line)

    # ---------------------------------------------------------------- selected pairs
    print("\nSelected pairs: estimated relative rotation / relative rotation error / relative translation error")
    print("pair    GT rot " + "".join(f"| {n:^26}" for n in names[1:]))
    for i in (0, 1, 9, 10, 25, 34):
        G = inv(gt[i + 1]) @ gt[i]
        line = f"{i:2d}->{i + 1:<3d} {angle(G[:3, :3]):6.2f} "
        for n in names[1:]:
            rel = inv(traj[n][i + 1]) @ traj[n][i]
            line += f"| {angle(rel[:3, :3]):7.2f} {angle(G[:3, :3].T @ rel[:3, :3]):7.2f} {np.linalg.norm(G[:3, 3] - rel[:3, 3]):7.4f}  "
        print(line)

    print("\nSummary (frame 35 position / rotation error; max look-at error over frames):")
    for n in names:
        print(f"  {n:18s} {table[-1][f'{n}_pos_err_m']:.4f} m  {table[-1][f'{n}_rot_err_deg']:7.2f} deg   "
              f"look-at max {max(r[f'{n}_lookat_deg'] for r in table):7.2f} deg")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"\nwrote {a.csv}")
    if fails:
        print(f"\n{len(fails)} identity check(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
