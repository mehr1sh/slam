#!/usr/bin/env python3
"""Diagnosis of the refined scratch PnP's systematic under-rotation (NumPy only).

For every pair i -> i+1 it takes the RANSAC inliers of the default run
(pnp_from_scratch/results/pipeline/correspondences/), and minimises the
reprojection error from the TRUE pose with Gauss-Newton (the same model as
src/refine.cpp) on four inlier subsets / conventions:
  all                  every inlier (= what the pipeline does)
  interior             inliers with no depth discontinuity within 3 px (no
                       background pixel and no depth jump > 1 %)
  half_pixel           all inliers, pixel coordinates shifted by +0.5 (the
                       renderer samples pixel (x, y) at (x + 0.5, y + 0.5))
  interior+half_pixel  both
and prints the signed rotation error (estimated - 10 deg) summed over 35 pairs.

Usage (repository root): pixi run -e results python docs/migration/04_scratch_refinement/contour_bias_experiment.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pnp_from_scratch" / "tools"))
from png_reader import read_png  # noqa: E402

fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7


def q2R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def exp(p):
    a = np.linalg.norm(p)
    W = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    return np.eye(3) + W if a < 1e-12 else np.eye(3) + np.sin(a) / a * W + (1 - np.cos(a)) / a ** 2 * W @ W


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def refine(X, uv, R, t):
    for _ in range(30):
        Xc = X @ R.T + t
        x, y, z = Xc.T
        e = np.stack([fx * x / z + cx - uv[:, 0], fy * y / z + cy - uv[:, 1]], 1).reshape(-1)
        Jp = np.zeros((len(X), 2, 3))
        Jp[:, 0, 0], Jp[:, 0, 2], Jp[:, 1, 1], Jp[:, 1, 2] = fx / z, -fx * x / z ** 2, fy / z, -fy * y / z ** 2
        J = np.zeros((len(X) * 2, 6))
        J[:, :3] = Jp.reshape(-1, 3)
        for k in range(3):  # d(R X + t)/d phi_k = e_k x Xc
            ek = np.zeros(3)
            ek[k] = 1
            J[:, 3 + k] = (Jp @ np.cross(ek, Xc)[:, :, None]).reshape(-1)
        dx = np.linalg.lstsq(J, -e, rcond=None)[0]
        dR = exp(dx[3:])
        R, t = dR @ R, dR @ t + dx[:3]
        if np.linalg.norm(dx) < 1e-12:
            break
    return R, t


gt = [list(map(float, l.split())) for l in open(ROOT / "data/synthetic_bunny/groundtruth.txt") if not l.startswith("#")]
Twc = [(q2R(*g[4:8]), np.array(g[1:4])) for g in gt]
out = {k: [] for k in ("all", "interior", "half_pixel", "interior+half_pixel")}
for i in range(len(Twc) - 1):
    d = read_png(ROOT / f"data/synthetic_bunny/{i:06d}_depth.png").astype(float)
    (Ri, ti), (Rj, tj) = Twc[i], Twc[i + 1]
    Rg, tg = Rj.T @ Ri, Rj.T @ (ti - tj)
    rows = [r for r in csv.DictReader(open(ROOT / f"pnp_from_scratch/results/pipeline/correspondences/pair_{i}_{i + 1}.csv"))
            if r["pnp_inlier"] == "1"]
    ui = np.array([[float(r["u_i"]), float(r["v_i"])] for r in rows])
    uj = np.array([[float(r["u_j"]), float(r["v_j"])] for r in rows])
    raw = np.array([float(r["depth_raw_i"]) for r in rows])
    inter = []
    for (u, v), z in zip(ui.astype(int), raw):
        w = d[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
        inter.append(not ((w == 0).any() or (np.abs(w[w > 0] - z) > 0.01 * z).any()))
    inter = np.array(inter)
    for name, sel, sh in (("all", slice(None), 0), ("interior", inter, 0), ("half_pixel", slice(None), 0.5),
                          ("interior+half_pixel", inter, 0.5)):
        a, b, zz = ui[sel] + sh, uj[sel] + sh, raw[sel] / 5000
        X = np.stack([(a[:, 0] - cx) / fx * zz, (a[:, 1] - cy) / fy * zz, zz], 1)
        R, t = refine(X, b, Rg.copy(), tg.copy())
        out[name].append((ang(R) - 10, ang(Rg.T @ R), np.linalg.norm(t - tg), len(zz)))
for k, v in out.items():
    v = np.array(v)
    print(f"{k:21s} signed rotation (est - 10): mean {v[:, 0].mean():+.3f} deg, sum {v[:, 0].sum():+.1f} deg, "
          f"negative {int((v[:, 0] < 0).sum())}/35;  |rotation error| mean {v[:, 1].mean():.3f} deg; "
          f"translation error mean {v[:, 2].mean():.4f} m; points per pair {v[:, 3].mean():.0f}")
