#!/usr/bin/env python3
"""Tests of the independent GT / PnP / ICP reconstructions (tum_reconstruction.py, export_tum_scene.py).

Usage (repository root): python3 visualization/scripts/test_tum_reconstruction.py [--export DIR ...]
Synthetic checks need no data; with --export (an export_tum_scene.py output), every exported cloud is also
recomputed independently from the dataset images and trajectory.csv and compared with the PLY. Exit 1 on failure.
"""

import argparse
import copy
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "evaluation"))
sys.path.insert(0, str(ROOT / "tools" / "evaluation" / "diagnostics"))
import tum_metrics as tm  # noqa: E402
import tum_reconstruction as TR  # noqa: E402
from synthetic_runs import write_run  # noqa: E402

failures = 0
K = {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "width": 640, "height": 480, "depth_factor": 5000.0}


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def synthetic(tmp):
    print("=== synthetic run, non-identity poses ===")
    rng = np.random.default_rng(3)
    n = 6
    gt = [(rot(rng.normal(size=3), 10 + 7 * k), np.array([0.3 * k, -0.1 * k, 1.0 + 0.05 * k])) for k in range(n)]
    pnp = [(rot([0, 0, 1], 4.0 * k) @ R, p + [0.02 * k, 0, 0]) for k, (R, p) in enumerate(gt)]
    pnp[0] = gt[0]
    pnp[3] = pnp[2]  # pair 2->3 failed: frame 3 holds frame 2's pose
    icp = [(R @ rot([1, 0, 0], 2.0 * k), p + [0, 0.03 * k, 0]) for k, (R, p) in enumerate(gt)]
    icp[0] = gt[0]
    icp[3] = icp[2]  # write_run marks a failed pair for PnP AND ICP: both hold frame 3
    write_run(tmp / "run", gt, pnp, icp, failed=(2,))
    run = tm.load_run(tmp / "run")
    depth = np.zeros((480, 640), np.uint16)
    depth[4::8, 4::8] = 5000 + (np.arange(60)[:, None] * 37 + np.arange(80)[None] * 11) % 4000
    depth[100:140, 100:300] = 0  # invalid depth region
    rgb = (np.arange(480 * 640 * 3) % 251).astype(np.uint8).reshape(480, 640, 3)
    read = lambda k: (depth, rgb)
    recon, sel, samples = TR.reconstruct(run, read, K, every=1, step=8)
    check(sel == list(range(n)), "frame selection: every frame, same list for all modes")
    check(all(len(samples[k]["xyz"]) == int((depth[4::8, 4::8] > 0).sum()) for k in sel),
          "only pixels with raw depth > 0 are sampled (invalid region excluded)")
    # transform direction with a known pixel
    u, v = 4 + 8 * 40, 4 + 8 * 30
    z = depth[v, u] / 5000.0
    X = np.array([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z])
    idx = int(np.nonzero((samples[1]["px"][:, 0] == u) & (samples[1]["px"][:, 1] == v))[0][0])
    off = len(samples[0]["xyz"])
    for mode, poses in (("gt", gt), ("pnp", pnp), ("icp", icp)):
        R, t = poses[1]
        Pm = recon[mode]["xyz"][off + idx]
        check(np.allclose(Pm, R @ X + t, atol=1e-9) and not np.allclose(Pm, R.T @ (X - t), atol=1e-6),
              f"{mode}: frame 1 pixel placed at R_wc X_cam + t_wc of the {mode} pose (not the inverse)")
    check(recon["gt"]["frames_used"] == list(range(n)), "GT: no frame excluded (ground truth is never held)")
    check(recon["icp"]["frames_used"] == [0, 1, 2, 4, 5], "ICP: its held frame 3 is excluded as well")
    check(recon["pnp"]["frames_used"] == [0, 1, 2, 4, 5] and recon["pnp"]["frames_skipped"][0]["frame"] == 3,
          "PnP: the held frame 3 is excluded and recorded with its reason")
    check("held" in recon["pnp"]["frames_skipped"][0]["reason"], "skip reason names the held pose")
    # ground truth is never used by an estimated mode: change GT, estimated clouds stay bit-identical
    run2 = copy.deepcopy(run)
    run2["gt_R"] = run2["gt_R"][::-1].copy()
    run2["gt_p"] = run2["gt_p"] + 5.0
    r2, _, _ = TR.reconstruct(run2, read, K, every=1, step=8)
    check(np.array_equal(r2["pnp"]["xyz"], recon["pnp"]["xyz"]) and np.array_equal(r2["icp"]["xyz"], recon["icp"]["xyz"]),
          "changing the ground truth leaves the PnP and ICP clouds bit-identical")
    check(not np.array_equal(r2["gt"]["xyz"], recon["gt"]["xyz"]), "control: the GT cloud does change")
    try:
        TR.pose_source(run, "gt_aligned")
        check(False, "unknown pose source rejected")
    except ValueError:
        check(True, "unknown pose source rejected")
    # comparison: identical poses -> zero displacement; a pure 0.1 m shift -> 0.1 m everywhere
    run3 = copy.deepcopy(run)
    run3["icp_R"], run3["icp_p"] = run3["gt_R"].copy(), run3["gt_p"].copy()
    run3["pnp_R"], run3["pnp_p"] = run3["gt_R"].copy(), run3["gt_p"] + [0.1, 0, 0]
    run3["pnp_held"] = np.zeros(n, bool)
    r3, _, s3 = TR.reconstruct(run3, read, K, every=1, step=8)
    c3 = TR.compare(r3, s3)
    check(c3["icp"]["displacement_vs_gt_m"]["max"] < 1e-12 and abs(c3["pnp"]["displacement_vs_gt_m"]["median"] - 0.1) < 1e-9,
          "compare: identical poses -> 0 displacement; 0.1 m shift -> 0.1 m")
    c = TR.compare(recon, samples)
    check(c["pnp"]["frames_compared"] == 5, "compare: only frames placed in both modes are compared")


def real(export):
    print(f"=== real export: {export} ===")
    from common import read_depth_raw, read_rgb
    exp = Path(export)
    D = json.loads((exp / "trajectory.json").read_text())
    run_dir = Path(json.loads((exp / f"manifest_preview_gt.json").read_text())["source_files"]["trajectory.csv"]).parent
    run = tm.load_run(run_dir)
    for density in D["clouds"]:
        mans = {m: json.loads((exp / f"manifest_{density}_{m}.json").read_text()) for m in TR.MODES}
        check(len({json.dumps(m["sampling"]["per_frame_pixel_sha256"], sort_keys=True) for m in mans.values()}) == 1 and
              len({tuple(m["frames_selected"]) for m in mans.values()}) == 1,
              f"{density}: same frames and identical sampled pixels in all modes")
        for mode, man in mans.items():
            from build_tum_scene_io import read_ply
            xyz, rgb = read_ply(exp / man["cloud_file"])
            check(man["uses_ground_truth"] == (mode == "gt") and man["pose_source"] == mode,
                  f"{density} {mode}: manifest pose source")
            off, worst = 0, 0.0
            for k in man["frames_used"]:  # independent recomputation from the images and trajectory.csv
                raw = read_depth_raw(D["dataset"], run["depth"][k])
                pts, col, _, _ = TR.sample_frame(raw, read_rgb(D["dataset"], run["rgb"][k]), D["intrinsics"],
                                                 man["sampling"]["pixel_step"])
                R, t, held = TR.pose_source(run, mode)
                exp_pts = pts @ R[k].T + t[k]
                got = xyz[off:off + len(pts)]
                worst = max(worst, float(np.abs(got - exp_pts).max()))
                off += len(pts)
            check(off == len(xyz) and worst < 1e-5,
                  f"{density} {mode}: every point equals R_wc X_cam + t_wc of the {mode} pose of its frame (max |d| {worst:.1e} m)")
            sk = [s["frame"] for s in man["frames_skipped"]]
            check(all(run[f"{mode}_held"][k] for k in sk) if mode != "gt" else not sk,
                  f"{density} {mode}: skipped frames {sk} are exactly held poses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="*", default=[])
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="tum_recon_"))
    try:
        synthetic(tmp)
    finally:
        shutil.rmtree(tmp)
    for e in a.export:
        real(e)
    print(f"\ntest_tum_reconstruction: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
