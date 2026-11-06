#!/usr/bin/env python3
"""Self-checking tests of the diagnostic export and the diagnostic modules.

Usage (repository root): python3 tools/evaluation/diagnostics/test_diagnostics.py [--dataset DIR]
Synthetic checks need no data. The evaluator checks (option parsing, export on/off giving identical
standard outputs, validation of a real export) run when the binary and --dataset (default
$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz if SLAM_TUM_ROOT is set, else data/rgbd_dataset_freiburg1_xyz) exist,
and are reported as SKIPPED otherwise. Exit 1 on failure.
"""

import argparse
import csv
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import failure_timeline as FT  # noqa: E402
import feature_tracks as TR  # noqa: E402
import build_viewer as BV  # noqa: E402
import pair_page as PG  # noqa: E402
import pair_report as PR  # noqa: E402
import pair_view as PV  # noqa: E402
import render3d as R3  # noqa: E402
from synthetic_runs import Rz, write_run  # noqa: E402

failures = 0
K = {"sensor": "freiburg1", "fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "width": 640, "height": 480,
     "depth_factor": 5000.0, "distortion_applied": False}


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def rodrigues(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = math.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def write_export(d, T_ji, n=40, outliers=(3, 7), no_depth_i=(5,), no_depth_j=(9,), icp_thr=0.01, pnp_failed=False,
                 swap_pnp=False, flip_icp=None, seed=1):
    """A synthetic export in the evaluator's format: n matches of points seen by both cameras."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    rows, X_all = [], []
    u_i = rng.uniform(40, 600, n)
    v_i = rng.uniform(40, 440, n)
    raw_i = rng.integers(4000, 12000, n)
    T_ij = np.linalg.inv(T_ji)  # native ICP direction T_i<-j
    pnp_rows, icp_rows = [], []
    for m in range(n):
        z = raw_i[m] / 5000.0
        X = np.array([(u_i[m] - K["cx"]) / K["fx"] * z, (v_i[m] - K["cy"]) / K["fy"] * z, z])
        Y = T_ji[:3, :3] @ X + T_ji[:3, 3]
        uj, vj = K["fx"] * Y[0] / Y[2] + K["cx"], K["fy"] * Y[1] / Y[2] + K["cy"]
        if m in outliers:
            uj, vj = uj + 60, vj - 45
        rj = int(round(Y[2] * 5000)) if m not in no_depth_j else 0
        ri = int(raw_i[m]) if m not in no_depth_i else 0
        rows.append(dict(match=m, kp_i=m, kp_j=m, hamming=10, u_i=u_i[m], v_i=v_i[m], u_j=uj, v_j=vj, depth_raw_i=ri,
                         depth_raw_j=rj, X=X, Y=Y))
    T_pnp = np.linalg.inv(T_ji) if swap_pnp else T_ji
    pnp_corr, icp_corr = 0, 0
    lines = []
    inl_idx = []
    n_icp_in = 0
    T_ransac = T_ij
    for r in rows:
        ri, rj = r["depth_raw_i"], r["depth_raw_j"]
        dm_i, dm_j = ri / 5000.0, rj / 5000.0
        base = [r["match"], r["kp_i"], r["kp_j"], r["hamming"], r["u_i"], r["v_i"], r["u_j"], r["v_j"], ri, rj, dm_i, dm_j]
        if ri == 0:
            pnp = [-1, "no_depth_i"] + [""] * 11
            icp = [-1, "no_depth_i"] + [""] * 6
        else:
            z = dm_i
            X = np.array([(r["u_i"] - K["cx"]) / K["fx"] * z, (r["v_i"] - K["cy"]) / K["fy"] * z, z])
            uvp, zp = C.project(T_pnp, X[None], K)
            uvg, zg = C.project(T_ji, X[None], K)
            rp = float(np.hypot(*(uvp[0] - [r["u_j"], r["v_j"]])))
            rg = float(np.hypot(*(uvg[0] - [r["u_j"], r["v_j"]])))
            if pnp_failed:
                cls, res = "unclassified", ["", "", "", ""]
            else:
                cls = "inlier" if rg < 8 else "outlier"
                res = [uvp[0, 0], uvp[0, 1], zp[0], rp]
                if cls == "inlier":
                    inl_idx.append(pnp_corr)
            pnp = [pnp_corr, cls, *X] + res + [uvg[0, 0], uvg[0, 1], zg[0], rg]
            pnp_corr += 1
            if rj == 0:
                icp = [-1, "no_depth_j"] + [""] * 6
            else:
                zj = dm_j
                Q = np.array([(r["u_j"] - K["cx"]) / K["fx"] * zj, (r["v_j"] - K["cy"]) / K["fy"] * zj, zj])
                res3 = float(np.linalg.norm(X - (T_ij[:3, :3] @ Q + T_ij[:3, 3])))
                lab = "inlier" if res3 < icp_thr else "rejected"
                if flip_icp is not None and icp_corr == flip_icp:
                    lab = "rejected" if lab == "inlier" else "inlier"
                n_icp_in += lab == "inlier"
                icp = [icp_corr, lab, *Q, res3, res3, res3]
                icp_corr += 1
        lines.append(base + pnp + icp)
    with open(d / "matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(C.MATCH_COLS)
        for l in lines:
            w.writerow([repr(float(x)) if isinstance(x, (float, np.floating)) else x for x in l])
    for side in "ij":
        with open(d / f"keypoints_{side}.csv", "w") as f:
            f.write("kp,x,y,angle_deg,response\n")
            for r in rows:
                f.write(f"{r['kp_i']},{r['u_' + side]},{r['v_' + side]},0,1\n")
    T_wc_i = np.eye(4)
    T_wc_j = np.linalg.inv(T_ji)
    meta = {"pair": 0, "frame_i": 0, "frame_j": 1, "t_rgb_i": 1.0, "t_rgb_j": 1.1, "t_depth_i": 1.0, "t_depth_j": 1.1,
            "rgb_i": "rgb/a.png", "rgb_j": "rgb/b.png", "depth_i": "depth/a.png", "depth_j": "depth/b.png",
            "intrinsics": K, "config": {"icp": {"threshold_m": icp_thr}},
            "counts": {"keypoints_i": n, "keypoints_j": n, "matches": n,
                       "depth_valid_i": int(sum(r["depth_raw_i"] > 0 for r in rows)),
                       "depth_valid_j": int(sum(r["depth_raw_j"] > 0 for r in rows)),
                       "pnp_correspondences": pnp_corr, "pnp_inliers": len(inl_idx), "icp_correspondences": icp_corr,
                       "icp_inliers": n_icp_in},
            "status": {"pnp_called": True, "pnp_solver_returned": not pnp_failed, "pnp_ok": not pnp_failed,
                       "icp_called": True, "icp_ransac_ok": True, "icp_ok": True},
            "pnp_inlier_indices": inl_idx, "T_wc_gt_i": T_wc_i.tolist(), "T_wc_gt_j": T_wc_j.tolist(),
            "T_j_from_i_gt": T_ji.tolist(), "T_j_from_i_pnp": None if pnp_failed else T_pnp.tolist(),
            "T_j_from_i_icp": T_ji.tolist(), "T_i_from_j_icp_ransac_native": T_ransac.tolist(),
            "T_i_from_j_icp_native": T_ij.tolist(), "export_only_pnp_refits_on_inliers": {"attempts": []}}
    (d / "pair.json").write_text(json.dumps(meta))
    return meta


def synthetic_tests(tmp):
    print("=== synthetic exports ===")
    T = C.se3(rodrigues([0.2, 1, 0.1], 3.0), np.array([0.05, -0.01, 0.02]))
    write_export(tmp / "good", T)
    p = C.load_pair(tmp / "good")
    pr = C.validate_pair(p)
    check(pr == [], f"consistent synthetic export validates ({pr})")
    check(int((p["pnp_class"] == "outlier").sum()) == 2 and int((p["pnp_class"] == "no_depth_i").sum()) == 1,
          "outliers and the missing-depth row keep their classes")
    check(int((p["icp_class"] == "no_depth_j").sum()) == 1, "a match without depth in frame j is no_depth_j for ICP")
    check(np.isnan(p["X_i"][p["pnp_class"] == "no_depth_i"]).all(), "a row without depth has no 3D point (NaN, not 0)")
    # transform direction: the PnP pose written as its inverse must be caught by the residual recomputation
    write_export(tmp / "swapped", T, swap_pnp=True)
    q = C.load_pair(tmp / "swapped")
    meta = json.loads((tmp / "swapped" / "pair.json").read_text())
    meta["T_j_from_i_pnp"] = T.tolist()  # residual columns were computed with the inverse: mismatch
    (tmp / "swapped" / "pair.json").write_text(json.dumps(meta))
    q = C.load_pair(tmp / "swapped")
    check(any("pnp_res_px" in s for s in C.validate_pair(q)), "PnP residuals computed with the inverted transform are detected")
    meta = json.loads((tmp / "good" / "pair.json").read_text())
    meta["T_j_from_i_gt"] = np.linalg.inv(T).tolist()
    (tmp / "dir").mkdir()
    for f in ("matches.csv", "keypoints_i.csv", "keypoints_j.csv"):
        shutil.copy(tmp / "good" / f, tmp / "dir" / f)
    (tmp / "dir" / "pair.json").write_text(json.dumps(meta))
    check(any("inv(T_wc_j) T_wc_i" in s for s in C.validate_pair(C.load_pair(tmp / "dir"))),
          "a ground-truth T_j<-i stored in the opposite direction is detected")
    # ICP classification must follow the 1 cm rule
    write_export(tmp / "flip", T, flip_icp=0)
    check(any("ICP labels disagree" in s for s in C.validate_pair(C.load_pair(tmp / "flip"))),
          "an ICP label that contradicts residual < threshold is detected")
    # failed PnP: no transform, unclassified rows, no residuals; indicators still computed
    write_export(tmp / "failed", T, pnp_failed=True)
    f = C.load_pair(tmp / "failed")
    check(C.validate_pair(f) == [] and (f["pnp_class"][f["pnp_corr"] >= 0] == "unclassified").all(),
          "a failed PnP export (no pose, unclassified rows) validates")
    imgs = {s: np.zeros((480, 640, 3), np.uint8) for s in "ij"}
    deps = {s: np.full((480, 640), 5000, np.uint16) for s in "ij"}
    ind = PR.indicators(f, imgs, deps, None)
    check(ind["pose_errors_from_exported_transforms"]["pnp"] is None and "within_8px_of_returned_pnp_pose" not in ind,
          "indicators report no PnP error for a failed solver (null, not 0)")
    # residual arithmetic
    uv, z = C.project(np.eye(4), np.array([[0.1, -0.05, 2.0]]), K)
    check(np.allclose(uv[0], [517.3 * 0.05 + 318.6, 516.5 * -0.025 + 255.3]) and z[0] == 2.0, "pinhole projection")
    X, px = C.backproject(np.pad(np.array([[5000]], np.uint16), ((255, 224), (318, 321))), K)
    check(len(X) == 1 and np.allclose(X[0], [(318 - 318.6) / 517.3, (255 - 255.3) / 516.5, 1.0]),
          "back-projection of one valid pixel (raw 5000 = 1 m); zero depth ignored")
    check(abs(PR.p_no_good_sample(10, 5) - (1 - 1 / 252) ** 100) < 1e-12, "sampling probability (1 - C(5,5)/C(10,5))^100")
    # malformed exports
    (tmp / "bad").mkdir()
    shutil.copy(tmp / "good" / "pair.json", tmp / "bad" / "pair.json")
    try:
        C.load_pair(tmp / "bad")
        check(False, "missing matches.csv raises ExportError")
    except C.ExportError:
        check(True, "missing matches.csv raises ExportError")
    txt = (tmp / "good" / "matches.csv").read_text().replace("pnp_class", "pnp_klass", 1)
    shutil.copy(tmp / "good" / "keypoints_i.csv", tmp / "bad")
    shutil.copy(tmp / "good" / "keypoints_j.csv", tmp / "bad")
    (tmp / "bad" / "matches.csv").write_text(txt)
    try:
        C.load_pair(tmp / "bad")
        check(False, "a renamed column raises ExportError")
    except C.ExportError as e:
        check("pnp_class" in str(e), "a renamed column raises ExportError naming it")


def track_tests():
    print("=== track chaining ===")
    xy = np.array([[10.0, 10], [20, 20], [30, 30]])

    def P(kp_i, kp_j, n_kp=3):
        return {"n": len(kp_i), "kp_i": np.array(kp_i, float), "kp_j": np.array(kp_j, float), "kp_i_xy": xy, "kp_j_xy": xy}
    pairs = {0: P([0, 1, 2], [1, 1, 2]), 1: P([1, 2], [0, 2])}
    tracks, consistent, merges = TR.build_tracks(pairs)
    links = sorted(l for t in tracks for l in t)
    check(consistent == {0: True} and len(links) == 5 and len(set(links)) == 5, "every match is in exactly one track")
    check(merges == 1 and max(len(t) for t in tracks) == 2, "two matches ending on one keypoint: the second track ends at the merge")
    pairs[1]["kp_i_xy"] = xy + 1
    tracks, consistent, _ = TR.build_tracks(pairs)
    check(consistent == {0: False} and all(len(t) == 1 for t in tracks), "no chaining where the keypoint lists differ")


def timeline_tests(tmp):
    print("=== failure timeline ===")
    gt = [(Rz(5 * k), np.array([0.1 * k, 0.0, 1.0])) for k in range(6)]
    est = list(gt)
    est[3] = est[2]  # pair 2 failed: pose held
    C3 = est[2][0] @ gt[3][0].T
    est[4:] = [(C3 @ R, C3 @ (p - gt[3][1]) + est[2][1]) for R, p in gt[4:]]
    write_run(tmp / "run", gt, est, est, failed=(2,))
    _, tl = FT.timeline(tmp / "run")
    d = tl[2]
    check(not d["pnp_solver_ok"] and d["pnp_icp_rot_deg"] is None, "failed pair: solver status false, disagreement null")
    check(all(x["pnp_icp_rot_deg"] is not None and x["pnp_icp_rot_deg"] < 1e-4 for x in tl if x["pair"] != 2),
          "identical PnP and ICP estimates: zero disagreement elsewhere")
    check(all(x["pnp_solver_ok"] for x in tl if x["pair"] != 2), "warnings do not change the solver status")


def view_tests(tmp):
    print("=== views: renderer, sampling, viewer payload, report ===")
    v = R3.View(eye=[0, 0, 0], target=[0, 0, 1], up=[0, -1, 0], fov_deg=60, size=(200, 100))
    check(np.allclose(v.R, np.eye(3)), "virtual camera looking along +z with up = -y has OpenCV axes (x right, y down)")
    img, zb = v.splat(np.array([[0, 0, 2.0], [0, 0, 1.0]]), np.array([[1, 0, 0], [0, 0, 1.0]]), radius=0)
    check(np.allclose(img[50, 100], [0, 0, 1]) and zb[50, 100] == 1.0, "z-buffer keeps the nearest point per pixel")
    uv, keep = v.segments_2d(np.array([[[0, 0, 1.0], [0, 0, -1.0]], [[0, 0, 1.0], [0.1, 0, 1.0]]]))
    check(keep.tolist() == [False, True], "segments reaching behind the virtual camera are dropped")
    T = C.se3(rodrigues([0.2, 1, 0.1], 3.0), np.array([0.05, -0.01, 0.02]))
    write_export(tmp / "v" / "diagnostics" / "pair_0_1", T, n=60, outliers=tuple(range(0, 40, 2)))
    p = C.load_pair(tmp / "v" / "diagnostics" / "pair_0_1")
    sel = PV.select(p, max_lines=20)
    drawn = [r for rows in sel.values() for r in rows]
    check(all(p["pnp_class"][r] == c for c, rows in sel.items() for r in rows),
          "pair view: every drawn line has the class the export gives that match row")
    again = PV.select(p, max_lines=20)
    check(len(drawn) <= 20 and all(list(sel[c]) == list(again[c]) for c in sel),
          "pair view: sampling is capped and deterministic")
    pay = BV.pair_payload(p, None, None)
    code = {"inlier": "I", "outlier": "O", "no_depth_i": "D", "unclassified": "U"}
    check(pay["m"]["cls"] == "".join(code[c] for c in p["pnp_class"]), "viewer: PnP class codes equal the export")
    q = p["icp_corr"] >= 0
    check(pay["icp"]["cls"] == "".join("I" if c == "inlier" else "R" for c in p["icp_class"][q]),
          "viewer: ICP flags equal the export")
    check(np.allclose(np.linalg.inv(np.array(pay["T_icp_ij"])), np.array(pay["T_icp_ji"]), atol=1e-9),
          "viewer: T_i<-j and T_j<-i of ICP are inverse transforms")
    check(pay["reg"]["icp"]["within_threshold"] == int((p["icp_class"][q] == "inlier").sum()),
          "viewer: under ICP's transform exactly the inliers are within the threshold (synthetic: exact data)")
    rows = {k: {"pnp_rot_err_deg": e, "icp_rot_err_deg": e, "pnp_inliers": n, "gt_motion_m": g}
            for k, (e, n, g) in enumerate([(5, 10, 0.05), (0.2, 200, 0.05), (0.3, 200, 0.05), (0.1, 200, 0.2),
                                           (9, 3, 0.05), (0.25, 200, 0.05), (2, 50, 0.05)])}
    # median error 0.3: candidates 1 and 5 (pair 3 moves 4x the median, pair 2 is not below the median)
    check(PR.choose_reference(rows, 4) == 5, "reference pair: nearest pair with typical errors, inliers and motion")
    h = PG.md_to_html("# T\n\n| a | b |\n|---|---|\n| <x> | `y` |\n\n1. item & more")
    check("&lt;x&gt;" in h and "<code>y</code>" in h and "&amp; more" in h and "<table>" in h,
          "report page: Markdown converted with HTML escaped")


def evaluator_tests(tmp, dataset):
    print("=== evaluator export (tum_trajectory_eval) ===")
    binary = HERE.parents[2] / "build" / "tum_trajectory_eval"
    if not binary.is_file() or not (Path(dataset) / "rgb.txt").is_file():
        print(f"  [SKIPPED] needs {binary} and {dataset}")
        return
    run = lambda *args: subprocess.run([str(binary), str(dataset), *args], capture_output=True, text=True)
    for bad in ("7-3", "x", "1,,2", "-1"):
        r = run("--frames", "4", "--out-dir", str(tmp / "bad"), "--export-pairs", bad)
        check(r.returncode == 2 and "export-pairs" in r.stderr, f"--export-pairs {bad!r} rejected with exit 2")
    r = run("--frames", "4", "--out-dir", str(tmp / "range"), "--export-pairs", "3")
    check(r.returncode == 2 and "does not exist" in r.stderr, "a pair beyond the run is rejected (4 frames: pairs 0..2)")
    a = run("--frames", "4", "--out-dir", str(tmp / "plain"))
    b = run("--frames", "4", "--out-dir", str(tmp / "export"), "--export-pairs", "1-2")
    check(a.returncode == 0 and b.returncode == 0, "evaluator runs with and without the export")
    check(all(filecmp.cmp(tmp / "plain" / f, tmp / "export" / f, shallow=False)
              for f in ("frames.csv", "pairs.csv", "trajectory.csv", "summary.txt")),
          "standard outputs byte-identical with and without --export-pairs")
    check(not (tmp / "plain" / "diagnostics").exists(), "no diagnostics without the option (off by default)")
    dirs = sorted(p.name for p in (tmp / "export" / "diagnostics").iterdir())
    check(dirs == ["pair_1_2", "pair_2_3"], f"only the requested pairs are exported ({dirs})")
    rows = C.pairs_rows(tmp / "export")
    for k in (1, 2):
        pr = C.validate_pair(C.load_pair(tmp / "export" / "diagnostics" / f"pair_{k}_{k + 1}"), rows[k])
        check(pr == [], f"real export of pair {k} is consistent with itself and with pairs.csv ({pr})")


def main():
    ap = argparse.ArgumentParser()
    tum_root = Path(os.environ["SLAM_TUM_ROOT"]) if os.environ.get("SLAM_TUM_ROOT") else HERE.parents[2] / "data"
    ap.add_argument("--dataset", default=str(tum_root / "rgbd_dataset_freiburg1_xyz"))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="diag_test_"))
    try:
        synthetic_tests(tmp)
        track_tests()
        timeline_tests(tmp)
        view_tests(tmp)
        evaluator_tests(tmp, a.dataset)
    finally:
        shutil.rmtree(tmp)
    print(f"\ntest_diagnostics: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
