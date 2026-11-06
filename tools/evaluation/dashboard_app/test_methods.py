#!/usr/bin/env python3
"""Tests of the method-source layer (methods.py): reference vs scratch results side by side.

Usage (repository root): python3 tools/evaluation/dashboard_app/test_methods.py      exit 1 on failure
Synthetic checks build a temporary output/ tree (reference run + scratch run); real checks run on the
repository's results/runs/ when a scratch run exists.
"""

import csv
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "diagnostics")]
import data as D  # noqa: E402
import methods as MS  # noqa: E402
import tum_metrics as tm  # noqa: E402
from synthetic_runs import R_to_q, Rz, write_run  # noqa: E402

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


def write_scratch(d, frames_csv, gt, est, failed=()):
    """A scratch_pipeline-format output (trajectory.csv with '#' comments, pairs.csv) for the frames of frames_csv."""
    d.mkdir(parents=True)
    shutil.copy(frames_csv, d / "frames.csv")
    (d / "command.txt").write_text("build/scratch_pipeline --tum-frames x --tum-sequence data/seq_fr1 "
                                   "--intrinsics 517.3,516.5,318.6,255.3 --out " + str(d) + "\n")
    n = len(gt)
    with open(d / "trajectory.csv", "w") as f:
        f.write("# Scratch pipeline trajectory\n# note: test\nframe,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22,qx,qy,qz,qw,"
                "rotation_error_deg,translation_error_m,pnp_inliers,pose_source\n")
        for k, (R, p) in enumerate(est):
            q = R_to_q(R)
            src = "gt_anchor" if k == 0 else ("held_previous_pose_pair_failed" if (k - 1) in failed else "scratch_pipeline")
            f.write(",".join([str(k)] + [f"{x:.9f}" for x in list(p) + list(R.ravel()) + list(q)] +
                             [f"{tm.rot_angle_deg(gt[k][0].T @ R):.6f}", f"{np.linalg.norm(p - gt[k][1]):.9f}", "50", src]) + "\n")
    cols = ["pair", "frame_i", "frame_j", "keypoints_i", "keypoints_j", "raw_matches", "filtered_matches",
            "correspondences_3d2d", "pnp_ok", "pnp_inliers", "rot_err_deg", "trans_err_m"] + \
        [f"t{a}" for a in "xyz"] + [f"r{i}{j}" for i in range(3) for j in range(3)]
    with open(d / "pairs.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(n - 1):
            ok = k not in failed
            Ri, pi = est[k]
            Rj, pj = est[k + 1]
            R, t = Rj.T @ Ri, Rj.T @ (pi - pj)
            Rg, tg = gt[k + 1][0].T @ gt[k][0], gt[k + 1][0].T @ (gt[k][1] - gt[k + 1][1])
            vals = [k, k, k + 1, 500, 500, 480, 300, 250, int(ok), 120 if ok else 0,
                    f"{tm.rot_angle_deg(Rg.T @ R):.6f}" if ok else "", f"{np.linalg.norm(tg - t):.6f}" if ok else ""]
            if ok:
                vals += [f"{x:.9f}" for x in list(t) + list(R.ravel())]
            else:
                vals += [f"{x:.9f}" for x in [0, 0, 0] + list(np.eye(3).ravel())]
            f.write(",".join(map(str, vals)) + "\n")


def synthetic():
    print("=== synthetic output tree ===")
    tmp = Path(tempfile.mkdtemp(prefix="methods_test_"))
    old = D.OUTPUT
    try:
        D.OUTPUT = tmp / "output"
        D._cache.clear()
        n = 7
        gt = [(rot([0.3, 1, 0.2], 6 * k), np.array([0.1 * k, 0.02 * k * k, 1.0])) for k in range(n)]
        ref = [gt[0]] + [(R @ rot([0, 0, 1], 0.5 * k), p + [0.01 * k, 0, 0]) for k, (R, p) in list(enumerate(gt))[1:]]
        write_run(D.OUTPUT / "stress" / "s1" / "c1", gt, ref, ref)
        scr = [gt[0]] + [(R @ rot([1, 0, 0], 1.0 * k), p + [0, 0.02 * k, 0]) for k, (R, p) in list(enumerate(gt))[1:]]
        scr[4] = scr[3]   # pair 3 failed: frame 4 held
        C3 = scr[3][0] @ gt[4][0].T
        scr[5:] = [(C3 @ R, C3 @ (p - gt[4][1]) + scr[3][1]) for R, p in gt[5:]]
        write_scratch(D.OUTPUT / "scratch" / "seq" / "c1", D.OUTPUT / "stress" / "s1" / "c1" / "frames.csv", gt, scr,
                      failed=(3,))
        srcs = MS.discover()
        ids = sorted(s["id"] for s in srcs)
        check(ids == ["scratch/seq/c1::scratch_pnp", "stress/s1/c1::ref_icp", "stress/s1/c1::ref_pnp"],
              f"discovery: one reference run (PnP, ICP) and one scratch run ({ids})")
        runs, _ = D.discover()
        check([r["id"] for r in runs] == ["stress/s1/c1"], "the scratch directory is not mistaken for a reference run")
        lab = {s["id"]: (s["label"], s["family"]) for s in srcs}
        check(lab["scratch/seq/c1::scratch_pnp"] == ("Scratch PnP", "scratch") and
              lab["stress/s1/c1::ref_pnp"] == ("Reference PnP", "reference"), "labels and families per source")
        check(all(s["source_files"][0].startswith(f"results/runs/{s['run']}/") for s in srcs), "each source names its own files")
        run = MS.load_scratch_run(D.OUTPUT / "scratch" / "seq" / "c1")
        check(np.array_equal(run["t"], tm.load_run(D.OUTPUT / "stress" / "s1" / "c1")["t"]),
              "timestamps of the scratch run = frames.csv of the reference run (frame k <-> same timestamp)")
        check(MS.check_scratch(run) == [], f"scratch files self-consistent (T_j<-i = T_wc[j]^-1 T_wc[i]) {MS.check_scratch(run)}")
        check(run["scratch_held"].tolist() == [False, False, False, False, True, False, False],
              "the failed scratch pair marks frame 4 as held")
        C = MS.compare(["stress/s1/c1::ref_pnp", "scratch/seq/c1::scratch_pnp"])
        mr, ms_ = C["sources"][0]["method"], C["sources"][1]["method"]
        e = [np.linalg.norm(p - g[1]) for (_, p), g in zip(scr, gt)]
        check(abs(ms_["ate_raw_rmse_m"] - math.sqrt(np.mean(np.square(e)))) < 1e-6 and ms_["failed_pairs"] == 1,
              "scratch metrics from the scratch poses: ATE raw and 1 failed pair")
        e = [np.linalg.norm(p - g[1]) for (_, p), g in zip(ref, gt)]
        check(abs(mr["ate_raw_rmse_m"] - math.sqrt(np.mean(np.square(e)))) < 1e-6 and mr["failed_pairs"] == 0,
              "reference metrics from the reference poses (different numbers: no cross-substitution)")
        check(ms_["ate_aligned_rmse_m"] <= ms_["ate_raw_rmse_m"] + 1e-12 and ms_["ate_raw_rmse_m"] != ms_["ate_aligned_rmse_m"],
              "aligned ATE reported separately from the raw ATE")
        # missing / incomplete scratch files are errors, never a fallback
        (D.OUTPUT / "scratch" / "seq" / "c1" / "trajectory.csv").rename(D.OUTPUT / "scratch" / "seq" / "c1" / "t.bak")
        D._cache.clear()
        try:
            MS.metrics("scratch/seq/c1::scratch_pnp")
            check(False, "missing scratch trajectory raises")
        except MS.SourceError as ex:
            check("missing" in str(ex), "missing scratch trajectory.csv -> SourceError (no fallback to PnP)")
        lines = (D.OUTPUT / "scratch" / "seq" / "c1" / "t.bak").read_text().splitlines()
        (D.OUTPUT / "scratch" / "seq" / "c1" / "trajectory.csv").write_text("\n".join(lines[:-2]) + "\n")
        D._cache.clear()
        try:
            MS.metrics("scratch/seq/c1::scratch_pnp")
            check(False, "incomplete scratch trajectory raises")
        except MS.SourceError as ex:
            check("incomplete" in str(ex), "incomplete scratch trajectory -> SourceError")
        (D.OUTPUT / "scratch" / "seq" / "c1" / "t.bak").rename(D.OUTPUT / "scratch" / "seq" / "c1" / "trajectory.csv")
        # a different frame selection is refused
        write_run(D.OUTPUT / "stress" / "s2" / "c2", gt[:5], ref[:5], ref[:5])
        try:
            MS.compare(["stress/s2/c2::ref_pnp", "scratch/seq/c1::scratch_pnp"])
            check(False, "different frames refused")
        except MS.SourceError:
            check(True, "sources with different frames.csv are refused (no comparison)")
        # GT must not be what places an estimate: the scene of a source carries that source's poses
        D._cache.clear()
        sc = MS.source_scene("scratch/seq/c1::scratch_pnp")  # no dataset images: poses only, no cloud
        P = np.array(sc["poses"]["scratch_pnp"]["raw"])
        check(sc["clouds"] == [] and np.allclose(P[:, 9:], [p for _, p in scr], atol=1e-6) and
              np.allclose(P[:, :9].reshape(-1, 3, 3), [R for R, _ in scr], atol=1e-6),
              "scene of a source: its own poses (non-identity rotations), no cloud without dataset images")
        check(sc["held"]["scratch_pnp"][4] and set(sc["poses"]) == {"gt", "scratch_pnp"},
              "scene marks the held frame and carries only GT + that source")
    finally:
        D.OUTPUT = old
        D._cache.clear()
        shutil.rmtree(tmp)


def real():
    print("=== real outputs ===")
    srcs = [s for s in MS.discover() if s["family"] == "scratch"]
    done = [s for s in srcs if (D.OUTPUT / s["run"] / "trajectory.csv").is_file()]
    if not done:
        print("  [SKIPPED] no finished scratch TUM run under results/runs/")
        return
    for s in done:
        run = MS.load_scratch_run(D.OUTPUT / s["run"])
        check(MS.check_scratch(run) == [], f"{s['run']}: scratch files self-consistent")
        cfg = [c for c in MS.configurations() if c["frames_key"] == s["frames_key"]][0]
        for r in cfg["reference_runs"][:1]:
            ref = tm.load_run(D.OUTPUT / r)
            check(np.array_equal(ref["t"], run["t"]) and np.allclose(ref["gt_p"], run["gt_p"]),
                  f"{s['run']} vs {r}: same timestamps and ground truth frame by frame")
            sc = MS.source_scene(s["id"])
            P = np.array(sc["poses"]["scratch_pnp"]["raw"])
            check(np.allclose(P[:, 9:], run["scratch_p"], atol=1e-6) and np.allclose(P[:, :9].reshape(-1, 3, 3), run["scratch_R"], atol=1e-6),
                  f"{s['run']}: the 3D scene draws the scratch poses (not GT, not reference PnP)")
            check(sc["held"]["scratch_pnp"] == [bool(x) for x in run["scratch_held"]], f"{s['run']}: held flags for cloud exclusion")
            sr = MS.source_scene(f"{r}::ref_pnp")
            check(not np.allclose(np.array(sr["poses"]["ref_pnp"]["raw"])[1:, 9:], P[1:, 9:]),
                  f"{s['run']}: reference PnP and scratch PnP panels show different poses")
            check(sc["cloud_frames"] == sr["cloud_frames"], f"{s['run']}: both panels use the same cloud frames")


def orb_tests():
    print("=== OpenCV ORB vs scratch correspondences ===")
    import json
    check(MS.ORB_LABEL[0] == "OpenCV ORB" and "cv::ORB" in MS.ORB_LABEL[1], "ORB label names the genuine cv::ORB implementation")
    check("not OpenCV ORB" in MS.LABELS["ref_pnp"][1] and "ORB" not in MS.LABELS["ref_pnp"][0],
          "the reference FAST + rotated BRIEF pipeline is not labelled ORB")
    check("ORB" not in MS.LABELS["scratch_pnp"][0], "the scratch pipeline is not labelled ORB")
    tmp = Path(tempfile.mkdtemp(prefix="corr_test_"))
    old = D.OUTPUT
    try:
        D.OUTPUT = tmp / "output"
        D._cache.clear()
        n = 4
        gt = [(rot([0, 1, 0], 3 * k), np.array([0.05 * k, 0, 1.0])) for k in range(n)]
        write_run(D.OUTPUT / "stress" / "s" / "c", gt, gt, gt)
        fr = D.OUTPUT / "stress" / "s" / "c" / "frames.csv"
        write_scratch(D.OUTPUT / "scratch" / "seq" / "c", fr, gt, gt)
        sd = D.OUTPUT / "scratch" / "seq" / "c" / "correspondences"
        sd.mkdir()
        (sd / "pair_1_2.csv").write_text("match,query,train,u_i,v_i,u_j,v_j,pnp_used,pnp_inlier\n0,0,0,10,10,12,10,1,1\n"
                                         "1,1,1,20,20,40,30,1,0\n2,2,2,30,30,31,30,0,0\n")
        od = D.OUTPUT / "orb_tum" / "seq" / "c"
        (od / "pair_1_2").mkdir(parents=True)
        shutil.copy(fr, od / "frames.csv")
        (od / "command.txt").write_text("build/tum_orb_correspondences data/seq --frames-csv x --out y\n")
        (od / "summary.csv").write_text("pair,pnp_rot_err_deg\n1,0.5\n")
        rows = list(csv.DictReader(open(fr)))
        (od / "pair_1_2" / "pair.json").write_text(json.dumps({
            "pair": 1, "frame_i": 1, "frame_j": 2, "t_rgb_i": float(rows[1]["t_rgb"]), "t_rgb_j": float(rows[2]["t_rgb"]),
            "rgb_i": rows[1]["rgb"], "rgb_j": rows[2]["rgb"], "config": {}, "pnp_solver_returned": True,
            "counts": {"keypoints_i": 3, "keypoints_j": 3, "raw_matches": 3, "matches": 2, "pnp_correspondences": 2,
                       "pnp_inliers": 1, "pnp_outliers": 1}, "pnp_rot_err_deg": 0.5, "pnp_trans_err_m": 0.01,
            "gt_motion_deg": 3.0, "gt_motion_m": 0.05}))
        (od / "pair_1_2" / "matches.csv").write_text("match,query,train,hamming,u_i,v_i,u_j,v_j,depth_raw_i,pnp_used,pnp_inlier\n"
                                                    "0,0,0,5,100,100,101,100,5000,1,1\n1,1,1,9,200,200,260,210,5000,1,0\n")
        for side in "ij":
            (od / "pair_1_2" / f"keypoints_{side}.csv").write_text("kp,x,y,size,angle_deg,response,octave\n0,100,100,31,0,1,0\n1,200,200,31,0,1,0\n")
        check([r["id"] for r in D.discover()[0]] == ["stress/s/c"] and D.discover()[1] == [], "ORB and scratch outputs are not listed as reference (or incomplete) runs")
        C = MS.corr_compare("orb_tum/seq/c", "scratch/seq/c", 1)
        check(C["frames"]["frame_i"] == 1 and C["orb"]["i"] == C["scratch"]["i"] == 1 and C["orb"]["j"] == C["scratch"]["j"] == 2
              and C["orb"]["t_i"] == C["scratch"]["t_i"] == C["frames"]["t_rgb_i"], "both panels show the same frame pair and timestamps")
        check(C["orb"]["drawn_counts"] == {"matches": 2, "inliers": 1, "outliers": 1, "no_depth": 0, "unclassified": 0} and
              C["scratch"]["drawn_counts"] == {"matches": 3, "inliers": 1, "outliers": 1, "no_depth": 1, "unclassified": 0},
              "each panel's counts come from its own export")
        C0 = MS.corr_compare("orb_tum/seq/c", "scratch/seq/c", 0)
        check(not C0["orb"]["available"] and "not exported" in C0["orb"]["reason"] and "--pairs 0" in C0["orb"]["command"]
              and not C0["scratch"]["available"], "missing exports for a pair: unavailable with a reason (and an ORB command)")
        rows2 = rows[:]
        with open(od / "frames.csv", "w", newline="") as f:  # a different frame selection must be refused
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            rows2[2] = dict(rows[2], t_rgb=str(float(rows[2]["t_rgb"]) + 1))
            w.writerows(rows2)
        try:
            MS.corr_compare("orb_tum/seq/c", "scratch/seq/c", 1)
            check(False, "different frames refused")
        except MS.SourceError:
            check(True, "ORB and scratch runs with different frames.csv are refused")
    finally:
        D.OUTPUT = old
        D._cache.clear()
        shutil.rmtree(tmp)
    runs = MS.orb_runs()
    for r in runs:
        scr = [s for s in MS.discover() if s["family"] == "scratch" and s["frames_key"] == r["frames_key"]]
        for k in (46, 73):
            if not scr or k not in r["pairs"]:
                continue
            C = MS.corr_compare(r["run"], scr[0]["run"], k)
            pj = json.loads((D.OUTPUT / r["run"] / f"pair_{k}_{k + 1}" / "pair.json").read_text())
            sp = MS._read_scratch_csv(D.OUTPUT / scr[0]["run"] / "pairs.csv")[k]
            check(C["orb"]["drawn_counts"]["inliers"] == pj["counts"]["pnp_inliers"] and
                  C["orb"]["drawn_counts"]["matches"] == pj["counts"]["matches"] and
                  C["scratch"]["drawn_counts"]["inliers"] == int(sp["pnp_inliers"]) and
                  C["scratch"]["drawn_counts"]["matches"] == int(sp["filtered_matches"]) and
                  C["orb"]["rgb_i"] == C["frames"]["rgb_i"],
                  f"real pair {k}->{k + 1} ({r['run']}): panel counts equal each method's files; same RGB frames")


def main():
    synthetic()
    real()
    orb_tests()
    print(f"\ntest_methods: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
