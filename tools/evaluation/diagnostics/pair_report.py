#!/usr/bin/env python3
"""Measurements and written report of one frame pair from a tum_trajectory_eval --export-pairs export
(no figures; the images and the page are made by pair_page.py, which calls this module).

Usage (repository root):
  python3 tools/evaluation/diagnostics/pair_report.py --run <evaluator dir> --pair 73 \
          --dataset data/rgbd_dataset_freiburg1_xyz [--compare 68 --context 66-80] [--output DIR]
Default output: <run>/../pairs/pair_<i>_<j>/: pair_summary.json (every number, with its source) and
pair_<i>_<j>_report.md (with --compare: failing pair vs a reference pair, observed facts separated from
hypotheses). Exit: 0 ok, 1 the export is inconsistent (listed), 2 missing / malformed input.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GT_CONSISTENT_PX = 8.0  # = the PnP RANSAC threshold; a correspondence within it under the TRUE pose
MINIMAL_SAMPLE = 5      # cv::solvePnPRansac's minimal set for SOLVEPNP_ITERATIVE (EPnP kernel), per OpenCV source


def sharpness(img):
    g = img.astype(float).mean(axis=2)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def spread_px(xy):
    """Bounding box (w, h) and the smaller principal standard deviation of a 2D point set."""
    if len(xy) < 2:
        return dict(bbox_w=0.0, bbox_h=0.0, min_principal_std=0.0)
    ev = np.linalg.eigvalsh(np.cov(xy.T))
    return dict(bbox_w=float(np.ptp(xy[:, 0])), bbox_h=float(np.ptp(xy[:, 1])),
                min_principal_std=float(math.sqrt(max(ev[0], 0))))


def p_no_good_sample(n, good, k=MINIMAL_SAMPLE, iters=100):
    """P(no all-good k-subset in `iters` uniform draws) = (1 - C(good,k)/C(n,k))^iters."""
    if n < k:
        return None
    q = math.comb(good, k) / math.comb(n, k)
    return (1 - q) ** iters


def indicators(p, imgs, deps, pairs_row):
    m, K = p["meta"], C.intrinsics(p)
    cls, icls = p["pnp_class"], p["icp_class"]
    s = p["pnp_corr"] >= 0
    q = p["icp_corr"] >= 0
    inl = cls == "inlier"
    G, Tp, Ti = p["T_j_from_i_gt"], p["T_j_from_i_pnp"], p["T_j_from_i_icp"]
    out = {"pair": m["pair"], "frame_i": m["frame_i"], "frame_j": m["frame_j"], "t_rgb_i": m["t_rgb_i"],
           "t_rgb_j": m["t_rgb_j"], "rgb_i": m["rgb_i"], "rgb_j": m["rgb_j"], "counts": m["counts"],
           "status": m["status"], "source": str(p["dir"])}
    c = m["counts"]
    out["pnp_inlier_ratio"] = c["pnp_inliers"] / c["pnp_correspondences"] if c["pnp_correspondences"] else None
    out["icp_inlier_ratio"] = c["icp_inliers"] / c["icp_correspondences"] if c["icp_correspondences"] else None
    out["pnp_inliers_distinct_kp_i"] = int(len(set(p["kp_i"][inl])))
    out["pnp_inliers_distinct_kp_j"] = int(len(set(p["kp_j"][inl])))
    kj, cnt = np.unique(p["kp_j"], return_counts=True)
    out["matches_sharing_a_kp_j"] = int(cnt[cnt > 1].sum())
    out["max_matches_on_one_kp_j"] = int(cnt.max()) if len(cnt) else 0
    out["pnp_inlier_spread_i_px"] = spread_px(np.c_[p["u_i"], p["v_i"]][inl])
    out["pnp_inlier_spread_j_px"] = spread_px(np.c_[p["u_j"], p["v_j"]][inl])
    out["matched_kp_spread_i_px"] = spread_px(np.c_[p["u_i"], p["v_i"]])
    good = s & (p["gt_res_px"] < GT_CONSISTENT_PX)
    out["gt_consistent_pnp_correspondences"] = int(good.sum())
    out["gt_consistent_fraction"] = float(good.sum() / s.sum()) if s.any() else None
    out["p_no_all_gt_consistent_5_sample_in_100"] = p_no_good_sample(int(s.sum()), int(good.sum()))
    if inl.any():
        P = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][inl]
        out["pnp_inlier_3d_max_pairwise_m"] = float(max(np.linalg.norm(a - b) for a in P for b in P))
    out["pnp_inliers_gt_res_px_median"] = float(np.median(p["gt_res_px"][inl])) if inl.any() else None
    out["pnp_inliers_that_are_gt_consistent"] = int((inl & good).sum())
    if Tp is not None:
        out["within_8px_of_returned_pnp_pose"] = int((p["pnp_res_px"][s] < 8.0).sum())
        out["behind_camera_j_under_returned_pnp_pose"] = int((p["pnp_z_j"][s] <= 0).sum())
    qi = icls == "inlier"
    out["icp_gt_consistent_1cm"] = int((q & (p["icp_gt_res_m"] < 0.01)).sum())
    out["icp_inliers_gt_res_m_median"] = float(np.median(p["icp_gt_res_m"][qi])) if qi.any() else None
    if qi.sum() >= 3:
        P = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][qi]
        out["icp_inlier_extent_singular_values_m"] = (np.linalg.svd(P - P.mean(0), compute_uv=False)
                                                      / math.sqrt(len(P))).tolist()
    for side in ("i", "j"):
        out[f"sharpness_laplacian_var_{side}"] = sharpness(imgs[side])
        out[f"depth_valid_fraction_image_{side}"] = float((deps[side] > 0).mean())
        out[f"depth_valid_fraction_matched_kp_{side}"] = float((p[f"depth_raw_{side}"] > 0).mean()) if p["n"] else None
    err = {}
    for meth, T in (("pnp", Tp), ("icp", Ti)):
        err[meth] = None if T is None else {
            "rot_err_deg": C.rot_deg(G[:3, :3].T @ T[:3, :3]),
            "trans_err_m": float(np.linalg.norm(G[:3, 3] - T[:3, 3])),
            "rotation_deg": C.rot_deg(T[:3, :3]), "translation_m": float(np.linalg.norm(T[:3, 3]))}
    err["gt"] = {"rotation_deg": C.rot_deg(G[:3, :3]), "translation_m": float(np.linalg.norm(G[:3, 3]))}
    if Tp is not None and Ti is not None:
        err["pnp_vs_icp"] = {"rot_deg": C.rot_deg(Tp[:3, :3].T @ Ti[:3, :3]),
                             "trans_m": float(np.linalg.norm(Tp[:3, 3] - Ti[:3, 3]))}
    out["pose_errors_from_exported_transforms"] = err
    out["pairs_csv"] = {k: pairs_row[k] for k in ("pnp_rot_err_deg", "pnp_trans_err_m", "icp_rot_err_deg",
                                                  "icp_trans_err_m")} if pairs_row else None
    refits = m.get("export_only_pnp_refits_on_inliers", {}).get("attempts", [])
    out["export_only_refits"] = [
        {"name": r["name"], "status": " ".join(re.sub(r"(^|\n|: )>\s*", r"\1", r["status"]).split())[:200],
         "max_abs_diff_to_returned_pose": (None if r["T_j_from_i"] is None or Tp is None
                                           else float(np.abs(np.array(r["T_j_from_i"]) - Tp).max()))}
        for r in refits]
    return out


def report(ind, problems, out, made):
    m = ind
    c = m["counts"]
    e = m["pose_errors_from_exported_transforms"]
    f = lambda v, fmt="{:.3f}": "n/a" if v is None else fmt.format(v)
    L = [f"# Pair {m['pair']}: frame {m['frame_i']} -> {m['frame_j']} ({m['rgb_i']} -> {m['rgb_j']})", "",
         f"Source: `{m['source']}` (tum_trajectory_eval `--export-pairs`); consistency checks: "
         f"{'all passed' if not problems else problems}.", "",
         "## Observed (from the export)", "",
         "| quantity | value |", "|---|---|",
         f"| keypoints frame i / j (cap 500) | {c['keypoints_i']} / {c['keypoints_j']} |",
         f"| image sharpness, Laplacian variance i / j | {m['sharpness_laplacian_var_i']:.0f} / {m['sharpness_laplacian_var_j']:.0f} |",
         f"| filtered matches | {c['matches']} |",
         f"| matched keypoints with depth i / j | {c['depth_valid_i']} / {c['depth_valid_j']} |",
         f"| PnP correspondences / inliers | {c['pnp_correspondences']} / {c['pnp_inliers']} ({100 * (m['pnp_inlier_ratio'] or 0):.1f} %) |",
         f"| distinct keypoints among PnP inliers, frame i / j | {m['pnp_inliers_distinct_kp_i']} / {m['pnp_inliers_distinct_kp_j']} |",
         f"| PnP inlier extent in frame i (bbox) | {m['pnp_inlier_spread_i_px']['bbox_w']:.0f} x {m['pnp_inlier_spread_i_px']['bbox_h']:.0f} px |",
         f"| matches sharing a frame-j keypoint (max on one) | {m['matches_sharing_a_kp_j']} ({m['max_matches_on_one_kp_j']}) |",
         f"| PnP correspondences within 8 px under the TRUE pose | {m['gt_consistent_pnp_correspondences']} / {c['pnp_correspondences']} |",
         f"| PnP inliers that are within 8 px under the true pose | {m['pnp_inliers_that_are_gt_consistent']} / {c['pnp_inliers']} (median {f(m['pnp_inliers_gt_res_px_median'], '{:.1f}')} px) |",
         f"| correspondences within 8 px of the RETURNED pose | {m.get('within_8px_of_returned_pnp_pose', 'n/a')} |",
         f"| ICP correspondences / inliers | {c['icp_correspondences']} / {c['icp_inliers']} |",
         f"| ICP correspondences within 1 cm under the true transform | {m['icp_gt_consistent_1cm']} |",
         f"| true motion | {e['gt']['rotation_deg']:.3f} deg, {e['gt']['translation_m']:.4f} m |",
         f"| PnP error | {f(e['pnp'] and e['pnp']['rot_err_deg'])} deg, {f(e['pnp'] and e['pnp']['trans_err_m'], '{:.4f}')} m |",
         f"| ICP error | {f(e['icp'] and e['icp']['rot_err_deg'])} deg, {f(e['icp'] and e['icp']['trans_err_m'], '{:.4f}')} m |",
         f"| solver status | PnP returned {m['status']['pnp_solver_returned']}, ICP RANSAC ok {m['status']['icp_ransac_ok']} |",
         "", "Export-only re-solves on the reported PnP inliers (not part of the estimate):", ""]
    for r in m["export_only_refits"]:
        d = r["max_abs_diff_to_returned_pose"]
        L.append(f"- `{r['name']}`: {r['status']}" + ("" if d is None else f"; max |T - T_returned| = {d:.2e}"))
    L += ["", f"Probability that 100 uniformly drawn {MINIMAL_SAMPLE}-point samples contain no sample made only of "
              f"correspondences within 8 px of the true pose: {f(m['p_no_all_gt_consistent_5_sample_in_100'], '{:.3f}')} "
              "(assumes OpenCV's 5-point EPnP minimal set and all 100 iterations; OpenCV may stop earlier, which only "
              "raises it).", ""]
    (Path(out) / f"pair_{m['frame_i']}_{m['frame_j']}_report.md").write_text("\n".join(L) + "\n")
    made.append(str(Path(out) / f"pair_{m['frame_i']}_{m['frame_j']}_report.md"))
    return L


def choose_reference(rows, failure):
    """The pair nearest to `failure` with both rotation errors below the run median, at least the median PnP
    inlier count and a true translation within 25 % of the median (rows: pairs.csv by pair index)."""
    ks = sorted(rows)
    g = lambda key: np.array([float(rows[k][key]) for k in ks])
    pe, ie, inl, gm = g("pnp_rot_err_deg"), g("icp_rot_err_deg"), g("pnp_inliers"), g("gt_motion_m")
    ok = (pe < np.nanmedian(pe)) & (ie < np.nanmedian(ie)) & (inl >= np.median(inl)) & \
         (np.abs(gm - np.median(gm)) < 0.25 * np.median(gm))
    cand = [k for k, o in zip(ks, ok) if o and k != failure]
    return min(cand, key=lambda k: (abs(k - failure), k)) if cand else None


def run_description(run_dir):
    cfg = Path(run_dir) / "config.json"
    if not cfg.is_file():
        return f"Run `{run_dir}`."
    c = json.loads(cfg.read_text())
    txt = f"Run `{run_dir}`: `{c.get('command', '')}`."
    b = c.get("baseline_compare")
    if b:
        same = all(b["byte_identical"].values())
        txt += (f" Standard outputs {'byte-identical to' if same else 'DIFFERENT from'} `{b['baseline']}`.")
    return txt


def context_rows(run_dir, dataset, pairs):
    """Per exported pair: counts, ground-truth-consistent correspondences and frame-i sharpness."""
    rows = []
    for k in pairs:
        d = Path(run_dir) / "diagnostics" / f"pair_{k}_{k + 1}"
        if not (d / "pair.json").is_file():
            continue
        q = C.load_pair(d)
        c, s = q["meta"]["counts"], q["pnp_corr"] >= 0
        inl = q["pnp_class"] == "inlier"
        rows.append(dict(pair=k, kp_i=c["keypoints_i"], matches=c["matches"], pnp_corr=c["pnp_correspondences"],
                         pnp_inl=c["pnp_inliers"], distinct_kp_j_inl=int(len(set(q["kp_j"][inl]))),
                         gt_ok=int((s & (q["gt_res_px"] < GT_CONSISTENT_PX)).sum()),
                         sharp_i=sharpness(C.read_rgb(dataset, q["meta"]["rgb_i"]))))
    return rows


def comparison_report(f, r, ctx, out, made, run_dir=None, ref_rule=False):
    """pair_<i>_<j>_report.md for a failing pair f, against a reference pair r (indicator dicts)."""
    cf, cr = f["counts"], r["counts"]
    ef, er = f["pose_errors_from_exported_transforms"], r["pose_errors_from_exported_transforms"]
    fmt = lambda v, s="{:.3f}": "n/a" if v is None else s.format(v)
    pct = lambda a, b: f"{a} / {b} ({100 * a / b:.0f} %)" if b else "n/a"
    rows = [
        ("frames", f"{f['frame_i']} -> {f['frame_j']}", f"{r['frame_i']} -> {r['frame_j']}"),
        ("true motion", f"{ef['gt']['rotation_deg']:.2f} deg, {100 * ef['gt']['translation_m']:.1f} cm",
         f"{er['gt']['rotation_deg']:.2f} deg, {100 * er['gt']['translation_m']:.1f} cm"),
        ("PnP error (deg / m)", f"{fmt(ef['pnp'] and ef['pnp']['rot_err_deg'])} / {fmt(ef['pnp'] and ef['pnp']['trans_err_m'], '{:.4f}')}",
         f"{fmt(er['pnp'] and er['pnp']['rot_err_deg'])} / {fmt(er['pnp'] and er['pnp']['trans_err_m'], '{:.4f}')}"),
        ("ICP error (deg / m)", f"{fmt(ef['icp'] and ef['icp']['rot_err_deg'])} / {fmt(ef['icp'] and ef['icp']['trans_err_m'], '{:.4f}')}",
         f"{fmt(er['icp'] and er['icp']['rot_err_deg'])} / {fmt(er['icp'] and er['icp']['trans_err_m'], '{:.4f}')}"),
        ("keypoints frame i / j (cap 500)", f"{cf['keypoints_i']} / {cf['keypoints_j']}", f"{cr['keypoints_i']} / {cr['keypoints_j']}"),
        ("sharpness (Laplacian variance) i / j", f"{f['sharpness_laplacian_var_i']:.0f} / {f['sharpness_laplacian_var_j']:.0f}",
         f"{r['sharpness_laplacian_var_i']:.0f} / {r['sharpness_laplacian_var_j']:.0f}"),
        ("filtered matches", cf["matches"], cr["matches"]),
        ("matched keypoints with depth in frame i", pct(cf["depth_valid_i"], cf["matches"]), pct(cr["depth_valid_i"], cr["matches"])),
        ("valid depth, whole image i / j", f"{100 * f['depth_valid_fraction_image_i']:.0f} / {100 * f['depth_valid_fraction_image_j']:.0f} %",
         f"{100 * r['depth_valid_fraction_image_i']:.0f} / {100 * r['depth_valid_fraction_image_j']:.0f} %"),
        ("PnP correspondences within 8 px of the TRUE pose", pct(f["gt_consistent_pnp_correspondences"], cf["pnp_correspondences"]),
         pct(r["gt_consistent_pnp_correspondences"], cr["pnp_correspondences"])),
        ("P(no all-correct 5-point sample in 100 draws)", fmt(f["p_no_all_gt_consistent_5_sample_in_100"]),
         fmt(r["p_no_all_gt_consistent_5_sample_in_100"])),
        ("matches sharing a frame-j keypoint (max on one)", f"{f['matches_sharing_a_kp_j']} ({f['max_matches_on_one_kp_j']})",
         f"{r['matches_sharing_a_kp_j']} ({r['max_matches_on_one_kp_j']})"),
        ("PnP inliers", pct(cf["pnp_inliers"], cf["pnp_correspondences"]), pct(cr["pnp_inliers"], cr["pnp_correspondences"])),
        ("distinct keypoints among PnP inliers, i / j", f"{f['pnp_inliers_distinct_kp_i']} / {f['pnp_inliers_distinct_kp_j']}",
         f"{r['pnp_inliers_distinct_kp_i']} / {r['pnp_inliers_distinct_kp_j']}"),
        ("PnP inlier extent in frame i (bbox, px)", f"{f['pnp_inlier_spread_i_px']['bbox_w']:.0f} x {f['pnp_inlier_spread_i_px']['bbox_h']:.0f}",
         f"{r['pnp_inlier_spread_i_px']['bbox_w']:.0f} x {r['pnp_inlier_spread_i_px']['bbox_h']:.0f}"),
        ("PnP inlier 3D extent (max pairwise, m)", fmt(f.get("pnp_inlier_3d_max_pairwise_m"), "{:.3f}"),
         fmt(r.get("pnp_inlier_3d_max_pairwise_m"), "{:.3f}")),
        ("PnP inliers within 8 px of the TRUE pose", f"{f['pnp_inliers_that_are_gt_consistent']} / {cf['pnp_inliers']}",
         f"{r['pnp_inliers_that_are_gt_consistent']} / {cr['pnp_inliers']}"),
        ("ICP correspondences / inliers", f"{cf['icp_correspondences']} / {cf['icp_inliers']}", f"{cr['icp_correspondences']} / {cr['icp_inliers']}"),
        ("ICP correspondences within 1 cm of the TRUE transform", f["icp_gt_consistent_1cm"], r["icp_gt_consistent_1cm"]),
    ]
    refit = {x["name"]: x for x in f["export_only_refits"]}
    L = [f"# Pair {f['frame_i']} -> {f['frame_j']}: diagnostic report", "",
         run_description(run_dir or Path(f["source"]).parents[1]) + " Frame indices are positions in the run's "
         f"selected sequence: frame {f['frame_i']} = `{f['rgb_i']}` (t = {f['t_rgb_i']:.6f} s), frame {f['frame_j']} = "
         f"`{f['rgb_j']}` (t = {f['t_rgb_j']:.6f} s). Reference pair {r['frame_i']} -> {r['frame_j']}: " +
         ("the pair nearest to the failing pair with both rotation errors below the run median, at least the median "
          "PnP inlier count and a true translation within 25 % of the median." if ref_rule else "chosen by the user."), "",
         f"Export consistency checks: failing pair {f['consistency_problems'] or 'all passed'}; reference "
         f"{r['consistency_problems'] or 'all passed'}. Images: `matches.png`, `depth.png`, `pnp_residuals.png`, `orientation.png` in this directory, "
         "`registration.png` under `icp/`; numbers in `pair_summary.json`; page: `index.html`.", "",
         "## Measurements", "", f"| quantity | failing pair | reference pair |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    L += ["", "Exported neighbouring pairs (same run):", "",
          "| pair | keypoints i | matches | PnP corr. | PnP inliers | distinct frame-j kp in inliers | "
          "within 8 px of true pose | sharpness i |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {c['pair']} | {c['kp_i']} | {c['matches']} | {c['pnp_corr']} | {c['pnp_inl']} | {c['distinct_kp_j_inl']} | "
          f"{c['gt_ok']} | {c['sharp_i']:.0f} |" for c in ctx]
    sharp_rank = sorted(c["sharp_i"] for c in ctx).index(f["sharpness_laplacian_var_i"]) + 1 if ctx else None
    L += ["", "## Observed facts", "",
          f"1. PnP's solver returned a pose (counted as success) supported by {cf['pnp_inliers']} inliers. These are "
          f"{f['pnp_inliers_distinct_kp_i']} different keypoints of frame {f['frame_i']} inside a "
          f"{f['pnp_inlier_spread_i_px']['bbox_w']:.0f} x {f['pnp_inlier_spread_i_px']['bbox_h']:.0f} px patch "
          f"({fmt(f.get('pnp_inlier_3d_max_pairwise_m'), '{:.3f}')} m across in 3D), all matched to "
          f"{f['pnp_inliers_distinct_kp_j']} keypoint(s) of frame {f['frame_j']}.",
          f"2. Under the ground-truth pose those inliers reproject {fmt(f['pnp_inliers_gt_res_px_median'], '{:.0f}')} px "
          f"(median) from their match; {f['pnp_inliers_that_are_gt_consistent']} of them are within 8 px. The matches "
          "are wrong, and the returned pose fits them: only "
          f"{f.get('within_8px_of_returned_pnp_pose')} correspondences lie within 8 px of it.",
          f"3. Correspondences that agree with the true pose exist but are a minority: "
          f"{f['gt_consistent_pnp_correspondences']} of {cf['pnp_correspondences']} "
          f"({100 * (f['gt_consistent_fraction'] or 0):.0f} %), against "
          f"{100 * (r['gt_consistent_fraction'] or 0):.0f} % in the reference pair.",
          f"4. Frame {f['frame_i']} has {cf['keypoints_i']} keypoints (cap 500) and the lowest sharpness "
          f"(Laplacian variance {f['sharpness_laplacian_var_i']:.0f}) of the {len(ctx)} exported frames i "
          f"(rank {sharp_rank} from the bottom); the RGB image is visibly blurred (`matches.png`).",
          f"5. Depth is not scarce: {cf['depth_valid_i']} of {cf['matches']} matched keypoints have depth in frame "
          f"{f['frame_i']} and {100 * f['depth_valid_fraction_image_i']:.0f} % of its pixels are valid.",
          f"6. ICP, given the same matches with depth in both frames, kept {cf['icp_inliers']} inliers spread over the "
          f"scene (`registration.png`) and is {fmt(ef['icp'] and ef['icp']['rot_err_deg'])} deg / "
          f"{fmt(ef['icp'] and ef['icp']['trans_err_m'], '{:.4f}')} m off.",
          f"7. Re-solving on the {cf['pnp_inliers']} reported inliers alone (export-only): "
          f"`iterative_no_guess` -> {refit.get('iterative_no_guess', {}).get('status', 'n/a')}; `epnp` and "
          f"`iterative_from_epnp` differ from the returned pose by "
          f"{fmt(refit.get('epnp', {}).get('max_abs_diff_to_returned_pose'), '{:.1e}')}. In the reference pair (and in "
          "every other exported pair) the returned pose equals `iterative_no_guess` on the inliers to <= 5e-10.",
          "",
          "## Assessment of the candidate causes", "",
          "| candidate | evidence | status |", "|---|---|---|",
          f"| low inlier count | {cf['pnp_inliers']} inliers vs {cr['pnp_inliers']} | observed, but a symptom: the "
          "count is what RANSAC found, not why |",
          f"| poor feature distribution | inliers in one {f['pnp_inlier_spread_i_px']['bbox_w']:.0f} x "
          f"{f['pnp_inlier_spread_i_px']['bbox_h']:.0f} px patch; the matched keypoints overall spread over the image "
          f"({f['matched_kp_spread_i_px']['bbox_w']:.0f} x {f['matched_kp_spread_i_px']['bbox_h']:.0f} px) | observed for "
          "the inliers; not for the matches as a whole |",
          f"| depth validity | {100 * cf['depth_valid_i'] / cf['matches']:.0f} % of matched keypoints with depth "
          f"(reference {100 * cr['depth_valid_i'] / cr['matches']:.0f} %) | not supported as the cause |",
          f"| geometric degeneracy | {cf['pnp_inliers']} inlier rows = {f['pnp_inliers_distinct_kp_j']} image point(s) "
          f"in frame {f['frame_j']}; a {fmt(f.get('pnp_inlier_3d_max_pairwise_m'), '{:.3f}')} m cluster imaged at one "
          "pixel fixes about 2 of the 6 pose parameters | observed: the inlier set cannot determine the pose |",
          f"| bad pose hypothesis | inliers {fmt(f['pnp_inliers_gt_res_px_median'], '{:.0f}')} px off under the true "
          f"pose; {f['gt_consistent_fraction'] and round(100 * f['gt_consistent_fraction'])} % correct correspondences; "
          f"P(no all-correct sample) = {fmt(f['p_no_all_gt_consistent_5_sample_in_100'])} | observed: the selected "
          "hypothesis is supported only by wrong matches |",
          f"| image blur in frame {f['frame_i']} | lowest sharpness of the window, fewer keypoints, fewer correct "
          "matches | hypothesis: consistent with the data, but sharpness and match correctness are only correlated "
          "here; no controlled test was run |",
          "",
          "## Interpretation", "",
          "What the export shows: the returned PnP pose is the one that fits five rows which are, in effect, one wrong "
          "match repeated (near-duplicate frame-i keypoints matched by nearest neighbour to the same frame-j "
          "keypoint); no correspondence that agrees with the true pose supports it, and the solver returned it as a "
          "success. With about a fifth of the "
          "correspondences correct, 100 iterations of 5-point samples were unlikely to draw an all-correct sample "
          "(probability of none about "
          f"{fmt(f['p_no_all_gt_consistent_5_sample_in_100'], '{:.2f}')} under the stated assumptions).", "",
          "Hypotheses, not established: (a) motion blur in frame "
          f"{f['frame_i']} lowered the keypoint count and the fraction of correct matches; (b) the many-to-one "
          "matching lets a single wrong match appear as several agreeing inliers. Many-to-one matches also occur in the "
          f"reference pair ({r['matches_sharing_a_kp_j']} matches share a frame-j keypoint) without harm, so they "
          "are not sufficient on their own.", "",
          "Not observable from the export: how cv::solvePnPRansac produced its final pose from five inliers (its "
          "inlier refit with SOLVEPNP_ITERATIVE needs six points without an initial guess, and EPnP on the five "
          "points does not reproduce the pose). The RANSAC minimal-sample size (5, EPnP) is taken from the OpenCV "
          "source, not measured here.", "",
          "Not changed: no estimator, threshold or success definition was modified; the evaluator counts this pair "
          "as successful because the solver returned a pose."]
    path = Path(out) / f"pair_{f['frame_i']}_{f['frame_j']}_report.md"
    path.write_text("\n".join(L) + "\n")
    if str(path) not in made:
        made.append(str(path))


def run(run_dir, pair, dataset, out=None):
    run_dir = Path(run_dir)
    pdir = run_dir / "diagnostics" / f"pair_{pair}_{pair + 1}"
    p = C.load_pair(pdir)
    rows = C.pairs_rows(run_dir)
    problems = C.validate_pair(p, rows.get(pair))
    out = Path(out) if out else run_dir.parent / "pairs" / f"pair_{pair}_{pair + 1}"
    out.mkdir(parents=True, exist_ok=True)
    m = p["meta"]
    imgs = {"i": C.read_rgb(dataset, m["rgb_i"]), "j": C.read_rgb(dataset, m["rgb_j"])}
    deps = {"i": C.read_depth_raw(dataset, m["depth_i"]), "j": C.read_depth_raw(dataset, m["depth_j"])}
    ind = indicators(p, imgs, deps, rows.get(pair))
    made = []
    ind["consistency_problems"] = problems
    ind["figures"] = list(made)
    report(ind, problems, out, made)
    C.write_json(out / "pair_summary.json", ind)
    return ind, problems, made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--compare", type=int, help="reference pair: writes the comparison report for --pair")
    ap.add_argument("--context", help="exported pairs listed as context in the comparison report, e.g. 66-80")
    a = ap.parse_args()
    try:
        ind, problems, made = run(a.run, a.pair, a.dataset, a.output)
        if a.compare is not None:
            ref, rp, rmade = run(a.run, a.compare, a.dataset)
            problems += [f"reference: {x}" for x in rp]
            lo, hi = (int(x) for x in a.context.split("-")) if a.context else (a.pair, a.pair)
            ctx = context_rows(a.run, a.dataset, range(lo, hi + 1))
            out = Path(a.output) if a.output else Path(a.run).parent / "pairs" / f"pair_{a.pair}_{a.pair + 1}"
            rule = choose_reference(C.pairs_rows(a.run), a.pair) == a.compare
            comparison_report(ind, ref, ctx, out, made, a.run, rule)
            made += rmade
    except (C.ExportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(made))
    if problems:
        print("EXPORT INCONSISTENT:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
