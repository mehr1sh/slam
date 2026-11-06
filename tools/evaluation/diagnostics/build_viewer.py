#!/usr/bin/env python3
"""Build the interactive viewer of one evaluator run (open <output>/index.html in a browser; works offline).

Usage (repository root):
  python3 tools/evaluation/diagnostics/build_viewer.py --run <evaluator dir> \
          --dataset data/rgbd_dataset_freiburg1_xyz [--output DIR] [--pair 73] [--scene-every 10] [--frames 73,74]
Default output: <run>/../viewer/ (index.html, data.js, frames/frame_<k>.jpg, viewer.json).

Data, all from the run and the dataset (nothing estimated here):
  frames      every frame of the run: index, RGB timestamp, the RGB image as JPEG (quality 85)
  poses       T_wc of GT, PnP, ICP per frame (trajectory.csv), raw; and for PnP / ICP one rigid transform
              per method fitted to all camera centres (tum_metrics.umeyama_rigid, no scale), as a separate
              "aligned" option; held poses (failed pair) are flagged, never interpolated
  scene       depth of frames 0, every --scene-every-th, the last and --frames, on an 8 px grid
              (pixels (4 + 8a, 4 + 8b) with raw depth > 0), back-projected with the official intrinsics,
              stored in each camera's frame to 1 mm; placed in the world by the chosen method's pose
  pairs       every exported pair (diagnostics/pair_<i>_<j>, validated): keypoints, all matches with the
              PnP class solvePnPRansac returned, the ICP correspondences with EstimateRigidRansac's flags,
              the transforms, and frames i and j on a 5 px grid for the registration view
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import viewer_html as V  # noqa: E402

CODE = {"inlier": "I", "outlier": "O", "no_depth_i": "D", "unclassified": "U"}


def jpeg(src_rgb, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(src_rgb).save(path, quality=85)


def r4(a, d=4):
    return [round(float(x), d) for x in np.asarray(a).reshape(-1)]


def pair_payload(p, row, detail):
    m = p["meta"]
    q = p["icp_corr"] >= 0
    Pi = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][q]
    Pj = np.c_[p["Xj_x"], p["Xj_y"], p["Xj_z"]][q]
    lab = p["icp_class"][q]
    n_rej = int((lab == "rejected").sum())
    thr = m["config"]["icp"]["threshold_m"]
    reg = {}
    G = np.linalg.inv(p["T_j_from_i_gt"])
    for state, T in (("before", np.eye(4)), ("icp", p["T_i_from_j_icp_native"]), ("gt", G)):
        if T is None or not q.any():
            reg[state] = None
            continue
        r = np.linalg.norm(Pi - (Pj @ T[:3, :3].T + T[:3, 3]), axis=1)
        inl = r[lab == "inlier"]
        reg[state] = {"inlier_median_m": float(np.median(inl)) if len(inl) else None,
                      "within_threshold": int((r < thr).sum())}
    T = lambda key: None if p[key] is None else p[key].tolist()
    return {
        "pair": m["pair"], "i": m["frame_i"], "j": m["frame_j"], "t_i": m["t_rgb_i"], "t_j": m["t_rgb_j"],
        "counts": m["counts"], "status": m["status"], "icp_threshold_m": thr, "detail": detail,
        "err": {k: (None if row is None or not math.isfinite(float(row[k])) else float(row[k]))
                for k in ("pnp_rot_err_deg", "pnp_trans_err_m", "icp_rot_err_deg", "icp_trans_err_m")},
        "kp_i": r4(p["kp_i_xy"], 2), "kp_j": r4(p["kp_j_xy"], 2),
        "m": {"ui": r4(p["u_i"], 2), "vi": r4(p["v_i"], 2), "uj": r4(p["u_j"], 2), "vj": r4(p["v_j"], 2),
              "cls": "".join(CODE[c] for c in p["pnp_class"])},
        "icp": {"pi": r4(Pi, 5), "pj": r4(Pj, 5), "cls": "".join("I" if c == "inlier" else "R" for c in lab),
                "rej_every": max(1, int(math.ceil(n_rej / 20)))},
        "T_gt_ji": T("T_j_from_i_gt"), "T_pnp_ji": T("T_j_from_i_pnp"), "T_icp_ji": T("T_j_from_i_icp"),
        "T_icp_ij": T("T_i_from_j_icp_native"), "reg": reg}


def build(run_dir, dataset, out=None, pair=None, scene_every=10, extra=(), notes=()):
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir.parent / "viewer"
    run = C.tm.load_run(run_dir)
    K = C.run_intrinsics(run_dir, dataset)
    n = run["n"]
    rows = C.pairs_rows(run_dir)
    frames = []
    for k in range(n):
        name = f"frames/frame_{k:04d}.jpg"
        jpeg(C.read_rgb(dataset, run["rgb"][k]), out / name)
        frames.append({"k": k, "t": float(run["t"][k]), "img": name, "rgb": run["rgb"][k]})
    poses, align = {}, {}
    for m in ("gt", "pnp", "icp"):
        poses[m] = {"raw": V.pose_list(run[f"{m}_R"], run[f"{m}_p"]), "aligned": None}
        if m != "gt":
            R, t = C.tm.umeyama_rigid(run[f"{m}_p"], run["gt_p"])
            poses[m]["aligned"] = V.pose_list(np.einsum("ij,njk->nik", R, run[f"{m}_R"]), run[f"{m}_p"] @ R.T + t)
            align[m] = {"rotation_deg": C.rot_deg(R), "translation_m": float(np.linalg.norm(t))}
    held = {"gt": [False] * n, "pnp": [bool(x) for x in run["pnp_held"]], "icp": [bool(x) for x in run["icp_held"]]}
    scene_frames = sorted(set(range(0, n, scene_every)) | {n - 1} | {k for k in extra if 0 <= k < n})
    scene, gt_pts = [], []
    for k in scene_frames:
        xyz, px = C.backproject(C.read_depth_raw(dataset, run["depth"][k]), K, step=8, offset=4)
        rgb = C.read_rgb(dataset, run["rgb"][k])[px[:, 1], px[:, 0]]
        scene.append(dict(V.pack_cloud(xyz, rgb), k=k))
        gt_pts.append(xyz @ run["gt_R"][k].T + run["gt_p"][k])
    pair_dirs = sorted((run_dir / "diagnostics").glob("pair_*_*"), key=lambda d: int(d.name.split("_")[1]))
    pairs, pair_clouds, problems = [], {}, {}
    for d in pair_dirs:
        p = C.load_pair(d)
        k = p["meta"]["pair"]
        pr = C.validate_pair(p, rows.get(k))
        if pr:
            problems[k] = pr
            continue
        detail = out.parent / "pairs" / f"pair_{k}_{k + 1}" / "index.html"
        pairs.append(pair_payload(p, rows.get(k), f"../pairs/pair_{k}_{k + 1}/index.html" if detail.is_file() else None))
        for f, side in ((p["meta"]["frame_i"], "i"), (p["meta"]["frame_j"], "j")):
            if str(f) not in pair_clouds:
                xyz, px = C.backproject(C.read_depth_raw(dataset, p["meta"][f"depth_{side}"]), K, step=5, offset=2)
                rgb = C.read_rgb(dataset, p["meta"][f"rgb_{side}"])[px[:, 1], px[:, 0]]
                pair_clouds[str(f)] = V.pack_cloud(xyz, rgb)
    P = np.concatenate(gt_pts)
    centre = (np.median(P, axis=0) + run["gt_p"].mean(0)) / 2
    # default scene view: from behind and above the mean GT camera, along its mean optical axis
    axis = run["gt_R"][:, :, 2].mean(0)
    axis /= np.linalg.norm(axis)
    eye = run["gt_p"].mean(0) - 1.6 * axis + np.array([0, 0, 0.8])
    d = eye - centre
    dist = float(np.linalg.norm(d))
    view = {"yaw": math.degrees(math.atan2(d[1], d[0])), "pitch": math.degrees(math.asin(d[2] / dist)), "dist": dist,
            "tx": float(centre[0]), "ty": float(centre[1]), "tz": float(centre[2])}
    default_pair = pair if pair is not None and any(q["pair"] == pair for q in pairs) else (pairs[0]["pair"] if pairs else None)
    data = {
        "run": {"title": f"{Path(dataset).name} / {run_dir.parent.name}: {n} frames", "dir": str(run_dir)},
        "K": {k: K[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}, "frames": frames, "poses": poses,
        "alignment": align, "held": held, "scene_clouds": scene, "pair_clouds": pair_clouds, "pairs": pairs,
        "defaults": {"pair": default_pair, "frame": (default_pair + 1) if default_pair is not None else 0,
                     "scene_view": view},
        "notes": [f"Scene: depth of frames {scene_frames} on an 8 px grid, {K['sensor']} intrinsics, depth = raw / "
                  f"{K['depth_factor']:g} m, no undistortion (as the evaluator).",
                  "Aligned view: one rigid transform per method (no scale): " +
                  "; ".join(f"{m}: {v['rotation_deg']:.1f} deg, {v['translation_m']:.3f} m" for m, v in align.items()) + ".",
                  "Held poses (failed pairs) are flagged, never interpolated. Drag: rotate; shift-drag: pan; wheel: "
                  "zoom; click a camera (scene mode) to select its frame."] + list(notes)}
    V.write(out, data)
    C.write_json(out / "viewer.json", {"run": str(run_dir), "dataset": str(dataset), "frames": n,
                                       "scene_frames": scene_frames, "scene_points": int(sum(c["n"] for c in scene)),
                                       "pairs": [q["pair"] for q in pairs], "skipped_inconsistent_pairs": problems,
                                       "alignment": align})
    return out, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--pair", type=int, help="pair selected when the page opens")
    ap.add_argument("--scene-every", type=int, default=10)
    ap.add_argument("--frames", default="", help="extra frames for the scene cloud, e.g. 73,74")
    a = ap.parse_args()
    try:
        out, problems = build(a.run, a.dataset, a.output, a.pair, a.scene_every,
                              [int(x) for x in a.frames.split(",") if x])
    except (C.ExportError, C.tm.SchemaError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(out / "index.html")
    if problems:
        print(f"skipped inconsistent exports: {problems}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
