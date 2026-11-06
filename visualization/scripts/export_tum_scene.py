#!/usr/bin/env python3
"""Export one TUM evaluator run for the Blender TUM scene (runs OUTSIDE Blender, `results` pixi environment).

Usage (repository root):
  PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/export_tum_scene.py \
      --run results/runs/stress/xyz_session3_pnp_validity/n100_stride4 [--dataset data/rgbd_dataset_freiburg1_xyz] \
      [--out results/runs/blender/<sequence>/<run name>] [--force]

Writes into --out (default results/runs/blender/<sequence>/<run>/, refused if it exists unless --force):
  trajectory.json   per frame: index, RGB/depth timestamps and files, T_wc of GT, PnP and ICP (centre + quaternion
                    qx qy qz qw, exactly the trajectory.csv values), errors, held flags; per pair: solver status,
                    acceptance and reason (when the evaluator recorded them); frame gaps; skipped frames (from the
                    evaluator log); intrinsics; one global SE(3) per estimate for an optional aligned view
  cloud_<density>_<mode>.ply   binary PLY, float x y z (metres, TUM world frame) + uchar r g b, one per pose
                    source mode gt / pnp / icp and density preview / render (tum_reconstruction.py)
  manifest_<density>_<mode>.json   pose source, frames used / skipped (with reasons), timestamps, point counts,
                    sampling, intrinsics, conventions, source files
  comparison_<density>.json   per-point displacement of the PnP / ICP clouds from the GT cloud (same pixels) and
                    occupied 1 cm voxels per mode
  depth/frame_<k>.png   colourised depth previews (0..4 m viridis, no measurement magenta), 320 x 240
  matches/pair_<i>_<j>.png   real feature correspondences (pair_view.py) for pairs with a match-level export
  export.json        what was written, sampling parameters, source files

Point clouds: see tum_reconstruction.py. Three independent reconstructions from the SAME sampled pixels
(frames 0, every <every>-th and the last; a <step> px grid; raw depth > 0), each frame placed with the pose of
one source: X_world = R_wc X_cam + t_wc. Estimated modes exclude frames whose pose is held (no estimate of its
own) and never read ground truth. (Before 2026-10-24 the exporter wrote ONE cloud placed with GT poses and
thinned it with a world-space voxel filter.)
"""

import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "evaluation"))
sys.path.insert(0, str(ROOT / "tools" / "evaluation" / "diagnostics"))
sys.path.insert(0, str(ROOT / "tools" / "evaluation" / "dashboard_app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tum_metrics as tm  # noqa: E402
import common as C  # noqa: E402

DENSITY = {"preview": dict(every=10, step=8), "render": dict(every=3, step=4)}
DEPTH_MAX_M = 4.0


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["r"], rec["g"], rec["b"] = rgb.T
        f.write(rec.tobytes())


def reconstructions(run, run_dir, dataset, K, out, density, every, step, include=()):
    import tum_reconstruction as TR
    read = lambda k: (C.read_depth_raw(dataset, run["depth"][k]), C.read_rgb(dataset, run["rgb"][k]))
    recon, sel, samples = TR.reconstruct(run, read, K, every, step, include=include)
    info = {}
    for mode, r in recon.items():
        f = f"cloud_{density}_{mode}.ply"
        write_ply(out / f, r["xyz"].astype(np.float32), r["rgb"].astype(np.uint8))
        man = {"dataset": str(dataset.resolve()), "sequence": dataset.name, "run": str(run_dir),
               "pose_source": mode, "pose_columns": f"trajectory.csv {mode}_x,y,z,qx,qy,qz,qw",
               "uses_ground_truth": mode == "gt",
               "source_files": {k: str((run_dir / k).resolve()) for k in ("trajectory.csv", "pairs.csv", "frames.csv")},
               "run_config": str((run_dir / "config.json").resolve()) if (run_dir / "config.json").is_file() else None,
               "frames_selected": sel, "frames_used": r["frames_used"],
               "timestamps_used": {str(k): float(run["t"][k]) for k in r["frames_used"]},
               "frames_processed": len(r["frames_used"]), "frames_skipped": r["frames_skipped"],
               "valid_depth_pixels_in_used_frames": r["valid_depth_pixels"], "points_after_sampling": r["points"],
               "per_frame_points": r["per_frame_points"],
               "sampling": {"every_frame": every, "pixel_step": step, "pixel_offset": step // 2,
                            "included_frames": list(include),
                            "rule": "pixels (offset + step a, offset + step b) with raw depth > 0; no world-space filter",
                            "per_frame_pixel_sha256": {str(k): samples[k]["px_sha256"] for k in sel}},
               "intrinsics": K, "depth_scale": f"metres = raw / {K['depth_factor']:g}; raw 0 = invalid",
               "convention": "X_cam = ((u-cx) z/fx, (v-cy) z/fy, z); X_world = R_wc X_cam + t_wc (T_wc of the pose "
                             "source at that frame); TUM world frame, metres",
               "cloud_file": f}
        (out / f"manifest_{density}_{mode}.json").write_text(json.dumps(man, indent=2))
        info[mode] = {"file": f, "points": r["points"], "frames_used": len(r["frames_used"]),
                      "frames_skipped": [x["frame"] for x in r["frames_skipped"]]}
    cmp = TR.compare(recon, samples)
    (out / f"comparison_{density}.json").write_text(json.dumps(cmp, indent=2))
    return info, cmp


def depth_preview(raw, path):
    import matplotlib
    from PIL import Image
    z = raw / 5000.0
    rgb = matplotlib.colormaps["viridis"](np.clip(z / DEPTH_MAX_M, 0, 1))[..., :3]
    rgb[raw == 0] = (0.84, 0.15, 0.56)
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((320, 240), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="tum_trajectory_eval output directory")
    ap.add_argument("--dataset", help="sequence directory (default: from the run's config.json / command)")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-frames", type=lambda s: [int(x) for x in s.split(",") if x], default=[],
                    help="frames added to the selection of every mode, e.g. 73,74")
    a = ap.parse_args()
    run_dir = Path(a.run)
    import data as dash
    rid = run_dir.resolve().relative_to((ROOT / "results" / "runs").resolve()).as_posix()
    meta = dash._run_meta(rid)
    dataset = Path(a.dataset or ROOT / meta["dataset"])
    if not (dataset / "rgb.txt").is_file():
        print(f"error: sequence {dataset} not found (nothing is downloaded)", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else ROOT / "results" / "runs" / "blender" / dataset.name / rid.replace("/", "__")
    if out.exists() and not a.force:
        print(f"error: {out} exists; pass --force or another --out", file=sys.stderr)
        return 2
    if out.exists():
        shutil.rmtree(out)
    (out / "depth").mkdir(parents=True)
    run = tm.load_run(run_dir)
    problems = tm.check_consistency(run)
    if problems:
        print(f"error: trajectory files inconsistent: {problems}", file=sys.stderr)
        return 1
    K = C.run_intrinsics(run_dir, str(dataset))
    n = run["n"]
    import csv
    traj_rows = list(csv.DictReader(open(run_dir / "trajectory.csv")))
    frames = []
    for k in range(n):
        r = traj_rows[k]
        f = {"k": k, "t_rgb": float(run["t_rgb"][k]), "t_depth": float(run["t_depth"][k]),
             "rgb": str((dataset / run["rgb"][k]).resolve()), "depth": str((dataset / run["depth"][k]).resolve()),
             "depth_preview": f"depth/frame_{k:04d}.png"}
        for m in ("gt", "pnp", "icp"):  # exactly the CSV numbers
            f[m] = {"p": [float(r[f"{m}_{c}"]) for c in "xyz"], "q_xyzw": [float(r[f"{m}_q{c}"]) for c in "xyzw"]}
        for m in ("pnp", "icp"):
            f[m].update(pos_err_m=float(r[f"{m}_translation_error"]), rot_err_deg=float(r[f"{m}_rotation_error"]),
                        held=bool(run[f"{m}_held"][k]))
        frames.append(f)
        depth_preview(C.read_depth_raw(dataset, run["depth"][k]), out / f["depth_preview"])
    pairs = []
    for k in range(n - 1):
        fin = lambda x: None if not np.isfinite(x) else float(x)
        p = {"pair": k, "pnp_accepted": bool(run["pnp_ok"][k]), "icp_ok": bool(run["icp_ok"][k]),
             "gt_motion_deg": float(run["gt_motion_deg"][k]), "gt_motion_m": float(run["gt_motion_m"][k]),
             "pnp_rot_err_deg": fin(run["pnp_rot_err_deg"][k]), "pnp_trans_err_m": fin(run["pnp_trans_err_m"][k]),
             "icp_rot_err_deg": fin(run["icp_rot_err_deg"][k]), "icp_trans_err_m": fin(run["icp_trans_err_m"][k]),
             "pnp_returned_rot_err_deg": fin(run["pnp_returned_rot_err_deg"][k]) if "pnp_returned_rot_err_deg" in run else None,
             "pnp_returned_trans_err_m": fin(run["pnp_returned_trans_err_m"][k]) if "pnp_returned_trans_err_m" in run else None,
             "pnp_inliers": int(run["pnp_inliers"][k]), "pnp_correspondences": int(run["pnp_correspondences"][k])}
        if "pnp_solver_returned" in run:
            p["pnp_solver_returned"] = bool(run["pnp_solver_returned"][k])
        pairs.append(p)
    with open(run_dir / "pairs.csv") as fh:
        for row, p in zip(csv.DictReader(fh), pairs):
            if row.get("pnp_reject_reason"):
                p["pnp_reject_reason"] = row["pnp_reject_reason"]
    dt = np.diff(run["t"])
    med = float(np.median(dt))
    gaps = [{"pair": int(k), "dt_s": float(dt[k])} for k in np.nonzero(dt > 1.5 * med)[0]]
    skipped = []
    log = run_dir / "stdout.log"
    if log.is_file():
        skipped = [l.strip()[len("skipped "):] for l in log.read_text().splitlines() if l.strip().startswith("skipped ")]
    aligned = {}
    for m in ("pnp", "icp"):
        R, t = tm.umeyama_rigid(run[f"{m}_p"], run["gt_p"])
        aligned[m] = {"R": R.tolist(), "t": t.tolist(), "rotation_deg": C.rot_deg(R)}
    clouds, comparison = {}, {}
    for name, d in DENSITY.items():
        clouds[name], comparison[name] = reconstructions(run, run_dir, dataset, K, out, name, include=a.include_frames, **d)
    matches = {}
    src = meta.get("export_source")
    if src:
        import pair_view as PV
        sd = ROOT / "results" / "runs" / src
        for k in dash._run_meta(src)["exports"]:
            made, pr = PV.run(sd, k, str(dataset), out / "matches" / f"pair_{k}_{k + 1}")
            if not pr:
                dst = out / "matches" / f"pair_{k}_{k + 1}.png"
                shutil.move(made[0], dst)
                shutil.rmtree(out / "matches" / f"pair_{k}_{k + 1}")
                matches[str(k)] = dst.relative_to(out).as_posix()
    doc = {"run": rid, "dataset": str(dataset.resolve()), "sequence": dataset.name, "intrinsics": K,
           "convention": "T_wc: X_world = R X_cam + t (TUM world frame, metres); camera axes x right, y down, z forward; "
                         "quaternion order qx qy qz qw",
           "frames": frames, "pairs": pairs, "frame_gaps": gaps, "skipped_frames_from_selection": skipped,
           "median_frame_interval_s": med, "aligned_for_optional_view": aligned, "clouds": clouds,
           "cloud_comparison": comparison,
           "matches": matches, "match_source_run": src,
           "summary": {m: {"final_pos_err_m": frames[-1][m]["pos_err_m"], "final_rot_err_deg": frames[-1][m]["rot_err_deg"]}
                       for m in ("pnp", "icp")}}
    (out / "trajectory.json").write_text(json.dumps(doc))
    (out / "export.json").write_text(json.dumps({k: v for k, v in doc.items() if k not in ("frames", "pairs")}, indent=2))
    print(f"wrote {out}: {n} frames, {len(matches)} match images")
    for name in clouds:
        for mode, c in clouds[name].items():
            cm = comparison[name][mode]
            dv = cm.get("displacement_vs_gt_m")
            print(f"  {name:7s} {mode}: {c['points']} points from {c['frames_used']} frames (skipped {c['frames_skipped']}); "
                  f"occupied 1 cm voxels per 1000 points {cm['occupied_voxels_per_1000_points']:.1f}"
                  + (f"; displacement vs GT median {100 * dv['median']:.1f} cm, p90 {100 * dv['p90']:.1f} cm, "
                     f"max {dv['max']:.2f} m, >5 cm {100 * dv['fraction_over_5cm']:.0f} %" if dv else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
