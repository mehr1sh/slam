#!/usr/bin/env python3
"""Generate all figures, CSVs and summary tables in results/ from the ACTUAL
outputs of this project's C++ pipeline.

Run (after `pixi run build`):
    pixi run -e results results

What it does:
  1. Runs build/slam_trajectory_test on a temporary COPY of
     data/synthetic_bunny/ with the export flags
     (--export-dir results/data/cpp_export --export-pair 0), and checks that
     this run reproduces the committed trajectory files byte-for-byte.
     (--skip-cpp reuses an existing results/data/cpp_export/.)
  2. Reads the dataset (intrinsics, RGB/depth PNGs, mesh), the trajectory
     files (slam_trajectory.csv, pnp/icp_trajectory.txt, groundtruth.txt) and
     the C++ per-pair export.
  3. Cross-checks the numbers: errors re-derived from the poses, the
     trajectories re-accumulated from the exported relative motions, the
     back-projection of one keypoint -- and aborts if anything disagrees.
  4. Writes results/data/*.csv, results/tables/summary.{md,json} and
     results/figures/fig*.{png,pdf}.

No number shown in a figure is typed in by hand: every value comes from the
files above. The only fixed values are the colors/layout and the structural
constants of the ORB matching rule / RANSAC settings used for LABELS, which
are read from the C++ source text for display.

Terminology: this is a synthetic RGB-D trajectory experiment comparing ground
truth, PnP-based trajectory estimation, and the project's
feature-correspondence-based 3D-to-3D ICP (SVD + g2o) -- not full SLAM, not
dense ICP, no loop closure, no global optimization.
"""

import argparse
import csv
import filecmp
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "synthetic_bunny"
MESH = ROOT / "data" / "meshes" / "bunny" / "bun_zipper.ply"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
DATA = RESULTS / "data"
TABLES = RESULTS / "tables"
EXPORT = DATA / "cpp_export"
EXPORT_PAIR = 0

# Same hues as the Blender visualization (visualization/scripts/build_scene.py METHOD_STYLES).
C = {"GT": (0.15, 0.82, 0.45), "PnP": (1.00, 0.62, 0.10), "ICP": (0.85, 0.40, 0.95)}
# Slightly darker variants for text/thin lines on a white background.
CD = {"GT": (0.05, 0.55, 0.28), "PnP": (0.85, 0.45, 0.00), "ICP": (0.62, 0.22, 0.72)}
GREY = "0.45"


# ----------------------------------------------------------------------------
# Small math helpers (quaternion (x, y, z, w) -> R, SE(3) as 4x4)
# ----------------------------------------------------------------------------

def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def se3(t, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*q)
    T[:3, 3] = t
    return T


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotations -- the same formula as
    tests/report_utils.hpp RotationAngleDeg(): acos((tr(Ra^T Rb) - 1) / 2)."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def read_intrinsics():
    w, h, fx, fy, cx, cy = (DATASET / "intrinsics.txt").read_text().split()[:6]
    return dict(width=int(w), height=int(h), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def read_tum(path):
    poses = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        poses.append(se3(v[1:4], v[4:8]))
    return poses


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ply_mesh(path):
    """Minimal ASCII PLY reader (vertex x,y,z + face lists), mirroring the
    C++ reader's behavior: extra vertex properties ignored, polygons
    fan-triangulated."""
    lines = Path(path).read_text().splitlines()
    nv = nf = 0
    vprops = []
    in_vertex = False
    end = 0
    for k, l in enumerate(lines):
        if l.startswith("element vertex"):
            nv, in_vertex = int(l.split()[2]), True
        elif l.startswith("element face"):
            nf, in_vertex = int(l.split()[2]), False
        elif l.startswith("property") and in_vertex:
            vprops.append(l.split()[-1])
        elif l.strip() == "end_header":
            end = k + 1
            break
    ix, iy, iz = vprops.index("x"), vprops.index("y"), vprops.index("z")
    V = np.array([[float(t) for t in lines[end + k].split()] for k in range(nv)])[:, [ix, iy, iz]]
    tris = []
    for k in range(nf):
        idx = [int(t) for t in lines[end + nv + k].split()]
        n, ids = idx[0], idx[1:]
        tris += [(ids[0], ids[m], ids[m + 1]) for m in range(1, n - 1)]
    return V, np.array(tris), nf


def read_png(path):
    return np.array(Image.open(path))


def source_constants():
    """Display-only: read the ORB filter rule and RANSAC arguments from the
    C++ source so labels never drift from the code."""
    feat = (ROOT / "src" / "features" / "features.cpp").read_text()
    traj = (ROOT / "tests" / "synthetic" / "slam_trajectory_test.cpp").read_text()
    filt = re.search(r"distance <= max\((\d+) \* min_dist, ([\d.]+)\)", feat)
    ransac = re.search(r"solvePnPRansac\([^;]*false, (\d+), ([\d.]+)f?, ([\d.]+), inliers\)", traj)
    return dict(filter_factor=int(filt.group(1)), filter_floor=float(filt.group(2)),
                ransac_iters=int(ransac.group(1)), ransac_px=float(ransac.group(2)),
                ransac_conf=float(ransac.group(3)))


# ----------------------------------------------------------------------------
# Step 1: run the C++ export
# ----------------------------------------------------------------------------

def run_cpp_export():
    exe = BUILD / "slam_trajectory_test"
    if not exe.exists():
        sys.exit(f"{exe} not found -- run `pixi run build` first")
    EXPORT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "synthetic_bunny"
        shutil.copytree(DATASET, ds)
        print(f"[results] running {exe.name} on a temporary copy of {DATASET.relative_to(ROOT)} ...")
        out = subprocess.run([str(exe), str(ds), "--export-dir", str(EXPORT),
                              "--export-pair", str(EXPORT_PAIR)],
                             cwd=BUILD, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(out.stdout[-2000:] + out.stderr[-2000:])
        (EXPORT / "slam_trajectory_test.log").write_text(out.stdout)
        for f in ("pnp_trajectory.txt", "icp_trajectory.txt", "slam_trajectory.csv"):
            same = filecmp.cmp(ds / f, DATASET / f, shallow=False)
            print(f"[results]   {f}: {'reproduced identically' if same else 'DIFFERS from the committed file'}")
            if not same:
                sys.exit("The C++ run did not reproduce the committed trajectory files -- "
                         "regenerate data/synthetic_bunny/ (see COMMANDS.md) before plotting.")


# ----------------------------------------------------------------------------
# Step 2+3: metrics and cross-checks
# ----------------------------------------------------------------------------

def check(label, ok, detail=""):
    print(f"[check] {'OK  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        sys.exit(f"cross-check failed: {label}")


def build_metrics():
    K = read_intrinsics()
    rows = read_csv(DATASET / "slam_trajectory.csv")
    pairs = read_csv(EXPORT / "pair_metrics.csv")
    n = len(rows)

    poses = {k: [] for k in ("GT", "PnP", "ICP")}
    for r in rows:
        for key, p in (("GT", "gt"), ("PnP", "pnp"), ("ICP", "icp")):
            poses[key].append(se3([float(r[f"{p}_{a}"]) for a in "xyz"],
                                  [float(r[f"{p}_q{a}"]) for a in "xyzw"]))

    # (a) CSV poses == TUM trajectory files
    for key, f in (("GT", "groundtruth.txt"), ("PnP", "pnp_trajectory.txt"), ("ICP", "icp_trajectory.txt")):
        tum = read_tum(DATASET / f)
        d = max(np.abs(a[:3, 3] - b[:3, 3]).max() for a, b in zip(poses[key], tum))
        check(f"slam_trajectory.csv {key} positions == {f}", len(tum) == n and d < 1e-5, f"(max {d:.1e} m)")

    # (b) the CSV's absolute errors == errors re-derived from its own poses
    for key, p in (("PnP", "pnp"), ("ICP", "icp")):
        dt = max(abs(np.linalg.norm(poses["GT"][k][:3, 3] - poses[key][k][:3, 3])
                     - float(rows[k][f"{p}_translation_error"])) for k in range(n))
        dr = max(abs(rot_angle_deg(poses["GT"][k][:3, :3], poses[key][k][:3, :3])
                     - float(rows[k][f"{p}_rotation_error"])) for k in range(n))
        check(f"{key} absolute errors re-derived from poses", dt < 1e-5 and dr < 1e-2,
              f"(max diff {dt:.1e} m, {dr:.1e} deg; CSV has 6 decimals)")

    # (c) exported relative motions, chained with T_wc[i+1] = T_wc[i] * T_rel^-1,
    #     reproduce the trajectory files (the C++ accumulation formula)
    rel = {"GT": [], "PnP": [], "ICP": []}
    for pr in pairs:
        for key, p in (("GT", "gt"), ("PnP", "pnp"), ("ICP", "icp")):
            rel[key].append(se3([float(pr[f"{p}_rel_t{a}"]) for a in "xyz"],
                                [float(pr[f"{p}_rel_q{a}"]) for a in "xyzw"]))
    for key in ("PnP", "ICP", "GT"):
        T = poses["GT"][0].copy()
        worst = 0.0
        for k in range(n - 1):
            T = T @ np.linalg.inv(rel[key][k])
            worst = max(worst, np.linalg.norm(T[:3, 3] - poses[key][k + 1][:3, 3]))
        check(f"{key}: chaining the exported relative motions reproduces the trajectory",
              worst < 2e-5, f"(max {worst:.1e} m)")

    # (d) exported per-pair relative errors == errors of the exported relative poses
    for key, p in (("PnP", "pnp"), ("ICP", "icp")):
        dt = max(abs(np.linalg.norm(rel["GT"][k][:3, 3] - rel[key][k][:3, 3])
                     - float(pairs[k][f"{p}_relative_translation_error_m"])) for k in range(n - 1))
        dr = max(abs(rot_angle_deg(rel["GT"][k][:3, :3], rel[key][k][:3, :3])
                     - float(pairs[k][f"{p}_relative_rotation_error_deg"])) for k in range(n - 1))
        check(f"{key} per-pair relative errors consistent with exported relative poses",
              dt < 1e-6 and dr < 1e-4, f"(max diff {dt:.1e} m, {dr:.1e} deg)")

    return K, rows, pairs, poses, rel


def write_csvs(rows, pairs):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "trajectory_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "gt_x", "gt_y", "gt_z", "pnp_x", "pnp_y", "pnp_z", "icp_x", "icp_y", "icp_z",
                    "pnp_translation_error", "icp_translation_error",
                    "pnp_rotation_error_deg", "icp_rotation_error_deg"])
        for r in rows:
            w.writerow([r["frame"], r["gt_x"], r["gt_y"], r["gt_z"], r["pnp_x"], r["pnp_y"], r["pnp_z"],
                        r["icp_x"], r["icp_y"], r["icp_z"], r["pnp_translation_error"],
                        r["icp_translation_error"], r["pnp_rotation_error"], r["icp_rotation_error"]])
    with open(DATA / "pair_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "frame_i", "frame_j", "orb_keypoints_i", "orb_keypoints_j", "raw_matches",
                    "filtered_matches", "pnp_correspondences", "pnp_inliers", "pnp_inlier_ratio",
                    "icp_correspondences", "pnp_relative_translation_error", "icp_relative_translation_error",
                    "pnp_relative_rotation_error_deg", "icp_relative_rotation_error_deg"])
        for p in pairs:
            corr, inl = int(p["pnp_correspondences"]), int(p["pnp_inliers"])
            w.writerow([p["pair"], p["frame_i"], p["frame_j"], p["orb_keypoints_i"], p["orb_keypoints_j"],
                        p["raw_matches"], p["filtered_matches"], corr, inl,
                        f"{inl / corr:.6f}" if corr > 0 and inl >= 0 else "",
                        p["icp_correspondences"], p["pnp_relative_translation_error_m"],
                        p["icp_relative_translation_error_m"], p["pnp_relative_rotation_error_deg"],
                        p["icp_relative_rotation_error_deg"]])


def dataset_stats(K, poses, mesh_V, n_tris, n_faces_file):
    lo, hi = mesh_V.min(0), mesh_V.max(0)
    centre = (lo + hi) / 2
    bound_radius = 0.5 * np.linalg.norm(hi - lo)
    C = np.array([T[:3, 3] for T in poses["GT"]])
    horiz = np.hypot(C[:, 0] - centre[0], C[:, 2] - centre[2])
    ang = np.degrees(np.arctan2(-(C[:, 0] - centre[0]), -(C[:, 2] - centre[2])))
    steps = np.diff(np.unwrap(np.radians(ang)))
    rgb0 = read_png(DATASET / "000000.png")
    return dict(
        bunny_vertices=int(len(mesh_V)), bunny_triangles=int(n_tris), bunny_faces_in_file=int(n_faces_file),
        bunny_bounding_radius_m=float(bound_radius), bunny_centre_m=[float(c) for c in centre],
        image_width=int(K["width"]), image_height=int(K["height"]),
        image_width_png=int(rgb0.shape[1]), image_height_png=int(rgb0.shape[0]),
        fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"], depth_scale=5000,
        frames=len(poses["GT"]), poses_per_trajectory=len(poses["GT"]), relative_motions=len(poses["GT"]) - 1,
        orbit_radius_m=float(horiz.mean()), orbit_height_m=float((C[:, 1] - centre[1]).mean()),
        angular_step_deg=float(np.degrees(steps).mean()),
        neighbour_distance_m=float(np.linalg.norm(np.diff(C, axis=0), axis=1).mean()),
    )


def method_summary(pairs, rows):
    corr = np.array([int(p["pnp_correspondences"]) for p in pairs])
    inl = np.array([int(p["pnp_inliers"]) for p in pairs])
    icp = np.array([int(p["icp_correspondences"]) for p in pairs])
    last = rows[-1]

    def stats(a):
        return dict(mean=float(a.mean()), min=int(a.min()), max=int(a.max()),
                    min_pair=int(a.argmin()), max_pair=int(a.argmax()))

    def errs(p):
        t = np.array([float(r[f"{p}_translation_error"]) for r in rows])
        r_ = np.array([float(r[f"{p}_rotation_error"]) for r in rows])
        rt = np.array([float(q[f"{p}_relative_translation_error_m"]) for q in pairs])
        rr = np.array([float(q[f"{p}_relative_rotation_error_deg"]) for q in pairs])
        return dict(final_translation_error_m=float(last[f"{p}_translation_error"]),
                    final_rotation_error_deg=float(last[f"{p}_rotation_error"]),
                    mean_translation_error_m=float(t.mean()), max_translation_error_m=float(t.max()),
                    max_translation_error_frame=int(t.argmax()),
                    mean_rotation_error_deg=float(r_.mean()), max_rotation_error_deg=float(r_.max()),
                    mean_relative_translation_error_m=float(rt.mean()),
                    mean_relative_rotation_error_deg=float(rr.mean()),
                    worst_relative_rotation_pair=int(rr.argmax()),
                    worst_relative_rotation_error_deg=float(rr.max()),
                    successful_pairs=int(sum(int(q[f"{p}_ok"]) for q in pairs)))

    return dict(
        PnP=dict(correspondences=stats(corr), inliers=stats(inl),
                 inlier_ratio_mean=float((inl / corr).mean()), **errs("pnp")),
        ICP=dict(correspondences=stats(icp), **errs("icp")),
        matches=dict(raw=stats(np.array([int(p["raw_matches"]) for p in pairs])),
                     filtered=stats(np.array([int(p["filtered_matches"]) for p in pairs]))),
    )


def write_tables(ds, ms, pair0):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "summary.json").write_text(json.dumps(dict(dataset=ds, methods=ms, pair_0_1=pair0), indent=2))
    P, I = ms["PnP"], ms["ICP"]
    md = f"""# Summary -- synthetic RGB-D Bunny trajectory experiment

Generated by `scripts/generate_results.py` from the C++ outputs; do not edit by hand.

## Dataset

| Quantity | Value |
|---|---|
| Bunny mesh vertices / triangles | {ds['bunny_vertices']:,} / {ds['bunny_triangles']:,} |
| Image resolution | {ds['image_width']} x {ds['image_height']} (RGB + 16-bit depth, depth = value / {ds['depth_scale']} m) |
| Intrinsics fx, fy, cx, cy | {ds['fx']}, {ds['fy']}, {ds['cx']}, {ds['cy']} |
| Frames = poses per trajectory | {ds['frames']} |
| Relative motions | {ds['relative_motions']} |
| Orbit radius / height above bunny centre | {ds['orbit_radius_m']:.4f} m / {ds['orbit_height_m']:.4f} m |
| Angular step | {ds['angular_step_deg']:.2f} deg |
| Neighbouring camera distance | {ds['neighbour_distance_m']:.4f} m |

## PnP vs ICP (over all {ds['relative_motions']} frame pairs)

| Metric | PnP (RANSAC) | ICP (feature 3D-3D, SVD + g2o) |
|---|---|---|
| Correspondences per pair: mean / min / max | {P['correspondences']['mean']:.1f} / {P['correspondences']['min']} / {P['correspondences']['max']} | {I['correspondences']['mean']:.1f} / {I['correspondences']['min']} / {I['correspondences']['max']} |
| RANSAC inliers per pair: mean / min / max | {P['inliers']['mean']:.1f} / {P['inliers']['min']} / {P['inliers']['max']} | n/a (no outlier rejection) |
| Mean inlier ratio | {P['inlier_ratio_mean']:.3f} | n/a |
| Successful pairs | {P['successful_pairs']} / {ds['relative_motions']} | {I['successful_pairs']} / {ds['relative_motions']} |
| Mean per-pair (local) error | {P['mean_relative_translation_error_m']:.4f} m, {P['mean_relative_rotation_error_deg']:.2f} deg | {I['mean_relative_translation_error_m']:.4f} m, {I['mean_relative_rotation_error_deg']:.2f} deg |
| Worst pair (rotation) | {P['worst_relative_rotation_pair']}->{P['worst_relative_rotation_pair']+1}: {P['worst_relative_rotation_error_deg']:.2f} deg | {I['worst_relative_rotation_pair']}->{I['worst_relative_rotation_pair']+1}: {I['worst_relative_rotation_error_deg']:.2f} deg |
| Mean accumulated (global) error | {P['mean_translation_error_m']:.4f} m, {P['mean_rotation_error_deg']:.2f} deg | {I['mean_translation_error_m']:.4f} m, {I['mean_rotation_error_deg']:.2f} deg |
| **Final error (frame {ds['frames']-1})** | **{P['final_translation_error_m']:.4f} m, {P['final_rotation_error_deg']:.2f} deg** | **{I['final_translation_error_m']:.4f} m, {I['final_rotation_error_deg']:.2f} deg** |

## Frame pair 0->1 (correspondence funnel)

| Stage | Count |
|---|---|
| ORB keypoints frame 0 / frame 1 | {pair0['orb_keypoints_i']} / {pair0['orb_keypoints_j']} |
| Raw matches | {pair0['raw_matches']} |
| Filtered matches | {pair0['filtered_matches']} |
| PnP 3D->2D correspondences | {pair0['pnp_correspondences']} |
| PnP RANSAC inliers | {pair0['pnp_inliers']} (reprojection error of inliers: mean {pair0['inlier_reprojection_mean_px']:.2f} px) |
| ICP 3D->3D correspondences | {pair0['icp_correspondences']} |
"""
    (TABLES / "summary.md").write_text(md)


# ----------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.8, "legend.frameon": False,
        "figure.dpi": 110,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[results] figures/{name}.png/.pdf")


def crop_box(kp_rows, shape, margin=45):
    """Pixel box (x0, x1, y0, y1) around the keypoints of both frames (same box for both)."""
    u = [float(r["u"]) for r in kp_rows]
    v = [float(r["v"]) for r in kp_rows]
    h, w = shape[:2]
    return (max(0, int(min(u)) - margin), min(w, int(max(u)) + margin),
            max(0, int(min(v)) - margin), min(h, int(max(v)) + margin))


def side_by_side(img_i, img_j, box, gap=24):
    x0, x1, y0, y1 = box
    a, b = img_i[y0:y1, x0:x1], img_j[y0:y1, x0:x1]
    sep = np.full((a.shape[0], gap, a.shape[2]), 255, dtype=a.dtype)
    return np.concatenate([a, sep, b], axis=1), (x1 - x0) + gap


def text_box(ax, x, y, s, **kw):
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", fontsize=kw.pop("fontsize", 9),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"), **kw)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig01_setup(ds, poses, mesh_V):
    c = np.array(ds["bunny_centre_m"])
    Cc = np.array([T[:3, 3] for T in poses["GT"]])
    fwd = np.array([T[:3, 2] for T in poses["GT"]])  # camera +Z (viewing direction) in world
    n = len(Cc)
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], wspace=0.12, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axi = fig.add_subplot(gs[1, 1])

    # --- top view: world X (right) vs world Z; Y is up (out of the page)
    sub = mesh_V[::7]
    ax.scatter(sub[:, 0], sub[:, 2], s=0.3, color="0.5", alpha=0.6, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 400)
    R = ds["orbit_radius_m"]
    ax.plot(c[0] + R * np.cos(th), c[2] + R * np.sin(th), color=C["GT"], lw=1.0, ls="--", alpha=0.6)
    for k, (p, f) in enumerate(zip(Cc, fwd)):
        tri = np.array([[0, 0], [-0.013, 0.032], [0.013, 0.032]])  # apex at the camera, opening = view
        ang = math.atan2(f[2], f[0]) - math.pi / 2
        Rz = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        pts = (Rz @ tri.T).T + [p[0], p[2]]
        ax.add_patch(patches.Polygon(pts, closed=True, fc=C["GT"], ec=CD["GT"], lw=0.6))
        if k % 3 == 0:
            out = (p[[0, 2]] - c[[0, 2]]) / np.linalg.norm(p[[0, 2]] - c[[0, 2]])
            ax.text(p[0] + out[0] * 0.05, p[2] + out[1] * 0.05, str(k), ha="center", va="center",
                    fontsize=7.5, color=CD["GT"])
    # label the last pose too (the orbit is not closed: no pose at 360 deg)
    out = (Cc[-1, [0, 2]] - c[[0, 2]]) / np.linalg.norm(Cc[-1, [0, 2]] - c[[0, 2]])
    ax.text(Cc[-1, 0] + out[0] * 0.05, Cc[-1, 2] + out[1] * 0.05, str(n - 1), ha="center", va="center",
            fontsize=7.5, color=CD["GT"])
    # relative motion 0 -> 1
    ax.annotate("", xy=(Cc[1, 0], Cc[1, 2]), xytext=(Cc[0, 0], Cc[0, 2]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.text(c[0] - 0.12, c[2] - 0.62, f"relative motion 0→1: |Δ| = {ds['neighbour_distance_m']:.4f} m",
            fontsize=8.5, ha="center")
    ax.annotate("", xy=((Cc[0, 0] + Cc[1, 0]) / 2, (Cc[0, 2] + Cc[1, 2]) / 2 - 0.01),
                xytext=(c[0] - 0.12, c[2] - 0.6), arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    # the 10 degree step between poses 0 and 1
    for k in (0, 1):
        ax.plot([c[0], Cc[k, 0]], [c[2], Cc[k, 2]], color="0.4", lw=0.8)
    a0, a1 = [math.degrees(math.atan2(Cc[k, 2] - c[2], Cc[k, 0] - c[0])) for k in (0, 1)]
    ax.add_patch(patches.Arc((c[0], c[2]), 0.36, 0.36, theta1=min(a0, a1), theta2=max(a0, a1), color="0.2"))
    ax.text(c[0] + 0.02, c[2] - 0.24, f"{ds['angular_step_deg']:.0f}°", fontsize=9)
    # orbit radius
    ang_r = math.radians(160)
    ax.annotate("", xy=(c[0] + R * math.cos(ang_r), c[2] + R * math.sin(ang_r)), xytext=(c[0], c[2]),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    ax.text(c[0] - 0.27, c[2] + 0.14, f"r = {R:.4f} m", fontsize=8.5, rotation=-20)
    ax.text(c[0], c[2] + 0.1, "BUNNY STAYS FIXED", ha="center", fontsize=9, weight="bold", color="0.25")
    ax.text(c[0], c[2] + R - 0.17, f"CAMERA MOVES ({n} poses)", ha="center", fontsize=9, weight="bold",
            color=CD["GT"])
    ax.set_aspect("equal")
    ax.set_xlim(c[0] - R - 0.1, c[0] + R + 0.1)
    ax.set_ylim(c[2] - R - 0.16, c[2] + R + 0.1)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Z (m)")
    ax.set_title("Top view (world Y is up, out of the page)")
    ax.grid(False)

    # --- side view: vertical plane through the bunny centre and cameras 0 and 18
    opp = n // 2
    axis = (Cc[0] - c)
    axis[1] = 0
    axis /= np.linalg.norm(axis)
    s_mesh = (mesh_V - c) @ axis
    ax2.scatter(s_mesh[::5], mesh_V[::5, 1], s=0.3, color="0.5", alpha=0.6, rasterized=True)
    for k in (0, opp):
        sk, yk = (Cc[k] - c) @ axis, Cc[k, 1]
        fk = np.array([fwd[k] @ axis, fwd[k][1]])
        ax2.plot(sk, yk, "o", color=C["GT"], mec=CD["GT"], ms=8)
        ax2.annotate("", xy=(sk + 0.3 * fk[0], yk + 0.3 * fk[1]), xytext=(sk, yk),
                     arrowprops=dict(arrowstyle="-|>", color=CD["GT"], lw=1.2))
        ax2.text(sk, yk + 0.025, f"camera {k}", ha="center", fontsize=8.5, color=CD["GT"])
    ax2.axhline(c[1], color="0.6", lw=0.8, ls=":")
    ax2.text(-R - 0.08, c[1] - 0.02, "bunny centre height", fontsize=7.5, color=GREY)
    ax2.annotate("", xy=(R + 0.07, Cc[0, 1]), xytext=(R + 0.07, c[1]),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    ax2.text(R + 0.09, (c[1] + Cc[0, 1]) / 2, f"h = {ds['orbit_height_m']:.4f} m", fontsize=8.5, va="center")
    ax2.set_aspect("equal")
    ax2.set_xlim(-R - 0.1, R + 0.32)
    ax2.set_ylim(min(mesh_V[:, 1]) - 0.03, Cc[0, 1] + 0.07)
    ax2.set_xlabel("horizontal distance from the bunny centre (m)")
    ax2.set_ylabel("world Y, up (m)")
    ax2.set_title("Side view: every camera looks at the bunny centre")
    ax2.grid(False)

    axi.axis("off")
    axi.text(0.02, 0.95, "Experiment values (read from the data)", fontsize=11, weight="bold", va="top")
    axi.text(0.02, 0.78,
             f"• {ds['frames']} camera poses = ground truth, one RGB-D frame each\n"
             f"• {ds['relative_motions']} relative motions (pose i → i+1); the orbit is not closed:\n"
             f"   pose {n - 1} is at {ds['angular_step_deg'] * (n - 1):.0f}°, no motion {n - 1}→0 is estimated\n"
             f"• orbit radius {R:.4f} m, height {ds['orbit_height_m']:.4f} m above the bunny centre\n"
             f"• angular step {ds['angular_step_deg']:.0f}°, neighbouring cameras {ds['neighbour_distance_m']:.4f} m apart\n"
             f"• {ds['image_width']}×{ds['image_height']} RGB + depth per pose, "
             f"fx = {ds['fx']}, fy = {ds['fy']}\n"
             f"• bunny mesh: {ds['bunny_vertices']:,} vertices (used only to render the images)",
             fontsize=9.5, va="top", linespacing=1.6)
    fig.suptitle("Figure 1 — Synthetic experiment: static Stanford Bunny, orbiting camera", weight="bold", y=0.99)
    save(fig, "fig01_experiment_setup")


def fig02_rgbd(K, corr_rows):
    rgb = read_png(DATASET / "000000.png")
    depth = read_png(DATASET / "000000_depth.png").astype(np.float64)
    # an actual PnP inlier of pair 0->1, and its back-projection
    row = next(r for r in corr_rows if r["pnp_inlier"] == "1")
    u, v = float(row["u_i"]), float(row["v_i"])
    d_raw = int(row["depth_raw_i"])
    check("depth PNG value at the keypoint == C++ exported depth",
          int(depth[int(v), int(u)]) == d_raw, f"({d_raw})")
    Z = d_raw / 5000.0
    X = (u - K["cx"]) * Z / K["fx"]
    Y = (v - K["cy"]) * Z / K["fy"]
    check("back-projection (u,v,Z) -> (X,Y,Z) matches the C++ 3D point",
          max(abs(X - float(row["X_i"])), abs(Y - float(row["Y_i"])), abs(Z - float(row["Z_i"]))) < 1e-5)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB image (frame 0)")
    dm = np.ma.masked_where(depth == 0, depth / 5000.0)
    im = axs[1].imshow(dm, cmap="viridis")
    axs[1].set_title(f"Depth (frame 0): {np.count_nonzero(depth):,} valid pixels")
    cb = fig.colorbar(im, ax=axs[1], fraction=0.035, pad=0.03)
    cb.set_label("depth Z (m)")
    for a in axs[:2]:
        a.plot(u, v, "o", mfc="none", mec="red", ms=12, mew=1.8)
        a.annotate(f"(u, v) = ({u:.1f}, {v:.1f})", (u, v), xytext=(u - 230, v - 80), color="red", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="red"))
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color("0.8")
    ax = axs[2]
    ax.axis("off")
    ax.text(0.0, 0.97, "Pixel + depth  →  3D point", fontsize=12, weight="bold", va="top")
    lines = [
        f"pixel (u, v) = ({u:.2f}, {v:.2f})",
        f"stored depth = {d_raw}  →  Z = {d_raw} / 5000 = {Z:.4f} m",
        "",
        r"$X = (u - c_x)\,Z / f_x$" + f" = {X:+.4f} m",
        r"$Y = (v - c_y)\,Z / f_y$" + f" = {Y:+.4f} m",
        r"$Z$" + f" = {Z:.4f} m",
        "",
        f"intrinsics.txt: fx = {K['fx']}, fy = {K['fy']}",
        f"                  cx = {K['cx']}, cy = {K['cy']}",
        "",
        "The 3D point is expressed in the",
        "coordinate frame of camera 0",
        "(+X right, +Y down, +Z forward).",
        "",
        "This is one real correspondence of",
        "pair 0→1 (a PnP RANSAC inlier).",
    ]
    ax.text(0.0, 0.86, "\n".join(lines), fontsize=9.5, va="top", family="DejaVu Sans")
    fig.suptitle("Figure 2 — One RGB-D observation and the back-projection of a feature", weight="bold", y=1.02)
    save(fig, "fig02_rgbd_backprojection")


def fig03_keypoints(kp_rows, pair):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, frame, which in ((axs[0], pair["frame_i"], "i"), (axs[1], pair["frame_j"], "j")):
        img = read_png(DATASET / f"{int(frame):06d}.png")
        pts = np.array([[float(r["u"]), float(r["v"])] for r in kp_rows if r["frame"] == which])
        x0, x1, y0, y1 = crop_box(kp_rows, img.shape)
        ax.imshow(img[y0:y1, x0:x1], extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, facecolors="none", edgecolors=(1, 0.25, 0.25), linewidths=0.9)
        ax.set_title(f"Frame {frame}: {len(pts)} ORB keypoints")
        ax.text(0.01, 0.01, f"crop of the {img.shape[1]}×{img.shape[0]} image (pixel coordinates shown)",
                transform=ax.transAxes, fontsize=8, color="0.85")
        ax.grid(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("Figure 3 — ORB keypoints detected by the C++ pipeline (OpenCV ORB, default settings)",
                 weight="bold", y=1.0)
    save(fig, "fig03_orb_keypoints")


def fig04_matches(kp_rows, raw_rows, pair, sc):
    img_i = read_png(DATASET / f"{int(pair['frame_i']):06d}.png")
    img_j = read_png(DATASET / f"{int(pair['frame_j']):06d}.png")
    w = img_i.shape[1]
    kpi = {int(r["index"]): (float(r["u"]), float(r["v"])) for r in kp_rows if r["frame"] == "i"}
    kpj = {int(r["index"]): (float(r["u"]), float(r["v"])) for r in kp_rows if r["frame"] == "j"}
    kept = [r for r in raw_rows if r["filtered"] == "1"]
    rej = [r for r in raw_rows if r["filtered"] == "0"]
    check("raw/filtered match counts in the dump == pair_metrics.csv",
          len(raw_rows) == int(pair["raw_matches"]) and len(kept) == int(pair["filtered_matches"]))
    box = crop_box(kp_rows, img_i.shape)
    canvas, dx = side_by_side(img_i, img_j, box)
    x0, y0 = box[0], box[2]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.imshow(canvas)
    for r in rej:
        a, b = kpi[int(r["query_index"])], kpj[int(r["train_index"])]
        ax.plot([a[0] - x0, b[0] - x0 + dx], [a[1] - y0, b[1] - y0], color=(0.95, 0.3, 0.3), lw=0.5, alpha=0.45)
    for r in kept:
        a, b = kpi[int(r["query_index"])], kpj[int(r["train_index"])]
        ax.plot([a[0] - x0, b[0] - x0 + dx], [a[1] - y0, b[1] - y0], color=(0.2, 0.85, 0.35), lw=0.8, alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"Frame {pair['frame_i']}  ←→  frame {pair['frame_j']}")
    ax.legend(handles=[Line2D([], [], color=(0.2, 0.85, 0.35), lw=2,
                              label=f"accepted: {len(kept)} filtered matches"),
                       Line2D([], [], color=(0.95, 0.3, 0.3), lw=2,
                              label=f"rejected by the distance filter: {len(rej)}")],
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)
    dist = [float(r["hamming_distance"]) for r in raw_rows]
    text_box(ax, 0.01, -0.09,
             f"raw matches: {len(raw_rows)} (best match for every frame-{pair['frame_i']} keypoint)\n"
             f"filter: Hamming distance ≤ max({sc['filter_factor']}·d_min, {sc['filter_floor']:.0f}),"
             f" d_min = {min(dist):.0f}\nfiltered matches: {len(kept)}")
    fig.suptitle("Figure 4 — ORB feature matches (brute-force Hamming)", weight="bold", y=1.0)
    save(fig, "fig04_feature_matches")


def fig05_pnp_vs_icp(sc):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax in axs:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")

    def frame_box(ax, x, label):
        ax.add_patch(patches.FancyBboxPatch((x, 2.2), 3.2, 3.2, boxstyle="round,pad=0.05", fc="0.97", ec="0.6"))
        ax.text(x + 1.6, 5.65, label, ha="center", fontsize=10, weight="bold")

    for ax, title, right, lines, col in (
        (axs[0], "PnP", "2D pixel  ×",
         ["3D → 2D", "depth from frame i only", f"OpenCV solvePnPRansac",
          f"({sc['ransac_iters']} iterations, {sc['ransac_px']:.0f} px threshold, "
          f"confidence {sc['ransac_conf']})", "minimizes reprojection error (pixels)", "output: T(i+1 ← i)"],
         C["PnP"]),
        (axs[1], "Our ICP implementation", "3D point  ●",
         ["3D → 3D", "depth from BOTH frames", "correspondences = the same ORB matches",
          "closed-form SVD alignment + g2o refinement", "minimizes Σ‖p − (R·q + t)‖² (metres)",
          "NOT dense nearest-neighbour ICP"],
         C["ICP"])):
        frame_box(ax, 0.3, "Frame i")
        frame_box(ax, 6.5, "Frame i+1")
        ax.plot(1.9, 3.8, "o", ms=16, color=col, mec="k")
        ax.text(1.9, 3.0, "3D point (from depth)", ha="center", fontsize=8.5)
        if "pixel" in right:
            ax.plot(8.1, 3.8, "x", ms=16, mew=3, color="k")
            ax.text(8.1, 3.0, "2D pixel (u, v)", ha="center", fontsize=8.5)
        else:
            ax.plot(8.1, 3.8, "o", ms=16, color=col, mec="k")
            ax.text(8.1, 3.0, "3D point (from depth)", ha="center", fontsize=8.5)
        ax.annotate("", xy=(7.6, 3.8), xytext=(2.5, 3.8), arrowprops=dict(arrowstyle="-|>", lw=2, color="0.25"))
        ax.text(5, 6.55, title, ha="center", fontsize=14, weight="bold", color=col)
        ax.text(5, 1.75, "\n".join(lines), ha="center", va="top", fontsize=9.5)
    fig.suptitle("Figure 5 — How the two methods use a feature correspondence", weight="bold", y=1.0)
    save(fig, "fig05_pnp_vs_icp_concept")


def fig06_funnel(pair, pairs):
    stages = [("ORB keypoints (frame i)", int(pair["orb_keypoints_i"]), "0.55"),
              ("raw matches", int(pair["raw_matches"]), "0.55"),
              ("filtered matches", int(pair["filtered_matches"]), "0.4"),
              ("PnP 3D→2D correspondences\n(depth valid in frame i)", int(pair["pnp_correspondences"]), C["PnP"]),
              ("PnP RANSAC inliers", int(pair["pnp_inliers"]), CD["PnP"]),
              ("ICP 3D→3D correspondences\n(depth valid in both frames)", int(pair["icp_correspondences"]), C["ICP"])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [s[1] for s in stages], color=[s[2] for s in stages], height=0.62)
    for yy, (lab, val, _) in zip(y, stages):
        ax.text(val + 5, yy, str(val), va="center", fontsize=10, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages])
    ax.set_xlabel("count")
    ax.set_title(f"Frame pair {pair['frame_i']}→{pair['frame_j']} "
                 f"(frame {pair['frame_j']}: {pair['orb_keypoints_j']} keypoints)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(s[1] for s in stages) * 1.15)
    # the same stages over all pairs
    keys = [("raw_matches", "raw matches", "0.55"), ("filtered_matches", "filtered", "0.4"),
            ("pnp_correspondences", "PnP corr.", C["PnP"]), ("pnp_inliers", "PnP inliers", CD["PnP"]),
            ("icp_correspondences", "ICP corr.", C["ICP"])]
    data = [[int(p[k]) for p in pairs] for k, _, _ in keys]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="k"))
    for patch, (_, _, col) in zip(bp["boxes"], keys):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax2.set_xticks(range(1, len(keys) + 1))
    ax2.set_xticklabels([k[1] for k in keys], rotation=15)
    ax2.set_ylabel("count per frame pair")
    ax2.set_title(f"Distribution over all {len(pairs)} pairs")
    fig.suptitle("Figure 6 — Correspondence funnel: from keypoints to the points each method uses",
                 weight="bold", y=1.02)
    save(fig, "fig06_correspondence_funnel")


def fig07_points(corr_rows, pair, ds, depth_valid):
    pnp = [r for r in corr_rows if r["pnp_used"] == "1"]
    icp = [r for r in corr_rows if r["icp_used"] == "1"]
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.08)
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1], projection="3d")
    axt = fig.add_subplot(gs[2])
    # camera-frame coordinates: plot X right, Z forward (depth), -Y up for a natural view
    def xyz(rows, s):
        return np.array([[float(r[f"X_{s}"]), float(r[f"Z_{s}"]), -float(r[f"Y_{s}"])] for r in rows])
    P = xyz(pnp, "i")
    inl = np.array([r["pnp_inlier"] == "1" for r in pnp])
    ax1.scatter(*P[inl].T, s=10, color=C["PnP"], label=f"RANSAC inliers ({inl.sum()})")
    ax1.scatter(*P[~inl].T, s=18, color="k", marker="x", label=f"rejected ({(~inl).sum()})")
    ax1.set_title(f"PnP: {len(P)} 3D points in camera {pair['frame_i']}")
    A, B = xyz(icp, "i"), xyz(icp, "j")
    ax2.scatter(*A.T, s=9, color=C["ICP"], label=f"camera {pair['frame_i']} points")
    ax2.scatter(*B.T, s=9, color="0.35", label=f"camera {pair['frame_j']} points")
    for a, b in zip(A, B):
        ax2.plot(*np.array([a, b]).T, color="0.6", lw=0.4)
    lens = np.linalg.norm(A - B, axis=1)
    ax2.set_title(f"ICP: {len(A)} 3D↔3D pairs (lines)")
    ax2.text2D(0.02, -0.08, f"median pair offset {np.median(lens) * 100:.1f} cm; {int((lens > 0.05).sum())} pairs "
               f"> 5 cm\n= mismatched features, kept (no outlier rejection)", transform=ax2.transAxes,
               fontsize=8, color=GREY)
    for ax in (ax1, ax2):
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z, depth (m)")
        ax.set_zlabel("−Y, up (m)")
        ax.legend(loc="upper left", fontsize=8)
        ax.view_init(elev=18, azim=-70)
        for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            a_.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=7)
    axt.axis("off")
    f0 = pair["frame_i"]
    rows_t = [
        ("Mesh vertices", f"{ds['bunny_vertices']:,}", "render / display only"),
        ("Image pixels per frame", f"{ds['image_width'] * ds['image_height']:,}", "input to ORB"),
        (f"Valid depth pixels (fr. {f0})", f"{depth_valid:,}", "not used as a cloud"),
        (f"ORB keypoints (fr. {f0})", pair["orb_keypoints_i"], "detection"),
        ("Filtered matches", pair["filtered_matches"], "shared by both"),
        ("PnP 3D→2D pairs", pair["pnp_correspondences"], "PnP input"),
        ("ICP 3D→3D pairs", pair["icp_correspondences"], "ICP input"),
        ("Trajectory poses", str(ds["poses_per_trajectory"]), "output (per method)"),
    ]
    tab = axt.table(cellText=[list(r) for r in rows_t], colLabels=["kind of 'point'", "count", "role"],
                    cellLoc="left", bbox=[0.0, 0.12, 1.0, 0.72], colWidths=[0.41, 0.17, 0.42])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    axt.set_title(f"Different 'points' — pair {pair['frame_i']}→{pair['frame_j']}", pad=12)
    fig.suptitle("Figure 7 — The 3D points actually used by the estimation (not the 35k-vertex mesh)",
                 weight="bold", y=1.0)
    save(fig, "fig07_correspondence_points")


def fig08_trajectories(poses, mesh_V, ms):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    sub = mesh_V[::6]
    # plot (X, Z, Y) so that world Y (up) is vertical
    ax.scatter(sub[:, 0], sub[:, 2], sub[:, 1], s=0.3, color="0.6", alpha=0.35, rasterized=True)
    for key in ("GT", "PnP", "ICP"):
        P = np.array([T[:3, 3] for T in poses[key]])
        ax.plot(P[:, 0], P[:, 2], P[:, 1], color=C[key], lw=2.2 if key == "GT" else 1.8,
                label={"GT": "Ground truth", "PnP": "PnP", "ICP": "ICP (ours: feature 3D-3D)"}[key])
        ax.scatter(*P[-1, [0, 2, 1]], s=60, color=C[key], edgecolor="k", zorder=5, marker="s")
    P0 = poses["GT"][0][:3, 3]
    ax.scatter(P0[0], P0[2], P0[1], s=90, color="w", edgecolor="k", marker="o", zorder=6, label="start (all three)")
    last = len(poses["GT"]) - 1
    ax.scatter([], [], s=50, color="0.8", edgecolor="k", marker="s", label=f"final pose (frame {last})")
    allp = np.concatenate([[T[:3, 3] for T in poses[k]] for k in poses])
    mid = (allp.max(0) + allp.min(0)) / 2
    half = (allp.max(0) - allp.min(0)).max() / 2 * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[2] - half, mid[2] + half)
    ax.set_zlim(mid[1] - half * 0.55, mid[1] + half * 0.55)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Z (m)")
    ax.set_zlabel("world Y, up (m)")
    ax.view_init(elev=28, azim=-60)
    ax.legend(loc="upper left", fontsize=9)
    P, I = ms["PnP"], ms["ICP"]
    ax.text2D(0.70, 0.96, f"Final error (frame {last})\nPnP:  {P['final_translation_error_m']:.3f} m, "
              f"{P['final_rotation_error_deg']:.1f}°\nICP:  {I['final_translation_error_m']:.3f} m, "
              f"{I['final_rotation_error_deg']:.1f}°", transform=ax.transAxes, fontsize=9.5, va="top",
              bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    ax.set_title("Figure 8 — Ground truth vs PnP vs ICP camera trajectories", weight="bold")
    save(fig, "fig08_trajectories_3d")


def _error_plot(rows, pairs, col, ylabel, title, name, unit):
    f = np.array([int(r["frame"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for key, p in (("PnP", "pnp"), ("ICP", "icp")):
        y = np.array([float(r[f"{p}_{col}"]) for r in rows])
        ax.plot(f, y, "-o", ms=3.5, lw=1.8, color=C[key], label=key)
        ax.annotate(f"{y[-1]:.3f} {unit}" if unit == "m" else f"{y[-1]:.1f}{unit}", (f[-1], y[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=CD[key], weight="bold")
    ax.set_xlim(0, f[-1] + 3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("frame index")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.text(0.99, 0.02, "absolute error vs ground truth, accumulated from frame 0 (no smoothing)",
            transform=ax.transAxes, ha="right", fontsize=8, color=GREY)
    save(fig, name)


def fig09_10(rows, pairs):
    _error_plot(rows, pairs, "translation_error", "camera position error (m)",
                "Figure 9 — Accumulated translation error per frame", "fig09_translation_error", "m")
    _error_plot(rows, pairs, "rotation_error", "camera rotation error (deg)",
                "Figure 10 — Accumulated rotation error per frame", "fig10_rotation_error", "°")


def fig11_local(pairs):
    k = np.array([int(p["pair"]) for p in pairs])
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    for ax, col, lab, unit in ((axs[0], "relative_translation_error_m", "relative translation error (m)", "m"),
                               (axs[1], "relative_rotation_error_deg", "relative rotation error (deg)", "°")):
        for off, (key, p) in zip((-0.2, 0.2), (("PnP", "pnp"), ("ICP", "icp"))):
            y = np.array([float(q[f"{p}_{col}"]) for q in pairs])
            ax.bar(k + off, y, width=0.4, color=C[key], label=key)
            w = int(y.argmax())
            ax.annotate(f"worst {key}: {w}→{w + 1}\n{y[w]:.3f} {unit}" if unit == "m"
                        else f"worst {key}: {w}→{w + 1}\n{y[w]:.2f}{unit}",
                        (w + off, y[w]), xytext=(-60 if key == "PnP" else 0, 8), textcoords="offset points",
                        ha="center", arrowprops=dict(arrowstyle="-", color=CD[key], lw=0.6) if key == "PnP" else None,
                        fontsize=8, color=CD[key])
        ax.set_ylabel(lab)
        ax.legend(loc="upper left")
    axs[1].set_xlabel("frame pair i → i+1 (index i)")
    axs[1].set_xticks(k[::2])
    fig.suptitle("Figure 11 — LOCAL error of each estimated relative motion (vs ground-truth motion)",
                 weight="bold")
    fig.text(0.5, -0.01, "Each bar is ONE frame pair, before accumulation. Many moderate errors, plus "
             "a few larger ones, add up to the global drift of Figures 9–10.", ha="center", fontsize=9, color=GREY)
    save(fig, "fig11_local_pair_errors")


def fig12_counts(pairs):
    k = np.array([int(p["pair"]) for p in pairs])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(k, [int(p["filtered_matches"]) for p in pairs], "-", color="0.5", lw=1.2, label="filtered matches")
    ax.plot(k, [int(p["pnp_correspondences"]) for p in pairs], "-o", ms=3.5, color=C["PnP"],
            label="PnP 3D→2D correspondences")
    ax.plot(k, [int(p["pnp_inliers"]) for p in pairs], "--o", ms=3, color=CD["PnP"], label="PnP RANSAC inliers")
    ax.plot(k, [int(p["icp_correspondences"]) for p in pairs], "-o", ms=3.5, color=C["ICP"],
            label="ICP 3D→3D correspondences")
    ax.set_xlabel("frame pair i → i+1 (index i)")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("Figure 12 — Correspondences available to each method, per frame pair")
    save(fig, "fig12_correspondence_counts")


def fig13_inlier_ratio(pairs):
    k = np.array([int(p["pair"]) for p in pairs])
    r = np.array([int(p["pnp_inliers"]) / int(p["pnp_correspondences"]) for p in pairs])
    rot = np.array([float(p["pnp_relative_rotation_error_deg"]) for p in pairs])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(k, r, "-o", ms=4, color=CD["PnP"], label="PnP inlier ratio = inliers / correspondences")
    ax.set_ylim(min(0.8, r.min() - 0.02), 1.0)
    ax.set_xlabel("frame pair i → i+1 (index i)")
    ax.set_ylabel("inlier ratio")
    ax2 = ax.twinx()
    ax2.bar(k, rot, width=0.5, color=C["PnP"], alpha=0.25, label="PnP relative rotation error")
    ax2.set_ylabel("PnP relative rotation error (deg)", color=GREY)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    corr = np.corrcoef(r, rot)[0, 1]
    ax.set_title("Figure 13 — PnP RANSAC inlier ratio per pair (correspondence quality)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    ax.text(0.99, 0.03, f"mean ratio {r.mean():.3f}; Pearson corr(ratio, rotation error) = {corr:+.2f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
    ax.text(0.99, 0.10, "ICP has no outlier rejection, so no ICP inlier ratio exists.",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=GREY)
    save(fig, "fig13_pnp_inlier_ratio")


def fig14_local_global(rows, pairs):
    rr = np.array([float(p["pnp_relative_rotation_error_deg"]) for p in pairs])
    glob = np.array([float(r["pnp_rotation_error"]) for r in rows])
    ri = np.array([float(p["icp_relative_rotation_error_deg"]) for p in pairs])
    globi = np.array([float(r["icp_rotation_error"]) for r in rows])
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 2.2], hspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.set_xlim(0, 12)
    ax0.set_ylim(0, 3)
    xs = [0.6, 3.0, 5.4, 7.8]
    for n, x in enumerate(xs):
        ax0.add_patch(patches.Circle((x, 1.9), 0.33, fc=C["PnP"], ec="k"))
        ax0.text(x, 1.9, f"{n}", ha="center", va="center", weight="bold")
        ax0.text(x, 1.3, f"pose {n}", ha="center", fontsize=8.5)
    for n in range(3):
        ax0.annotate("", xy=(xs[n + 1] - 0.38, 1.9), xytext=(xs[n] + 0.38, 1.9),
                     arrowprops=dict(arrowstyle="-|>", lw=1.5))
        ax0.text((xs[n] + xs[n + 1]) / 2, 2.35, f"T{n + 1}←{n}\nerr {rr[n]:.2f}°", ha="center", fontsize=8.5)
    ax0.text(8.9, 1.9, "…", fontsize=16, va="center")
    ax0.text(9.5, 2.5, r"$T_{wc}[i{+}1] = T_{wc}[i]\cdot T_{rel}(i,i{+}1)^{-1}$", fontsize=11)
    ax0.text(9.5, 1.55, f"every pose inherits all earlier errors\nPnP after 3 steps: {glob[3]:.2f}° "
             f"| after 35: {glob[-1]:.1f}°", fontsize=9)
    ax0.set_title("Local motions are chained into the global trajectory (PnP, actual values)", fontsize=11)

    ax = fig.add_subplot(gs[1])
    f = np.arange(len(glob))
    for key, loc, g, off in (("PnP", rr, glob, -0.2), ("ICP", ri, globi, 0.2)):
        ax.bar(np.arange(1, len(loc) + 1) + off, loc, width=0.4, color=C[key], alpha=0.55,
               label=f"{key} local error of motion i-1→i")
        ax.plot(f, np.concatenate([[0], np.cumsum(loc)]), ":", color=CD[key], lw=1.4,
                label=f"{key} running sum of local errors")
        ax.plot(f, g, "-", color=CD[key], lw=2.2, label=f"{key} accumulated (global) error")
    ax.set_xlabel("frame index")
    ax.set_ylabel("rotation error (deg)")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.set_title("Small local errors → accumulation → large global drift (rotation)")
    ax.text(0.5, -0.17,
            f"PnP: mean local {rr.mean():.2f}°/pair, sum {rr.sum():.1f}°, final global {glob[-1]:.1f}°   |   "
            f"ICP: mean local {ri.mean():.2f}°/pair, sum {ri.sum():.1f}°, final global {globi[-1]:.1f}°\n"
            f"Largest single pair: PnP {rr.max():.2f}° at {int(rr.argmax())}→{int(rr.argmax()) + 1} "
            f"({100 * rr.max() / rr.sum():.0f}% of its sum); ICP {ri.max():.2f}° at {int(ri.argmax())}→"
            f"{int(ri.argmax()) + 1} ({100 * ri.max() / ri.sum():.0f}% of its sum). The global error stays below "
            "the running sum (errors partly cancel) and is mostly gradual accumulation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=GREY)
    fig.suptitle("Figure 14 — Local vs global error", weight="bold", y=0.98)
    save(fig, "fig14_local_vs_global_error")


def fig15_summary(ds, ms, poses, mesh_V, pair, kp_rows, raw_rows):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1, 1.15, 1.1], hspace=0.28, wspace=0.18)
    c = np.array(ds["bunny_centre_m"])
    # left: setup
    ax = fig.add_subplot(gs[0, 0])
    sub = mesh_V[::7]
    ax.scatter(sub[:, 0], sub[:, 2], s=0.3, color="0.55", alpha=0.6, rasterized=True)
    Cg = np.array([T[:3, 3] for T in poses["GT"]])
    ax.plot(Cg[:, 0], Cg[:, 2], "o", color=C["GT"], mec=CD["GT"], ms=5)
    ax.annotate("", xy=(Cg[1, 0], Cg[1, 2]), xytext=(Cg[0, 0], Cg[0, 2]), arrowprops=dict(arrowstyle="-|>"))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Static bunny, {ds['frames']} camera poses\n{ds['angular_step_deg']:.0f}° apart on a "
                 f"{ds['orbit_radius_m']:.2f} m orbit")
    # middle: RGB-D + matches
    axm = fig.add_subplot(gs[0, 1])
    img_i = read_png(DATASET / f"{int(pair['frame_i']):06d}.png")
    img_j = read_png(DATASET / f"{int(pair['frame_j']):06d}.png")
    box = crop_box(kp_rows, img_i.shape)
    canvas, dx = side_by_side(img_i, img_j, box)
    axm.imshow(canvas)
    kpi = {int(r["index"]): (float(r["u"]), float(r["v"])) for r in kp_rows if r["frame"] == "i"}
    kpj = {int(r["index"]): (float(r["u"]), float(r["v"])) for r in kp_rows if r["frame"] == "j"}
    for r in raw_rows:
        if r["filtered"] == "1":
            a, b = kpi[int(r["query_index"])], kpj[int(r["train_index"])]
            axm.plot([a[0] - box[0], b[0] - box[0] + dx], [a[1] - box[2], b[1] - box[2]],
                     color=(0.2, 0.85, 0.35), lw=0.5, alpha=0.85)
    axm.axis("off")
    axm.set_title(f"RGB-D → ORB matching → PnP (3D→2D) / our ICP (3D→3D)\n"
                  f"pair 0→1: {pair['filtered_matches']} matches, {pair['pnp_correspondences']} PnP, "
                  f"{pair['icp_correspondences']} ICP correspondences")
    # right: trajectories (top view)
    axr = fig.add_subplot(gs[0, 2])
    axr.scatter(sub[:, 0], sub[:, 2], s=0.3, color="0.7", alpha=0.5, rasterized=True)
    for key in ("GT", "PnP", "ICP"):
        P = np.array([T[:3, 3] for T in poses[key]])
        axr.plot(P[:, 0], P[:, 2], color=C[key], lw=2.2, label=key)
        axr.plot(P[-1, 0], P[-1, 2], "s", color=C[key], mec="k", ms=7)
    axr.plot(Cg[0, 0], Cg[0, 2], "o", color="w", mec="k", ms=8, label="start")
    axr.set_aspect("equal")
    axr.axis("off")
    axr.legend(loc="center", fontsize=9, frameon=True, framealpha=0.9)
    axr.set_title(f"Trajectories (top view), ■ = frame {ds['frames'] - 1}")
    # bottom: table
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    P, I = ms["PnP"], ms["ICP"]
    cells = [
        ["Correspondences per pair (mean / min–max)",
         f"{P['correspondences']['mean']:.0f} / {P['correspondences']['min']}–{P['correspondences']['max']}",
         f"{I['correspondences']['mean']:.0f} / {I['correspondences']['min']}–{I['correspondences']['max']}"],
        ["RANSAC inliers per pair (mean / min–max)",
         f"{P['inliers']['mean']:.0f} / {P['inliers']['min']}–{P['inliers']['max']}", "— (no outlier rejection)"],
        ["Mean local error per motion",
         f"{P['mean_relative_translation_error_m'] * 100:.2f} cm, {P['mean_relative_rotation_error_deg']:.2f}°",
         f"{I['mean_relative_translation_error_m'] * 100:.2f} cm, {I['mean_relative_rotation_error_deg']:.2f}°"],
        [f"Final accumulated error (frame {ds['frames'] - 1})",
         f"{P['final_translation_error_m']:.3f} m, {P['final_rotation_error_deg']:.1f}°",
         f"{I['final_translation_error_m']:.3f} m, {I['final_rotation_error_deg']:.1f}°"],
    ]
    tab = axt.table(cellText=cells, colLabels=["", "PnP (RANSAC)", "ICP (ours: feature 3D-3D, SVD + g2o)"],
                    loc="upper center", cellLoc="center", colWidths=[0.34, 0.24, 0.3])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 1.9)
    for (r_, c_), cell in tab.get_celld().items():
        if r_ == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor({1: (1, 0.93, 0.82), 2: (0.97, 0.9, 0.99)}.get(c_, "white"))
    axt.text(0.5, 0.02, f"Frames {ds['frames']}  ·  relative motions {ds['relative_motions']}  ·  "
             f"{ds['image_width']}×{ds['image_height']} RGB-D  ·  fx={ds['fx']} fy={ds['fy']} cx={ds['cx']} "
             f"cy={ds['cy']}  ·  frame-to-frame odometry: no loop closure, no global optimization",
             ha="center", transform=axt.transAxes, fontsize=10, color=GREY)
    fig.suptitle("Synthetic RGB-D Bunny: ground truth vs PnP vs feature-based 3D–3D ICP", fontsize=16,
                 weight="bold", y=0.99)
    save(fig, "fig15_summary")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-cpp", action="store_true",
                    help="reuse results/data/cpp_export/ instead of re-running slam_trajectory_test")
    args = ap.parse_args()
    style()
    if not args.skip_cpp:
        run_cpp_export()

    K, rows, pairs, poses, rel = build_metrics()
    write_csvs(rows, pairs)
    mesh_V, tris, n_faces = read_ply_mesh(MESH)
    ds = dataset_stats(K, poses, mesh_V, len(tris), n_faces)
    check("intrinsics.txt resolution == PNG size",
          (ds["image_width"], ds["image_height"]) == (ds["image_width_png"], ds["image_height_png"]))
    ms = method_summary(pairs, rows)
    sc = source_constants()

    prefix = EXPORT / f"pair_{EXPORT_PAIR}_{EXPORT_PAIR + 1}"
    kp_rows = read_csv(f"{prefix}_keypoints.csv")
    raw_rows = read_csv(f"{prefix}_raw_matches.csv")
    corr_rows = read_csv(f"{prefix}_correspondences.csv")
    pair = pairs[EXPORT_PAIR]
    check("dump counts == pair_metrics.csv (keypoints, PnP/ICP correspondences, inliers)",
          sum(r["frame"] == "i" for r in kp_rows) == int(pair["orb_keypoints_i"])
          and sum(r["frame"] == "j" for r in kp_rows) == int(pair["orb_keypoints_j"])
          and sum(r["pnp_used"] == "1" for r in corr_rows) == int(pair["pnp_correspondences"])
          and sum(r["pnp_inlier"] == "1" for r in corr_rows) == int(pair["pnp_inliers"])
          and sum(r["icp_used"] == "1" for r in corr_rows) == int(pair["icp_correspondences"]))

    # reprojection error of the PnP inliers with the exported PnP relative pose
    T = rel["PnP"][EXPORT_PAIR]
    errs = []
    for r in corr_rows:
        if r["pnp_inlier"] != "1":
            continue
        X = T[:3, :3] @ np.array([float(r["X_i"]), float(r["Y_i"]), float(r["Z_i"])]) + T[:3, 3]
        u = K["fx"] * X[0] / X[2] + K["cx"]
        v = K["fy"] * X[1] / X[2] + K["cy"]
        errs.append(math.hypot(u - float(r["u_j"]), v - float(r["v_j"])))
    pair0 = {k: int(pair[k]) for k in ("orb_keypoints_i", "orb_keypoints_j", "raw_matches", "filtered_matches",
                                       "pnp_correspondences", "pnp_inliers", "icp_correspondences")}
    pair0["inlier_reprojection_mean_px"] = float(np.mean(errs))
    pair0["inlier_reprojection_max_px"] = float(np.max(errs))
    write_tables(ds, ms, pair0)
    depth_valid = int(np.count_nonzero(read_png(DATASET / f"{int(pair['frame_i']):06d}_depth.png")))

    fig01_setup(ds, poses, mesh_V)
    fig02_rgbd(K, corr_rows)
    fig03_keypoints(kp_rows, pair)
    fig04_matches(kp_rows, raw_rows, pair, sc)
    fig05_pnp_vs_icp(sc)
    fig06_funnel(pair, pairs)
    fig07_points(corr_rows, pair, ds, depth_valid)
    fig08_trajectories(poses, mesh_V, ms)
    fig09_10(rows, pairs)
    fig11_local(pairs)
    fig12_counts(pairs)
    fig13_inlier_ratio(pairs)
    fig14_local_global(rows, pairs)
    fig15_summary(ds, ms, poses, mesh_V, pair, kp_rows, raw_rows)
    print(f"[results] done: {len(list(FIG.glob('*.png')))} figures in {FIG.relative_to(ROOT)}/, "
          f"CSVs in {DATA.relative_to(ROOT)}/, tables in {TABLES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
