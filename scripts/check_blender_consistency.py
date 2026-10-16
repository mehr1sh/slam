"""Read-only Blender/CSV consistency check (diagnostics). Changes nothing in
the scene and saves nothing.

Run from the repository root:
    blender -b visualization/scenes/bunny_slam_demo.blend \
        --python scripts/check_blender_consistency.py -- diagnostics/blender_consistency.csv

For every frame 0..N-1 and each of GT / PnP / ICP it compares, against
data/synthetic_bunny/slam_trajectory.csv read directly from disk (not the
scene's stored copy):
  - <M>_Animated_Camera world position vs WorldRoot @ camera centre
  - its world rotation vs WorldRoot @ (R_wc @ R_FIX)
  - its viewing direction (Blender camera -Z) vs WorldRoot @ (R_wc @ +Z_cv)
  - the <M>_Trajectory curve point i vs WorldRoot @ camera centre i
  - the HUD value slam_animation.error_at() vs the CSV error columns
and checks scene.frame_start / frame_end.
"""

import csv
import math
import sys
from pathlib import Path

import bpy
import mathutils

ROOT = Path(bpy.path.abspath("//")).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visualization" / "scripts"))
import slam_animation  # noqa: E402
from slam_coords import R_FIX  # noqa: E402

out_path = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else ROOT / "diagnostics" / "blender_consistency.csv"
rows = list(csv.DictReader(open(ROOT / "data" / "synthetic_bunny" / "slam_trajectory.csv")))
rows = [{k: float(v) for k, v in r.items()} for r in rows]
# Optional 4th trajectory (pnp_from_scratch); only its existing frames are checked.
# The file the scene was built from is recorded by build_scene.py in
# scene["slam_scratch_source"] (scenes built before that fall back to the
# milestone-2 file they used).
scratch_path = ROOT / bpy.context.scene.get("slam_scratch_source",
                                            "pnp_from_scratch/results/scratch_pnp_trajectory.csv")
n_scratch = 0
if scratch_path.exists() and "ScratchPnP_Animated_Camera" in bpy.data.objects:
    srows = list(csv.DictReader(l for l in open(scratch_path) if not l.startswith("#")))
    for d in srows:
        r = rows[int(d["frame"])]
        for k_csv, k in (("tx", "x"), ("ty", "y"), ("tz", "z"), ("qx", "qx"), ("qy", "qy"), ("qz", "qz"),
                         ("qw", "qw"), ("translation_error_m", "translation_error"),
                         ("rotation_error_deg", "rotation_error")):
            r[f"scratch_{k}"] = float(d[k_csv])
    n_scratch = len(srows)
N = len(rows)
scene = bpy.context.scene

# Bunny centre (native frame): least-squares intersection of the GT optical
# axes (every GT camera looks at it) -- independent of Blender.
A, b = mathutils.Matrix(((0, 0, 0), (0, 0, 0), (0, 0, 0))), mathutils.Vector((0, 0, 0))
for r in rows:
    C = mathutils.Vector((r["gt_x"], r["gt_y"], r["gt_z"]))
    d = mathutils.Quaternion((r["gt_qw"], r["gt_qx"], r["gt_qy"], r["gt_qz"])).to_matrix().col[2].normalized()
    P = mathutils.Matrix.Identity(3) - mathutils.Matrix(((d.x * d.x, d.x * d.y, d.x * d.z),
                                                         (d.y * d.x, d.y * d.y, d.y * d.z),
                                                         (d.z * d.x, d.z * d.y, d.z * d.z)))
    A += P
    b += P @ C
bunny_centre = A.inverted() @ b
root = bpy.data.objects["WorldRoot"]
methods = {m["key"]: m for m in slam_animation.methods(scene)}
PREFIX = {"GT": "gt", "PnP": "pnp", "ICP": "icp"}
if n_scratch:
    PREFIX["ScratchPnP"] = "scratch"

worst = {}
records = []
for f in range(N):
    scene.frame_set(f)
    W = root.matrix_world.copy()
    Wrot = W.to_3x3().normalized()
    r = rows[f]
    for key, p in PREFIX.items():
        if key == "ScratchPnP" and f >= n_scratch:
            continue  # no scratch pose at this frame
        pos = mathutils.Vector((r[f"{p}_x"], r[f"{p}_y"], r[f"{p}_z"]))
        R_wc = mathutils.Quaternion((r[f"{p}_qw"], r[f"{p}_qx"], r[f"{p}_qy"], r[f"{p}_qz"])).to_matrix()
        cam = bpy.data.objects[f"{key}_Animated_Camera"]
        loc, rot, _ = cam.matrix_world.decompose()
        exp_loc = W @ pos
        exp_rot = (Wrot @ R_wc @ R_FIX).to_quaternion()
        pos_err = (loc - exp_loc).length
        rot_err = math.degrees(rot.rotation_difference(exp_rot).angle)
        rot_err = min(rot_err, 360.0 - rot_err)
        view_dir = rot.to_matrix() @ mathutils.Vector((0, 0, -1))
        exp_dir = Wrot @ (R_wc @ mathutils.Vector((0, 0, 1)))
        dir_err = math.degrees(view_dir.angle(exp_dir, 0.0))
        # look-at: optical axis vs direction to the bunny centre -- from the data
        # (CV axes, native frame) and from the Blender camera (-Z, world frame)
        lookat_data = math.degrees((R_wc @ mathutils.Vector((0, 0, 1))).angle(bunny_centre - pos, 0.0))
        lookat_blender = math.degrees(view_dir.angle((W @ bunny_centre) - loc, 0.0))
        curve = bpy.data.objects[f"{key}_Trajectory"]
        pt = curve.data.splines[0].points[f].co
        curve_err = ((curve.matrix_world @ mathutils.Vector(pt[:3])) - exp_loc).length
        hud = slam_animation.error_at(scene, methods[key], f)
        if hud is None:
            hud_t_err = hud_r_err = 0.0 if key == "GT" else float("nan")
        else:
            hud_t_err = abs(hud[0] - r[f"{p}_translation_error"])
            hud_r_err = abs(hud[1] - r[f"{p}_rotation_error"])
        rec = dict(frame=f, method=key, camera_position_error_m=pos_err,
                   camera_rotation_error_deg=rot_err, view_direction_error_deg=dir_err,
                   curve_point_error_m=curve_err, lookat_data_deg=lookat_data,
                   lookat_blender_deg=lookat_blender, lookat_diff_deg=abs(lookat_blender - lookat_data), hud_translation_value_diff=hud_t_err,
                   hud_rotation_value_diff=hud_r_err)
        records.append(rec)
        for k, v in rec.items():
            if k not in ("frame", "method"):
                worst[k] = max(worst.get(k, 0.0), v)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
    w.writeheader()
    for rec in records:
        w.writerow({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in rec.items()})

print(f"BLENDER_CHECK frames={N} frame_start={scene.frame_start} frame_end={scene.frame_end} "
      f"methods={list(PREFIX)} scratch_frames={n_scratch}")
for k, v in worst.items():
    print(f"BLENDER_CHECK max_{k}={v:.3e}")

# Per-method look-at summary and the requested frames, straight from the scene.
print(f"BLENDER_CHECK scratch_source={scratch_path.relative_to(ROOT) if n_scratch else 'none'}")
for key in PREFIX:
    lk = [rec["lookat_blender_deg"] for rec in records if rec["method"] == key]
    print(f"BLENDER_LOOKAT {key:<10} max {max(lk):7.2f} deg  mean {sum(lk) / len(lk):6.2f} deg  ({len(lk)} frames)")
for f in (0, 1, 2, 10, 11, 35):
    if f >= N:
        continue
    scene.frame_set(f)
    for key in PREFIX:
        rec = next((x for x in records if x["frame"] == f and x["method"] == key), None)
        if rec is None:
            continue
        cam = bpy.data.objects[f"{key}_Animated_Camera"]
        loc = cam.matrix_world.translation
        print(f"BLENDER_FRAME {f:2d} {key:<10} world pos ({loc.x:+.4f}, {loc.y:+.4f}, {loc.z:+.4f})  "
              f"pos diff {rec['camera_position_error_m']:.1e} m  rot diff {rec['camera_rotation_error_deg']:.1e} deg  "
              f"look-at {rec['lookat_blender_deg']:7.2f} deg (data {rec['lookat_data_deg']:7.2f})  "
              f"hidden {cam.hide_viewport}")
