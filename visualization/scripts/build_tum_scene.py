"""Blender scene of a real TUM RGB-D run: reconstructed RGB-D point cloud + GT, PnP and ICP camera motion.

Input is an export of export_tum_scene.py (run it first, outside Blender); Blender never decodes the dataset
images to build the scene. The Bunny scene (build_scene.py, scenes/bunny_slam_demo.blend) is not touched; this
reuses its method colours (build_scene.METHOD_STYLES) and its single pose conversion
(slam_coords.trajectory_to_blender_pose).

Usage (repository root):
  blender --background --python visualization/scripts/build_tum_scene.py -- \
      --export results/runs/blender/rgbd_dataset_freiburg1_xyz/<run> [--density preview|render] [--frame 74] \
      [--save visualization/scenes/tum/<name>.blend] [--render out.png] [--view overview|gt|pnp|icp] [--check]
  blender visualization/scenes/tum/<name>.blend          (interactive; allow scripts for the "TUM SLAM" panel)

Coordinate conventions (no other transform anywhere):
  - Blender world = the TUM world frame (ground truth's motion-capture frame, z up), 1 Blender unit = 1 m.
    There is no WorldRoot rotation or recentring here (unlike the Bunny scene, whose C++ world is y up).
  - Poses are T_wc as exported (X_world = R_wc X_cam + t_wc; camera axes x right, y down, z forward).
  - Point cloud vertices are already world points (exporter: X_world = R_wc^GT X_cam + t_wc^GT).
  - Frustum wireframes are built from T_wc directly: corner rays ((u - cx)/fx, (v - cy)/fy, 1) * s mapped by T_wc.
  - Blender Camera objects get R_wc @ R_FIX (their local axes look down -z, y up); position unchanged.
    Lens and principal point from the intrinsics: lens = fx * 36 / W (sensor width 36 mm, fit horizontal),
    shift_x = (W/2 - cx) / W, shift_y = (cy - H/2) / W.

Objects (collections): Reconstruction_placed_with_GT / _PnP / _ICP (three independent clouds from the same
pixels, export_tum_scene.py + tum_reconstruction.py; the panel's "Reconstruction" selector shows one and draws
the trajectory that placed it 2.5x thicker); GT / PnP / ICP (trajectory curve, animated camera keyframed at every frame,
frustums at selected frames, start sphere, end cube); Connectors (GT centre -> estimated centre of the same
frame, every N frames); Aligned_optional (one global SE(3) per estimate, hidden by default); Overview_Camera.
Blender frame f shows trajectory frame f - 1. --check verifies the built scene numerically against the export.
"""

import argparse
import json
import math
import os
import sys

import bpy
import mathutils
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
from slam_coords import R_FIX, trajectory_to_blender_pose  # noqa: E402
from build_scene import METHOD_STYLES  # noqa: E402  (module-level constants only; its main() is not run)
from build_tum_scene_io import read_ply  # noqa: E402

METHODS = (("gt", "GT"), ("pnp", "PnP"), ("icp", "ICP"))
POINT_RADIUS = {"preview": 0.006, "render": 0.0035}
TUBE = 0.004            # trajectory curve radius (m)
FRUSTUM_DEPTH = 0.07    # frustum length along the optical axis (m)
FRUSTUM_EVERY = 25      # static frustums every N frames (+ first, last, --frame)
CONNECTOR_EVERY = 5
PANEL_TEXT = "tum_scene_panel.py"


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--density", choices=("preview", "render"), default="preview")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--save")
    ap.add_argument("--render")
    ap.add_argument("--view", default="overview", choices=("overview", "closeup", "gt", "pnp", "icp"))
    ap.add_argument("--mode", default="gt", choices=("gt", "pnp", "icp"), help="reconstruction shown on load")
    ap.add_argument("--render-all", help="render every reconstruction mode from the same camera: <prefix>_<mode>.png")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--check", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def collection(name, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def emission_material(name, srgb, attribute=None, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    if attribute:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "GEOMETRY"
        at.attribute_name = attribute
        nt.links.new(at.outputs["Color"], em.inputs["Color"])
    else:
        em.inputs["Color"].default_value = (*srgb_to_linear(srgb), 1.0)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*srgb_to_linear(srgb), 1.0) if srgb is not None else (0.7, 0.7, 0.7, 1)
    return mat


def point_cloud(path, radius, col, name, mat):
    """One mesh object (vertices only) + a geometry-nodes Mesh to Points modifier: a single object for the
    whole cloud, drawn as points in the viewport and in renders. Vertices are the exported world points as is."""
    xyz, rgb = read_ply(path)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(xyz))
    me.vertices.foreach_set("co", xyz.astype(np.float32).ravel())
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    lin = np.c_[srgb_to_linear(rgb / 255.0), np.ones(len(rgb))].astype(np.float32)
    ca.data.foreach_set("color", lin.ravel())
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ng = bpy.data.node_groups.new(f"{name}_points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    mp = ng.nodes.new("GeometryNodeMeshToPoints")
    mp.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    ng.links.new(gi.outputs[0], mp.inputs["Mesh"])
    ng.links.new(mp.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    mod = ob.modifiers.new("Points", "NODES")
    mod.node_group = ng
    return ob, xyz


def curve(name, segments, mat, col, radius=TUBE):
    """A curve object with one poly spline per segment list [[p0, p1, ...], ...] (rendered tubes)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    for pts in segments:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        sp.points.foreach_set("co", np.c_[np.asarray(pts, float), np.ones(len(pts))].ravel())
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    col.objects.link(ob)
    return ob


def R_of(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array(mathutils.Quaternion((w, x, y, z)).to_matrix())


def frustum_segments(p, R, K, s):
    W, H = K["width"], K["height"]
    c = np.asarray(p, float)
    corners = [c + R @ (np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0]) * s)
               for u, v in ((0, 0), (W, 0), (W, H), (0, H))]
    segs = [[c, q] for q in corners] + [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    segs.append([corners[0] + 0.35 * (corners[1] - corners[0]), (corners[0] + corners[1]) / 2 - R[:, 1] * 0.25 * s,
                 corners[0] + 0.65 * (corners[1] - corners[0])])  # "up" tick on the top edge (image row 0)
    return segs


def set_intrinsics(cam, K):
    W, H = K["width"], K["height"]
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.lens = K["fx"] * 36.0 / W
    cam.shift_x = (W / 2 - K["cx"]) / W
    cam.shift_y = (K["cy"] - H / 2) / W
    cam.clip_start, cam.clip_end = 0.05, 50.0


def build(a):
    exp = os.path.abspath(a.export)
    D = json.load(open(os.path.join(exp, "trajectory.json")))
    K, F, n = D["intrinsics"], D["frames"], len(D["frames"])
    bpy.ops.wm.read_homefile(use_empty=True)
    sc = bpy.context.scene
    sc.name = "TUM_SLAM"
    sc["tum_data"] = json.dumps(D)
    sc["tum_export_dir"] = exp
    sc["tum_density"] = a.density
    sc.frame_start, sc.frame_end = 1, n
    sc.render.fps = 10
    sc.render.resolution_x, sc.render.resolution_y = K["width"], K["height"]
    world = bpy.data.worlds.new("World")
    world.color = (0.02, 0.02, 0.025)
    sc.world = world
    # three independent reconstructions (same pixels, different pose source); one visible at a time
    cloud_mat = emission_material("PointCloud_RGB", None, attribute="Col")
    clouds = {}
    for m, key in METHODS:
        c = collection(f"Reconstruction_placed_with_{key}")
        info = D["clouds"][a.density][m]
        clouds[m], _ = point_cloud(os.path.join(exp, info["file"]), POINT_RADIUS[a.density], c,
                                   f"PointCloud_placed_with_{key}_poses", cloud_mat)
        clouds[m]["tum_pose_source"] = m
        clouds[m]["tum_manifest"] = os.path.join(exp, f"manifest_{a.density}_{m}.json")
    sel = sorted(set(range(0, n, FRUSTUM_EVERY)) | {0, n - 1} | ({a.frame} if a.frame is not None else set()))
    objs = {}
    for m, key in METHODS:
        st = METHOD_STYLES[key]
        col = collection(key)
        mat = emission_material(f"{key}_color", st["color"], strength=2.0)
        P = np.array([f[m]["p"] for f in F])
        objs[f"{key}_Trajectory"] = curve(f"{key}_Trajectory", [P], mat, col)
        segs = []
        for k in sel:
            segs += frustum_segments(F[k][m]["p"], R_of(F[k][m]["q_xyzw"]), K, FRUSTUM_DEPTH)
        objs[f"{key}_Frustums"] = curve(f"{key}_Frustums", segs, mat, col, radius=TUBE * 0.45)
        objs[f"{key}_Frustums"]["tum_frames"] = sel
        for k, (kind, nm) in ((0, ("ico", "Start")), (n - 1, ("cube", "End"))):
            if kind == "ico":
                bpy.ops.mesh.primitive_ico_sphere_add(radius=0.012, location=F[k][m]["p"])
            else:
                bpy.ops.mesh.primitive_cube_add(size=0.02, location=F[k][m]["p"])
            ob = bpy.context.active_object
            ob.name = f"{key}_{nm}"
            ob.data.materials.append(mat)
            for c in ob.users_collection:
                c.objects.unlink(ob)
            col.objects.link(ob)
        cam = bpy.data.cameras.new(f"{key}_Camera")
        set_intrinsics(cam, K)
        cam.display_size = 0.08
        co = bpy.data.objects.new(f"{key}_Camera", cam)
        col.objects.link(co)
        co.rotation_mode = "QUATERNION"
        for k, f in enumerate(F):
            x, y, z, w = f[m]["q_xyzw"]
            pos, q = trajectory_to_blender_pose(mathutils.Vector(f[m]["p"]), mathutils.Quaternion((w, x, y, z)))
            co.location, co.rotation_quaternion = pos, q
            co.keyframe_insert("location", frame=k + 1)
            co.keyframe_insert("rotation_quaternion", frame=k + 1)
        for fc in (co.animation_data.action.fcurves if hasattr(co.animation_data.action, "fcurves") else []):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"  # a pose belongs to one frame; nothing between frames
        objs[f"{key}_Camera"] = co
    con = collection("Connectors")
    for m, key in METHODS[1:]:
        st = METHOD_STYLES[key]
        mat = emission_material(f"{key}_connector", tuple(0.55 * c for c in st["color"]), strength=1.5)
        segs = [[F[k]["gt"]["p"], F[k][m]["p"]] for k in sorted(set(range(0, n, CONNECTOR_EVERY)) | {n - 1})]
        curve(f"{key}_Connectors", segs, mat, con, radius=TUBE * 0.4)
    al = collection("Aligned_optional (one global SE(3) per estimate)")
    for m, key in METHODS[1:]:
        A = D["aligned_for_optional_view"][m]
        P = np.array([f[m]["p"] for f in F]) @ np.array(A["R"]).T + np.array(A["t"])
        curve(f"{key}_Trajectory_aligned", [P], emission_material(f"{key}_aligned", METHOD_STYLES[key]["color"], strength=0.6), al)
    lc = bpy.context.view_layer.layer_collection.children[al.name]
    lc.exclude = True
    # overview camera: behind and above the mean GT camera, along its mean optical axis
    Rg = np.array([R_of(f["gt"]["q_xyzw"]) for f in F])
    axis = Rg[:, :, 2].mean(0)
    axis /= np.linalg.norm(axis)
    Pg = np.array([f["gt"]["p"] for f in F])
    # frames the camera motion and the first ~0.6 m of scene in front of it (same rule as diagnostics/scene_3d.py)
    target = Pg.mean(0) + 0.45 * axis
    direction = -axis + np.array([0, 0, 0.9])
    direction /= np.linalg.norm(direction)
    eye = target + direction * (np.linalg.norm(np.ptp(Pg, axis=0)) / 2 + 0.45) / math.tan(math.radians(25)) * 1.1
    oc = bpy.data.objects.new("Overview_Camera", bpy.data.cameras.new("Overview_Camera"))
    oc.data.lens = 24
    sc.collection.objects.link(oc)
    oc.location = eye
    oc.rotation_euler = (mathutils.Vector(target) - mathutils.Vector(eye)).to_track_quat("-Z", "Z").to_euler()
    # close-up: from 0.9 m behind the mean GT camera towards the scene in front of it (the monitors)
    cc = bpy.data.objects.new("Closeup_Camera", bpy.data.cameras.new("Closeup_Camera"))
    cc.data.lens = 30
    sc.collection.objects.link(cc)
    tgt = Pg.mean(0) + 0.9 * axis
    cc.location = Pg.mean(0) - 0.25 * axis + np.array([0, 0, 0.25])
    cc.rotation_euler = (mathutils.Vector(tgt) - cc.location).to_track_quat("-Z", "Z").to_euler()
    views = {"overview": oc, "closeup": cc}
    sc.camera = views[a.view] if a.view in views else objs[f"{dict(METHODS)[a.view]}_Camera"]
    if a.view != "overview":  # close-up or looking through an estimated / GT camera: render the point cloud only
        for c in bpy.data.collections:
            if not c.name.startswith("Reconstruction_"):
                c.hide_render = True
    sc.render.engine = "BLENDER_EEVEE"
    sc.view_settings.view_transform = "Standard"
    sc["tum_methods"] = "GT green, PnP orange, ICP magenta (build_scene.METHOD_STYLES)"
    # panel (embedded, registered on load)
    txt = bpy.data.texts.new(PANEL_TEXT)
    txt.write(open(os.path.join(SCRIPT_DIR, PANEL_TEXT)).read())
    txt.use_module = True
    exec(compile(txt.as_string(), PANEL_TEXT, "exec"), {"__name__": "__main__"})
    k0 = a.frame if a.frame is not None else 0
    sc.tum_frame = k0
    sc.frame_set(k0 + 1)
    sc.tum_recon = a.mode  # shows that reconstruction, emphasises the trajectory that placed it
    return D, objs, clouds, sel


def check(D, objs, clouds, density, export):
    """Numeric checks of the built scene against the export (which holds the trajectory.csv values)."""
    sc, F, K = bpy.context.scene, D["frames"], D["intrinsics"]
    worst = {"pos": 0.0, "rot": 0.0}
    for k in range(len(F)):
        sc.frame_set(k + 1)
        for m, key in METHODS:
            M = objs[f"{key}_Camera"].matrix_world
            R_wc = np.array(M.to_3x3()) @ np.array(R_FIX)
            worst["pos"] = max(worst["pos"], float(np.abs(np.array(M.translation) - F[k][m]["p"]).max()))
            worst["rot"] = max(worst["rot"], float(np.abs(R_wc - R_of(F[k][m]["q_xyzw"])).max()))
    ok = worst["pos"] < 1e-5 and worst["rot"] < 1e-5
    print(f"[check] animated cameras vs trajectory (all {len(F)} frames x 3): max |dp| {worst['pos']:.2e} m, "
          f"max |dR| {worst['rot']:.2e} -> {'OK' if ok else 'FAIL'}")
    # Blender camera looks along -Z local: its world view direction must equal R_wc[:, 2] (CV optical axis)
    sc.frame_set(1)
    view = np.array(objs["GT_Camera"].matrix_world.to_3x3()) @ np.array([0, 0, -1.0])
    ok_dir = np.allclose(view, R_of(F[0]["gt"]["q_xyzw"])[:, 2], atol=1e-6)
    print(f"[check] Blender camera view direction = CV optical axis R_wc[:,2]: {'OK' if ok_dir else 'FAIL'}")
    # static frustum apex = camera centre, far corners in front of the camera
    cu = objs["GT_Frustums"].data
    k = objs["GT_Frustums"]["tum_frames"][0]
    pts = np.array([tuple(p.co)[:3] for p in cu.splines[0].points])
    R = R_of(F[k]["gt"]["q_xyzw"])
    ok_fr = np.allclose(pts[0], F[k]["gt"]["p"], atol=1e-6) and (R.T @ (pts[1] - pts[0]))[2] > 0
    print(f"[check] frustum apex at the camera centre and corners at +z in camera coordinates: {'OK' if ok_fr else 'FAIL'}")
    ok_pc = True
    mans = {}
    for m, ob in clouds.items():
        man = mans[m] = json.load(open(ob["tum_manifest"]))
        xyz, _ = read_ply(os.path.join(export, man["cloud_file"]))
        co = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        same = len(co) == xyz.size and np.array_equal(co.reshape(-1, 3), xyz.astype(np.float32))
        src_ok = man["pose_source"] == m and man["uses_ground_truth"] == (m == "gt")
        ok_pc &= same and src_ok
        print(f"[check] {ob.name}: {len(xyz)} vertices identical to {man['cloud_file']}: {same}; manifest pose source "
              f"{man['pose_source']}, uses ground truth {man['uses_ground_truth']}; frames used {man['frames_processed']}, "
              f"skipped {[x['frame'] for x in man['frames_skipped']]}")
    shared = all(mans[m]["frames_selected"] == mans["gt"]["frames_selected"] and
                 mans[m]["sampling"]["per_frame_pixel_sha256"] == mans["gt"]["sampling"]["per_frame_pixel_sha256"]
                 for m in mans)
    ok_pc &= shared
    print(f"[check] same selected frames and identical sampled pixels (hash per frame) in all three modes: {shared}")
    worst_k = int(np.argmax([f["pnp"]["pos_err_m"] for f in F]))
    sc.frame_set(worst_k + 1)
    d = np.linalg.norm(np.array(objs["PnP_Camera"].matrix_world.translation) - np.array(objs["GT_Camera"].matrix_world.translation))
    print(f"[check] largest PnP position error: frame {worst_k}: scene {d:.4f} m, trajectory.csv {F[worst_k]['pnp']['pos_err_m']:.4f} m")
    return ok and ok_dir and ok_fr and ok_pc and abs(d - F[worst_k]["pnp"]["pos_err_m"]) < 1e-5


def main():
    a = args()
    D, objs, clouds, sel = build(a)
    good = check(D, objs, clouds, a.density, os.path.abspath(a.export)) if a.check else True
    sc = bpy.context.scene
    sc.frame_set((a.frame if a.frame is not None else 0) + 1)  # back to the selected frame after the checks
    if a.render:
        sc.eevee.taa_render_samples = a.samples
        sc.render.filepath = os.path.abspath(a.render)
        bpy.ops.render.render(write_still=True)
        print(f"[build_tum_scene] rendered {sc.render.filepath} through {sc.camera.name} at frame {sc.frame_current - 1}")
    if a.render_all:
        sc.eevee.taa_render_samples = a.samples
        for m in ("gt", "pnp", "icp"):  # same camera, frame, density and overlays; only the reconstruction differs
            sc.tum_recon = m
            sc.render.filepath = os.path.abspath(f"{a.render_all}_{m}.png")
            bpy.ops.render.render(write_still=True)
            print(f"[build_tum_scene] rendered {sc.render.filepath} (cloud placed with {m} poses) through {sc.camera.name}")
        sc.tum_recon = a.mode
    save = a.save or os.path.join(REPO_ROOT, "visualization", "scenes", "tum",
                                  f"{D['sequence']}__{os.path.basename(os.path.abspath(a.export))}__{a.density}.blend")
    if os.path.abspath(save) == os.path.join(REPO_ROOT, "visualization", "scenes", "bunny_slam_demo.blend"):
        raise SystemExit("refusing to overwrite the Bunny scene")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=save)
    print(f"[build_tum_scene] saved {save} ({len(D['frames'])} frames, frustums at {sel})")
    if not good:
        print("[build_tum_scene] CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
