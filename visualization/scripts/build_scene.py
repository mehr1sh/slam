"""Builds the Blender visualization comparing three camera trajectories
around the Stanford Bunny: ground truth (GT), PnP(RANSAC)-estimated, and
ICP-estimated. It answers one question: where did PnP and ICP think the
camera was, compared with where it actually was?

This script does NOT compute any trajectory, pose-estimation, or pose-
composition math itself -- it only reads the files the existing C++
pipeline already produces:
  - data/meshes/bunny/bun_zipper.ply          (Stanford Bunny mesh)
  - data/synthetic_bunny/intrinsics.txt       ("width height fx fy cx cy")
  - data/synthetic_bunny/slam_trajectory.csv  (per-frame GT/PnP/ICP positions,
                                                quaternions, and errors --
                                                see tests/synthetic/slam_trajectory_test.cpp)
  - data/synthetic_bunny/groundtruth.txt,
    pnp_trajectory.txt, icp_trajectory.txt    (cross-checked against the
                                                CSV's own columns below,
                                                not blindly trusted to agree)
PnP/ICP/trajectory-composition math (T_wc[i+1] = T_wc[i] * T_rel(i,i+1)^-1)
stays owned entirely by tests/synthetic/slam_trajectory_test.cpp in C++; this
script only reads, converts to Blender's coordinate representation, and
builds visualization objects from already-computed numbers. The drift
between GT/PnP/ICP is the real, unmodified C++ result -- nothing here
rescales, aligns, or smooths it.

Exactly THREE real Blender Camera objects (obj.type == 'CAMERA') show the
camera poses: GT_Animated_Camera, PnP_Animated_Camera, ICP_Animated_Camera.
Each is keyframed at every frame i with its trajectory's pose i
(location + quaternion rotation, LINEAR between frames), so the same three
objects move along their full trajectory lines as the timeline plays. Any of
them can be made the scene camera (bpy.context.scene.camera = ...) and looked
through with Numpad 0 -- the view then follows that trajectory during
playback. Blender draws their frustums; the viewport overlay only re-draws
each camera's own frustum in the method's color. slam_hud_panel.py's
frame_change_post handler updates the HUD, labels and (optional) debug
extras; it does not move the cameras -- the keyframes do.

Run with:
    blender --python visualization/scripts/build_scene.py            (interactive GUI)
    blender --background --python visualization/scripts/build_scene.py  (headless, just saves the .blend)

=== Coordinate system (see slam_coords.py for the shared constants) ===

Every object is first placed/sized using the RAW C++ world frame (Y-up,
unmodified numbers straight from the trajectory files). Everything is then
parented under one "WorldRoot" empty carrying a fixed Y-up -> Z-up display
rotation PLUS a translation that recenters the bunny at the world origin
with its base at Z=0. Because this same WorldRoot transform is applied
identically to the bunny AND all trajectories, their relative geometry --
including the GT/PnP/ICP drift itself -- is completely unaffected; only the
whole rig's orientation in Blender's world changes.

Trajectory poses are T_wc (camera -> world), camera axes in CV/OpenCV
convention: +X right, +Y down, +Z forward. Blender cameras look down their
local -Z with +Y up, so every camera placement goes through
slam_coords.trajectory_to_blender_pose() (a fixed local 180-degree rotation
about the camera's own X axis -- the only pose conversion, unchanged from
the previous frustum visualization).
"""

import csv
import math
import os
import sys

import bpy
import mathutils
from bpy_extras import anim_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATION_ROOT = os.path.dirname(SCRIPT_DIR)  # visualization/scripts/ -> visualization/
REPO_ROOT = os.path.dirname(VISUALIZATION_ROOT)    # visualization/ -> repo root

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import slam_animation
import slam_hud_panel
from slam_coords import WORLD_ROOT_ROTATION, trajectory_to_blender_pose

MESH_PATH = os.path.join(REPO_ROOT, "data", "meshes", "bunny", "bun_zipper.ply")
DATASET_DIR = os.path.join(REPO_ROOT, "data", "synthetic_bunny")
INTRINSICS_PATH = os.path.join(DATASET_DIR, "intrinsics.txt")
GROUNDTRUTH_PATH = os.path.join(DATASET_DIR, "groundtruth.txt")
PNP_TRAJECTORY_PATH = os.path.join(DATASET_DIR, "pnp_trajectory.txt")
ICP_TRAJECTORY_PATH = os.path.join(DATASET_DIR, "icp_trajectory.txt")
TRAJECTORY_CSV_PATH = os.path.join(DATASET_DIR, "slam_trajectory.csv")
SAVE_PATH = os.path.join(VISUALIZATION_ROOT, "scenes", "bunny_slam_demo.blend")
RENDER_OUTPUT_PATH = os.path.join(VISUALIZATION_ROOT, "output", "render", "frame_")
# Stored in the .blend relative to the .blend itself (visualization/scenes/),
# so the saved scene contains no machine-specific absolute path.
RENDER_OUTPUT_BLEND_PATH = "//../output/render/frame_"

# =========================================================================
# Visualization options. These are the INITIAL values of the toggles in the
# View3D sidebar (N) > "Trajectories" tab, where every one of them can also
# be changed live (and is saved in the .blend).
# =========================================================================
SHOW_GROUND_TRUTH = True
SHOW_PNP = True
SHOW_ICP = True

SHOW_CAMERAS = True               # the three animated GT/PnP/ICP cameras
SHOW_WORLD_AXES = True            # small X/Y/Z triad of the C++ world frame
SHOW_HUD = True                   # top-left viewport HUD

# Optional debug extra, OFF by default: additional STATIC real cameras at
# every CAMERA_SAMPLE_STEP-th pose (plus the last), named
# <key>_History_Camera_###, in <key>_Historical_Cameras collections, drawn
# smaller than the animated cameras. Build-time: rerun build_scene.py after
# changing either value (the sidebar toggle then shows/hides them).
SHOW_HISTORICAL_CAMERAS = False
CAMERA_SAMPLE_STEP = 5

# True: clean comparison. False (debug): additionally frame numbers at the
# cameras, red GT->estimate error vectors, and each method's camera center
# in the HUD.
PRESENTATION_MODE = True

PROGRESSIVE_TRAJECTORIES = False  # False: full trajectory lines all the time; True: grow frame by frame

ANIMATION_FPS = 8                 # one Blender frame per pose; 8 fps -> 36 poses in 4.5 s

# --- Purely visual/display sizes, all as a fraction of the bunny's bounding
# radius (~0.125 m) -- independent of camera calibration and trajectory math.
LINE_THICKNESS_SCALE = 0.008      # trajectory tube radius (x per-method thickness below)
ANIMATED_CAMERA_DISPLAY_SCALE = 0.45   # camera.display_size of the three animated cameras
HISTORY_CAMERA_DISPLAY_SCALE = 0.25    # camera.display_size of the optional historical cameras
AXES_LENGTH_SCALE = 0.45          # world-axes arrow length

# Visual-only uniform bunny scale about its own centroid. Kept at 1.0: the
# C++ renderer that produced the dataset images drew the mesh unscaled, and
# the orbit radius is only 4x the bunny's bounding radius, so any other
# value makes the bunny/camera geometry disagree with the data (2.5 put the
# bunny's bounding sphere within ~0.19 m of the camera path). The bunny is
# made prominent by framing and lighting instead.
BUNNY_DISPLAY_SCALE = 1.0
ADD_GROUND_PLANE = True           # subtle plane in renders; the viewport uses a fine floor grid

# Colorblind-safe palette (sRGB), readable on the dark background. Every
# method keeps ONE color across its trajectory line, camera frustums,
# labels and legend.
METHOD_STYLES = {
    "GT": dict(label="Ground Truth", short="GT", prefix="gt", color=(0.15, 0.82, 0.45),
               color_name="green", thickness=1.0),
    "PnP": dict(label="PnP", short="PnP", prefix="pnp", color=(1.00, 0.62, 0.10),
                color_name="orange", thickness=1.0),
    "ICP": dict(label="ICP", short="ICP", prefix="icp", color=(0.85, 0.40, 0.95),
                color_name="magenta", thickness=1.0),
}
METHOD_COLLECTIONS = {"GT": "GroundTruth", "PnP": "PnP", "ICP": "ICP"}
DRIFT_COLOR = (0.95, 0.15, 0.12)
AXIS_COLORS = {"X": (0.80, 0.36, 0.36), "Y": (0.42, 0.72, 0.42), "Z": (0.40, 0.52, 0.85)}  # muted
AUTOLOAD_TEXT_NAME = "slam_viz_autoload.py"


# =========================================================================
# Data loading (read-only)
# =========================================================================

def read_intrinsics(path):
    with open(path) as f:
        w, h, fx, fy, cx, cy = f.readline().split()
    return dict(width=int(w), height=int(h), fx=float(fx), fy=float(fy),
                cx=float(cx), cy=float(cy))


def read_tum_trajectory(path):
    """TUM-format T_wc poses ("timestamp tx ty tz qx qy qz qw"). Only used
    for the cross-checks in verify_trajectory_conventions() -- the CSV is the
    actual source for everything drawn."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in line.split())
            poses.append(dict(pos=mathutils.Vector((tx, ty, tz)),
                              quat_cv=mathutils.Quaternion((qw, qx, qy, qz))))
    return poses


def read_slam_trajectory_csv(path):
    """Reads the C++-produced per-frame rows (positions, quaternions, errors)
    as a list of dicts of floats ("frame" as int). Stored verbatim on the
    scene for slam_animation.py / the HUD; poses are looked up per method
    with slam_coords.pose_from_row(row, prefix)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "frame" else int(float(v))) for k, v in row.items()})
    return rows


def resolve_methods(rows):
    """Legend-ordered method table (GT, PnP, ICP) stored as
    scene["slam_viz_methods"]: labels, CSV column prefix, color, and which
    C++ error columns exist. No pose math."""
    columns = set(rows[0].keys())
    has_err = lambda p: {f"{p}_translation_error", f"{p}_rotation_error"} <= columns
    table = []
    for key in ("GT", "PnP", "ICP"):
        st = METHOD_STYLES[key]
        table.append(dict(key=key, label=st["label"], short=st["short"], prefix=st["prefix"],
                          color=list(st["color"]), color_name=st["color_name"],
                          error_prefix=st["prefix"] if (key != "GT" and has_err(st["prefix"])) else ""))
    return table


def verify_trajectory_conventions(rows):
    """Checks, rather than assumes, that the CSV-sourced trajectories share
    the same convention/anchor before anything is drawn: (a) the CSV's GT,
    PnP and ICP columns match the standalone groundtruth.txt /
    pnp_trajectory.txt / icp_trajectory.txt files (confirms the CSV wasn't
    regenerated from something stale/different), and (b) all trajectories
    start at the same frame-0 pose (confirms PnP/ICP were actually anchored
    to the GT frame-0 pose as documented, not left in some other frame).
    """
    from slam_coords import pose_from_row

    print("\n=== TRAJECTORY CONVENTION CHECK ===")
    for label, path, prefix in (("groundtruth.txt", GROUNDTRUTH_PATH, "gt"),
                                ("pnp_trajectory.txt", PNP_TRAJECTORY_PATH, "pnp"),
                                ("icp_trajectory.txt", ICP_TRAJECTORY_PATH, "icp")):
        if not os.path.isfile(path):
            print(f"{label}: not found, skipped")
            continue
        txt = read_tum_trajectory(path)
        if len(txt) != len(rows):
            print(f"{label}: {len(txt)} poses vs CSV {len(rows)} -- MISMATCH in frame count!")
            continue
        diff = max((pose_from_row(r, prefix)[0] - p["pos"]).length for r, p in zip(rows, txt))
        print(f"CSV {prefix}_* vs {label}: max position diff {diff:.6f} m "
              f"({'OK' if diff < 1e-4 else 'MISMATCH -- CSV disagrees with ' + label + '!'})")

    gt0 = pose_from_row(rows[0], "gt")[0]
    for prefix in ("pnp", "icp"):
        d = (gt0 - pose_from_row(rows[0], prefix)[0]).length
        print(f"frame 0: GT vs {prefix.upper()} position diff: {d:.6f} m "
              f"({'OK, common anchor confirmed' if d < 1e-4 else 'MISMATCH -- not anchored to GT!'})")
    print("=== END TRAJECTORY CONVENTION CHECK ===\n")


def native_poses(rows, prefix):
    from slam_coords import pose_from_row
    return [dict(zip(("pos", "quat_cv"), pose_from_row(r, prefix))) for r in rows]


# =========================================================================
# Scene housekeeping
# =========================================================================

OWNED_COLLECTIONS = (
    # current layout
    "Bunny", "GroundTruth", "PnP", "ICP", "Visualization",
    "GT_Historical_Cameras", "PnP_Historical_Cameras", "ICP_Historical_Cameras",
    "WorldAxes", "Error Vectors", "Lighting",
    # previous layouts (so rebuilding on top of an old .blend leaves no leftovers,
    # including the removed SLAM / Interactive_Camera parts)
    "SLAM", "SLAM_Cameras", "SLAM_Markers", "SLAM_Current_Camera", "SLAM_Historical_Cameras",
    "GT_Cameras", "PnP_Cameras", "ICP_Cameras",
    "GT_Current_Camera", "PnP_Current_Camera", "ICP_Current_Camera",
    "GT_Markers", "PnP_Markers", "ICP_Markers",
    "Axes", "Reference Cameras", "Overview Camera", "Presentation Camera",
    "GT Camera", "Ghost Cameras", "PnP Camera", "ICP Camera", "Drift Vectors",
    "Interactive Camera", "Visualization Camera",
)


def get_or_create_collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def clear_previous_scene_objects():
    # Current objects plus everything earlier versions created (SLAM_*,
    # Interactive_Camera, *_Marker dots, *_History_Camera_### / *_Current_Camera
    # curve frustums and their *_Center dots, Presentation_Camera, ...).
    names_prefixes = ("Bunny", "GT_", "PnP_", "ICP_", "SLAM_", "Interactive_Camera",
                      "Frustum_", "WorldAxes", "GazeLine", "WorldRoot", "GroundPlane",
                      "Visualization_Camera", "Presentation_Camera",
                      "Key_Light", "Fill_Light", "Rim_Light", "FillLight")
    exact_names = ("Cube", "Camera", "Light")  # Blender's default-startup-scene clutter
    for obj in list(bpy.data.objects):
        if obj.name.startswith(names_prefixes) or obj.name in exact_names:
            bpy.data.objects.remove(obj, do_unlink=True)
    for name in OWNED_COLLECTIONS:
        coll = bpy.data.collections.get(name)
        if coll is not None:
            bpy.data.collections.remove(coll)
    text = bpy.data.texts.get(AUTOLOAD_TEXT_NAME)
    if text is not None:
        bpy.data.texts.remove(text)
    # Free the old meshes/curves/materials so rebuilt datablocks get their
    # exact names back instead of ".001" suffixes.
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)


def remove_empty_default_collection():
    coll = bpy.data.collections.get("Collection")
    if coll is not None and len(coll.objects) == 0 and len(coll.children) == 0:
        for scene in bpy.data.scenes:
            if coll.name in scene.collection.children:
                scene.collection.children.unlink(coll)
        bpy.data.collections.remove(coll)


def link_only_to(obj, collection):
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    collection.objects.link(obj)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_material(name, srgb, emission=0.0, roughness=0.5):
    """Principled material; `emission` > 0 makes it self-lit so line colors
    read the same regardless of lighting. diffuse_color is what Solid
    viewport shading (color = Material) shows."""
    lin = tuple(srgb_to_linear(c) for c in srgb) + (1.0,)
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = lin
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = lin
        bsdf.inputs["Roughness"].default_value = roughness
        if emission > 0:
            bsdf.inputs["Emission Color"].default_value = lin
            bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def tag(obj, method, role, frame=None):
    obj["slam_method"] = method
    obj["slam_role"] = role
    if frame is not None:
        obj["slam_frame"] = frame


def overlay_object(obj):
    """Lines/markers are annotations: they must not cast shadows onto the
    bunny or be picked by accident."""
    obj.visible_shadow = False
    obj.hide_select = True


# =========================================================================
# Bunny, ground, lighting
# =========================================================================

def import_bunny_raw():
    """Just the import + material/shading -- no transform decisions here."""
    bpy.ops.wm.ply_import(filepath=MESH_PATH)
    obj = bpy.context.view_layer.objects.active
    obj.name = "Bunny"
    obj.data.materials.clear()
    obj.data.materials.append(make_material("BunnyClay", (0.86, 0.84, 0.80), roughness=0.45))

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    return obj


def measure_bounds(vertices_local):
    xs = [v.x for v in vertices_local]
    ys = [v.y for v in vertices_local]
    zs = [v.z for v in vertices_local]
    bounds = dict(x=(min(xs), max(xs)), y=(min(ys), max(ys)), z=(min(zs), max(zs)))
    center = mathutils.Vector(((bounds["x"][0] + bounds["x"][1]) / 2,
                               (bounds["y"][0] + bounds["y"][1]) / 2,
                               (bounds["z"][0] + bounds["z"][1]) / 2))
    return bounds, center


def print_bounds(label, bounds):
    print(f"{label}:")
    print(f"  X: [{bounds['x'][0]:.4f}, {bounds['x'][1]:.4f}]  (dim {bounds['x'][1]-bounds['x'][0]:.4f})")
    print(f"  Y: [{bounds['y'][0]:.4f}, {bounds['y'][1]:.4f}]  (dim {bounds['y'][1]-bounds['y'][0]:.4f})")
    print(f"  Z: [{bounds['z'][0]:.4f}, {bounds['z'][1]:.4f}]  (dim {bounds['z'][1]-bounds['z'][0]:.4f})")


def build_bunny_hierarchy(bunny_obj, bunny_center, collection):
    """Bunny_Root (uniform scale about the centroid) -> Bunny (mesh, whose
    own .location cancels the centroid offset so the parent's scale pivots
    exactly on it). No rotation here -- WorldRoot (applied later, to
    everything uniformly) handles that.
    """
    root = bpy.data.objects.new("Bunny_Root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.02
    root.location = bunny_center
    root.scale = (BUNNY_DISPLAY_SCALE,) * 3
    collection.objects.link(root)

    link_only_to(bunny_obj, collection)
    bunny_obj.parent = root
    bunny_obj.location = -bunny_center
    return root


def build_ground_plane(ground_native_pos, plane_size, collection):
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=ground_native_pos,
                                     rotation=(-math.radians(90), 0, 0))
    plane = bpy.context.view_layer.objects.active
    plane.name = "GroundPlane"
    plane.data.materials.clear()
    # Only slightly lighter than the background once lit: a quiet surface
    # that catches the bunny's contact shadow in renders. Hidden in the
    # viewport, where Blender's fine floor grid (set_default_viewport_view)
    # gives the spatial reference instead -- the same Z = 0 plane.
    plane.data.materials.append(make_material("GroundMatte", (0.085, 0.088, 0.095), roughness=1.0))
    plane.hide_viewport = True
    plane.hide_select = True
    link_only_to(plane, collection)
    return plane


def setup_lighting_and_world(collection):
    """Three sun lights (key with soft shadow, fill, rim) in Blender world
    space + a dark neutral world for low ambient light. Lights are not
    under WorldRoot; they only need a direction."""
    def sun(name, energy, rotation, shadow, angle_deg=4.0):
        data = bpy.data.lights.get(name) or bpy.data.lights.new(name, type="SUN")
        data.energy = energy
        data.use_shadow = shadow
        data.angle = math.radians(angle_deg)
        obj = bpy.data.objects.new(name, data)
        collection.objects.link(obj)
        obj.rotation_euler = rotation
        return obj

    sun("Key_Light", 3.2, (math.radians(50), 0.0, math.radians(35)), shadow=True)
    sun("Fill_Light", 0.7, (math.radians(65), 0.0, math.radians(-140)), shadow=False)
    sun("Rim_Light", 1.4, (math.radians(-60), 0.0, math.radians(20)), shadow=False)

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.030, 0.033, 0.040, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    world.color = (0.030, 0.033, 0.040)


# =========================================================================
# Cameras
# =========================================================================

def build_camera_data(intr, name, display_size):
    """Blender camera data equivalent to the dataset's pinhole camera
    (intrinsics.txt: width height fx fy cx cy -- the same K the C++
    renderer projects with, u = fx*X/Z + cx, v = fy*Y/Z + cy).

    Blender's model: sensor_fit HORIZONTAL means the sensor WIDTH maps to
    the image WIDTH, so a focal length f_px in pixels is
        lens_mm = f_px / width_px * sensor_width_mm.
    The sensor width is arbitrary (only the ratio matters); 36 mm is used,
    giving lens = 520.9 / 640 * 36 = 29.30 mm, i.e. horizontal FOV
    2*atan(640 / (2*520.9)) = 63.1 deg.
    The principal point is Blender's lens shift, in units of the larger
    image dimension (the width here), measured from the image centre with
    +x right and +y UP (image v runs down):
        shift_x = (width/2 - cx) / width  = -0.00797
        shift_y = (cy - height/2) / width = +0.01516
    fy is honoured through the render pixel aspect (setup_render():
    pixel_aspect_y = fy / fx = 1.000192) at resolution width x height
    (640 x 480). This is the camera set-up verified earlier by rendering
    through a Blender camera placed along ground truth: the bunny
    silhouettes matched the dataset images (IoU 0.999).
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = intr["fx"] / intr["width"] * cam_data.sensor_width
    cam_data.shift_x = (intr["width"] / 2 - intr["cx"]) / intr["width"]
    cam_data.shift_y = (intr["cy"] - intr["height"] / 2) / intr["width"]
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam_data.display_size = display_size
    return cam_data


# Default external viewport view: looking at the bunny from ~30 degrees
# above the orbit plane, far enough that the whole orbit fits with margin at
# Blender's default 50 mm viewport lens.
EXTERNAL_VIEW_DIRECTION = mathutils.Vector((1.1, -1.4, 0.95)).normalized()
EXTERNAL_VIEW_DISTANCE_SCALE = 3.9   # x orbit radius
EXTERNAL_VIEW_LENS = 50.0


def external_view(orbit_extent, bunny_height):
    center = mathutils.Vector((0.0, 0.0, 1.1 * bunny_height))
    return dict(center=list(center),
                rotation=list(EXTERNAL_VIEW_DIRECTION.to_track_quat("Z", "Y")),
                distance=EXTERNAL_VIEW_DISTANCE_SCALE * orbit_extent)


# =========================================================================
# Trajectories, frustums, markers, axes
# =========================================================================

def make_poly_curve(name, polylines, bevel_depth, material):
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 3
    curve_data.use_fill_caps = True
    for pts in polylines:
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for k, p in enumerate(pts):
            spline.points[k].co = (p.x, p.y, p.z, 1.0)
    curve_data.materials.append(material)
    return curve_data


def trail_point_spec(n, dashed):
    """Splines of a trajectory curve as lists of (a, t) = "fraction t along
    the drawn segment pose a -> pose a+1" (see slam_animation._update_trail).
    Solid: one spline through every pose (t = 0). Dashed: one short spline
    per segment covering its first 55% -- dash ends lie ON the straight
    segment between two recorded poses, so no new trajectory values appear."""
    if not dashed:
        return [[(a, 0.0) for a in range(n)]]
    return [[(a, 0.0), (a, 0.55)] for a in range(n - 1)]


def create_trajectory_curve(key, poses, unit, collection, dashed=False):
    """One thin, solid curve object per trajectory (straight segments
    between the recorded poses -- no smoothing, so the line passes exactly
    through the data). The per-point (a, t) spec is stored in
    obj["slam_trail_points"] so slam_animation.py can reveal the curve
    progressively by clamping not-yet-reached points."""
    st = METHOD_STYLES[key]
    splines = trail_point_spec(len(poses), dashed)
    polylines = [[poses[a]["pos"].lerp(poses[min(a + 1, len(poses) - 1)]["pos"], t) for a, t in sp]
                 for sp in splines]
    name = f"{key}_Trajectory"
    data = make_poly_curve(name, polylines, LINE_THICKNESS_SCALE * unit * st["thickness"],
                           make_material(f"{key}_Line", st["color"], emission=1.4))
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj["slam_trail_points"] = [v for sp in splines for a, t in sp for v in (float(a), t)]
    tag(obj, key, slam_animation.ROLE_TRAJECTORY)
    overlay_object(obj)
    return obj


def sampled_indices(n, step):
    """Pose indices of the optional historical cameras: every `step`-th pose
    plus the last."""
    idx = list(range(0, n, max(1, step)))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def place_camera(obj, pose):
    """Pose -> Blender transform: trajectory_to_blender_pose() (see
    slam_coords) -- the same conversion used by every previous camera
    visualization (location = camera centre, rotation = R_wc @ R_FIX)."""
    obj.rotation_mode = "QUATERNION"
    obj.location, obj.rotation_quaternion = trajectory_to_blender_pose(pose["pos"], pose["quat_cv"])


def new_camera_object(key, name, intr, display_size, collection, role):
    cam_data = build_camera_data(intr, name, display_size)
    obj = bpy.data.objects.new(name, cam_data)
    collection.objects.link(obj)
    tag(obj, key, role)
    obj.color = tuple(METHOD_STYLES[key]["color"]) + (1.0,)  # object color (Solid mode / outliner)
    return obj


def create_animated_camera(key, poses, intr, unit, collection):
    """<key>_Animated_Camera: ONE real Blender Camera object per method,
    keyframed at EVERY frame i = 0..N-1 with that method's pose i, so it
    moves along the trajectory as the timeline plays (and in renders, even
    without the Python handler).

    Interpolation between frames is LINEAR for both location and the
    quaternion (Blender interpolates the 4 quaternion channels and
    normalizes, i.e. nlerp -- a shortest-arc blend for these <~10 degree
    steps). At integer frames the pose is exactly the data. Consecutive
    quaternions are kept in the same hemisphere (q and -q are the SAME
    rotation) so the blend never takes the long way round."""
    obj = new_camera_object(key, f"{key}_Animated_Camera", intr,
                            ANIMATED_CAMERA_DISPLAY_SCALE * unit, collection, slam_animation.ROLE_ANIMATED_CAMERA)
    prev = None
    for i, pose in enumerate(poses):
        place_camera(obj, pose)
        q = obj.rotation_quaternion.copy()
        if prev is not None and prev.dot(q) < 0:
            q.negate()  # same rotation, continuous sign for interpolation
            obj.rotation_quaternion = q
        prev = q
        obj.keyframe_insert(data_path="location", frame=i)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=i)
    # LINEAR between frames, so the camera stays on the drawn (straight-
    # segment) trajectory line; Blender's default BEZIER would bow off it.
    ad = obj.animation_data
    for fc in anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return obj


def create_historical_cameras(key, poses, intr, unit, collection):
    """Optional debug extra (SHOW_HISTORICAL_CAMERAS): STATIC real cameras
    at every CAMERA_SAMPLE_STEP-th pose, smaller than the animated one."""
    objs = []
    for i in sampled_indices(len(poses), CAMERA_SAMPLE_STEP):
        obj = new_camera_object(key, f"{key}_History_Camera_{i:03d}", intr,
                                HISTORY_CAMERA_DISPLAY_SCALE * unit, collection,
                                slam_animation.ROLE_HISTORY_CAMERA)
        obj["slam_frame"] = i
        place_camera(obj, poses[i])
        objs.append(obj)
    return objs


def build_drift_vector(key, unit, collection):
    """A 2-point curve whose endpoints slam_animation.update_drift_vectors()
    pins every frame to the current GT and `key` estimate positions -- NOT
    parented to WorldRoot, so its points are written in final world space."""
    name = f"{key}_Error_Vector"
    data = make_poly_curve(name, [[mathutils.Vector()] * 2], LINE_THICKNESS_SCALE * unit * 0.5,
                           make_material("ErrorVector", DRIFT_COLOR, emission=1.5))
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    tag(obj, key, slam_animation.ROLE_DRIFT)
    overlay_object(obj)
    return obj


def create_world_axes(origin, length, unit, collection):
    """Small arrow triad of the C++ WORLD frame (the frame every trajectory
    number is expressed in: Y-up). Parented under WorldRoot with the other
    native-frame objects, so it rotates with them -- in Blender's Z-up view,
    the 'Y' arrow therefore points up. Placed beside the bunny, not at the
    true origin (which is inside the bunny); only its directions matter."""
    objs = []
    head = 0.2 * length
    for axis, direction in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
        d = mathutils.Vector(direction)
        side = mathutils.Vector((0, 1, 0)) if axis != "Y" else mathutils.Vector((1, 0, 0))
        tip = origin + d * length
        polylines = [[origin, tip],
                     [tip - d * head + side * head * 0.5, tip, tip - d * head - side * head * 0.5]]
        data = make_poly_curve(f"WorldAxes_{axis}", polylines, 0.005 * unit,
                               make_material(f"Axis_{axis}", AXIS_COLORS[axis], emission=1.0))
        obj = bpy.data.objects.new(f"WorldAxes_{axis}", data)
        collection.objects.link(obj)
        obj["slam_axis_label"] = axis
        obj["slam_axis_tip"] = list(tip + d * 0.1 * length)
        obj["slam_axis_color"] = list(AXIS_COLORS[axis])
        obj["slam_role"] = slam_animation.ROLE_AXES
        overlay_object(obj)
        objs.append(obj)
    return objs


def build_world_root(rotation, translation):
    root = bpy.data.objects.new("WorldRoot", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.03
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = rotation
    root.location = translation
    bpy.context.scene.collection.objects.link(root)
    return root


def parent_under_root(root, objects):
    for obj in objects:
        obj.parent = root


# =========================================================================
# Render / viewport / runtime
# =========================================================================

def setup_render(scene, intr, n_frames):
    scene.frame_start = 0
    scene.frame_end = n_frames - 1
    scene.render.fps = ANIMATION_FPS
    # The trajectory cameras ARE the dataset camera: render (and camera view)
    # at its 640x480 with pixel aspect fy/fx (see build_camera_data()).
    scene.render.resolution_x = intr["width"]
    scene.render.resolution_y = intr["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = intr["fy"] / intr["fx"]
    # No depth-EXR compositor (it belonged to the removed trajectory-
    # following camera); clear it (also when rebuilding over an old .blend).
    scene.compositing_node_group = None
    scene.view_layers[0].use_pass_z = False
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = RENDER_OUTPUT_BLEND_PATH
    # The frame-change handler edits curves/visibility; lock the UI while
    # rendering so it never races the render thread.
    scene.render.use_lock_interface = True
    # 'Standard' keeps the legend colors as specified (AgX desaturates them).
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    # Render-side HUD: burned-in stamp. The frame number is Blender's own
    # stamp field (always correct); the note (title, legend, errors) is
    # refreshed per frame by slam_animation.update_animation().
    r = scene.render
    r.use_stamp = True
    for field in ("date", "time", "render_time", "frame_range", "memory", "hostname", "camera",
                  "lens", "scene", "marker", "filename", "sequencer_strip"):
        setattr(r, f"use_stamp_{field}", False)
    r.use_stamp_frame = True
    r.use_stamp_note = True
    r.use_stamp_labels = True
    r.stamp_font_size = 11
    r.stamp_foreground = (0.95, 0.95, 0.95, 1.0)
    r.stamp_background = (0.0, 0.0, 0.0, 0.6)


def embed_autoload_text():
    """Text block registered as a module: when the saved .blend is opened
    with Python auto-run allowed (blender -y, or Preferences > Save & Load >
    Auto Run Python Scripts), it re-registers the sidebar, HUD overlay
    and frame-change handler. Without auto-run, open it in the Text Editor
    and press Run Script (or pass --python visualization/scripts/slam_hud_panel.py)."""
    text = bpy.data.texts.new(AUTOLOAD_TEXT_NAME)
    text.write(
        "# Re-registers the trajectory-comparison runtime (sidebar, HUD, frame handler).\n"
        "# Generated by visualization/scripts/build_scene.py.\n"
        "import os, sys, bpy\n"
        "_candidates = [bpy.path.abspath('//../scripts')]  # relative to this .blend\n"
        "for _d in _candidates:\n"
        "    if os.path.isfile(os.path.join(_d, 'slam_hud_panel.py')):\n"
        "        if _d not in sys.path:\n"
        "            sys.path.insert(0, _d)\n"
        "        break\n"
        "import slam_hud_panel\n"
        "slam_hud_panel.register()\n")
    text.use_module = True


def set_default_viewport_view(view):
    """External view + clean technical shading for every 3D viewport."""
    found = False
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            space = area.spaces.active
            if region is None or space is None:
                continue
            rv3d = space.region_3d
            rv3d.view_perspective = "PERSP"
            rv3d.view_rotation = mathutils.Quaternion(view["rotation"])
            rv3d.view_location = mathutils.Vector(view["center"])
            rv3d.view_distance = view["distance"]
            space.lens = EXTERNAL_VIEW_LENS
            space.clip_start = 0.001
            space.clip_end = 100.0
            shading = space.shading
            shading.type = "SOLID"
            shading.light = "STUDIO"
            shading.color_type = "MATERIAL"
            shading.show_shadows = False
            shading.background_type = "VIEWPORT"
            shading.background_color = (0.035, 0.038, 0.045)
            overlay = space.overlay
            overlay.show_floor = True          # fine 10 cm grid in place of the ground plane
            overlay.show_axis_x = False        # Blender's own X/Y axis lines would clash with
            overlay.show_axis_y = False        # the C++-world axes triad
            overlay.grid_scale = 0.1
            overlay.grid_subdivisions = 10
            overlay.show_relationship_lines = False
            overlay.show_object_origins = False
            found = True
    return found


def apply_initial_settings(scene):
    s = scene.slam_viz
    s.show_groundtruth = SHOW_GROUND_TRUTH
    s.show_pnp = SHOW_PNP
    s.show_icp = SHOW_ICP
    s.show_cameras = SHOW_CAMERAS
    s.show_historical_cameras = SHOW_HISTORICAL_CAMERAS
    s.show_world_axes = SHOW_WORLD_AXES
    s.show_hud = SHOW_HUD
    s.presentation_mode = PRESENTATION_MODE
    s.progressive = PROGRESSIVE_TRAJECTORIES


# =========================================================================
# Main
# =========================================================================

def main():
    clear_previous_scene_objects()

    intr = read_intrinsics(INTRINSICS_PATH)
    rows = read_slam_trajectory_csv(TRAJECTORY_CSV_PATH)
    n = len(rows)
    print(f"[build_scene] {n} frames, intrinsics: {intr}")
    verify_trajectory_conventions(rows)
    methods = resolve_methods(rows)
    for m in methods:
        print(f"[build_scene] {m['key']:<4} <- CSV columns {m['prefix']}_*")
    poses = {m["key"]: native_poses(rows, m["prefix"]) for m in methods}

    scene = bpy.context.scene
    scene_coll = scene.collection
    coll_bunny = get_or_create_collection("Bunny", scene_coll)
    coll_method = {key: get_or_create_collection(cname, scene_coll)
                   for key, cname in METHOD_COLLECTIONS.items()}
    coll_viz = get_or_create_collection("Visualization", scene_coll)
    coll_axes = get_or_create_collection("WorldAxes", coll_viz)
    coll_drift = get_or_create_collection("Error Vectors", coll_viz)
    coll_lighting = get_or_create_collection("Lighting", coll_viz)

    # === Stage 1: measure the RAW mesh (native, pre-transform frame) ===
    bunny_obj = import_bunny_raw()
    original_bounds, bunny_center = measure_bounds([v.co for v in bunny_obj.data.vertices])
    bound_radius = max((mathutils.Vector(c) - bunny_center).length for c in [
        (original_bounds["x"][0], original_bounds["y"][0], original_bounds["z"][0]),
        (original_bounds["x"][1], original_bounds["y"][1], original_bounds["z"][1])])
    h_low = bunny_center.y - original_bounds["y"][0]
    unit = bound_radius  # size unit for every line/marker/frustum

    print("\n=== BUNNY VISUALIZATION ===")
    print_bounds("Original bounds (raw PLY, native Y-up frame)", original_bounds)
    print(f"Native centroid: {tuple(round(x, 4) for x in bunny_center)}")
    print(f"Display scale (BUNNY_DISPLAY_SCALE): {BUNNY_DISPLAY_SCALE} (uniform)")

    setup_lighting_and_world(coll_lighting)

    orbit_extent = max((pose["pos"] - bunny_center).length for pose in poses["GT"])

    # === Stage 2: per-method trajectory + ONE animated real Camera -- all in
    # the native frame, all cameras from the same intrinsics.txt. ===
    native_objects = []
    for m in methods:
        key = m["key"]
        native_objects.append(create_trajectory_curve(key, poses[key], unit, coll_method[key]))
        native_objects.append(create_animated_camera(key, poses[key], intr, unit, coll_method[key]))
        if SHOW_HISTORICAL_CAMERAS:
            coll_hist = get_or_create_collection(f"{key}_Historical_Cameras", coll_method[key])
            native_objects += create_historical_cameras(key, poses[key], intr, unit, coll_hist)

    for key in ("PnP", "ICP"):
        build_drift_vector(key, unit, coll_drift)  # debug-mode only (see slam_animation)

    bunny_root = build_bunny_hierarchy(bunny_obj, bunny_center, coll_bunny)
    native_objects.append(bunny_root)

    post_scale_min_y = bunny_center.y - BUNNY_DISPLAY_SCALE * h_low
    if ADD_GROUND_PLANE:
        ground_pos = (bunny_center.x, post_scale_min_y, bunny_center.z)
        native_objects.append(build_ground_plane(ground_pos, orbit_extent * 2.6, coll_bunny))

    axes_origin = mathutils.Vector((bunny_center.x - 0.85 * orbit_extent, post_scale_min_y,
                                    bunny_center.z + 0.85 * orbit_extent))
    native_objects += create_world_axes(axes_origin, AXES_LENGTH_SCALE * unit, unit, coll_axes)

    # === Stage 3: WorldRoot -- one rotation + one translation, applied
    # identically to the bunny AND all trajectories, so the GT/PnP/ICP
    # drift is preserved exactly as computed in C++. Error vectors are NOT
    # parented -- they're updated directly in final world space. ===
    world_shift = mathutils.Vector((-bunny_center.x, bunny_center.z, -post_scale_min_y))
    root = build_world_root(WORLD_ROOT_ROTATION, world_shift)
    parent_under_root(root, native_objects)

    view = external_view(orbit_extent, BUNNY_DISPLAY_SCALE * (original_bounds["y"][1] - original_bounds["y"][0]))
    bpy.ops.object.select_all(action="DESELECT")
    remove_empty_default_collection()

    setup_render(scene, intr, n)
    scene["slam_intrinsics"] = intr
    scene["slam_trajectory"] = rows  # HUD/animation read positions+errors from this directly
    scene["slam_viz_methods"] = methods
    scene["slam_viz_external_view"] = view  # restored by the sidebar's "Reset external view" button
    os.makedirs(os.path.dirname(RENDER_OUTPUT_PATH), exist_ok=True)

    slam_hud_panel.register()
    apply_initial_settings(scene)
    # Blender needs *a* scene camera for F12 renders; start with the GT one.
    # Nothing switches it automatically -- choose any camera yourself.
    slam_hud_panel.set_active_camera(slam_hud_panel.get_gt_camera())
    embed_autoload_text()
    scene.frame_set(0)
    slam_animation.update_animation(scene)  # initialize for frame 0 before first draw

    # === Diagnostics ===
    bpy.context.view_layer.update()
    final_corners = [bunny_obj.matrix_world @ mathutils.Vector((x, y, z))
                     for x in original_bounds["x"] for y in original_bounds["y"]
                     for z in original_bounds["z"]]
    final_bounds, final_center = measure_bounds(final_corners)
    print_bounds("Final bounds (world space)", final_bounds)
    print(f"Final center (X,Y should be ~0): {tuple(round(x, 4) for x in final_center)}")
    print(f"Final base Z (should be ~0): {final_bounds['z'][0]:.4f}")

    traj_world = [root.matrix_world @ p["pos"] for p in poses["GT"]]
    traj_bounds, _ = measure_bounds(traj_world)
    print_bounds("GT trajectory bounds (world space)", traj_bounds)
    cams = [o.name for o in bpy.data.objects if o.get("slam_role") == slam_animation.ROLE_ANIMATED_CAMERA]
    hist = [o for o in bpy.data.objects if o.get("slam_role") == slam_animation.ROLE_HISTORY_CAMERA]
    print(f"Animated Blender cameras: {cams} | historical cameras: {len(hist)}")
    print(f"Scene camera: {scene.camera.name if scene.camera else None}")
    print("=== END BUNNY VISUALIZATION ===\n")

    framed = set_default_viewport_view(view)
    print(f"[build_scene] default viewport view set: {framed}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    # relative_remap=False: keep the "//"-relative paths set above exactly as written.
    bpy.ops.wm.save_as_mainfile(filepath=SAVE_PATH, relative_remap=False)
    print(f"[build_scene] saved {SAVE_PATH}")


if __name__ == "__main__":
    main()
