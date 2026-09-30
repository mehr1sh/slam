"""User-facing runtime of the GT / PnP / ICP trajectory-comparison scene:

  * SLAMVizSettings -- display toggles (scene.slam_viz), shown in the
    View3D sidebar (N) > "Trajectories" tab and seeded from build_scene.py's
    SHOW_* / CAMERA_SAMPLE_STEP / PRESENTATION_MODE constants. Changing one
    re-runs slam_animation.update_animation().
  * Camera helpers for the Python console -- get_gt_camera(),
    get_pnp_camera(), get_icp_camera(), set_active_camera(obj) -- plus the
    SLAM_OT_use_selected_camera / SLAM_OT_reset_view sidebar buttons. The
    three GT/PnP/ICP_Animated_Camera objects are real, keyframed Blender
    Camera objects (build_scene.py).
  * SLAM_PT_trajectories -- the sidebar: frame, display toggles, intrinsics
    (K), per-frame GT/PnP/ICP camera centers + errors, legend, GT extrinsics.
  * A viewport HUD overlay (POST_PIXEL draw handler): title, frame counter,
    color legend with the CSV's per-frame errors, one label per method at its
    moving camera, world-axis names (debug mode adds pose indices at the
    optional historical cameras), and -- in 3D -- each real camera's own frustum
    (camera.view_frame()) re-drawn in its method's color, because Blender
    draws camera gizmos in theme colors only. Viewport-only; rendered frames get the same
    information through the render stamp (see slam_animation.py).
  * The frame_change_post handler, which (a) calls
    slam_animation.update_animation() and (b) tags every VIEW_3D area for
    redraw so the sidebar/HUD visibly update during playback, not only
    after it stops.

Positions and errors come DIRECTLY from data/synthetic_bunny/slam_trajectory.csv
(stored by build_scene.py as scene["slam_trajectory"]) -- this module never
recomputes a pose-estimation or trajectory-composition number itself.

Registered by build_scene.py (slam_hud_panel.register()), by the
"slam_viz_autoload.py" text block embedded in the saved .blend (when Python
auto-run is allowed), or manually with
    blender visualization/scenes/bunny_slam_demo.blend --python visualization/scripts/slam_hud_panel.py
"""

import os
import sys

import bpy
import mathutils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:  # allow `blender file.blend --python slam_hud_panel.py`
    sys.path.insert(0, SCRIPT_DIR)

import slam_animation
from slam_coords import pose_from_row

# Key under which the overlay's draw-handler handle survives module reloads
# (re-running build_scene.py in one session must not stack two overlays).
_DRAW_HANDLE_KEY = "slam_viz_hud_draw_handle"


# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------

def _on_setting_changed(self, context):
    slam_animation.update_animation(context.scene)
    _tag_view3d_redraw()


class SLAMVizSettings(bpy.types.PropertyGroup):
    show_groundtruth: bpy.props.BoolProperty(name="Ground Truth", default=True, update=_on_setting_changed)
    show_pnp: bpy.props.BoolProperty(name="PnP", default=True, update=_on_setting_changed)
    show_icp: bpy.props.BoolProperty(name="ICP", default=True, update=_on_setting_changed)
    show_cameras: bpy.props.BoolProperty(
        name="Cameras", default=True, update=_on_setting_changed,
        description="The three animated GT/PnP/ICP Blender cameras")
    show_historical_cameras: bpy.props.BoolProperty(
        name="Historical cameras", default=False, update=_on_setting_changed,
        description="Optional static cameras at sampled poses (only if built with "
                    "SHOW_HISTORICAL_CAMERAS = True)")
    show_world_axes: bpy.props.BoolProperty(name="World axes", default=True, update=_on_setting_changed)
    show_hud: bpy.props.BoolProperty(name="HUD", default=True, update=_on_setting_changed)
    presentation_mode: bpy.props.BoolProperty(
        name="Presentation mode", default=True, update=_on_setting_changed,
        description="On: clean comparison. Off (debug): pose indices at historical cameras, "
                    "red GT->estimate error vectors, camera centers in the HUD")
    progressive: bpy.props.BoolProperty(
        name="Progressive reveal", default=False, update=_on_setting_changed,
        description="Grow the trajectory lines frame by frame (off: full lines all the time)")


# ---------------------------------------------------------------------------
# Camera helpers (Python console:  import slam_hud_panel as viz
#                                  viz.set_active_camera(viz.get_pnp_camera()) )
# ---------------------------------------------------------------------------

def get_camera(method):
    """The animated real Blender Camera of `method` ("GT", "PnP" or "ICP"):
    <method>_Animated_Camera."""
    obj = bpy.data.objects.get(f"{method}_Animated_Camera")
    if obj is None or obj.type != "CAMERA":
        raise KeyError(f"No camera {method}_Animated_Camera in this scene")
    return obj


def get_gt_camera():
    return get_camera("GT")


def get_pnp_camera():
    return get_camera("PnP")


def get_icp_camera():
    return get_camera("ICP")


def set_active_camera(camera_obj, view=False):
    """bpy.context.scene.camera = camera_obj (Numpad 0 then looks through it,
    and the view follows the camera's keyframes during playback).
    view=True also switches every 3D viewport into that camera view. Never
    called automatically on frame changes."""
    if camera_obj is None or camera_obj.type != "CAMERA":
        raise TypeError(f"{camera_obj!r} is not a Blender Camera object")
    scene = bpy.context.scene
    scene.camera = camera_obj
    camera_obj.hide_viewport = False  # the active camera is never hidden (see slam_animation)
    if view:
        wm = bpy.context.window_manager
        for window in (wm.windows if wm else []):
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.region_3d.view_perspective = "CAMERA"
                    area.tag_redraw()
    slam_animation.update_animation(scene)
    return camera_obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SLAM_OT_use_selected_camera(bpy.types.Operator):
    bl_idname = "slam.use_selected_camera"
    bl_label = "Look through selected camera"
    bl_description = ("Make the selected GT/PnP/ICP camera the scene camera and view through it "
                      "(same as Ctrl+Numpad0)")

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "CAMERA":
            self.report({"ERROR"}, "Select a GT/PnP/ICP camera first (Outliner or viewport).")
            return {"CANCELLED"}
        set_active_camera(obj, view=True)
        self.report({"INFO"}, f"Scene camera: {obj.name}")
        return {"FINISHED"}


class SLAM_OT_reset_view(bpy.types.Operator):
    bl_idname = "slam.reset_view"
    bl_label = "Reset external view"
    bl_description = "Restore the default framing: bunny centred, all three trajectories in view"

    def execute(self, context):
        view = context.scene.get("slam_viz_external_view")
        if view is None:
            self.report({"ERROR"}, "No stored external view in this scene.")
            return {"CANCELLED"}
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                rv3d = area.spaces.active.region_3d
                rv3d.view_perspective = "PERSP"
                rv3d.view_rotation = mathutils.Quaternion(view["rotation"])
                rv3d.view_location = mathutils.Vector(view["center"])
                rv3d.view_distance = view["distance"]
                area.tag_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------

def _matrix_rows(label, M, box):
    col = box.column(align=True)
    col.label(text=label)
    for row in range(3):
        col.label(text=f"  [{M[row][0]:8.4f}  {M[row][1]:8.4f}  {M[row][2]:8.4f}]")


def _current_trajectory_row(scene):
    rows = slam_animation.trajectory_rows(scene)
    if rows is None:
        return None
    return rows[slam_animation.current_pose_index(scene)]


class SLAM_PT_trajectories(bpy.types.Panel):
    bl_label = "Trajectory Comparison"
    bl_idname = "SLAM_PT_trajectories"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Trajectories"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        intr = scene.get("slam_intrinsics")
        if intr is None:
            layout.label(text="No trajectory scene loaded.", icon="ERROR")
            return

        rows = slam_animation.trajectory_rows(scene)
        n = len(rows) if rows else 0
        layout.label(text=f"FRAME: {scene.frame_current} / {scene.frame_end}   "
                          f"(pose {slam_animation.current_pose_index(scene)} of 0..{n - 1})", icon="TIME")
        box = layout.box()
        box.label(text="CAMERA VIEW", icon="VIEW_CAMERA")
        cam = scene.camera
        box.label(text=f"Scene camera: {cam.name if cam else 'none'}",
                  icon="OUTLINER_OB_CAMERA" if cam else "ERROR")
        box.operator("slam.use_selected_camera", icon="CAMERA_DATA")
        box.operator("slam.reset_view", icon="WORLD")

        s = slam_animation.settings(scene)
        if s is not None:
            box = layout.box()
            box.label(text="DISPLAY", icon="HIDE_OFF")
            box.prop(s, "presentation_mode")
            row = box.row(align=True)
            for prop in ("show_groundtruth", "show_pnp", "show_icp"):
                row.prop(s, prop, toggle=True)
            grid = box.grid_flow(columns=2, align=True)
            for prop in ("show_cameras", "show_world_axes", "show_hud", "progressive"):
                grid.prop(s, prop)
            if slam_animation.history_cameras():
                box.prop(s, "show_historical_cameras")

        box = layout.box()
        box.label(text="INTRINSICS (K) -- used for every frustum", icon="CAMERA_DATA")
        col = box.column(align=True)
        col.label(text=f"fx = {intr['fx']:.3f}")
        col.label(text=f"fy = {intr['fy']:.3f}")
        col.label(text=f"cx = {intr['cx']:.3f}")
        col.label(text=f"cy = {intr['cy']:.3f}")
        col.label(text=f"image = {intr['width']} x {intr['height']}")
        K = mathutils.Matrix((
            (intr["fx"], 0.0, intr["cx"]),
            (0.0, intr["fy"], intr["cy"]),
            (0.0, 0.0, 1.0)))
        _matrix_rows("K =", K, box)
        cams = slam_animation.animated_cameras()
        if cams:
            d = cams[0].data
            col = box.column(align=True)
            col.label(text="Blender camera (all GT/PnP/ICP cameras):")
            col.label(text=f"  lens {d.lens:.3f} mm on {d.sensor_width:.0f} mm sensor (horizontal fit)")
            col.label(text=f"  shift x {d.shift_x:+.5f}, y {d.shift_y:+.5f}")
            col.label(text=f"  render {scene.render.resolution_x} x {scene.render.resolution_y}, "
                           f"pixel aspect y {scene.render.pixel_aspect_y:.6f}")

        row = _current_trajectory_row(scene)
        if row is None:
            layout.label(text="slam_trajectory.csv not loaded.", icon="ERROR")
            return

        i = slam_animation.current_pose_index(scene)
        layout.separator()
        for m in slam_animation.methods(scene):
            box = layout.box()
            box.label(text=f"{m['label'].upper()}  ({m['color_name']})",
                      icon="CHECKMARK" if m["key"] == "GT" else "OUTLINER_OB_CAMERA")
            pos, _ = pose_from_row(row, m["prefix"])
            col = box.column(align=True)
            col.label(text=f"X = {pos.x:.4f}")
            col.label(text=f"Y = {pos.y:.4f}")
            col.label(text=f"Z = {pos.z:.4f}")
            err = slam_animation.error_at(scene, m, i)
            if err is not None:
                col.separator()
                col.label(text=f"Translation Error: {err[0]:.4f} m")
                col.label(text=f"Rotation Error: {err[1]:.3f} deg")

        layout.separator()
        box = layout.box()
        box.label(text="LEGEND", icon="INFO")
        col = box.column(align=True)
        col.label(text="Green = where the camera actually was (GT)")
        col.label(text="Orange = where PnP thinks the camera was")
        col.label(text="Magenta = where ICP thinks the camera was")
        col.label(text="Camera = real Blender camera, moving along its line")
        col.label(text="Debug mode: red lines = position error (GT -> estimate)")

        layout.separator()
        box = layout.box()
        box.label(text="Advanced: GT extrinsics (T_cw)", icon="EMPTY_AXIS")
        # From the CSV row: R_cw = R_wc^T, t_cw = -R_cw @ center.
        center, quat_cv = pose_from_row(row, "gt")
        R_cw = quat_cv.to_matrix().transposed()
        t_cw = -(R_cw @ center)
        _matrix_rows("R =", R_cw, box)
        col = box.column(align=True)
        col.label(text="t =")
        col.label(text=f"  [{t_cw.x:8.4f}  {t_cw.y:8.4f}  {t_cw.z:8.4f}]")


# ---------------------------------------------------------------------------
# Viewport HUD overlay
# ---------------------------------------------------------------------------

def _draw_rect(x, y, w, h, color):
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h))},
                             indices=((0, 1, 2), (0, 2, 3)))
    shader.uniform_float("color", color)
    batch.draw(shader)


def _text(font, x, y, size, color, body):
    import blf
    blf.size(font, size)
    blf.color(font, *color)
    blf.position(font, x, y, 0)
    blf.draw(font, body)
    return blf.dimensions(font, body)[0]


def _draw_labels(scene, project, s, ui, font):
    """Screen-space labels at 3D points, always readable at any zoom.
    project(world_co) -> 2D region coordinates or None (behind the view).
    One label per method at its moving camera ("PnP Camera"), in the
    method's color; debug mode adds pose indices at historical cameras."""
    ms = {m["key"]: m for m in slam_animation.methods(scene)}
    placed = []
    for key, m in ms.items():
        if not slam_animation.method_visible(scene, key):
            continue
        if not s.presentation_mode and s.show_historical_cameras:
            for o in slam_animation.history_cameras(key=key):
                if o.visible_get():
                    p2 = project(o.matrix_world.translation)
                    if p2 is not None:
                        _text(font, p2.x + 5 * ui, p2.y - 12 * ui, int(9 * ui), (*m["color"], 0.6),
                              f"{o['slam_frame']}")
        if not s.show_cameras:
            continue
        for cam in slam_animation.animated_cameras(key=key):
            if cam == scene.camera and _viewing_through(scene, cam):
                continue
            p2 = project(cam.matrix_world.translation)
            if p2 is None:
                continue
            # De-overlapped vertically: at frame 0 all three share the anchor pose.
            x, y = p2.x + 12 * ui, p2.y + 6 * ui
            while any(abs(y - py) < 14 * ui and abs(x - px) < 90 * ui for px, py in placed):
                y -= 14 * ui
            placed.append((x, y))
            _text(font, x, y, int(11 * ui), (*m["color"], 1.0), f"{m['short']} Camera")

    if s.show_world_axes:
        for obj in bpy.data.objects:
            tip = obj.get("slam_axis_label")
            if tip is None:
                continue
            p2 = project(obj.matrix_world @ mathutils.Vector(obj["slam_axis_tip"]))
            if p2 is not None:
                _text(font, p2.x + 3 * ui, p2.y + 2 * ui, int(10 * ui),
                      tuple(obj["slam_axis_color"]) + (0.9,), tip)


def _draw_hud_panel(scene, s, region_height, ui, font):
    """Compact top-left panel. Errors are the C++ values from the CSV
    (absolute translation m / rotation deg vs ground truth at this frame);
    nothing is computed or invented here."""
    rows = slam_animation.trajectory_rows(scene)
    i = slam_animation.current_pose_index(scene)
    ms = slam_animation.methods(scene)
    debug = not s.presentation_mode
    line = 17 * ui
    x0 = 58 * ui
    col_t = x0 + 130 * ui
    col_r = x0 + 215 * ui
    dim = (0.58, 0.58, 0.62, 1.0)

    lines = 3.3 + len(ms) + (1.2 + len(ms) if debug else 0)
    width = 312 * ui
    top = region_height - 28 * ui
    height = lines * line + 8 * ui
    _draw_rect(x0 - 12 * ui, top - height, width, height + 6 * ui, (0.02, 0.022, 0.028, 0.62))

    y = top - 14 * ui
    _text(font, x0, y, int(13 * ui), (0.95, 0.95, 0.96, 1.0), "STANFORD BUNNY — TRAJECTORY COMPARISON")
    y -= line
    _text(font, x0, y, int(11 * ui), (0.80, 0.80, 0.83, 1.0), f"Frame: {i} / {len(rows) - 1}")
    y -= line * 1.3
    _text(font, x0, y, int(9 * ui), dim, "TRAJECTORY")
    _text(font, col_t, y, int(9 * ui), dim, "TRANS. ERR")
    _text(font, col_r, y, int(9 * ui), dim, "ROT. ERR")
    for m in ms:
        y -= line
        alpha = 1.0 if slam_animation.method_visible(scene, m["key"]) else 0.3
        _draw_rect(x0, y + 4 * ui, 14 * ui, 3 * ui, (*m["color"], alpha))
        _text(font, x0 + 22 * ui, y, int(11 * ui), (*m["color"], alpha), m["label"].upper())
        err = slam_animation.error_at(scene, m, i)
        if m["key"] == "GT":
            _text(font, col_t, y, int(10 * ui), dim, "reference")
        elif err is not None:
            grey = (0.85, 0.85, 0.88, alpha)
            _text(font, col_t, y, int(10 * ui), grey, f"{err[0]:.3f} m")
            _text(font, col_r, y, int(10 * ui), grey, f"{err[1]:.1f}°")
        else:
            _text(font, col_t, y, int(10 * ui), dim, "n/a")
    if debug:
        y -= line * 1.2
        _text(font, x0, y, int(9 * ui), dim, "CAMERA CENTER (C++ world, m)")
        for m in ms:
            y -= line
            p, _ = pose_from_row(rows[i], m["prefix"])
            _text(font, x0, y, int(9 * ui), (*m["color"], 0.9),
                  f"{m['short']:<4} {p.x:+.3f} {p.y:+.3f} {p.z:+.3f}")


_VIEW_STATE = {"camera_view": False}


def _viewing_through(scene, cam):
    return _VIEW_STATE["camera_view"] and scene.camera == cam


def camera_frustum_lines(scene, cam):
    """World-space line segments of the frustum Blender itself draws for
    `cam`: apex -> 4 corners of camera.view_frame() (lens, sensor fit, shift
    and render aspect, at unit size -- scaled by display_size exactly as
    Blender scales its own camera gizmo), the image rectangle, and the "up"
    triangle over the top edge."""
    mw = cam.matrix_world
    size = cam.data.display_size
    corners = [mw @ (v * size) for v in cam.data.view_frame(scene=scene)]  # TR, BR, BL, TL
    apex = mw.translation
    segs = []
    for c in corners:
        segs += [apex, c]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        segs += [a, b]
    tr, br, bl, tl = corners
    mid = (tl + tr) / 2
    tip = mid + (tl - bl) * 0.35
    a, b = tl.lerp(tr, 0.25), tl.lerp(tr, 0.75)
    segs += [a, tip, tip, b]
    return segs


def draw_camera_colors(scene, skip=None):
    """Re-draw every visible GT/PnP/ICP camera's frustum in its method's
    color (Blender draws camera gizmos in theme colors only). Pure drawing:
    no objects are created."""
    import gpu
    from gpu_extras.batch import batch_for_shader
    colors = {m["key"]: m["color"] for m in slam_animation.methods(scene)}
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    vp = gpu.state.viewport_get()
    shader.uniform_float("viewportSize", (vp[2], vp[3]))
    shader.uniform_float("lineWidth", 2.0)
    for cam in slam_animation.animated_cameras() + slam_animation.history_cameras():
        if cam == skip or not cam.visible_get():
            continue
        batch = batch_for_shader(shader, "LINES", {"pos": camera_frustum_lines(scene, cam)})
        shader.uniform_float("color", (*colors.get(cam["slam_method"], (1, 1, 1)), 1.0))
        batch.draw(shader)


def _draw_cameras_3d():
    """SpaceView3D POST_VIEW callback."""
    import gpu
    context = bpy.context
    rv3d = context.region_data
    scene = context.scene
    s = slam_animation.settings(scene)
    if rv3d is None or s is None or not s.show_cameras:
        return
    _VIEW_STATE["camera_view"] = rv3d.view_perspective == "CAMERA"
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.blend_set("ALPHA")
    draw_camera_colors(scene, skip=scene.camera if _VIEW_STATE["camera_view"] else None)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _draw_hud():
    """SpaceView3D POST_PIXEL callback."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    context = bpy.context
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    draw_overlay(context.scene, region.height, context.preferences.system.ui_scale,
                 lambda co: location_3d_to_region_2d(region, rv3d, co))


def draw_overlay(scene, region_height, ui, project):
    import blf
    import gpu

    s = slam_animation.settings(scene)
    rows = slam_animation.trajectory_rows(scene)
    if s is None or not rows:
        return
    font = 0
    gpu.state.blend_set("ALPHA")
    blf.enable(font, blf.SHADOW)
    blf.shadow(font, 3, 0.0, 0.0, 0.0, 0.7)
    blf.shadow_offset(font, 1, -1)

    _draw_labels(scene, project, s, ui, font)
    if s.show_hud:
        _draw_hud_panel(scene, s, region_height, ui, font)

    blf.disable(font, blf.SHADOW)
    gpu.state.blend_set("NONE")


# ---------------------------------------------------------------------------
# Handlers / registration
# ---------------------------------------------------------------------------

def _tag_view3d_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _on_frame_change(scene, depsgraph=None):
    """During animation playback, Blender redraws the 3D viewport itself
    every frame but does NOT automatically redraw other regions of that
    same area -- including this panel's own sidebar -- for performance
    reasons; without tag_redraw() the HUD only catches up once playback
    stops. frame_change_post fires on every frame change (playback, manual
    scrubbing, AND each frame of an animation render); anything the
    animation update changes here is re-evaluated by Blender before the
    frame is drawn/rendered, so there is no one-frame lag. Purely
    event-driven (no polling/sleep).
    """
    slam_animation.update_animation(scene)
    _tag_view3d_redraw()


_CLASSES = (SLAMVizSettings, SLAM_OT_use_selected_camera, SLAM_OT_reset_view, SLAM_PT_trajectories)


def _remove_handlers():
    # Match by name, not identity: a re-run of build_scene.py in the same
    # session imports a fresh copy of this module whose function objects
    # differ from the ones already installed.
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == "_on_frame_change" and \
                getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.frame_change_post.remove(h)
    for key in (_DRAW_HANDLE_KEY, _DRAW_HANDLE_KEY + "_3d"):
        handle = bpy.app.driver_namespace.pop(key, None)
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
            except (ValueError, RuntimeError):
                pass


def register():
    for cls in _CLASSES:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass  # already registered -- harmless if build_scene.py is re-run in the same session
    if not hasattr(bpy.types.Scene, "slam_viz"):
        bpy.types.Scene.slam_viz = bpy.props.PointerProperty(type=SLAMVizSettings)
    _remove_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if not bpy.app.background:
        bpy.app.driver_namespace[_DRAW_HANDLE_KEY + "_3d"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_cameras_3d, (), "WINDOW", "POST_VIEW")
        bpy.app.driver_namespace[_DRAW_HANDLE_KEY] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hud, (), "WINDOW", "POST_PIXEL")


def unregister():
    _remove_handlers()
    if hasattr(bpy.types.Scene, "slam_viz"):
        del bpy.types.Scene.slam_viz
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
    if bpy.context.scene is not None:
        slam_animation.update_animation(bpy.context.scene)
