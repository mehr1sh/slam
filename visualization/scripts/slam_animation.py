"""Per-frame animation of the GT / PnP / ICP trajectory comparison -- the ONE
place that maps the current Blender frame to a trajectory pose and updates
the scene for it.

Called from slam_hud_panel.py's frame_change_post handler (playback,
scrubbing, and animation rendering) and from the sidebar's display toggles.
Everything is read from data build_scene.py stored on the scene:

  scene["slam_trajectory"]   raw slam_trajectory.csv rows (C++ output)
  scene["slam_viz_methods"]  per-method key / label / CSV prefix / color
  obj["slam_method"], obj["slam_role"], obj["slam_frame"], obj["slam_trail_points"]
                             tags on the objects build_scene.py created

so it works identically on a fresh build and on a re-opened .blend. No pose
is estimated, composed, interpolated, or aligned here: pose i is looked up in
row i and written into the scene verbatim.

Frame mapping: Blender frame f shows pose index i = clamp(f, 0, N-1)
(scene.frame_start = 0, scene.frame_end = N-1, one Blender frame per pose).

A method may have fewer poses than the dataset (method["n_poses"]; the
Scratch PnP trajectory of milestone 1 has frames 0-1 only). Its line then
ends at its last pose, and its camera is hidden at frames without a pose
instead of standing still at the last one.
"""

import bpy

from slam_coords import pose_from_row

# Object-tag roles (obj["slam_role"]), set by build_scene.py.
ROLE_TRAJECTORY = "trajectory"            # progressive trail curve
ROLE_ANIMATED_CAMERA = "animated_camera"  # the ONE keyframed real Camera per method
ROLE_HISTORY_CAMERA = "history_camera"    # optional static real Camera at pose obj["slam_frame"]
ROLE_DRIFT = "drift"                      # GT -> estimate position-error segment (debug)
ROLE_AXES = "axes"                        # world-axes triad


def trajectory_rows(scene):
    return scene.get("slam_trajectory")


def methods(scene):
    """List of plain dicts: key, label, short, prefix, color (sRGB),
    color_name, error_prefix, n_poses, note. Order = legend order
    (GT, PnP, ICP, ScratchPnP)."""
    raw = scene.get("slam_viz_methods") or []
    return [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in raw]


def pose_count(scene, method):
    """Number of leading frames that have a pose for `method`."""
    rows = trajectory_rows(scene) or []
    return min(int(method.get("n_poses", len(rows))), len(rows))


def has_pose(scene, method, i):
    return 0 <= i < pose_count(scene, method)


def settings(scene):
    """The sidebar's display toggles (slam_hud_panel.SLAMVizSettings), or
    None if the panel module isn't registered in this session."""
    return getattr(scene, "slam_viz", None)


def current_pose_index(scene):
    rows = trajectory_rows(scene)
    if not rows:
        return 0
    return max(0, min(scene.frame_current, len(rows) - 1))


def method_visible(scene, key):
    s = settings(scene)
    if s is None:
        return True
    return {"GT": s.show_groundtruth, "PnP": s.show_pnp, "ICP": s.show_icp,
            "ScratchPnP": getattr(s, "show_scratch_pnp", True)}.get(key, True)


def presentation(scene):
    s = settings(scene)
    return True if s is None else s.presentation_mode


def animated_cameras(key=None):
    """The keyframed GT/PnP/ICP/ScratchPnP Camera objects (or the one of `key`)."""
    order = {"GT": 0, "PnP": 1, "ICP": 2, "ScratchPnP": 3}
    cams = [o for o in bpy.data.objects
            if o.get("slam_role") == ROLE_ANIMATED_CAMERA and (key is None or o.get("slam_method") == key)]
    return sorted(cams, key=lambda o: order.get(o["slam_method"], 9))


def history_cameras(key=None):
    """Optional static historical cameras (debug extra), by pose index."""
    cams = [o for o in bpy.data.objects
            if o.get("slam_role") == ROLE_HISTORY_CAMERA and (key is None or o.get("slam_method") == key)]
    return sorted(cams, key=lambda o: (o["slam_method"], o["slam_frame"]))


def world_root():
    return bpy.data.objects.get("WorldRoot")


def world_position(scene, method, i):
    """Final Blender-world position of `method`'s camera center at pose i --
    the raw T_wc translation pushed through WorldRoot's display transform."""
    rows = trajectory_rows(scene)
    root = world_root()
    pos, _ = pose_from_row(rows[i], method["prefix"])
    return root.matrix_world @ pos if root is not None else pos


def _set_hidden(obj, hidden):
    # Only write when changed: toggling hide_* tags the depsgraph.
    if obj.hide_viewport != hidden:
        obj.hide_viewport = hidden
    if obj.hide_render != hidden:
        obj.hide_render = hidden


def _update_trail(obj, positions, reveal_upto):
    """Trail point k is stored as (a, t): the point at fraction t along the
    drawn segment from pose a to pose a+1 (t = 0 is pose a itself). Poses
    not yet reached are clamped to the latest revealed pose, so the curve
    grows frame by frame without being rebuilt. Positions are native
    (WorldRoot-local)."""
    spec = obj.get("slam_trail_points")
    if spec is None:
        return
    last = len(positions) - 1
    k = 0
    for spline in obj.data.splines:
        for point in spline.points:
            a, t = int(spec[2 * k]), spec[2 * k + 1]
            p0 = positions[min(a, reveal_upto)]
            p1 = positions[min(a + 1, reveal_upto, last)]
            p = p0.lerp(p1, t)
            point.co = (p.x, p.y, p.z, 1.0)
            k += 1


def update_drift_vectors(scene, i):
    """Pins each GT -> estimate error segment to the current poses. These
    curves are NOT parented to WorldRoot, so their points are written in
    final world space."""
    by_key = {m["key"]: m for m in methods(scene)}
    gt = by_key.get("GT")
    for obj in bpy.data.objects:
        if obj.get("slam_role") != ROLE_DRIFT or gt is None:
            continue
        est = by_key.get(obj.get("slam_method"))
        if est is None:
            continue
        a = world_position(scene, gt, i)
        b = world_position(scene, est, i)
        pts = obj.data.splines[0].points
        pts[0].co = (a.x, a.y, a.z, 1.0)
        pts[1].co = (b.x, b.y, b.z, 1.0)


def error_at(scene, method, i):
    """(translation_error_m, rotation_error_deg) for `method` at pose i, read
    straight from the C++-computed CSV columns -- or None when the CSV has
    no such columns (GT). Never recomputed here."""
    prefix = method.get("error_prefix")
    if not prefix:
        return None
    row = trajectory_rows(scene)[i]
    t_key, r_key = f"{prefix}_translation_error", f"{prefix}_rotation_error"
    if t_key not in row or r_key not in row:
        return None
    return row[t_key], row[r_key]


def _render_stamp_note(scene, i):
    """Text burned into rendered frames (render stamp) next to Blender's own
    frame-number field."""
    legend = []
    for m in methods(scene):
        if not method_visible(scene, m["key"]):
            continue
        err = error_at(scene, m, i)
        if m["key"] == "GT":
            tail = " reference"
        elif not has_pose(scene, m, i):
            tail = " no pose"
        else:
            tail = f" {err[0]:.3f}m {err[1]:.1f}deg" if err is not None else " n/a"
        legend.append(f"{m['short']} ({m['color_name']}){tail}")
    return "VISUAL SLAM - TRAJECTORY COMPARISON (synthetic test scene) | " + " | ".join(legend)


def update_animation(scene):
    rows = trajectory_rows(scene)
    if not rows:
        return
    n = len(rows)
    i = current_pose_index(scene)
    s = settings(scene)
    progressive = s.progressive if s is not None else True
    show_cameras = s.show_cameras if s is not None else True
    show_history = s.show_historical_cameras if s is not None else False
    show_axes = s.show_world_axes if s is not None else True
    debug = not presentation(scene)
    reveal = i if progressive else n - 1

    by_key = {m["key"]: m for m in methods(scene)}
    positions = {key: [pose_from_row(r, m["prefix"])[0] for r in rows[:pose_count(scene, m)]]
                 for key, m in by_key.items()}

    for obj in bpy.data.objects:
        role = obj.get("slam_role")
        if role == ROLE_AXES:
            _set_hidden(obj, not show_axes)
            continue
        key = obj.get("slam_method")
        if role is None or key not in by_key:
            continue
        shown = method_visible(scene, key)

        if role == ROLE_TRAJECTORY:
            _update_trail(obj, positions[key], reveal)
            _set_hidden(obj, not shown)
        elif role == ROLE_ANIMATED_CAMERA:
            # Moved by its own keyframes, never here; only visibility. The
            # active scene camera is never hidden.
            # Hidden at frames without a pose (partial trajectories) rather
            # than left standing at the last keyframe.
            exists = has_pose(scene, by_key[key], i)
            _set_hidden(obj, not (shown and show_cameras and exists) and obj != scene.camera)
        elif role == ROLE_HISTORY_CAMERA:
            k = obj.get("slam_frame", 0)
            _set_hidden(obj, not (shown and show_history and k <= reveal) and obj != scene.camera)
        elif role == ROLE_DRIFT:
            _set_hidden(obj, not (debug and shown))

    update_drift_vectors(scene, i)

    if scene.render.use_stamp and scene.render.use_stamp_note:
        note = _render_stamp_note(scene, i)
        if scene.render.stamp_note_text != note:
            scene.render.stamp_note_text = note
