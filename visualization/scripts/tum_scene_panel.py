"""Frame inspection panel of the TUM Blender scene (View3D sidebar, tab "TUM SLAM").

Embedded into the .blend as a text block that registers on load (like the Bunny scene's HUD; Blender asks to
allow scripts). Everything shown comes from scene["tum_data"] (a copy of the exporter's trajectory.json):
nothing is recomputed. Selecting a frame (or moving the timeline: Blender frame = trajectory frame + 1):
  - moves the timeline, so the animated GT / PnP / ICP cameras show that frame's poses (keyframed values);
  - loads that frame's RGB (dataset PNG), colourised depth and, for an exported pair, its real
    correspondence image into the images TUM_RGB, TUM_Depth, TUM_Matches (open them in an Image Editor;
    the buttons below switch the largest non-3D area to an Image Editor showing them);
  - lists timestamps, position / orientation errors and the pose status (estimated, held after a failed or
    rejected pair, and the solver / acceptance result of the pair that produced it).
"""

import json
import os

import bpy

_IMG = {"rgb": "TUM_RGB", "depth": "TUM_Depth", "matches": "TUM_Matches"}
_KEY = {"gt": "GT", "pnp": "PnP", "icp": "ICP"}
TUBE = 0.004  # = build_tum_scene.TUBE


def apply_recon(scene, mode):
    """Show only the reconstruction placed with `mode` poses (viewport and render); draw the trajectory that placed
    it 2.5x thicker. The other trajectories stay as overlays (their collections' visibility is untouched)."""
    for m, key in _KEY.items():
        col = bpy.data.collections.get(f"Reconstruction_placed_with_{key}")
        if col is not None:
            col.hide_render = col.hide_viewport = (m != mode)
        tr = bpy.data.objects.get(f"{key}_Trajectory")
        if tr is not None:
            tr.data.bevel_depth = TUBE * (2.5 if m == mode else 1.0)


def _on_recon(self, context):
    apply_recon(context.scene, context.scene.tum_recon)


def data(scene):
    if "_tum_cache" not in bpy.app.driver_namespace or bpy.app.driver_namespace["_tum_cache"][0] != scene.name:
        bpy.app.driver_namespace["_tum_cache"] = (scene.name, json.loads(scene["tum_data"]))
    return bpy.app.driver_namespace["_tum_cache"][1]


def current_frame(scene):
    n = len(data(scene)["frames"])
    return max(0, min(n - 1, scene.frame_current - 1))


def _load(name, path):
    img = bpy.data.images.get(name)
    if not path or not os.path.isfile(path):
        if img:
            img.filepath = ""
        return
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = name
    elif bpy.path.abspath(img.filepath) != path:
        img.filepath = path
        img.reload()


def load_images(scene, k):
    d = data(scene)
    root = scene["tum_export_dir"]
    f = d["frames"][k]
    _load(_IMG["rgb"], f["rgb"])
    _load(_IMG["depth"], os.path.join(root, f["depth_preview"]))
    m = d["matches"].get(str(k - 1)) if k > 0 else None  # pair (k-1) -> k ends at the selected frame
    _load(_IMG["matches"], os.path.join(root, m) if m else "")


def _on_select(self, context):
    context.scene.frame_set(context.scene.tum_frame + 1)


def _on_frame_change(scene, *args):
    if "tum_data" not in scene:
        return
    k = current_frame(scene)
    if scene.get("_tum_loaded") != k:
        scene["_tum_loaded"] = k
        try:
            load_images(scene, k)
        except RuntimeError:
            pass


def status(d, m, k):
    if k == 0:
        return "frame 0: anchored to ground truth"
    p = d["pairs"][k - 1]
    held = d["frames"][k][m]["held"]
    if m == "pnp":
        solver = p.get("pnp_solver_returned")
        if not held:
            return f"estimated (pair {k - 1}->{k}: {p['pnp_inliers']}/{p['pnp_correspondences']} inliers)"
        if solver is True:
            return f"HELD: pair {k - 1}->{k} pose REJECTED ({p.get('pnp_reject_reason', 'validity check')})"
        return f"HELD: pair {k - 1}->{k} solver failed" if solver is False else f"HELD: pair {k - 1}->{k} failed"
    return "estimated" if not held else f"HELD: pair {k - 1}->{k} failed"


class TUM_OT_show_image(bpy.types.Operator):
    bl_idname = "tum.show_image"
    bl_label = "Show image"
    bl_description = "Show this image in an Image Editor (switches the largest area that is not the 3D viewport)"
    which: bpy.props.StringProperty()

    def execute(self, context):
        img = bpy.data.images.get(_IMG[self.which])
        if img is None or not img.filepath:
            self.report({"WARNING"}, "not available for this frame")
            return {"CANCELLED"}
        areas = [a for a in context.screen.areas if a.type == "IMAGE_EDITOR"]
        if not areas:
            cand = sorted((a for a in context.screen.areas if a.type != "VIEW_3D"), key=lambda a: -a.width * a.height)
            if not cand:
                self.report({"WARNING"}, "no area to switch: open an Image Editor and pick " + img.name)
                return {"CANCELLED"}
            cand[0].type = "IMAGE_EDITOR"
            areas = [cand[0]]
        areas[0].spaces.active.image = img
        return {"FINISHED"}


class TUM_PT_frame(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TUM SLAM"
    bl_label = "Frame inspection"

    def draw(self, context):
        sc, L = context.scene, self.layout
        if "tum_data" not in sc:
            L.label(text="no TUM data in this scene")
            return
        d = data(sc)
        k = current_frame(sc)
        f = d["frames"][k]
        box = L.box()
        box.label(text="Point cloud placed with:")
        box.row().prop(sc, "tum_recon", expand=True)
        key = _KEY[sc.tum_recon]
        ob = bpy.data.objects.get(f"PointCloud_placed_with_{key}_poses")
        if ob is not None and ob.get("tum_manifest") and os.path.isfile(ob["tum_manifest"]):
            man = json.load(open(ob["tum_manifest"]))
            box.label(text=f"{key} poses (raw): {man['frames_processed']} frames, {man['points_after_sampling']} points")
            sk = [x["frame"] for x in man["frames_skipped"]]
            box.label(text=f"skipped frames: {sk if sk else 'none'}" + (" (pose held: no estimate)" if sk else ""))
        box.label(text=f"thick line = {key} trajectory (placed the cloud); others are overlays")
        L.prop(sc, "tum_frame", text="frame")
        L.label(text=f"{d['sequence']}  ({d['run']})")
        L.label(text=f"frame {k}   t_rgb {f['t_rgb']:.6f} s")
        L.label(text=f"depth t {f['t_depth']:.6f} s ({1000 * (f['t_depth'] - f['t_rgb']):+.1f} ms)")
        for m in ("pnp", "icp"):
            b = L.box()
            b.label(text=f"{m.upper()}: {f[m]['pos_err_m'] * 100:.2f} cm, {f[m]['rot_err_deg']:.2f} deg vs GT")
            b.label(text=status(d, m, k))
        if k > 0:
            p = d["pairs"][k - 1]
            b = L.box()
            b.label(text=f"pair {k - 1}->{k} relative pose T_j<-i (pairs.csv), error vs GT:")
            b.label(text=f"true motion {p.get('gt_motion_deg', float('nan')):.2f} deg, {100 * p.get('gt_motion_m', float('nan')):.1f} cm")
            for m in ("pnp", "icp"):
                e = (p.get(f"{m}_rot_err_deg"), p.get(f"{m}_trans_err_m"))
                txt = f"{e[0]:.2f} deg, {e[1]:.3f} m" if e[0] is not None else "not used (pose held)"
                if m == "pnp" and e[0] is None and p.get("pnp_returned_rot_err_deg") is not None:
                    txt += f"; returned pose was {p['pnp_returned_rot_err_deg']:.2f} deg, {p['pnp_returned_trans_err_m']:.3f} m off"
                b.label(text=f"{m.upper()}: {txt}")
            gap = [g for g in d["frame_gaps"] if g["pair"] == k - 1]
            if gap:
                L.label(text=f"frame gap before this frame: {gap[0]['dt_s']:.3f} s", icon="ERROR")
        row = L.row(align=True)
        for w, lab in (("rgb", "RGB"), ("depth", "Depth"), ("matches", "Matches")):
            op = row.operator("tum.show_image", text=lab)
            op.which = w
        if k == 0 or str(k - 1) not in d["matches"]:
            L.label(text="matches: no match-level export for this pair")
        L.label(text="Raw estimates; T_wc in the TUM world frame (metres)")


_CLASSES = (TUM_OT_show_image, TUM_PT_frame)


def register():
    for c in _CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:
            pass
    bpy.types.Scene.tum_recon = bpy.props.EnumProperty(
        name="reconstruction", update=_on_recon, default="gt",
        items=(("gt", "GT", "points placed with the ground-truth poses (reference)"),
               ("pnp", "PnP", "points placed with the raw PnP poses (no ground truth used)"),
               ("icp", "ICP", "points placed with the raw ICP poses (no ground truth used)")))
    bpy.types.Scene.tum_frame = bpy.props.IntProperty(name="frame", min=0, update=_on_select,
                                                      description="trajectory frame index (timeline = frame + 1)")
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == "_on_frame_change":
            bpy.app.handlers.frame_change_post.remove(h)
    bpy.app.handlers.frame_change_post.append(_on_frame_change)


def unregister():
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == "_on_frame_change":
            bpy.app.handlers.frame_change_post.remove(h)
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
