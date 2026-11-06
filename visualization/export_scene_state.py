"""Read-only export of the evaluated state of a Blender scene, for comparing two builds of the same scene.

Changes nothing and never saves. Run from anywhere:
    blender -b --factory-startup SCENE.blend --python visualization/export_scene_state.py -- OUT.json

Exported (all floats as Python repr, i.e. exact):
  meta     Blender / file version, frame range, fps, render engine and settings, active camera
  objects  per object: type, parent, local scale, dimensions; mesh vertex/polygon counts and a SHA-256 of the
           float32 vertex coordinates; curve control points; camera intrinsics (lens, sensor, shift, clipping)
  frames   per frame and object: world matrix (4x4, row-major), hide_viewport, hide_render
  props    the scene's custom properties (slam_*), converted to plain JSON
  texts    the embedded text blocks (the autoload script)
Compare two exports with visualization/compare_scene_state.py.
"""
import hashlib
import json
import sys
from array import array

import bpy


def plain(v):
    """IDProperty / bpy value -> JSON-serialisable Python value."""
    if hasattr(v, "to_dict"):
        return {k: plain(x) for k, x in v.to_dict().items()}
    if hasattr(v, "to_list"):
        return [plain(x) for x in v.to_list()]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, float):
        return repr(v)
    return v


out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
r = sc.render
state = {
    "meta": {
        "blender": bpy.app.version_string,
        "file_version": list(bpy.data.version),
        "frame_start": sc.frame_start, "frame_end": sc.frame_end, "fps": r.fps,
        "engine": r.engine, "resolution": [r.resolution_x, r.resolution_y, r.resolution_percentage],
        "render_filepath": r.filepath, "camera": sc.camera.name if sc.camera else None,
    },
    "objects": {}, "frames": {}, "props": {k: plain(sc[k]) for k in sorted(sc.keys())},
    "texts": {t.name: t.as_string() for t in bpy.data.texts},
}
objs = sorted(bpy.data.objects, key=lambda o: o.name)
for o in objs:
    d = {"type": o.type, "parent": o.parent.name if o.parent else None,
         "scale": [repr(v) for v in o.scale], "dimensions": [repr(v) for v in o.dimensions]}
    if o.type == "MESH":
        me = o.data
        co = array("f", [0.0]) * (3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
        d["mesh"] = {"vertices": len(me.vertices), "polygons": len(me.polygons),
                     "co_sha256": hashlib.sha256(co.tobytes()).hexdigest()}
    if o.type == "CURVE":
        d["curve_points"] = [[[repr(v) for v in p.co] for p in s.points] for s in o.data.splines]
    if o.type == "CAMERA":
        c = o.data
        d["camera"] = [repr(v) for v in (c.lens, c.sensor_width, c.sensor_height, c.shift_x, c.shift_y,
                                         c.clip_start, c.clip_end)] + [c.sensor_fit]
    state["objects"][o.name] = d
for f in range(sc.frame_start, sc.frame_end + 1):
    sc.frame_set(f)
    state["frames"][str(f)] = {
        o.name: {"matrix": [repr(o.matrix_world[i][j]) for i in range(4) for j in range(4)],
                 "hide_viewport": o.hide_viewport, "hide_render": o.hide_render}
        for o in objs}
# Second pass with the scene's own runtime registered (what the autoload script does when the .blend is opened
# with auto-run): imports visualization/scripts/slam_hud_panel.py next to this file, registers it, exports the
# sidebar settings (scene.slam_viz, stored by Blender 5 outside scene.keys()) and re-evaluates every frame with
# its frame-change handler active ("frames_live": matrices, visibility, curve points as the user sees them).
import importlib  # noqa: E402
import os  # noqa: E402
scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, scripts)
try:
    panel = importlib.import_module("slam_hud_panel")
    panel.register()
    state["runtime"] = {"registered": True, "module": os.path.relpath(panel.__file__, scripts)}
except Exception as exc:  # recorded, never hidden
    state["runtime"] = {"registered": False, "error": f"{type(exc).__name__}: {exc}"}
if state["runtime"]["registered"]:
    viz = getattr(sc, "slam_viz", None)
    state["runtime"]["slam_viz"] = ({p.identifier: plain(getattr(viz, p.identifier)) for p in viz.bl_rna.properties
                                     if p.identifier not in ("rna_type", "name")} if viz is not None else None)
    state["frames_live"] = {}
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        fr = {}
        for o in objs:
            fr[o.name] = {"matrix": [repr(o.matrix_world[i][j]) for i in range(4) for j in range(4)],
                          "hide_viewport": o.hide_viewport, "hide_render": o.hide_render}
            if o.type == "CURVE":
                fr[o.name]["curve_points"] = [[[repr(v) for v in p.co] for p in s.points] for s in o.data.splines]
        state["frames_live"][str(f)] = fr
with open(out, "w") as fh:
    json.dump(state, fh, indent=1, sort_keys=True)
print(f"EXPORTED {out}: frames {sc.frame_start}..{sc.frame_end}, {len(objs)} objects, "
      f"{len(state['props'])} scene properties, {len(state['texts'])} text blocks")
