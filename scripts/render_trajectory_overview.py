"""Read-only overview render of the saved Blender scene: the static Bunny and
the GT / PnP / ICP / ScratchPnP trajectory curves, seen from a temporary
camera above the orbit. Nothing in the scene is changed on disk (the file is
not saved); the four animated cameras stay as they are.

Run from the repository root:
    blender -b visualization/scenes/bunny_slam_demo.blend \
        --python scripts/render_trajectory_overview.py -- OUT.png
"""
import sys
from pathlib import Path

import bpy
import mathutils

out = Path(sys.argv[sys.argv.index("--") + 1]).resolve() if "--" in sys.argv else Path("trajectory_overview.png").resolve()
sc = bpy.context.scene
sc.frame_set(sc.frame_end)
# temporary overview camera: 50 deg above the orbit plane, looking at the bunny
bunny = bpy.data.objects["Bunny"]
target = sum((bunny.matrix_world @ mathutils.Vector(c) for c in bunny.bound_box), mathutils.Vector()) / 8
cam_data = bpy.data.cameras.new("Overview_tmp")
cam_data.lens = 35
cam = bpy.data.objects.new("Overview_tmp", cam_data)
sc.collection.objects.link(cam)
cam.location = target + mathutils.Vector((0.75, -0.95, 1.15))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
# keep the render to the bunny and the four trajectory curves
keep = {"Bunny", "GT_Trajectory", "PnP_Trajectory", "ICP_Trajectory", "ScratchPnP_Trajectory"}
for o in sc.objects:
    if o.type in {"MESH", "CURVE", "FONT", "GPENCIL", "EMPTY"} and o.name not in keep and not o.name.startswith("Ground"):
        o.hide_render = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 1600, 1200, 100
sc.render.film_transparent = False
# the scene's HUD text is written by a frame-change handler that does not run in this
# background render; replace it with a static colour legend instead of stale numbers
sc.render.use_stamp_note = True
sc.render.stamp_note_text = ("Stanford Bunny, frames 0-35 | GT (green) | Reference PnP (orange) | "
                             "Reference ICP (magenta) | Scratch PnP (cyan)")
for flag in ("use_stamp_frame", "use_stamp_date", "use_stamp_time", "use_stamp_render_time", "use_stamp_camera",
             "use_stamp_lens", "use_stamp_scene", "use_stamp_filename", "use_stamp_marker", "use_stamp_memory",
             "use_stamp_frame_range", "use_stamp_hostname", "use_stamp_sequencer_strip"):
    if hasattr(sc.render, flag):
        setattr(sc.render, flag, False)
sc.render.filepath = str(out)
bpy.ops.render.render(write_still=True)
print("OVERVIEW_RENDER", out)
