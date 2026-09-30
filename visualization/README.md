# Blender visualization — GT vs PnP vs ICP

An animated Blender scene that compares three camera trajectories around the
Stanford Bunny:

- **Ground truth (green):** where the camera actually was.
- **PnP (orange):** where PnP (RANSAC) thinks the camera was.
- **ICP (magenta):** where ICP thinks the camera was.

Blender is only a viewer. No trajectory, pose-estimation or pose-composition
maths lives here. Everything is computed in C++
(`tests/synthetic/slam_trajectory_test.cpp`); these scripts only read the
results, convert them to Blender's coordinate representation, and animate
them.

## Layout

```
scripts/
  build_scene.py       builds/rebuilds the whole scene from data/synthetic_bunny/*;
                       all options (SHOW_*, CAMERA_SAMPLE_STEP, PRESENTATION_MODE, ...) at its top
  slam_animation.py    per-frame update: Blender frame -> pose, progressive trajectories,
                       HUD/labels, optional debug extras (cameras are keyframed)
  slam_hud_panel.py    sidebar "Trajectories" tab (toggles, poses, errors), viewport HUD,
                       frame_change_post handler
  slam_coords.py       coordinate-convention constants + trajectory_to_blender_pose()
scenes/
  bunny_slam_demo.blend  written by build_scene.py (generated -- not tracked by git)
output/
  render/                rendered animation frames (frame_####.png, generated)
```

## Run

```bash
# from the repository root
blender --python visualization/scripts/build_scene.py      # rebuild + open
blender -y visualization/scenes/bunny_slam_demo.blend      # open the saved scene
blender -b -y visualization/scenes/bunny_slam_demo.blend -a  # render the animation
```

`-y` lets the embedded `slam_viz_autoload.py` text block re-register the
sidebar, HUD and frame handler. Without it, add
`--python visualization/scripts/slam_hud_panel.py`. Rerun `build_scene.py`
whenever the files in `data/synthetic_bunny/` change.

## What you see

- **Data source:** all three trajectories come from `slam_trajectory.csv`. The
  build checks them against `groundtruth.txt`, `pnp_trajectory.txt` and
  `icp_trajectory.txt`.
- **Lines:** a thin line in the method's colour shows its full trajectory
  (all 36 poses) for the whole animation.
- **Cameras:** exactly **three real Blender Camera objects** move along those
  lines: `GT_Animated_Camera`, `PnP_Animated_Camera` and `ICP_Animated_Camera`,
  in the GroundTruth / PnP / ICP collections. Each is keyframed at every frame:
  Blender frame *i* = that trajectory's pose *i* (frames 0–35), exact at whole
  frames, with LINEAR interpolation in between. Blender draws their frustums.
  The viewport overlay re-draws each camera's own frustum in its trajectory
  colour and labels it ("PnP Camera"). There are no static per-pose cameras by
  default.
- **HUD:** the frame counter and each method's translation and rotation error
  against ground truth at the current frame. The errors are the C++ values
  from the CSV.
- **Axes:** a small triad of the C++ world frame, which is Y-up.
- **Floor:** a fine floor grid in the viewport and a quiet ground plane in renders.

## Looking through a camera

- **In the viewport:** click `GT_Animated_Camera`, `PnP_Animated_Camera` or
  `ICP_Animated_Camera` in the Outliner, then press **Ctrl+Numpad0**, or use
  the sidebar's *Look through selected camera* button. Press **Space** to play:
  the view follows that trajectory.
- **From Blender's Python console:**

  ```python
  bpy.context.scene.camera = bpy.data.objects["GT_Animated_Camera"]
  bpy.context.scene.camera = bpy.data.objects["PnP_Animated_Camera"]
  bpy.context.scene.camera = bpy.data.objects["ICP_Animated_Camera"]
  # or, with the helpers:
  import slam_hud_panel as viz
  viz.set_active_camera(viz.get_pnp_camera())              # then Numpad 0
  viz.set_active_camera(viz.get_icp_camera(), view=True)   # also enters camera view
  ```

- **Which camera is active:** nothing changes the scene camera automatically.
  The build starts with `GT_Animated_Camera` because renders need a camera.

**Camera model.** The cameras use the renderer's pinhole model from
`intrinsics.txt`:

- lens = fx / 640 × 36 mm = 29.30 mm, on a 36 mm sensor with horizontal fit
- shift_x = (320 − cx) / 640 = −0.00797 and shift_y = (cy − 240) / 640 = +0.01516
- render resolution 640 × 480, with pixel aspect y = fy / fx

**Pose conversion.** The pose goes through `slam_coords.trajectory_to_blender_pose()`,
the same conversion as before, under `WorldRoot`.

## Options

Set the options at the top of `build_scene.py`, or change them live in the
sidebar (N) > **Trajectories**:

- `SHOW_GROUND_TRUTH`, `SHOW_PNP`, `SHOW_ICP`
- `SHOW_CAMERAS` (the three animated cameras)
- `SHOW_HISTORICAL_CAMERAS` (default `False`) and `CAMERA_SAMPLE_STEP`: an
  optional debug extra that adds static, smaller cameras at every Nth pose.
  These are build-time settings: rerun `build_scene.py` after changing them.
- `SHOW_WORLD_AXES`, `SHOW_HUD`
- `PRESENTATION_MODE`, `PROGRESSIVE_TRAJECTORIES`

With `PRESENTATION_MODE = False` (debug mode), the scene also shows red
GT → estimate error vectors, each method's camera centre in the HUD, and pose
indices at the optional historical cameras. `PROGRESSIVE_TRAJECTORIES = True`
makes the lines grow frame by frame instead of always showing the full path.
