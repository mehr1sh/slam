# Visualization (Blender)

Blender scenes and their Python scripts. Visualization only **reads** files written by the C++ programs
(trajectories, correspondences, exported clouds); it never estimates or corrects a pose. The C++ code is the
source of truth for all geometry.

- Bunny scene: `results/blender/bunny_slam_demo.blend` (tracked), built by `visualization/scripts/build_scene.py`;
  full rebuild with `bash visualization/rebuild_blender.sh` (or `pixi run blender-rebuild`).
- TUM scenes: built from a `tum_trajectory_eval` run by the TUM scripts (section below; ported from main on
  2026-11-04). Output goes to `visualization/scenes/tum/` (ignored).
- Coordinate conventions: `docs/architecture/COORDINATE_CONVENTIONS.md`.

**Blender version.** The Bunny scene `results/blender/bunny_slam_demo.blend` was rebuilt on 2026-11-06 with
**Blender 5.2.0** (file version 5.2.44, SHA-256 `508ab5af…`) and **requires Blender 5.x** (≥ 5.0) to open;
Blender 4.x cannot read it. The previous scene, saved by Blender 4.0 (SHA-256 `ed569c33…`), is preserved outside
the repository in `../backups/stage_c_pre_rebuild/bunny_slam_demo.blend` and in Git
(`outputs/blender/bunny_slam_demo.blend` at commit `aadfd13`).

**Equivalence of the rebuilt scene.** Compared with the Blender-4.0 scene (same inputs, both exported with
`export_scene_state.py` under Blender 5.2), everything the scene shows is identical: every camera's position,
orientation and viewing direction at every frame, all 720 per-frame object matrices, the Bunny's placement,
scale and mesh, the trajectory curves, camera intrinsics and visibility; the consistency checker's 144 result
rows are identical. **One non-functional difference remains:** the saved default viewport orientation
`scene["slam_viz_external_view"].rotation` differs by 1–2 float32 ulps per quaternion component (5.9e-6°),
because Blender 4.0 and 5.2 evaluate `mathutils` `to_track_quat` on the same constant slightly differently.
It is a UI view, not a scene camera.

**Blender warnings.** `build_scene.py` triggers `DeprecationWarning`s for `Material.use_nodes` and
`World.use_nodes` (removed in Blender 6.0; the scene will need updating then). Blender's own `bl_pkg` add-on
prints a harmless `cattrs` traceback at start-up; run headless commands with `--factory-startup`.
In background mode (`blender -b`) frame-change handlers do not run, so the HUD text is not updated in
background renders; `tools/figures/render_trajectory_overview.py` therefore draws a static legend.

**Inputs.** The scene reads only tracked or frozen files, never a runtime output:
`results/benchmarks/rgbd_icp/` (corrected ICP) and the frozen T0 scratch trajectory
`tests/fixtures/bunny_t0_scene/trajectory.csv`; both paths are defined once, in `scripts/trajectory_data.py`
(`ICP_RESULTS`, `SCRATCH_SOURCE`).

**Locating the helper scripts.** When the scene is opened with auto-run (`blender -y …`), its embedded
`slam_viz_autoload.py` finds the repository at run time: `$SLAM_REPO_ROOT` if set, otherwise the nearest
directory at or above the `.blend` that holds `data/synthetic_bunny/intrinsics.txt` and
`visualization/scripts/trajectory_data.py`. The scene can therefore be moved to any depth inside the repository;
outside it, set `SLAM_REPO_ROOT`, or it stops with a clear error.

**Comparing two builds of the scene** (e.g. after a rebuild):

```bash
blender -b --factory-startup OLD.blend --python visualization/export_scene_state.py -- old.json
blender -b --factory-startup NEW.blend --python visualization/export_scene_state.py -- new.json
python3 visualization/compare_scene_state.py old.json new.json [--expect PROP] [--expect-text NAME]
```

## Bunny scene: corrected ICP workflow

The tracked `results/blender/bunny_slam_demo.blend` is rebuilt from the corrected C++ run.
Open it from the extracted project, keeping the folder structure intact:

```bash
blender -y results/blender/bunny_slam_demo.blend
```

The corrected ICP camera is selected initially. Press Space to animate, or select
GT / PnP / ICP cameras in the Trajectories sidebar. Corrected ICP almost overlaps
GT at this scale. Orange PnP remains the explicitly labelled **legacy comparison**.

To recompile, estimate, rebuild and validate everything:

```bash
bash visualization/rebuild_blender.sh
```

Requirements: C++17 compiler, CMake, OpenCV, Eigen and Blender. Estimation is all
C++; Python scripts here only build and validate the Blender scene.

The scene reads `results/benchmarks/rgbd_icp/selected_trajectory.txt`, choosing the current
run's pose-graph trajectory when optimization succeeds, otherwise point-to-plane.
`trajectory_data.py` combines it with GT/legacy PnP and recomputes all ICP HUD
errors. `results/benchmarks/rgbd_icp/blender_trajectory.csv` and `blender_source.json` are
regenerated from those authoritative sources each build. Historical
`data/synthetic_bunny/slam_trajectory.csv` is **not** the corrected viewer source.
A completed-run marker and source fingerprint prevent silent use of a stale scene.

Tested in Blender 5.2.0 (2026-11-06, `check_blender_consistency.py` with auto-run): all 36 frames, camera poses,
viewing axes, curve points, HUD values, animation range, reopen handler and intrinsic projection checks. Earlier
(Blender 4.0.2) the same checks passed on the previous scene. No final image render was made.

## Bunny scene: controls and design

Blender visualization — GT vs PnP vs ICP vs Scratch PnP

A visualization and debugging aid for the SLAM pipeline: an animated Blender
scene that compares the estimated camera trajectories with ground truth on the
synthetic evaluation sequence (the test object is the Stanford Bunny mesh):

- **Ground truth (green):** where the camera actually was.
- **PnP (orange):** where PnP (OpenCV RANSAC) thinks the camera was.
- **ICP (magenta):** where ICP thinks the camera was.
- **Scratch PnP (cyan):** where the production pipeline (`build/scratch_pipeline`) thinks the
  camera was. It uses its own features (8-level pyramid, FAST, rotated BRIEF, Hamming), RANSAC
  around its linear PnP and Levenberg–Marquardt refinement, and ends 0.040 m / 3.86° from GT
  (the frozen T0 run, last row of the fixture).
  - It is read from the frozen fixture `tests/fixtures/bunny_t0_scene/trajectory.csv` (T0 defaults; a
    byte-identical copy of the run the scene was originally built from). A fresh `scratch_pipeline` run never
    changes the scene; `rebuild_blender.sh` only reports whether today's run still equals the fixture.
  - The scene records the file it was built from in `scene["slam_scratch_source"]`; the consistency check
    requires it to equal `trajectory_data.SCRATCH_SOURCE`, so another source (e.g. the milestone-2 file
    `results/runs/milestones/scratch_pnp_trajectory.csv`, linear PnP on raw matches) can only be shown by changing
    `SCRATCH_SOURCE` deliberately and rebuilding. Frames reached through a physically invalid pair are flagged in the HUD.
  - If that file holds only a prefix of the frames, the line ends there and the camera is hidden afterwards; nothing is filled in.

Blender is only a viewer. No trajectory, pose-estimation or pose-composition
maths lives here. Everything is computed in C++
(`reference/opencv_slambook2/tests/synthetic/slam_trajectory_test.cpp`); these scripts only read the
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
scenes/tum/             TUM scenes saved by build_tum_scene.py --save (generated, ignored)
```

The Bunny scene itself is `results/blender/bunny_slam_demo.blend` (tracked); rendered animation frames go
to `results/render/frame_####.png` (generated, ignored).

## Run

```bash
# from the repository root
blender --python visualization/scripts/build_scene.py      # rebuild + open
blender -y results/blender/bunny_slam_demo.blend      # open the saved scene
blender -b -y results/blender/bunny_slam_demo.blend -a  # render the animation
```

`-y` lets the embedded `slam_viz_autoload.py` text block re-register the
sidebar, HUD and frame handler. Without it, add
`--python visualization/scripts/slam_hud_panel.py`. Rerun `build_scene.py`
whenever the files in `data/synthetic_bunny/` change.

## What you see

- **Data source:** GT and legacy PnP come from their TUM files; ICP comes from the corrected C++ run. See the workflow above.
- **Scratch PnP data:** it comes from the frozen `tests/fixtures/bunny_t0_scene/trajectory.csv`
  (same T_wc format, same conversion as the other cameras). The build checks:
  - that frame 0 is the GT anchor
  - its stored errors against errors it recomputes from the poses
  - that it differs from the reference PnP

- **Lines:** a thin line in the method's colour shows its full trajectory
  (all 36 poses) for the whole animation.
- **Cameras:** exactly **four real Blender Camera objects** move along those
  lines: `GT_Animated_Camera`, `PnP_Animated_Camera`, `ICP_Animated_Camera` and
  `ScratchPnP_Animated_Camera`, in the GroundTruth / PnP / ICP / ScratchPnP
  collections. Each is keyframed at every frame where its trajectory has a pose:
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
  The build starts with `ICP_Animated_Camera` because renders need a camera.

**Camera model.** The cameras use the renderer's pinhole model from
`intrinsics.txt`:

- lens = fx / 640 × 36 mm = 29.30 mm, on a 36 mm sensor with horizontal fit
- shift_x = (320 − cx) / 640 = −0.00797 and shift_y = (cy − 240) / 640 × fx/fy = +0.01515
- render resolution 640 × 480, with pixel aspect ratio y/x = fx / fy (encoded with both components >= 1)

**Pose conversion.** The pose goes through `slam_coords.trajectory_to_blender_pose()`,
the same conversion as before, under `WorldRoot`.

## Options

Set the options at the top of `build_scene.py`, or change them live in the
sidebar (N) > **Trajectories**:

- `SHOW_GROUND_TRUTH`, `SHOW_PNP`, `SHOW_ICP`
- `SHOW_SCRATCH_PNP` (the 4th trajectory, if its file exists)
- `SHOW_CAMERAS` (the animated cameras)
- `SHOW_HISTORICAL_CAMERAS` (default `False`) and `CAMERA_SAMPLE_STEP`: an
  optional debug extra that adds static, smaller cameras at every Nth pose.
  These are build-time settings: rerun `build_scene.py` after changing them.
- `SHOW_WORLD_AXES`, `SHOW_HUD`
- `PRESENTATION_MODE`, `PROGRESSIVE_TRAJECTORIES`

With `PRESENTATION_MODE = False` (debug mode), the scene also shows red
GT → estimate error vectors, each method's camera centre in the HUD, and pose
indices at the optional historical cameras. `PROGRESSIVE_TRAJECTORIES = True`
makes the lines grow frame by frame instead of always showing the full path.

## TUM RGB-D scenes (real sequences)

Runs are read from `results/runs/` (copy an archived run there first; see `docs/DATA_AND_ARCHIVE.md`), and the
exports are written there too.

A separate scene for a `tum_trajectory_eval` run. It shows the real RGB-D point cloud plus the GT, PnP and ICP
camera motion. The Bunny scene and its scripts are unchanged; this reuses their colours (`METHOD_STYLES`) and
their single pose conversion (`slam_coords.trajectory_to_blender_pose`).

```bash
# 1. export, outside Blender (decodes the images once):
#    trajectory JSON, PLY clouds, depth previews, real match images
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/export_tum_scene.py \
    --run results/runs/stress/xyz_session3_pnp_validity/n100_stride4   # -> results/runs/blender/<sequence>/<run>/
# 2. build the scene (headless) with numeric checks, save it and render a preview
blender --background --python visualization/scripts/build_tum_scene.py -- \
    --export results/runs/blender/rgbd_dataset_freiburg1_xyz/stress__xyz_session3_pnp_validity__n100_stride4 \
    --density preview --frame 74 --check --render /tmp/tum_preview.png
#    --density render   ~650k points instead of ~33k
#    --view gt|pnp|icp  render through that camera (point cloud only)
# 3. open it (allow scripts so the "TUM SLAM" sidebar panel registers)
blender visualization/scenes/tum/rgbd_dataset_freiburg1_xyz__stress__xyz_session3_pnp_validity__n100_stride4__preview.blend
```

**Three independent reconstructions.** Each scene holds three point clouds:
`PointCloud_placed_with_GT_poses`, `..._PnP_poses` and `..._ICP_poses`.
- **Shared:** all three use the same frames, the same sampled pixels (a hash per frame is in each manifest), the
  same intrinsics and the same depth scale (`tum_reconstruction.py`). Only the pose that places each frame differs.
- **No ground truth in the estimated clouds:** the PnP and ICP clouds never read it. A frame whose estimated pose
  is held (failed or rejected pair) is excluded from that cloud and listed in the manifest with the reason.
- **Selecting:** the panel's *Point cloud placed with* selector shows one cloud and draws the trajectory that
  placed it 2.5× thicker; the other trajectories stay as overlays.
- **Comparing:** `--render-all <prefix>` renders all three from one camera (`--view overview|closeup`).
  `compose_reconstructions.py` puts them side by side, captioned with the numbers in `comparison_<density>.json`:
  the per-point displacement from the GT cloud (same pixels) and the occupied 1 cm voxels per 1000 points.

```bash
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/export_tum_scene.py \
    --run results/runs/stress/xyz_session1/n100_stride4 --include-frames 73,74 --force
E=results/runs/blender/rgbd_dataset_freiburg1_xyz/stress__xyz_session1__n100_stride4
blender --background --python visualization/scripts/build_tum_scene.py -- --export $E --density render \
    --frame 74 --check --render-all $E/render_frame74          # add --view closeup for the close-up camera
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/compose_reconstructions.py \
    --export $E --prefix $E/render_frame74 --density render
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/test_tum_reconstruction.py --export $E
```

The manifests are `manifest_<density>_<mode>.json` in the export directory.

**Conventions**
- The Blender world is the TUM world frame (z up, metres), with no extra rotation or recentring.
- Poses are the trajectory.csv T_wc (X_world = R X_cam + t, camera axes x right, y down, z forward).
- Point clouds: `X_world = R_wc^S ((u - cx) z / fx, (v - cy) z / fy, z) + t_wc^S`, where S is the pose source
  (GT, PnP or ICP), z = raw / 5000 m, the official intrinsics and no undistortion. Points come from every 10th
  frame on an 8 px grid (preview) or every 3rd frame on a 4 px grid (render), plus `--include-frames`. There is no
  world-space filter.
  - The earlier version placed a single cloud with GT poses only and voxel-filtered it in world space.
- Blender cameras get `R_wc @ R_FIX`, with lens = fx·36/W, shift_x = (W/2 − cx)/W and shift_y = (cy − H/2)/W.
  They are keyframed at every frame (Blender frame = trajectory frame + 1), with constant interpolation.

**Objects**
- `GT_*`, `PnP_*`, `ICP_*`: trajectory curve, animated camera, frustums (every 25 frames, plus first, last and
  `--frame`), start sphere, end cube.
- `Connectors`: GT centre → estimated centre of the same frame.
- `Aligned_optional`: one global SE(3) per estimate. Excluded from the view layer by default.
- `Overview_Camera`: the render viewpoint.

**Frame inspection**
- The *TUM SLAM* sidebar tab has a frame selector, which also moves the timeline. It shows timestamps, errors,
  and the pose status: estimated, held after a failed pair, or held after a rejected pair (with the reason).
- Buttons show the frame's RGB, depth preview and real matches. Matches are available only for pairs with an
  `--export-pairs` export.

**`--check` verifies:**
- every animated camera against the exported poses at every frame
- that the view direction equals R_wc[:,2]
- the frustum apex and direction
- that the point cloud is identical to the PLY
- the largest PnP error

**Limitations**
- Only `freiburg1_xyz` is downloaded. `freiburg1_desk` uses the same commands once downloaded; its freiburg1
  intrinsics are chosen from the sequence name.
- Frustums are drawn every 25 frames, not at every frame.
