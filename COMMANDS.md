# Commands

All commands start from the repository root unless a `cd` is shown. The C++
programs are run from `build/`, because they use paths relative to it.

## Environment and build

```bash
pixi install            # create the environment from pixi.lock
pixi run build          # cmake -B build -S . && cmake --build build -j
```

## Self-checking tests (exit non-zero on failure)

```bash
cd build
./fast_test
./brief_test
./essential_matrix_test
./vo_pipeline_test
./trajectory_validation_test ../data/synthetic_bunny
```

## Synthetic Stanford Bunny pipeline

```bash
cd build

# 1. Regenerate the RGB-D dataset. This reproduces the committed data byte-for-byte.
#    It also writes ../output/synthetic_bunny/scene_visualization.ply.
./render_bunny_test ../data/meshes/bunny/bun_zipper.ply ../data/synthetic_bunny \
    --num-frames 36 --radius-scale 4 --height-scale 1

# 2. Per-pair diagnostics against ground truth (arguments: dataset, frame i, frame j or end frame)
./pnp_synthetic_test    ../data/synthetic_bunny 0 1
./icp_synthetic_test    ../data/synthetic_bunny 0 1
./pnp_multiframe_test   ../data/synthetic_bunny 0 10
./synthetic_pose_eval_test ../data/synthetic_bunny/groundtruth.txt 0 1

# 3. Full PnP (RANSAC) and ICP trajectories. This writes
#    pnp_trajectory.txt, icp_trajectory.txt and slam_trajectory.csv into the dataset directory.
./slam_trajectory_test  ../data/synthetic_bunny

# 4. Independently check the accumulation maths and the written trajectory files
./trajectory_validation_test ../data/synthetic_bunny
```

## Results: figures, CSVs and tables

This writes to `results/`; see `results/README.md` for details.

```bash
pixi run build                # once, for the C++ programs
pixi run -e results results   # = python scripts/generate_results.py (Python environment with matplotlib)
pixi run -e results results --skip-cpp   # reuse results/data/cpp_export/
```

Behind the scenes, the script runs the C++ trajectory program with its
export-only flags. These don't change any trajectory output:

```bash
cd build
./slam_trajectory_test ../data/synthetic_bunny --export-dir <dir> --export-pair 0
# writes <dir>/pair_metrics.csv and <dir>/pair_0_1_{keypoints,raw_matches,correspondences}.csv
# --export-pair all writes the per-pair files for every pair
```

## Correspondence figures

```bash
# correspondence maps: results/figures/bunny_correspondences_<i>_<j>{,_inliers,_outliers}.{png,pdf,svg}
pixi run -e results correspondence-map              # pair 0 -> 1
pixi run -e results correspondence-map --pair 17    # pair 17 -> 18

# annotated explanatory versions: bunny_correspondences_annotated_<i>_<j>.*, bunny_pnp_icp_correspondences*.*
pixi run -e results correspondences [--pair K]
```

## Trajectory error diagnostics

This writes to `diagnostics/`; the findings are in `diagnostics/REPORT.md`.
It is diagnostic only: no algorithm, parameter or dataset is changed.

```bash
pixi run build
pixi run -e results diagnostics              # = python scripts/run_diagnostics.py
pixi run -e results diagnostics --skip-cpp   # reuse diagnostics/cpp/

# the C++ part on its own (same solver calls, measured against ground truth)
cd build && ./trajectory_diagnostics_test ../data/synthetic_bunny <out_dir>

# read-only check that the Blender scene shows exactly the CSV poses and errors
blender -b visualization/scenes/bunny_slam_demo.blend \
    --python scripts/check_blender_consistency.py -- diagnostics/blender_consistency.csv
```

## Real TUM RGB-D frame pair (book chapter 7 programs)

Images are written to `../output/`.

```bash
cd build
./feature_matching_test ../data/tum_sample/1.png ../data/tum_sample/2.png
./two_view_pose_test    ../data/tum_sample/1.png ../data/tum_sample/2.png
./triangulation_test    ../data/tum_sample/1.png ../data/tum_sample/2.png
./pnp_test  ../data/tum_sample/1.png ../data/tum_sample/2.png ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
./icp_test  ../data/tum_sample/1.png ../data/tum_sample/2.png ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
./orb_from_scratch_test          # reads ../data/tum_sample/{1,2}.png
```

## Book chapter 6 optimization demos

These write CSV files to `../output/optimization/`.

```bash
cd build
./gauss_newton_curve_fit_test
./g2o_curve_fit_test
```

## Blender visualization (Blender 5.x)

```bash
# optional 4th trajectory (Scratch PnP, all 36 poses): the self-contained scratch pipeline
(cd pnp_from_scratch && cmake -S . -B build && cmake --build build -j && ./build/scratch_pipeline)

# read-only check: every camera equals its data, and looks at the bunny (look-at angle)
blender -b visualization/scenes/bunny_slam_demo.blend --python scripts/check_blender_consistency.py -- /tmp/blender_check.csv

# build the scene, save it to visualization/scenes/bunny_slam_demo.blend, and open it
blender --python visualization/scripts/build_scene.py

# headless build (save only)
blender --background --python visualization/scripts/build_scene.py

# reopen the saved scene (-y lets its embedded script re-register the HUD/sidebar)
blender -y visualization/scenes/bunny_slam_demo.blend

# render the animation (from the active scene camera) to visualization/output/render/
blender -b -y visualization/scenes/bunny_slam_demo.blend -a
```

To choose the camera to look through, run one of these in Blender's Python
console, then press Numpad 0 (Space plays the animation):

```python
bpy.context.scene.camera = bpy.data.objects["GT_Animated_Camera"]
bpy.context.scene.camera = bpy.data.objects["PnP_Animated_Camera"]
bpy.context.scene.camera = bpy.data.objects["ICP_Animated_Camera"]
```

## Presentation

The slides include figures from `../output/`, so run `feature_matching_test`
and `triangulation_test` first.

```bash
cd presentation && latexmk -pdf slam_progress.tex
```
