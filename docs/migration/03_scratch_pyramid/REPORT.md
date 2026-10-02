# Checkpoint 03: multiscale pyramid in the scratch front end

Branch `vision-library-removal`. Two commits:
- `979d69a`: the pyramid, multiscale extraction and tests, plus the scratch `--pyramid-levels` option
- the next commit: figures, docs and this report

## Why

Checkpoint 02 showed that the single-scale scratch detector finds 108 keypoints in frame 0,
against 366 for the reference ORB. The difference is the reference's 8-level pyramid
(scale 1.2): at full resolution the counts are 97 (ORB) and 108 (scratch).

## What was implemented (no OpenCV)

| Module | Content |
|---|---|
| `include/features/pyramid.hpp`, `src/features/pyramid.cpp` | `BuildPyramid`: each level is resampled from the previous one by centre-aligned bilinear interpolation, at size round(W/1.2^l) × round(H/1.2^l). `scale_x/y` = original size / level size. `LevelToBaseX/Y`: x0 = (x + 0.5)·scale − 0.5 |
| `include/features/multiscale.hpp`, `src/features/multiscale.cpp` | `ExtractMultiscale`: per level, the unchanged `DetectFast` (threshold 20, same NMS, border, cap), `AssignOrientations` and `ComputeBrief`, all on the level image. Keypoints keep level, level x/y, scale, score, angle and descriptor |
| `pnp_from_scratch/src/frontend.*` | `FrontendParams::pyramid` (default 1 level = single scale). `FrameFeatures` gains level, scale and level x/y; keypoint x/y are original coordinates. `Correspondence` gains the levels of both keypoints |
| `pnp_from_scratch/src/pipeline_main.cpp` | `--pyramid-levels L`, `--out DIR`. The keypoint and correspondence CSVs gain level columns; the summary prints the per-level counts |
| `pnp_from_scratch/tools/make_figures.py` | `--pipeline`, `--out`. Keypoints are drawn with colour = level, circle ∝ scale and an orientation tick. New `pyramid_levels_0` figure. The PnP comparison covers single scale, pyramid and reference. The correspondence maps are unchanged (no orientation ticks) |

There is no Harris ranking and no per-level quota. RANSAC, PnP, trajectory accumulation,
the Blender scene and the coordinate conventions are unchanged.

Level sizes: 640x480, 533x400, 444x333, 370x278, 309x231, 257x193, 214x161, 179x134.

## Results

Frame-0 keypoints per level (`test_multiscale_0_1.txt`):

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | total |
|---|---|---|---|---|---|---|---|---|---|
| scratch | 108 | 87 | 53 | 45 | 35 | 29 | 21 | 20 | 398 |
| reference ORB | 97 | 73 | 50 | 46 | 36 | 22 | 23 | 19 | 366 |

The pyramid alone gives 398, 9 % above ORB's 366. It was not tuned to reach 366.

| | Single scale (before) | Pyramid (after) | Reference |
|---|---|---|---|
| pair 0→1 keypoints (frame 0 / 1) | 108 / 107 | 398 / 385 | 366 / 367 |
| pair 0→1 raw / filtered matches | 108 / 86 | 398 / 346 | 366 / 202 |
| pair 0→1 valid 3D→2D / RANSAC inliers | 76 / 57 | 249 / 206 | 136 / 121 |
| pair 0→1 inlier reprojection (mean) | 1.53 px | 2.07 px | 2.01 px |
| pair 0→1 rotation / translation error | 2.306° / 0.0197 m | 2.766° / 0.0232 m | 0.978° / 0.0073 m |
| mean keypoints / filtered matches / 3D→2D / inliers | 131.4 / 94.2 / 83.6 / 56.5 | 484.1 / 395.6 / 313.6 / 240.3 | 430.4 / 191.7 / 142.5 / 132.0 |
| per-pair rotation error, mean / median / max | 4.74° / 2.75° / 29.66° | 2.07° / 1.83° / 5.58° | 1.39° / 1.17° / 3.28° |
| per-pair translation error, mean / median | 0.0416 / 0.0241 m | 0.0181 / 0.0153 m | 0.0125 / 0.0109 m |
| trajectory error, mean over 36 frames | 0.187 m / 21.0° | 0.076 m / 6.9° | 0.180 m / 21.4° |
| final error (frame 35) | 0.095 m / 9.9° | 0.106 m / 13.5° | 0.277 m / 32.1° |

How to read this:
- **Pair 0→1** is slightly worse with the pyramid (+0.46°). The coarse levels add keypoints
  that are localised less precisely after mapping back (the inlier reprojection rises from
  1.53 to 2.07 px). The PnP is linear and has no reprojection refinement to absorb this.
- **Over the sequence** the pyramid is much better: the mean per-pair rotation error halves,
  the worst pair drops from 29.7° to 5.6°, and the mean trajectory error falls 2.5×.
- **The final-frame error** is not a fair summary of the single-scale run: its mixed-sign
  per-pair errors partly cancel.
- 190 of the 346 filtered matches of pair 0→1 join keypoints from different levels.

Full summaries: `pipeline_single_summary.txt`, `pipeline_pyramid_summary.txt`,
`pnp_comparison.md`.

## Regression

- **Scratch ctest:** 14/14 pass (`scratch_ctest.txt`), including `features_pyramid_test`
  (`features_pyramid_test.txt`) and `test_multiscale_0_1`.
- **Single scale reproducible:** with `--pyramid-levels 1` (the default), `pairs.csv` and
  `trajectory.csv` are byte-identical to checkpoint 02, and `features_pyramid_test` checks that
  one level equals the single-scale path bit for bit.
- **Reference suite:** run with `scripts/migration/run_test_suite.sh` (`reference_suite_summary.tsv`).
  - The dataset regenerates byte-identically (74 files), and the reference metrics are unchanged.
  - 24 of 25 programs exited 0. `orb_from_scratch_test` exited 139 in this run (see below).
- **OpenCV:**
  - `pnp_from_scratch/`, `features/{types,fast,orientation,brief,matcher,pyramid,multiscale}`,
    `cmake/feature_core.cmake`: no text references.
  - `ldd` of `scratch_pipeline` and `test_multiscale_0_1`: no OpenCV libraries.
  - `make_figures.py` run: `cv2` not loaded.
  - The OpenCV references in `include/features/features.hpp` and `src/features/features.cpp`
    belong to the reference pipeline's adapter. They are not part of `feature_core`.

### Pre-existing intermittent crash in `orb_from_scratch_test` (not caused by this checkpoint)

The book demo `tests/unit/orb_from_scratch_test.cpp` (OpenCV, independent of `feature_core`,
source and build rules identical to `baseline-pre-migration`) segfaults in about 1 of 10 runs
(3/30 locally). gdb puts the crash in its `ComputeORB`.
- **Cause:** the rotated BRIEF pattern reaches radius 18.38 px, but the demo only rejects
  keypoints within 16 px of the border. Keypoints near the bottom or right edge therefore read
  up to 3 pixels past the image buffer.
- **Why it is intermittent:** whether that read faults depends on the memory layout (with
  address randomisation off it never crashed in 40 runs). The baseline run happened to pass.
- **Not fixed here:** it is reference code, outside this checkpoint. The fix is a border of
  19 px (or ⌈max pattern radius⌉ + 1).

## Not changed (by instruction)

- **Blender:** `ScratchPnP_Animated_Camera` still reads the single-scale
  `pnp_from_scratch/results/pipeline/trajectory.csv`.
- **Pipeline:** trajectory accumulation, conventions and the Blender conversion are untouched.
- **Default:** the default stays single scale.

Recommended next steps:
- make 8 levels the default
- point the Blender camera at the pyramid run
- add a nonlinear reprojection refinement of the PnP
