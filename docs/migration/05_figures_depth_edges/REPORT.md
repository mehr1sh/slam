# Checkpoint 05: result figures, Blender overview, depth-edge audit, descriptor distances

Branch `vision-library-removal`.

**No change to the algorithm.** The pipeline, its defaults, the Blender transforms and the
reference pipeline are unchanged. What was added:
- figure tools
- a read-only experiment program
- a read-only Blender render script
- this report and its evidence

## 1. Figures (generated in `pnp_from_scratch/results/figures/`, PNG + PDF)

| Figure | Tool |
|---|---|
| `trajectory_3d_comparison`, `trajectory_position_error`, `trajectory_rotation_error`, `relative_pose_error`, `final_trajectory_comparison` | `pnp_from_scratch/tools/trajectory_figures.py` |
| `blender_trajectories_overview` | `scripts/render_trajectory_overview.py` (renders the saved scene from a temporary camera; does not save it) |
| `descriptor_distances_0_1` | `pnp_from_scratch/tools/descriptor_figure.py` |
| `depth_edge/depth_edge_{correspondences_0_1,map_0,statistics,experiment}` | `pnp_from_scratch/tools/depth_edge_audit.py` (after `build/depth_edge_experiment`) |
| `bunny_correspondences_0_1{,_inliers,_outliers}`, `feature_comparison_0_1`, `fig03_orb_keypoints`, `pyramid_levels_0`, `pnp_comparison` | `pnp_from_scratch/tools/make_figures.py` (unchanged since checkpoint 04) |

**Data sources:**
- Reference PnP and ICP: `data/synthetic_bunny/slam_trajectory.csv`, byte-identical to the
  frozen `docs/migration/baseline/slam_trajectory.csv`; the Blender scene reads the same file.
- Scratch: the default run (`results/pipeline/`).

**Blender:**
- The scene contains the static bunny, the GT / PnP / ICP / ScratchPnP trajectory curves and
  the four animated cameras (`GT_`, `PnP_`, `ICP_`, `ScratchPnP_Animated_Camera`).
- `check_blender_consistency.py` (`blender_consistency_check.txt`): camera position within
  3.3e-8 m, rotation 0°, curve points exact, HUD values exact.
- The overview render uses a static colour legend. The scene's HUD text is written by a
  frame-change handler that does not run in a background render, and it would have shown
  stale numbers.

## 2. Consolidated result (35 pairs, 36 frames; measured values, no ranking)

| | Scratch (default: pyramid + refinement) | Reference PnP | Reference ICP |
|---|---|---|---|
| keypoints per frame, mean (pair 0→1) | 484.1 (398 / 385) | 430.4 (366 / 367) | same features as reference PnP |
| filtered matches per pair, mean (0→1) | 395.6 (346) | 191.7 (202) | same as reference PnP |
| valid correspondences, mean (0→1) | 313.6 3D→2D (249) | 142.5 3D→2D (136) | 128.5 3D→3D (119) |
| RANSAC inliers, mean (0→1) | 240.3 (206) | 132.0 (121) | — (no RANSAC) |
| mean pair rotation error | 1.164° | 1.391° | 2.810° |
| mean pair translation error | 0.0107 m | 0.0125 m | 0.0228 m |
| mean trajectory position error | 0.1466 m | 0.1798 m | 0.2701 m |
| mean trajectory rotation error | 18.16° | 21.36° | 30.33° |
| final position error (frame 35) | 0.2835 m | 0.2767 m | 0.5253 m |
| final rotation error (frame 35) | 31.90° | 32.12° | 64.92° |

**Scratch development** (each row is a separate run; the first two use `--no-refine`):

| | Single scale | → 8-level pyramid | → nonlinear refinement (default) |
|---|---|---|---|
| keypoints per frame, mean (frame 0) | 131.4 (108) | 484.1 (398) | 484.1 (398) |
| filtered matches / 3D→2D / inliers, means | 94.2 / 83.6 / 56.5 | 395.6 / 313.6 / 240.3 | 395.6 / 313.6 / 240.3 |
| mean pair rotation / translation error | 4.741° / 0.0416 m | 2.075° / 0.0181 m | 1.164° / 0.0107 m |
| worst pair rotation error | 29.66° | 5.58° | 2.56° |
| summed signed rotation error | +39.3° | +0.3° | −30.4° |
| mean trajectory error | 0.1871 m / 21.01° | 0.0759 m / 6.89° | 0.1466 m / 18.16° |
| final error (frame 35) | 0.0952 m / 9.88° | 0.1062 m / 13.52° | 0.2835 m / 31.90° |

Each step lowers the per-pair error. The trajectory does not improve monotonically,
because the remaining error is a systematic bias (§3) and the linear runs' errors partly
cancel.

## 3. Depth-edge audit (read-only; the production pipeline is unchanged)

**Experiment program:** `pnp_from_scratch/experiments/depth_edge_experiment.cpp`.
- It runs the unchanged front end, then the unchanged geometric stage (RANSAC around the
  linear PnP, then LM refinement) twice per pair: on all depth correspondences, and on the
  interior ones only.
- Its "all" run reproduces the production pipeline exactly (mean pair rotation error
  1.164°, final error 0.2835 m / 31.90°).

### The rule, and why these values

A depth correspondence is a **depth edge** if, within radius r = ⌈3·s⌉ of its depth sample
(s = the keypoint's pyramid scale), either:
- a pixel has no depth, or
- two 4-adjacent pixels differ in depth by more than τ(Z) = (Z/fx)·tan 80°.

Otherwise it is **interior**.

- **Radius:** 3 is the radius of the FAST segment-test circle, mapped from the keypoint's
  level to the original image. If that circle crosses a discontinuity, the detected corner is
  formed by it.
- **Threshold τ:**
  - One pixel spans Z/fx ≈ 0.93 mm at the median depth (0.486 m).
  - A continuous surface changes depth by (Z/fx)·tan θ between neighbouring pixels, where θ
    is the incidence angle, so a jump above τ means the surface is within 10° of tangent to
    the view ray, or there is an occlusion.
  - At Z = 0.486 m, τ ≈ 5.3 mm.
  - Cross-check: with the 8.7 cm step between frames, a 5 mm depth error moves the
    reprojection in the next frame by about 1 px (f·b/Z² ≈ 192 px/m).
  - Over all 36 depth images, the 99th percentile of adjacent-pixel jumps is 6.4 mm.
- **Not tuned on trajectory error.** τ and r come from the depth geometry and the detector,
  and were not adjusted after running the experiment.

### Statistics (`depth_edge_audit.txt`, `depth_edge_statistics`)

Over all 35 pairs:
- **Edge share:** 7812 of the 10977 depth correspondences (71.2 %), and 5628 of the 8412
  RANSAC inliers (66.9 %), are depth edges.
- **Pair 0→1:** 174 of 249 correspondences and 135 of 206 inliers.

Inlier percentiles (10 / 50 / 90):

| | interior (2784) | edge (5628) |
|---|---|---|
| valid depth pixels in the window | 1.00 / 1.00 / 1.00 | 0.51 / 0.66 / 1.00 |
| depth range in the window (mm) | 2.6 / 6.6 / 14.4 | 4.8 / 16.0 / 40.2 |
| largest adjacent jump (mm) | 0.8 / 1.4 / 3.4 | 1.6 / 6.0 / 26.2 |
| depth gradient at the sample (mm/px) | 0.30 / 0.76 / 1.60 | 0.51 / 1.64 / 4.61 (undefined for 1768: a neighbour has no depth) |

### Is the fixed-3D-point assumption violated? (ground truth, all inliers)

**Test:** move X_i, the frame-i depth point, into camera j with the **true** motion, and
compare it with X_j, the matched frame-j keypoint back-projected with frame-j depth.

| | interior | edge |
|---|---|---|
| \|T_gt X_i − X_j\|, percentiles 10 / 50 / 90 | 0.38 / 1.17 / 3.51 mm | 0.83 / 2.76 / 7.74 mm |
| above 5 mm | 4.9 % | 22.5 % |
| matched frame-j pixel has **no depth** | 3 | **520** |
| image residual vs the true projection, median \|r\| | 1.01 px | 1.89 px |
| mean residual (u, v) | (+0.22, +0.02) px | (+0.34, −0.10) px |

**Supported.** Edge correspondences violate the fixed-point assumption measurably more often:
- 520 of them match a frame-j pixel that lies off the object, i.e. the frame-j feature sits
  on the silhouette, not on the surface point seen in frame i;
- their 3D discrepancy under the true motion is about 2.4× larger.

**The per-point effect is small.** The mean image offset differs by only about 0.1 px. The
bias appears because about 70 % of the points share it, and the orbit geometry turns small
image errors into rotation errors (checkpoint 02: the motion is mostly cancelled in the
image).

**Phrasing.** This does not show that the matches are "bad matches": most edge
correspondences describe the same RGB corner in both frames. They are geometrically
unstable observations for PnP.

### Experiment: all depth correspondences vs interior only (`depth_edge_experiment`)

| | all (production) | interior only |
|---|---|---|
| correspondences per pair, mean | 313.6 | 90.4 (min 42) |
| RANSAC inliers, mean (min) | 240.3 (154) | 76.9 (**2**) |
| mean pair rotation / translation error, 35 pairs | 1.164° / 0.0107 m | 3.672° / 0.0245 m |
| median pair rotation error | 1.025° | 0.914° |
| pairs with lower rotation error | — | 20 / 35 |
| **33 pairs (8→9 and 28→29 excluded):** mean rotation / translation error | 1.157° / 0.0106 m | **1.018° / 0.0093 m** |
| **33 pairs:** summed signed rotation error | −28.0° (31/33 negative) | **−2.8°** (17/33 negative) |
| **33 pairs:** inlier reprojection error, mean | 1.98 px | 1.37 px |
| trajectory error at frame 8 (before the first failure) | 0.1261 m / 14.18° | **0.0128 m / 2.15°** |
| mean trajectory error, 36 frames | 0.1466 m / 18.16° | 0.6493 m / 63.48° |
| final error (frame 35) | 0.2835 m / 31.90° | 0.4728 m / 88.95° |

Two interior-only pairs fail:
- **8→9:** 46 inputs, 2 inliers after the refit, 82.7° error.
- **28→29:** 42 inputs, 10 inliers, 12.2° error.

In both, interior points are scarce and clustered, and RANSAC with the 6-point linear PnP
fails; those two pairs dominate the interior trajectory.

### Conclusions

1. **The depth-edge hypothesis is supported.** Edge correspondences violate the
   fixed-3D-point assumption more often, and removing them removes the systematic
   under-rotation: −28.0° → −2.8° over the 33 pairs where RANSAC succeeds, with smaller
   per-pair errors.
2. **Plain rejection is not yet a safe production change.** It leaves 42–46 correspondences
   on some pairs, too few for the current RANSAC; two pairs fail and the trajectory is worse.
3. **Candidate next steps** (not implemented; each is an algorithm change):
   - **Fallback:** reject edges only when enough interior points remain, otherwise keep all.
   - **Down-weighting:** keep edge points in RANSAC but give them lower weight in the
     refinement.
   - **Sanity check:** reject a RANSAC result whose inlier ratio collapses.

   Any of them must be validated on all 35 pairs without tuning on trajectory error.

## 4. Descriptor distances (`descriptor_distances_0_1`)

Both pipelines use the same filter, distance ≤ max(2·d_min, 30) = 30 on pair 0→1.

| | Scratch | Reference ORB |
|---|---|---|
| unrelated pairs (all frame-0 × frame-1): median, share ≤ 30 | 69, 5.28 % | 109, 0.48 % |
| best match: median | 16 | 29 |
| best matches kept by the filter | 346 / 398 | 202 / 366 |
| bits with P(1) outside 0.2–0.8 | 144 / 256 | 29 / 256 |

Scratch distances are smaller overall, because the rotated random Gaussian pattern has
strongly biased bits; the learned ORB pattern was chosen for balanced, uncorrelated bits.
The threshold was not changed.

**Reference histogram source.** The reference all-pairs histogram
(`reference_orb_distances_0_1.csv`) was exported once by `export_reference_orb_distances.py`
with the frozen baseline's library. The script asserts that it reproduces the frozen
best-match distances. The scratch figure tools only read the CSV.

## 5. Regression and audit

- **Scratch:** ctest 15/15 (`scratch_ctest.txt`).
- **Reference:** `run_test_suite.sh`, 25/25 programs exit 0, and the dataset regenerates
  byte-identically (`reference_suite_summary.tsv`). Reference code is untouched.
- **OpenCV** (`opencv_audit.txt`):
  - no OpenCV text, include, `find_package` or link target in the scratch path
  - `ldd` of all 16 scratch binaries, including `depth_edge_experiment`: none linked
  - the four figure tools run in one process: `cv2` never loaded
  - the only OpenCV use is the reference-side, one-off export script in this directory
