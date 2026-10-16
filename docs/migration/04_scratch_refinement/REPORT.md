# Checkpoint 04: pyramid by default, nonlinear PnP refinement, Blender verification

Branch `vision-library-removal`. Commits:
- `118fdf3`: 8-level pyramid by default; Levenberg–Marquardt refinement; tests
- `694847c`: figures and tables for single scale / pyramid linear / pyramid + refinement
- the next commit: descriptor export, this report and the README

## 1. Pyramid as the default

- `FrontendParams` no longer overrides `PyramidParams::levels`, so the scratch default is
  8 levels × 1.2.
- FAST is unchanged: threshold 20, own score, 3×3 NMS, border 16, strongest 500 per level.
- `--pyramid-levels 1` is still available. `--pyramid-levels 1 --no-refine` reproduces the
  checkpoint-03 single-scale summary exactly, and `--no-refine` the checkpoint-03 pyramid
  summary (`summary_pipeline_single.txt`, `summary_pipeline_linear.txt`, compared with
  `diff`).
- **Frame 0 keypoints per level:** 108 87 53 45 35 29 21 20 = 398, against ORB's 366.
- **Not tuned to reach 366.** The reference ORB ranks by Harris score and applies per-level
  quotas; the scratch detector uses its FAST score, with no Harris and no quotas.

## 2. Nonlinear refinement (`pnp_from_scratch/src/refine.{hpp,cpp}`, Eigen only)

    RANSAC (unchanged) → best inlier set → linear PnP refit (unchanged) → LM refinement → final pose

- **Cost:** ½ Σ ‖π(K(RX + t)) − u‖² over the RANSAC inliers. The inlier set is not changed.
- **Update:** a left SE(3) perturbation, R ← Exp(φ)R, t ← Exp(φ)t + ρ, with the analytic
  Jacobian ∂π/∂X_c·[I | −[X_c]×].
- **Damping:** steps are accepted only if the cost drops; λ /= 3 on success, ×4 on failure.
- **Failures:** fewer than 3 points, or a point behind the camera: the linear pose is kept.
- `--no-refine` skips the refinement. `pairs.csv` keeps the linear pose's errors and
  reprojection statistics (`linear_*`).

## 3. Validation

Inliers are the same for linear and refined (same RANSAC). Reprojection error is over the
inliers, in pixels. The frozen baseline has no maximum inlier reprojection error for the
reference.

| pair | estimate | rotation error | translation error | reprojection mean / median / max | inliers |
|---|---|---|---|---|---|
| 0→1 | scratch linear | 2.766° | 0.0232 m | 2.07 / 1.79 / 7.97 | 206 |
| | scratch refined | **1.303°** | **0.0095 m** | 1.96 / 1.68 / 7.51 | 206 |
| | reference | 0.978° | 0.0073 m | 2.01 / 1.75 / — | 121 |
| 1→2 | scratch linear | 0.806° | 0.0059 m | 2.00 / 1.48 / 7.82 | 197 |
| | scratch refined | 1.686° | 0.0148 m | 1.92 / 1.45 / 7.74 | 197 |
| | reference | 2.055° | 0.0165 m | 1.77 / 1.59 / — | 135 |
| 10→11 | scratch linear | 1.767° | 0.0145 m | 2.13 / 1.75 / 7.87 | 208 |
| | scratch refined | **0.616°** | **0.0059 m** | 1.94 / 1.47 / 8.55 | 208 |
| | reference | 0.722° | 0.0063 m | 1.80 / 1.50 / — | 131 |

All 35 pairs:

| | Pyramid, linear | Pyramid + refinement | Reference |
|---|---|---|---|
| per-pair rotation error, mean / median / max | 2.07° / 1.83° / 5.58° | **1.16° / 1.03° / 2.56°** | 1.39° / 1.17° / 3.28° |
| per-pair translation error, mean / median | 0.0181 / 0.0153 m | **0.0107 / 0.0095 m** | 0.0125 / 0.0109 m |
| inlier reprojection error, mean over pairs | 2.28 px | 2.00 px | 1.94 px |
| RANSAC inliers, mean / min | 240.3 / 154 | 240.3 / 154 | 132.0 / 67 |
| summed signed rotation error | +0.3° | **−30.4°** (33/35 pairs negative) | −30.4° |
| trajectory error, mean over 36 frames | **0.076 m / 6.9°** | 0.147 m / 18.2° | 0.180 m / 21.4° |
| final error (frame 35) | 0.106 m / 13.5° | 0.284 m / 31.9° | 0.277 m / 32.1° |

**Conclusions.**
- **Refinement works.** Per pair, the mean rotation error halves (2.07° → 1.16°) and is now
  below the reference's (1.39°).
- **It lowers the reprojection error.** The RMS over the inliers (the minimised quantity)
  drops on 35/35 pairs and the mean on 34/35; it also drops in 40/40 synthetic trials
  (`test_refine.txt`).
- **The rotation error improves on 25 of 35 pairs.** For example, 1→2 is worse
  (0.81° → 1.69°), while still better than the reference's 2.06°.
- **The trajectory gets worse because the remaining error is systematic.** The refinement
  removes the random part of each pair's error, and what is left is an under-rotation of
  about 0.87° per pair that accumulates. The reference has the same one (−30.4° in total).
  The linear run's smaller trajectory error is a cancellation of larger, mixed-sign errors.

### Cause of the systematic under-rotation (`contour_bias_experiment.py/.txt`)

The experiment refines each pair's inliers from the true pose, with the same model as
`refine.cpp`, on different subsets:

| subset | summed signed rotation error | negative pairs | mean \|rotation error\| |
|---|---|---|---|
| all inliers (as in the pipeline) | −30.5° | 33/35 | 1.16° |
| interior only (no background or depth jump > 1 % within 3 px) | **+3.8°** | 16/35 | 1.23° |
| all, pixel coordinates + 0.5 (the renderer's sample position) | −30.3° | 33/35 | 1.15° |

- **The bias comes from silhouette features.** About 73 % of the inliers lie on the
  bunny's outline. A corner on the occluding contour is not a fixed 3D point: as the camera
  orbits it slides along the surface, so its image motion is smaller than the scene's.
- **The geometry amplifies it.** The camera orbits while looking at the bunny, so rotation
  and translation nearly cancel in the image (checkpoint 02: 4.3 px mean motion). A small
  systematic pixel bias therefore becomes a large rotation bias.
- **The half-pixel convention is not the cause.** The renderer samples pixel (x, y) at
  (x + 0.5, y + 0.5) while the pipeline treats x as the pixel centre; this constant offset
  changes nothing.
- **Not fixed here,** because the instruction was not to change feature detection or
  matching. The fix would be a front-end rule: reject, or down-weight, 3D points with a
  depth discontinuity in their neighbourhood.

## 4. Pyramid information and the depth mapping (`test_multiscale_0_1.txt`)

- **Every keypoint keeps its pyramid data:** level, level x/y, original x/y, x scale, FAST
  score, angle and descriptor. In memory this is `FrameFeatures`; on disk,
  `keypoints/frame_*.csv` now includes the 256-bit descriptor in hex.
- **The mapping is checked explicitly, per axis.** Every correspondence uses
  (level + 0.5)·s − 0.5 with s_x = 640 / level width and s_y = 480 / level height (these
  differ by up to 0.2 %, e.g. level 1: 1.20075 vs 1.2).
- **Depth lookup:** depth is read from the 640×480 depth image at the truncated mapped
  pixel, and the 3D point back-projects that pixel.
- **Ground-truth check:** 87 coarse-level (≥ 3) matches with depth reproject under the true
  motion with a median error of 1.94 px.
- **Control:** used directly as pixels, their level coordinates would have no depth for
  87/87.

## 5. Figures

- **Keypoints:** small fixed dots coloured by level, with a 6 px orientation tick; no scale
  circles.
- **Correspondence maps:** unchanged; no orientation ticks, no circles.
- **New and updated figures:**
  - `pyramid_levels_0`: per-level counts.
  - `feature_comparison_0_1`: names the detector differences (Harris ranking, quotas).
  - `pnp_comparison`: shows single scale, pyramid linear and pyramid + refinement against
    the reference PnP and ICP.

## 6. Blender camera

**Chain (unchanged, verified):**

    T_rel = T_{i+1←i}
      → T_wc[i+1] = T_wc[i]·T_rel⁻¹
      → trajectory.csv (C_w = t_wc, R_wc)
      → trajectory_to_blender_pose (rotation R_wc·R_FIX)
      → WorldRoot
      → ScratchPnP_Animated_Camera

**Numerical checks** (`trajectory_check.txt`):
- frame 0 equals GT
- quaternion equals the matrix
- T_wc[i+1]⁻¹·T_wc[i] equals the exported T_rel for all 35 pairs (1.5e-9)
- C = −R_cwᵀ·t_cw (9e-10 m)
- frame by frame against GT: the look-at error is at most 3.81°

**Blender checks** (`blender_consistency_check.txt`, `blender_orbit_check.txt`):
- the camera equals the CSV: position within 3.3e-8 m, rotation within 0°
- the HUD values equal the CSV
- look-at in Blender equals look-at from the data, within 8e-4°

| frame | azimuth GT / scratch | step GT / scratch | camera forward · direction to bunny | camera up · world up (GT / scratch) | distance to GT |
|---|---|---|---|---|---|
| 0 | 0.00° / 0.00° | — | 1.00000 | 0.9701 / 0.9701 | 0.0000 m |
| 1 | 10.00° / 8.98° | 10.00° / 8.98° | 1.00000 | 0.9701 / 0.9721 | 0.0103 m |
| 2 | 20.00° / 17.38° | 10.00° / 8.41° | 1.00000 | 0.9701 / 0.9750 | 0.0254 m |
| 10 | 100.00° / 82.94° | 10.00° / 8.78° | 0.99999 | 0.9701 / 0.9760 | 0.1505 m |
| 11 | 110.00° / 92.34° | 10.00° / 9.40° | 0.99999 | 0.9701 / 0.9755 | 0.1553 m |
| 35 | 350.00° / 319.43° | 10.00° / 8.34° | 0.99917 | 0.9701 / 0.9880 | 0.2835 m |

The orbit runs the same way as GT on 35/35 steps, at 7.6°–11.0° per step.

**Root cause and fix.**
- **Original defect:** the scene loaded the milestone-2 trajectory (linear PnP, no RANSAC),
  whose camera looked up to 141.7° away from the bunny. It was fixed in checkpoint 02 by
  reading `pnp_from_scratch/results/pipeline/trajectory.csv`.
- **This checkpoint:** no Blender code changed. The default run (pyramid + refinement)
  writes that file, and the audit above shows the camera reproduces it exactly, with correct
  position, orientation, forward direction, orbit direction and frame-to-frame motion.
- **What remains is not a visualization error:** the camera's offset from GT (0.28 m at
  frame 35) is the estimate's accumulated under-rotation (§3).

## 7. Regression and audit

- **Scratch:** ctest 15/15 (`scratch_ctest.txt`). New: `test_refine`. Extended:
  `test_multiscale_0_1` and `test_pipeline_0_1`.
- **Reference:**
  - `run_test_suite.sh`: 25/25 programs exit 0, and the dataset regenerates
    byte-identically (74 files).
  - `bunny_metrics.py`: `slam_trajectory.csv`, `pair_metrics.csv`, `metrics.json` and all
    correspondence files are identical to `docs/migration/01_features/`.
  - So the reference behaviour is unchanged.
  - `orb_from_scratch_test` passed in this run. Its intermittent border crash (checkpoint 03)
    is unchanged and still documented there; it was not modified.
- **OpenCV** (`opencv_audit.txt`), across `pnp_from_scratch/`, the scratch feature modules
  and their tests, and `cmake/feature_core.cmake`:
  - no OpenCV text (the only hit is a README sentence saying `cv2` is not imported)
  - no OpenCV includes
  - no `find_package(OpenCV)` and no OpenCV link targets
  - `ldd` of all 15 scratch binaries: none linked
  - the Python tools: `cv2` not loaded at runtime
  - the Blender scene builder imports no OpenCV
