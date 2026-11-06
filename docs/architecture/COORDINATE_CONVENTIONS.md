# Coordinate conventions

This page states the geometric conventions of the production pipeline (`scratch_pipeline`) and of the
components around it (reference ICP variants, the experimental RGB-D ICP, the evaluation tools, the
Blender bridge). Each statement cites the line that implements it. Paths are relative to the repository
root. Section 12 lists every place where two components differ.

## 1. Symbols

| Symbol | Meaning | Where defined |
|---|---|---|
| (u, v) | pixel coordinates: u = column (x, right), v = row (y, down), origin at the top-left pixel | `include/slam/features/types.hpp:13-23, 35` |
| K = (fx, fy, cx, cy) | pinhole intrinsics, no skew, no distortion | `include/slam/pipeline/projection.hpp:5-8` |
| Z | depth in metres = raw 16-bit value / 5000 | `include/slam/pipeline/frontend.hpp:41`, `src/pipeline/frontend.cpp:139` |
| X_i | 3D point in the camera frame of image i (metres) | `include/slam/pipeline/frontend.hpp:90-91` |
| `Pose {R, t}` | rigid transform with X_b = R X_a + t | `include/slam/pipeline/common.hpp:27-30` |
| T_{j<-i} = (R, t) | relative pose of a pair: X_j = R X_i + t | `include/slam/pipeline/ransac.hpp:43`, `include/slam/pipeline/pipeline.hpp:18-19, 79` |
| T_wc | camera-to-world pose: X_w = R_wc X_c + t_wc; t_wc = camera centre | `include/slam/pipeline/common.hpp:110` (reader), `apps/pipeline_main.cpp:451, 473` (writer) |
| T_cw | world-to-camera, T_wc^-1 | `reference/opencv_slambook2/src/render/rasterizer.cpp` (render), `results/reports/diagnostics/transform_convention.md` |
| G_k | ground-truth relative pose T_{k+1<-k} = T_gt[k+1]^-1 T_gt[k] | `apps/pipeline_main.cpp:346` |
| q = (qx, qy, qz, qw) | unit Hamilton quaternion of R_wc, stored x, y, z, w | `include/slam/pipeline/common.hpp:32-71` |
| E, F | essential (normalised coords) and fundamental (pixels) matrices, x2^T E x1 = 0, p2^T F p1 = 0 | `include/slam/pipeline/essential.hpp:7-13, 72-76`, `include/slam/geometry/fundamental.hpp:7` |

## 2. Camera frame axes

The production code never names the axes explicitly in the camera model, but they follow from the code:

- Projection u = fx X/Z + cx, v = fy Y/Z + cy with fx, fy > 0 (`src/pipeline/projection.cpp:5-8`), and
  image x right / y down (`include/slam/features/orientation.hpp:6`, `src/features/brief.cpp:83`). So camera
  +X points right and +Y points down in the image.
- A point is in front of the camera iff Zc > 0: RANSAC residuals are +inf for Zc <= 0
  (`src/pipeline/ransac.cpp:53`), refinement fails if a point has Zc <= 0 (`src/pipeline/refine.cpp:31-32`).
  So +Z points forward (optical axis).

Result: **x right, y down, z forward** (the OpenCV convention). The data sources use the same axes:
the renderer builds the camera with columns (right, -up, forward) (`reference/opencv_slambook2/src/render/trajectory.cpp:38-45`);
the TUM adapter states that TUM ground truth uses x right, y down, z forward with no axis change
(`reference/opencv_slambook2/include/dataset/tum.hpp:10-12`).

## 3. Pixel coordinates

- Origin (0, 0) at the top-left pixel; `GrayImage::at(x, y)` indexes `pixels[y * width + x]`
  (`include/slam/features/types.hpp:20-21`). The same row-major layout is used for the depth image
  (`src/pipeline/frontend.cpp:115-118`).
- **Pixel centres are at integer coordinates** in the production pipeline:
  - FAST keypoints are the integer pixel indices (`src/features/fast.cpp:92-93`);
  - the pyramid is centre-aligned with integer centres: level coordinate x maps to
    x0 = (x + 0.5) s_x - 0.5 (`include/slam/features/pyramid.hpp:13-16, 40-41`,
    `src/features/pyramid.cpp:16, 20`);
  - back-projection and projection use the file's cx, cy unchanged (`src/pipeline/frontend.cpp:140-141`,
    `src/pipeline/projection.cpp:7-8`).
- Keypoints from coarse pyramid levels have fractional level-0 coordinates. Image lookups at a keypoint
  **truncate** the coordinate: depth (`src/pipeline/frontend.cpp:115-118, 121`), orientation
  (`src/features/orientation.cpp:24`), BRIEF centre (`src/features/brief.cpp:79`), Harris
  (`src/features/multiscale.cpp:77-79`; level coordinates are integers there). Truncation equals floor for
  the non-negative coordinates that occur, so a fractional coordinate looks up the pixel whose centre is up
  to 1 px to the upper-left.
- The geometric stages (back-projection, PnP, refinement) use the fractional coordinates
  (`src/pipeline/frontend.cpp:112-113, 140-141`).

## 4. Intrinsic matrix

```
    | fx  0  cx |
K = |  0 fy  cy |        x_norm = ((u - cx)/fx, (v - cy)/fy)
    |  0  0   1 |
```

Stored as four doubles (`include/slam/pipeline/projection.hpp:6-8`); never as a 3x3 matrix in production.
K^-1 is applied component-wise (`src/pipeline/pnp.cpp:21`, `src/pipeline/pipeline.cpp:78-81`).
Source: Bunny `data/synthetic_bunny/intrinsics.txt` read as `width height fx fy cx cy`
(`apps/pipeline_main.cpp:201-203`); TUM `--intrinsics fx,fy,cx,cy` (`apps/pipeline_main.cpp:210`).
No distortion model exists (`include/slam/pipeline/projection.hpp:5`).

## 5. Depth and back-projection

- Depth PNG: 16-bit, 1 channel (`src/pipeline/frontend.cpp:28-32`), big-endian samples
  (`src/pipeline/png.cpp:323`). Metres = raw / 5000 (`include/slam/pipeline/frontend.hpp:41`); the same
  factor for Bunny and TUM, with no CLI option. The renderer stores camera-frame Z (not ray length):
  it interpolates 1/z of camera-frame vertices (`reference/opencv_slambook2/src/render/rasterizer.cpp:67, 83-86`)
  and writes round(z * scale) (`reference/opencv_slambook2/src/render/rasterizer.cpp:100-112`).
- Invalid depth: raw value 0 means "no measurement" (`data/synthetic_bunny/README.md`;
  `reference/opencv_slambook2/include/dataset/tum.hpp:13`). A correspondence gets a 3D point only if the
  raw depth at the truncated frame-i pixel is > 0 (`src/pipeline/frontend.cpp:119`). In T0, additionally all
  3x3 neighbours must be > 0, the pixel must not be on the 1-px border, and max - min of the 3x3 depths
  must be < 0.012 m (`src/pipeline/frontend.cpp:120-137`). Frame-j depth is recorded but never used.
- Back-projection (`src/pipeline/frontend.cpp:138-142`):

  X_i = ( (u - cx) Z / fx , (v - cy) Z / fy , Z ),  Z = raw_i(int(u), int(v)) / 5000.

## 6. Meaning of the PnP and refinement result

- `solvePnPDLT` returns (R, t) with X_camera = R X + t, X in the object frame (`include/slam/pipeline/pnp.hpp:23-28`).
- In the pipeline the object frame is camera i and the image is frame j (`src/pipeline/pipeline.cpp:102-110`),
  so **(R, t) = T_{j<-i}: X_j = R X_i + t** (`include/slam/pipeline/ransac.hpp:43`, `include/slam/pipeline/pipeline.hpp:79`).
- `RefinePnP` keeps the same meaning; it perturbs on the left, R <- Exp(phi) R, t <- Exp(phi) t + rho
  (`include/slam/pipeline/refine.hpp:9-13`, `src/pipeline/refine.cpp:96-98`).
- The diagnostic essential-matrix pose uses the same direction, X_2 = R X_1 + t with |t| = 1
  (`include/slam/pipeline/essential.hpp:7-13`).
- Rotations returned by PnP are proper (det +1) by construction (`src/pipeline/pnp.cpp:82-87`) and
  re-orthonormalised after refinement (`src/pipeline/refine.cpp:118-120`).

## 7. Trajectory composition, world frame and initial pose

`AccumulateTrajectory` (`include/slam/pipeline/accumulate.hpp:39-55`) with
`Inverse(T) = (R^T, -R^T t)` and `Compose(A, B) = (A.R B.R, A.R B.t + A.t)`
(`include/slam/pipeline/accumulate.hpp:15-26`):

```
T_wc[0]   = T_gt[0]                                          (accumulate.hpp:44)
T_wc[k+1] = Compose(T_wc[k], Inverse(T_{k+1<-k}))            (accumulate.hpp:48)   accepted pair
T_wc[k+1] = T_wc[k]                                          (accumulate.hpp:50)   failed / rejected pair
```

Derivation: a point in camera k+1 is X_{k+1}; X_k = T_{k<-k+1} X_{k+1} = T_{k+1<-k}^-1 X_{k+1};
X_w = T_wc[k] X_k. Hence T_wc[k+1] = T_wc[k] T_{k+1<-k}^-1, i.e.

R_wc[k+1] = R_wc[k] R^T,   t_wc[k+1] = t_wc[k] - R_wc[k] R^T t.

- **World frame:** the ground-truth world frame. For the Bunny it is the render world, Y up
  (`reference/opencv_slambook2/include/render/trajectory.hpp:11`); for TUM it is the motion-capture frame.
- **Initial pose:** ground-truth anchored, not identity (`apps/pipeline_main.cpp:335`; written as
  `pose_source = gt_anchor`, `include/slam/pipeline/accumulate.hpp:45`).
- Ground truth relative poses for evaluation are built the same way: G = T_gt[k+1]^-1 T_gt[k] = T_{k+1<-k}
  (`apps/pipeline_main.cpp:346`).

## 8. Trajectory file formats and quaternions

**`trajectory.txt`** (`apps/pipeline_main.cpp:470-487`): two `#` lines, then
`timestamp tx ty tz qx qy qz qw`, T_wc, 6 decimals. The "timestamp" is the frame index k
(`apps/pipeline_main.cpp:485`), also for TUM runs.

**`trajectory.csv`** (`apps/pipeline_main.cpp:441-484`): five `#` lines (configuration, convention, notes,
accumulation rule), then

`frame, tx, ty, tz, r00..r22, qx, qy, qz, qw, rotation_error_deg, translation_error_m, pnp_inliers, pose_source`

with (tx, ty, tz) = t_wc (camera centre, metres), r_ab = R_wc(a, b) row-major, quaternion of R_wc,
absolute errors against ground truth, the inlier count of the pair that produced the frame (empty for
frame 0), and `gt_anchor` / `scratch_pipeline` / `held_previous_pose_pair_failed` /
`held_previous_pose_pair_rejected`.

**`pairs.csv`** stores T_{j<-i}: `tx,ty,tz` = t and `r00..r22` = R row-major (`apps/pipeline_main.cpp:393-397`).
For a failed pair these are I and 0; for a gate-rejected pair they are the rejected pose.

**Input ground truth:** Bunny `groundtruth.txt`, `timestamp tx ty tz qx qy qz qw`, T_wc
(`include/slam/pipeline/common.hpp:110-125`); TUM `frames.csv` columns `gt_x..gt_z, gt_qx..gt_qw`, T_wc
(`apps/pipeline_main.cpp:225-226`).

**Quaternion order and convention.**
- Files and the production helpers use **(x, y, z, w)** order (`include/slam/pipeline/common.hpp:32, 51, 68`).
- `QuatToR` is the Hamilton convention (R = I + 2w[v]x + 2[v]x^2 written out; e.g. R01 = 2(xy - zw))
  (`include/slam/pipeline/common.hpp:33-41`). This is the same matrix Eigen's `Quaterniond(w, x, y, z)`
  produces, which the reference TUM reader uses (`reference/opencv_slambook2/src/dataset/tum.cpp:92-94`).
- `RToQuat` (Shepperd) returns a unit quaternion with **w >= 0** (`include/slam/pipeline/common.hpp:52-71`).
- Constructors that take **(w, x, y, z)**: Eigen `Quaterniond` (`reference/opencv_slambook2/src/dataset/tum.cpp:60, 93`,
  `experiments/rgbd_icp/apps/icp_sequence.cpp:57`) and Blender `mathutils.Quaternion`
  (`visualization/scripts/slam_coords.py:46-47`, `visualization/scripts/build_scene.py:272`). Each of these
  call sites reorders correctly from the x, y, z, w file order.

## 9. Error definitions used in evaluation

Production (`scratch_pipeline`):

| Quantity | Formula | Code |
|---|---|---|
| rotation error between A and B | angle(A^T B) = acos(clamp((tr(A^T B) - 1)/2)) in degrees | `include/slam/pipeline/common.hpp:45-48` |
| per-pair rotation error | angle(G_k^T R_est) | `apps/pipeline_main.cpp:348, 352` |
| per-pair translation error | ||t(G_k) - t_est|| in metres, i.e. difference of the translations of T_{k+1<-k} (expressed in camera k+1) | `apps/pipeline_main.cpp:349, 353` |
| absolute position error at frame k | ||t_wc,gt[k] - t_wc,est[k]|| (camera centres, world frame) | `apps/pipeline_main.cpp:364` |
| absolute rotation error at frame k | angle(R_wc,gt[k]^T R_wc,est[k]) | `apps/pipeline_main.cpp:365` |
| essential translation-direction error | angle between t(G_k) and the unit t_E | `apps/pipeline_main.cpp:70-75, 359` |

Per-pair errors are filled only for accepted pairs (rotation/translation) or solved pairs ("returned"
errors in `pose_validity.csv`) (`apps/pipeline_main.cpp:347-356`).

**ATE, raw versus aligned.**
- `scratch_pipeline` reports per-frame raw errors and the final-frame error; it computes no ATE and no
  alignment (`apps/pipeline_main.cpp:362-366, 621-623`).
- `scratch::baseline::RawAte` (C++ regression harness): raw ATE RMSE of `trajectory.csv` positions, no
  alignment (`include/slam/evaluation/baseline.hpp:201-210`, `src/evaluation/baseline.cpp:579-606`).
- `tools/evaluation/tum_metrics.py` (for `tum_trajectory_eval` outputs): `ate_raw_rmse_m` (no alignment) and
  `ate_aligned_rmse_m` after one global rigid SE(3) Umeyama/Kabsch alignment **without scale**
  (`tools/evaluation/tum_metrics.py:21-23, 181-187, 226-234`); RPE per Sturm et al.,
  E_k = (Q_k^-1 Q_{k+d})^-1 (P_k^-1 P_{k+d}) (`tools/evaluation/tum_metrics.py:25-28, 190-210`).
  Rotation angle uses the same acos-trace formula (`tools/evaluation/tum_metrics.py:89-90`).
- All of them keep frame 0 anchored to ground truth, so the raw ATE includes frame 0 with error 0.

## 10. ICP variants

All three variants work on 3D-3D pairs in metres, in the same x right / y down / z forward camera frames.
Their native output direction is the **opposite** of the production PnP.

**(a) Book ICP, `reference/icp_main_variant/src/icp.cpp` (main's version, built as `tracking_main_icp`).**
- `pose_estimation_3d3d(pts1, pts2)`: centroids, W = sum q1 q2^T, R = U V^T, t = p1 - R p2
  (`reference/icp_main_variant/src/icp.cpp:19-63`). So **pts1 = R pts2 + t**; with pts1 = frame i and
  pts2 = frame j this is T_{i<-j}.
- Reflection handling: if det(U V^T) < 0 the code negates the whole matrix, R = -U V^T
  (`reference/icp_main_variant/src/icp.cpp:51-54`), which is not the Kabsch correction
  U diag(1, 1, -1) V^T.
- `bundleAdjustment` (g2o, 10 LM iterations) minimises ||pts1 - T pts2||, i.e. the same direction, and
  starts from the identity, not from the SVD estimate (`reference/icp_main_variant/src/icp.cpp:72-75, 105-118`).
- `FitRigid` / `EstimateRigidRansac`: same direction, residual ||p1 - (R p2 + t)||, proper Kabsch
  reflection fix (`reference/icp_main_variant/src/icp.cpp:144-212`).
- The reference trajectory test converts to the common form, R' = R^T, t' = -R^T t, and accumulates like
  production: T_wc[i+1] = T_wc[i] T_{i+1<-i}^-1, frame 0 = ground truth
  (`reference/opencv_slambook2/tests/synthetic/slam_trajectory_test.cpp:17-30, 234-236, 413`).

**(b) Teammate ICP, `reference/opencv_slambook2/src/tracking/icp.cpp`.**
- `pose_estimation_3d3d(pts1, pts2)` calls `rgbd_icp::rigidSVD(source = pts2, target = pts1)`
  (`reference/opencv_slambook2/src/tracking/icp.cpp:5-24`), so again **pts1 = R pts2 + t** (T_{i<-j}).
- `bundleAdjustment` is the same SVD solve; no g2o optimisation runs
  (`reference/opencv_slambook2/src/tracking/icp.cpp:26-33`). The function name is shared with (a) but the
  behaviour differs.

**(c) Experimental dense RGB-D ICP, `experiments/rgbd_icp`.**
- `rigidSVD(source, target)` returns (R, t) with **target = R source + t**, Kabsch with
  D = diag(1, 1, det(V U^T)) (`experiments/rgbd_icp/src/rgbd_icp.cpp:44-65`); `Pose::apply` is R p + t
  (`experiments/rgbd_icp/include/tracking/rgbd_icp.hpp:12-18`).
- In `registerFramesWithFeatures(a = frame i, b = frame j)`: target = frame-i points, source = frame-j points
  (`experiments/rgbd_icp/src/rgbd_icp.cpp:288-306, 538-543`). All stage poses (baseline, robust, point,
  plane) are therefore **T_{i<-j}**.
- `icp_sequence` accumulates without an inverse: T[i+1] = T[i] * step (`experiments/rgbd_icp/apps/icp_sequence.cpp:133-136`),
  which is correct for T_{i<-j}. It starts from the identity (`experiments/rgbd_icp/apps/icp_sequence.cpp:91-92`)
  and, if ground truth exists, writes gt[0] * T[i] (`experiments/rgbd_icp/apps/icp_sequence.cpp:189`); this is
  equivalent to the production anchor. A failed pair reuses the last accepted step (constant velocity)
  (`experiments/rgbd_icp/apps/icp_sequence.cpp:133-135`).
- Back-projection uses the same formula and depth scale 5000, but the depth pixel is the **rounded**
  keypoint (`std::nearbyint`) and the stable-depth window requires a 2-px border
  (`experiments/rgbd_icp/src/rgbd_icp.cpp:234-251, 300-305`; config `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp:24-25`).
- Errors: rotation angle via log(R) norm (`experiments/rgbd_icp/src/rgbd_icp.cpp:43`), RPE
  (gt[i-1]^-1 gt[i])^-1 (est[i-1]^-1 est[i]), raw ATE RMSE with no alignment
  (`experiments/rgbd_icp/apps/icp_sequence.cpp:189-210`).
- Output `*_trajectory.txt`: `frame tx ty tz qx qy qz qw` (frame index, T_wc, Eigen quaternion without sign
  normalisation) (`experiments/rgbd_icp/apps/icp_sequence.cpp:183-193`).

## 11. Blender bridge (`visualization/scripts/slam_coords.py`)

- Input poses are T_wc as stored; nothing is inverted or re-composed (`visualization/scripts/slam_coords.py:7-15`).
- Camera axes: a local 180 degree rotation about camera X, R_FIX = diag(1, -1, -1), converts CV axes
  (x right, y down, z forward) to Blender camera axes (x right, y up, looking down -Z); the camera object gets
  R_blender = R_wc R_FIX and position t_wc (`visualization/scripts/slam_coords.py:22-27, 51-68`).
- World axes: the C++ world is Y up; the whole rig is parented under an empty rotated +90 degrees about X,
  mapping (x, y, z) -> (x, -z, y) so that the world is Z up in Blender (`visualization/scripts/slam_coords.py:29-37`).
- Quaternions are read from the (x, y, z, w) columns into `mathutils.Quaternion((w, x, y, z))`
  (`visualization/scripts/slam_coords.py:46-47`).
- The production (T0) trajectory is read from the frozen `tests/fixtures/bunny_t0_scene/trajectory.csv` and checked:
  frames 0..k-1, quaternion equals the matrix columns, frame 0 equals ground truth, stored errors equal
  recomputed ones (rotation error from the quaternion difference, folded to [0, 180] degrees)
  (`visualization/scripts/build_scene.py:93, 225-288`).

## 12. Places where conventions differ between components

| # | Topic | Component A | Component B | Consequence |
|---|---|---|---|---|
| 1 | Pixel centre | production: integer centres (`include/slam/features/pyramid.hpp:13-16`, cx used as is, `src/pipeline/frontend.cpp:140-141`) | Bunny renderer samples pixel (x, y) at (x + 0.5, y + 0.5) (`reference/opencv_slambook2/src/render/rasterizer.cpp:77`) | exact intrinsics for the Bunny would be cx - 0.5, cy - 0.5; reported as a negligible effect in `results/reports/diagnostics/transform_convention.md` |
| 2 | Depth pixel at a fractional keypoint | production truncates (`src/pipeline/frontend.cpp:115-118, 121`) | `rgbd_icp` rounds (`experiments/rgbd_icp/src/rgbd_icp.cpp:300-303`) | different depth sample for coarse-level keypoints |
| 3 | Depth-edge border | 1 px (`src/pipeline/frontend.cpp:123`) | 2 px (`experiments/rgbd_icp/src/rgbd_icp.cpp:238`) | minor |
| 4 | Relative-pose direction | production PnP / essential: T_{j<-i} (`include/slam/pipeline/ransac.hpp:43`) | all ICP variants: T_{i<-j} (`reference/icp_main_variant/src/icp.cpp:55`, `reference/opencv_slambook2/src/tracking/icp.cpp:5`, `experiments/rgbd_icp/src/rgbd_icp.cpp:543`) | accumulation must invert one of them; production does `T * T^-1` (`include/slam/pipeline/accumulate.hpp:48`), `icp_sequence` does `T * step` (`experiments/rgbd_icp/apps/icp_sequence.cpp:136`); both are consistent with their own direction |
| 5 | Failed pair | production holds the previous pose (`include/slam/pipeline/accumulate.hpp:50`) | `icp_sequence` repeats the last accepted step (`experiments/rgbd_icp/apps/icp_sequence.cpp:133`) | trajectories after a failure are not comparable |
| 6 | Initial pose | production: T_gt[0] at accumulation (`apps/pipeline_main.cpp:335`) | `icp_sequence`: identity, left-multiplied by gt[0] only when written, identity world without ground truth (`experiments/rgbd_icp/apps/icp_sequence.cpp:92, 189`) | equivalent when ground truth exists |
| 7 | SVD reflection fix | book `pose_estimation_3d3d`: R = -U V^T (`reference/icp_main_variant/src/icp.cpp:51-54`) | `FitRigid`, `rigidSVD`: D = diag(1, 1, -1) (`reference/icp_main_variant/src/icp.cpp:161-163`, `experiments/rgbd_icp/src/rgbd_icp.cpp:60-63`) | the book variant gives a different (non-optimal) rotation in the reflection case |
| 8 | `bundleAdjustment` | main variant: g2o from identity (`reference/icp_main_variant/src/icp.cpp:107`) | teammate variant: SVD only (`reference/opencv_slambook2/src/tracking/icp.cpp:29-33`) | same name, different behaviour |
| 9 | Quaternion order | files and production helpers (x, y, z, w) (`include/slam/pipeline/common.hpp:32, 51`) | Eigen and Blender constructors (w, x, y, z) | handled correctly at each call site checked (Section 8) |
| 10 | Quaternion sign | production forces w >= 0 (`include/slam/pipeline/common.hpp:69`) | `icp_sequence` writes Eigen's quaternion unchanged (`experiments/rgbd_icp/apps/icp_sequence.cpp:191-193`) | same rotation, possibly opposite sign in files |
| 11 | Timestamp column | `trajectory.txt` writes the frame index (`apps/pipeline_main.cpp:485`), also in TUM mode | TUM tools expect real timestamps equal to `t_rgb` (`tools/evaluation/tum_metrics.py:117`) | `trajectory.txt` of a TUM run is not directly usable by TUM timestamp-based tools; `frames.csv` (copied to the output) holds the timestamps |
| 12 | Lens distortion | production: none (`include/slam/pipeline/projection.hpp:5`) | TUM freiburg1 has non-zero distortion (`reference/opencv_slambook2/src/dataset/tum.cpp:127`) | ignored on TUM freiburg1 data |
| 13 | Per-pair translation error | production: ||t(T_{j<-i},gt) - t(T_{j<-i},est)|| (`apps/pipeline_main.cpp:349`) | RPE: ||t((Q^-1 Q')^-1 (P^-1 P'))|| (`tools/evaluation/tum_metrics.py:200-206`, `experiments/rgbd_icp/apps/icp_sequence.cpp:197-200`) | different quantities; numbers are not interchangeable |
| 14 | Rotation-error formula | acos of trace (`include/slam/pipeline/common.hpp:45-48`) | log-map norm (`experiments/rgbd_icp/src/rgbd_icp.cpp:43`); quaternion difference in Blender (`visualization/scripts/build_scene.py:276-277`) | same geodesic angle, numerically different near 0 and 180 degrees |
| 15 | Name "R_cw" | renderer names the camera-to-world rotation `R_cw` (`reference/opencv_slambook2/src/render/trajectory.cpp:38-45`) | everywhere else T_cw is world-to-camera (`results/reports/diagnostics/transform_convention.md`, production `T_wc`) | naming only; the stored pose is T_wc |
| 16 | Bunny RGB channel order | `data/synthetic_bunny/README.md` says "BGR as written by OpenCV" | the PNG files are standard RGB (colour type 2) and production reads them as R, G, B (`src/pipeline/frontend.cpp:47-49`); `rgbd_icp` reads through `cv::imread` (BGR) and uses the BGR converter (`experiments/rgbd_icp/src/rgbd_icp.cpp:504-505, 513-514`) | both paths compute the same grey values; only the README wording is misleading |
