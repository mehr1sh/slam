# PnP from scratch: linear DLT (milestone 1: one pair; milestone 2: all 35 pairs)

A self-contained Perspective-n-Point solver: projection, the linear system, its
SVD solution and the recovery of a valid rotation are all written here.
**OpenCV is not linked** (`ldd pnp_from_scratch | grep opencv` is empty); Eigen
provides only matrices, vectors and SVD/QR. The solver runs on the **existing**
frame 0 → 1 correspondences of the synthetic Bunny experiment and is checked
against ground truth and against the repository's OpenCV `solvePnPRansac` result.

The existing pipeline in `../src`, `../tests` is untouched.

## Build and run

```bash
cd pnp_from_scratch
mkdir -p build && cd build
cmake .. && make
./pnp_from_scratch            # finds the repository root automatically
```

The program writes `results/frame_0_1_pose.txt` and `results/frame_0_1_reprojection.csv`.
The console output of the run below is saved in `results/frame_0_1_run.txt`.
The exit status is 0 only if every counted sanity check passes.

The input `../results/data/cpp_export/pair_0_1_correspondences.csv` is generated, not
tracked. If it is missing, the program prints the command that creates it
(`pixi run -e results results`, which runs `slam_trajectory_test` with export-only flags).

## 1. What PnP solves

Given n 3D points whose coordinates are known in some frame (the "object"
frame), and the pixels where a camera sees them, find the camera pose (R, t):

    X_c = R · X + t

- **X**: 3D point in the object frame
- **X_c**: the same point in the camera frame
- **R**: 3×3 rotation
- **t**: translation

## 2. Inputs in this experiment

All inputs come from the existing pipeline (`RunPnpPair` in
`tests/synthetic/slam_trajectory_test.cpp`), exported to CSV:

- **3D points X:** each ORB keypoint in frame 0 with valid depth Z = raw/5000,
  back-projected with `pixel2cam`: X = ((u−cx)/fx·Z, (v−cy)/fy·Z, Z). The object
  frame is therefore **camera 0**.
- **2D points (u, v):** the matched keypoint in frame 1.
- **K:** `data/synthetic_bunny/intrinsics.txt`, with fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7.
- **Result:** (R, t) = **T_{1←0}**, i.e. x₁ = R·x₀ + t. This is the same convention
  as the repository's PnP.
- **Ground truth:** T_{1←0} = T_wc[1]⁻¹ · T_wc[0] from `groundtruth.txt` (T_wc =
  camera → world). That is a 10.000° rotation and a 0.0872 m translation.

## 3. Projection (`src/projection.cpp`)

    X_c = R X + t
    u = fx · X_c / Z_c + cx
    v = fy · Y_c / Z_c + cy

- **fx, fy:** focal lengths in pixels
- **cx, cy:** principal point
- **Z_c:** the depth, which must be > 0 for a visible point

`projectPoint()` implements exactly this, and is tested against hand-computed values.

## 4. The linear (DLT) equations (`src/pnp.cpp`)

**Remove K.** Write the normalized image coordinates x = (u−cx)/fx, y = (v−cy)/fy.
Then

    λ [x y 1]ᵀ = M [X Y Z 1]ᵀ,   M = [R | t]  (3×4, 12 unknowns m)

where λ = Z_c is an unknown per-point scale. With m₁, m₂, m₃ the rows of M and
P = [X Y Z 1]ᵀ, eliminate λ = m₃·P:

    x (m₃·P) − m₁·P = 0
    y (m₃·P) − m₂·P = 0

Each correspondence gives two rows of A·m = 0, with A of size 2n × 12:

    [ Pᵀ   0ᵀ  −x Pᵀ ]
    [ 0ᵀ   Pᵀ  −y Pᵀ ]

12 unknowns up to scale means 11 degrees of freedom, so n ≥ 6 points are needed.

**Normalization (Hartley).** Before building A:
- the 2D points are shifted to zero mean and scaled to a mean distance of √2
- the 3D points are shifted to zero mean and scaled to a mean distance of √3

This keeps the columns of A at similar magnitudes, so the SVD is well
conditioned. The result is mapped back with M = T₂⁻¹ M̃ T₃.

## 5. Solving with SVD, and why

With noise, A·m = 0 has no exact solution. The least-squares solution of
min ‖A m‖ subject to ‖m‖ = 1 is the right singular vector of A that belongs to
the smallest singular value σ₁₂.

The ratio σ₁₂/σ₁₁ shows how distinct that solution is: close to 0 means well
determined, close to 1 means ambiguous.

## 6. Recovering a valid R and t

m is only defined up to scale and sign, so M = λ·[R | t] with an unknown λ.

1. **Sign.** The depths are λZ_c = m₃·P. Choose the sign of M that puts the
   majority of points in front of the camera (cheirality).
2. **Rotation.** The DLT ignores that R must be a rotation (R has 3 degrees of
   freedom, the 3×3 block has 9). With noise, the block B = M[:, 0:3] is not
   λ times a rotation: its three singular values differ. The closest rotation
   in the Frobenius norm comes from the SVD B = U S Vᵀ:

       R = U · diag(1, 1, det(U Vᵀ)) · Vᵀ,   λ = mean(S)

   This guarantees Rᵀ R = I and det R = +1.
3. **Translation.** t = m₄/λ would mix the uncorrected block with the corrected R.
   Instead, t is re-solved by linear least squares with R fixed. With X_r = R X:

       [ −1  0  x ] t = X_r,x − x · X_r,z
       [  0 −1  y ] t = X_r,y − y · X_r,z

   This is 2n equations in 3 unknowns, solved by QR.

## 7. Reprojection error

For each point, e_k = ‖projectPoint(X_k, R, t) − (u_k, v_k)‖ in pixels. The
program reports the mean, median, max and RMSE = √(mean e_k²).

## 8. Result on the Bunny, frame 0 → 1

| | Scratch DLT, all 136 | **Scratch DLT, 121-point subset** | Existing OpenCV PnP (RANSAC) |
|---|---|---|---|
| rotation error | 18.645° | **1.875°** | 0.978° |
| translation error | 0.1626 m | **0.0163 m** | 0.0073 m |
| reprojection on the 121 points, mean / median | 17.87 / 16.54 px | **2.23 / 2.02 px** | 2.02 / 1.75 px |
| reprojection on all 136, mean / median | 25.67 / 17.80 px | 9.89 / 2.16 px | 9.76 / 1.93 px |
| inliers | — | — | 121 of 136 |

For the 121-point subset:
- max reprojection error 7.12 px, RMSE 2.64 px
- estimated rotation 10.87° (truth 10.00°)
- ‖RᵀR − I‖ = 9e-16, det R = 1
- 121/121 points in front of the camera

**Self-test.** With noise-free projections of the same 3D points, the solver
recovers the ground-truth pose to 2e-6° and 4e-16 m. The 2e-6° is the
precision floor of the acos-based metric.

**How to read this:**

- **All 136 correspondences: fails, and that is expected.** 15 matches are gross
  mismatches, up to 160 px off. An unweighted linear least-squares fit cannot
  ignore them: the smallest singular value is barely distinct (σ₁₂/σ₁₁ = 0.87),
  and the recovered block is far from a scaled rotation (singular values
  5.16 / 1.30 / 0.78). Outlier rejection (RANSAC) is the next milestone.
- **121-point subset: a valid pose.** The subset is the set of points that the
  repository's OpenCV RANSAC kept. Only this **selection** is borrowed; the pose is
  computed entirely by the scratch DLT. It is within 1.9° and 1.6 cm of the truth,
  with a reprojection error close to OpenCV's.
- **Remaining gap to OpenCV.** OpenCV refines its pose by minimizing the
  geometric reprojection error. The DLT minimizes an algebraic error and
  estimates 11 parameters instead of 6, so it is less precise. The points span
  only about 0.13 m of depth at 0.5 m, which also limits the linear estimate.

## 9. Why this is independent of OpenCV PnP

- `CMakeLists.txt` links only `Eigen3::Eigen`, and the binary has no OpenCV library.
- No `cv::` symbol appears anywhere in `src/`.
- Projection, normalization, the DLT matrix, sign selection, rotation
  projection, the translation solve, reprojection and the error metrics are all
  implemented in `src/`.
- Eigen is used only for matrix arithmetic, `JacobiSVD` and `colPivHouseholderQr`.
- The OpenCV result appears only as numbers read from the existing export, for comparison.

## Milestone 2: the same solver on all 35 pairs (raw correspondences)

`pnp_full_sequence` (`src/sequence_main.cpp`) runs the **unchanged** solver
(`src/pnp.cpp`) on every consecutive pair 0→1 … 34→35. It accumulates the 35
results into a 36-pose trajectory. **No RANSAC, no OpenCV inlier mask, no
OpenCV pose.**

**Input.** For each pair, every Hamming-filtered ORB match whose frame-i pixel
has depth, i.e. the pipeline's own PnP input (136 points for 0→1). The
`pnp_inlier` column is never read. Each 3D point is re-checked against
(u_i, v_i, depth) with `intrinsics.txt`. The input files come from the existing
export logic:

```bash
pnp_from_scratch/export_correspondences.sh
```

This runs `slam_trajectory_test --export-dir results/data/raw_correspondences --export-pair all`
on a temporary copy of the dataset and checks that the trajectory outputs stay
byte-identical.

**Accumulation.** This is the project convention:

    T_wc[0] = T_gt[0],   T_wc[i+1] = T_wc[i] · T_{i+1←i}⁻¹

A pair where the solver fails would keep the previous pose and be flagged; none failed.

**Outputs** (`results/`):

| File | Content |
|---|---|
| `scratch_pnp_trajectory.csv` | 36 poses, T_wc: `tx ty tz`, `r00…r22`, `qx…qw`, absolute errors vs GT (`rotation_error_deg`, `translation_error_m` = camera-centre distance), and the stats of the pair into each frame (reprojection mean/median/max, `num_correspondences`, `num_positive_depth`, `success`, `cheirality_ok`, `pose_source`) |
| `scratch_pnp_trajectory.txt` | the same poses in TUM format |
| `scratch_pnp_pair_results.csv` | one row per pair: counts, success, cheirality, estimated R/t, GT motion, relative errors, reprojection stats, DLT conditioning |
| `scratch_pnp_full_sequence_summary.txt`, `scratch_pnp_full_sequence_run.txt` | the summary and the full console output |

**Result.** Pure linear PnP on raw matches:

| | Value |
|---|---|
| pairs attempted / solved / failed | 35 / 35 / 0 |
| per-pair rotation error, mean / median / max | 58.14° / 29.29° / 170.23° (pair 19→20) |
| per-pair translation error, mean / median / max | 0.293 m / 0.252 m / 0.726 m |
| final trajectory error (frame 35) | 0.477 m, 143.8° |
| pair 0→1 (136 points) | 18.645°, 0.1626 m, 25.67 px. This is identical to the milestone-1 "all correspondences" run; the solver code is unchanged |

For comparison, the existing OpenCV PnP with RANSAC ends 0.277 m and 32.1° off.

**Physically invalid poses.** In 5 pairs, the returned pose puts most points
**behind** the camera (`cheirality_ok = 0`):

| Pair | Points in front |
|---|---|
| 5→6 | 56/155 |
| 17→18 | 0/206 |
| 18→19 | 0/193 |
| 19→20 | 0/184 |
| 21→22 | 30/175 |

They are still accumulated exactly as returned (nothing is substituted), and they
are flagged in both CSVs, the summary and the Blender HUD. In these pairs the
gross mismatches dominate the algebraic least-squares fit so strongly that its
3×3 block is far from a scaled rotation, and the sign fix cannot save it.

**Why it is so much worse than the 121-point milestone-1 result.** About 7–10% of
the raw matches are gross mismatches, tens to hundreds of pixels off. The DLT
weights every equation equally, so these points pull the null vector away from
the true pose: σ₁₂/σ₁₁ is typically 0.4–0.9, against 0.098 on the clean set.
This is the expected motivation for the next milestone, **RANSAC from scratch**.

## Blender scene

`visualization/scripts/build_scene.py` reads `results/scratch_pnp_trajectory.csv` and adds
`ScratchPnP_Animated_Camera` (36 keyframes) and `ScratchPnP_Trajectory` (36 points), in cyan.
The HUD label is "Scratch Linear PnP", with the note "pure linear PnP on raw matches, no
RANSAC". Frames reached through a physically invalid pair are flagged.

Before drawing, the build checks:
- the frame-0 anchor
- the quaternion columns against the matrix columns
- the stored errors against errors recomputed from the poses
- that no frame equals OpenCV PnP

The scene supports partial trajectories: a prefix of frames 0..k−1 is shown, and the
camera is hidden after it.

## Reproduce everything

```bash
pixi run build                                          # repository C++ (for the export)
pnp_from_scratch/export_correspondences.sh              # raw correspondences, all 35 pairs
cd pnp_from_scratch && mkdir -p build && cd build && cmake .. && make
./pnp_from_scratch                                      # milestone 1 (pair 0->1, comparison)
./pnp_full_sequence                                     # milestone 2 (all pairs, trajectory)
cd ../.. && blender --python visualization/scripts/build_scene.py
```

## Limitations

- **No outlier rejection.** This is why milestone 2 drifts badly. Milestone 1's valid pose
  used an inlier set borrowed from OpenCV RANSAC. **Next: RANSAC from scratch.**
- Algebraic least squares only; there is no nonlinear (Gauss-Newton/LM) refinement.
- The input correspondences (ORB, matching, depth lookup) are reused from the existing pipeline.
