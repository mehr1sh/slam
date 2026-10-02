# PnP from scratch: a self-contained visual-odometry pipeline

Everything in this directory is written in the project. It depends on
**Eigen** (matrices, SVD, QR), the **C++ standard library** and the project's
own feature modules (`../include/features`, `../src/features`, library target
`feature_core`). No computer-vision, image or compression library is used:
`ldd` of every binary lists only the C++ runtime.

| Milestone | Program | What it does |
|---|---|---|
| 1 | `pnp_from_scratch` | linear (DLT) PnP on the pair-0→1 correspondences of the frozen baseline export, checked against ground truth (sections 1–9) |
| 2 | `pnp_full_sequence` | the same solver on all 35 pairs of the baseline export, without outlier rejection |
| **3** | **`scratch_pipeline`** | **the complete pipeline from the RGB-D images:** PNG decoding → grayscale → FAST → orientation → rotated BRIEF → Hamming matching → correspondences → essential matrix (8-point, decomposition, cheirality) → RANSAC around the linear PnP → 36-pose trajectory |

The repository's normal pipeline (`../src`, `../tests`) is the **reference**.
It is not modified by anything here.

## Build, test and run

```bash
cd pnp_from_scratch
cmake -S . -B build && cmake --build build -j
ctest --test-dir build        # 12 self-checking tests (see "Tests")
./build/scratch_pipeline      # milestone 3: images -> trajectory (about 1.5 s)
./build/pnp_from_scratch      # milestone 1
./build/pnp_full_sequence     # milestone 2
```

Every program finds the repository root automatically. All outputs go to
`results/`, which is generated and gitignored.

The program writes `results/frame_0_1_pose.txt` and `results/frame_0_1_reprojection.csv`.
The console output of the run below is saved in `results/frame_0_1_run.txt`.
The exit status is 0 only if every counted sanity check passes.

The default input is the **frozen baseline export**:
`../docs/migration/baseline/correspondences/pair_0_1_correspondences.csv` and
`../docs/migration/baseline/pair_metrics.csv`. They are tracked, so the numbers below
reproduce without running the reference pipeline. `--corr FILE` selects another
export, for example one made with `export_correspondences.sh`.

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

| | Scratch DLT, all 136 | **Scratch DLT, 121-point subset** | Reference pipeline PnP (RANSAC) |
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
  reference pipeline's RANSAC kept. Only this **selection** is borrowed; the pose is
  computed entirely by the scratch DLT. It is within 1.9° and 1.6 cm of the truth,
  with a reprojection error close to the reference pose's.
- **Remaining gap to the reference.** The reference PnP refines its pose by
  minimizing the geometric reprojection error. The DLT minimizes an algebraic error and
  estimates 11 parameters instead of 6, so it is less precise. The points span
  only about 0.13 m of depth at 0.5 m, which also limits the linear estimate.

## 9. Why this is independent of the reference pipeline's PnP

- `CMakeLists.txt` links only Eigen and project-owned code, and the binaries depend on no
  computer-vision library (`ldd`).
- No third-party vision type or call appears anywhere in `src/`.
- Projection, normalization, the DLT matrix, sign selection, rotation
  projection, the translation solve, reprojection and the error metrics are all
  implemented in `src/`.
- Eigen is used only for matrix arithmetic, `JacobiSVD` and `colPivHouseholderQr`.
- The reference result appears only as numbers read from the export, for comparison.

## Milestone 2: the same solver on all 35 pairs (raw correspondences)

`pnp_full_sequence` (`src/sequence_main.cpp`) runs the **unchanged** solver
(`src/pnp.cpp`) on every consecutive pair 0→1 … 34→35. It accumulates the 35
results into a 36-pose trajectory. **No RANSAC, and no inlier mask or pose from
the reference pipeline.**

**Input.** For each pair, every Hamming-filtered ORB match whose frame-i pixel
has depth, i.e. the pipeline's own PnP input (136 points for 0→1). The
`pnp_inlier` column is never read. Each 3D point is re-checked against
(u_i, v_i, depth) with `intrinsics.txt`.

By default the inputs are the 35 **frozen baseline exports** in
`../docs/migration/baseline/correspondences/` (tracked). `--corr-dir DIR` selects
another set, for example a fresh export of the current reference pipeline:

```bash
pnp_from_scratch/export_correspondences.sh    # -> results/data/raw_correspondences
./build/pnp_full_sequence --corr-dir ../results/data/raw_correspondences
```

The script runs `slam_trajectory_test --export-dir … --export-pair all` on a temporary
copy of the dataset and checks that the trajectory outputs stay byte-identical.

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

For comparison, the reference pipeline's RANSAC PnP (baseline) ends 0.277 m and 32.1° off.

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

## Milestone 3: the complete scratch pipeline (`scratch_pipeline`)

### Stages and modules

| Stage | Module | Notes |
|---|---|---|
| PNG decoding | `src/png.{hpp,cpp}` | DEFLATE (RFC 1951; stored, fixed and dynamic Huffman blocks), zlib wrapper with Adler-32, chunk CRC-32, the five scanline filters; 8/16-bit grey/RGB(A) |
| grayscale | `feature_core` `GrayFromRGB` | BT.601 luma, 14-bit fixed point |
| FAST | `feature_core` `DetectFast` | 16-pixel circle, arc ≥ 9, **threshold 20** (configurable), corner score, 3×3 non-maximum suppression, border 16, strongest 500 |
| orientation | `feature_core` `AssignOrientations` | intensity centroid, angle = atan2(m01, m10), radius 15 |
| rotated BRIEF | `feature_core` `ComputeBrief` | Gaussian smoothing (σ 2, 9×9), 256 tests from a fixed Gaussian pattern (seed `0x5B21EF`), rotated by the keypoint angle |
| matching | `feature_core` `MatchBruteForce`, `FilterMatchesByDistance` | brute-force Hamming nearest neighbour; keep distance ≤ max(2·d_min, 30) (floor configurable) |
| correspondences | `src/frontend.{hpp,cpp}` | depth at the truncated keypoint pixel, Z = raw/5000, X = ((u−cx)/fx·Z, (v−cy)/fy·Z, Z) in camera i; project-owned `Correspondence` type |
| essential matrix | `src/essential.{hpp,cpp}` | normalized 8-point (Hartley normalization), (s, s, 0) enforcement, four (R, t) candidates, cheirality by two-view depths; Sampson distance |
| RANSAC | `src/ransac.{hpp,cpp}` | sample → fit → residuals of all points → inliers → score (count, then residual sum) → best → refit on inliers → final mask |
| PnP | `src/pnp.{hpp,cpp}` (milestone 1, unchanged) | linear DLT, rotation projection by SVD, cheirality sign, translation re-solve |
| trajectory | `src/pipeline_main.cpp` | T_wc[0] = T_gt[0], T_wc[i+1] = T_wc[i]·T_{i+1←i}⁻¹ |

**Convention.** For every pair i → i+1, both the essential-matrix pose and the
PnP pose satisfy X_{i+1} = R·X_i + t (T_{i+1←i}). The essential matrix uses
x_{i+1}ᵀ E x_i = 0 with E = [t]× R and returns a unit t, because the scale is
unobservable from two views.

### Essential matrix

With normalized coordinates x = K⁻¹(u, v, 1)ᵀ, each match gives one linear
equation in the nine entries of E:

    [x₂x₁, x₂y₁, x₂, y₂x₁, y₂y₁, y₂, x₁, y₁, 1] · e = 0

1. **Estimate.** Hartley normalization; with 8 or more matches, e is the right
   singular vector of the smallest singular value. The normalization is undone
   with E = T₂ᵀ·Ẽ·T₁.
2. **Constraint enforcement.** An essential matrix has two equal singular values
   and one zero. With SVD E = U·S·Vᵀ, it is replaced by U·diag(s, s, 0)·Vᵀ with
   s = (s₁ + s₂)/2.
3. **Decomposition.** U and V are made proper (det = +1). With W = [[0,−1,0],[1,0,0],[0,0,1]]:
   - R ∈ {U·W·Vᵀ, U·Wᵀ·Vᵀ}
   - t = ±u₃, the third column of U

   This gives four candidates.
4. **Cheirality.** For each candidate, solve d₂·x₂ − d₁·R·x₁ = t in least
   squares for the two depths of every match. The physical candidate is the
   one with the most matches in front of both cameras (d₁, d₂ > 0).

### RANSAC around PnP

- **Sampling:** a minimal sample of 6 correspondences (the DLT's minimum), drawn by a
  partial Fisher–Yates shuffle on `std::mt19937`. The generator's output sequence is
  fixed by the C++ standard; `std::uniform_int_distribution` is avoided because its
  algorithm is implementation-defined.
- **Hypotheses:** the scratch DLT fits each sample. Every correspondence is
  reprojected; points behind the camera are never inliers.
- **Scoring:** inliers are points with error < threshold. Hypotheses are scored by inlier
  count, and ties go to the smaller sum of inlier errors.
- **Final model:** the best hypothesis is **refit with the same DLT on all its inliers**.
  The exported mask and statistics are those of the refit pose.
- **Defaults:** 300 iterations, 8 px, seed 12345, all configurable
  (`--pnp-iterations`, `--pnp-threshold`, `--seed`).

The essential stage uses the same engine: 8-point samples, Sampson distance in
pixels, 2 px, 2000 iterations.

### Result: all 35 pairs (`results/pipeline/summary.txt`)

| | Value |
|---|---|
| keypoints per frame, filtered matches, 3D→2D correspondences (means) | 131.4, 94.2, 83.6 |
| RANSAC PnP inliers (mean / min) | 56.5 / 30 |
| inlier reprojection error (mean over pairs) | 2.42 px |
| pairs solved / failed | 35 / 0 |
| per-pair rotation error, mean / median / max | 4.74° / 2.75° / 29.7° (pair 10→11) |
| per-pair translation error, mean / median / max | 0.042 m / 0.024 m / 0.255 m |
| **final trajectory error (frame 35)** | **0.095 m, 9.88°** |
| pair 0→1 | 108/107 keypoints, 86 matches, 76 3D→2D, 57 inliers; **2.31°, 0.0197 m**; inlier reprojection 1.53 / 1.28 / 4.55 px (mean / median / max) |

**How it compares.**
- The same matches of pair 0→1 without RANSAC give 47.5° (pure linear PnP).
- The reference pipeline's baseline ends at 0.277 m / 32.1°, and its checkpoint-01 run
  (threshold 30) at 0.357 m / 56.2°.
- The trajectories are not expected to match: different features, and a linear
  rather than a nonlinear PnP refinement.

**Weak pairs.** Several pairs keep only 30–36 inliers and have 10–30° errors:
- 9→10, 10→11
- 26→27 to 29→30

These are the frames where the bunny shows the least texture to FAST at
threshold 20 (88–112 keypoints).

### Essential-matrix stage: what it can and cannot do here

The essential stage runs on every pair and is exported (`essential_*` columns,
`essential_inlier` flags). Its pose is a **diagnostic only**: it is not
accumulated into the trajectory. On this dataset it reports a rotation error of
about 10° and a translation direction about 90° off, i.e. it returns R ≈ I.

The cause is the geometry, not the code:
- **On exact data it is exact.** Pair 0→1's real 3D points, projected with the true
  motion, give the pose to 0.000° (`test_essential`).
- **The motion nearly cancels in the image.** The camera orbits while looking at
  the bunny, so the 10° rotation and the 8.7 cm translation almost cancel: the
  points move only **4.3 px on average** between frames, against about 91 px for
  the rotation alone.
- **The two-view problem is therefore nearly degenerate.** With 1 px of noise, even
  all 76 points give 3.6° of error. The minimal 8-point samples inside RANSAC are
  dominated by noise and favour the rotation-free solution.

PnP does not have this problem, because it uses the depth (3D points). On a
scene with a wide field of view, the same RANSAC essential estimate is accurate
to 0.14° (`test_ransac`).

### Outputs (`results/pipeline/`, generated)

| File | Content |
|---|---|
| `pairs.csv` | per pair: keypoints, raw/filtered matches, d_min, 3D→2D count, PnP ok/inliers, reprojection mean/median/max, rotation/translation error, t, estimated rotation, essential ok/inliers/errors, PnP-vs-essential rotation |
| `trajectory.csv`, `trajectory.txt` | the 36 poses (T_wc; CSV with matrix and quaternion, absolute errors, inliers, source) |
| `correspondences/pair_<i>_<j>.csv` | every filtered match: pixels, Hamming distance, depths, 3D point, `essential_inlier`, `pnp_used`, `pnp_inlier`, residual |
| `summary.txt` | the summary above |

### Tests (`ctest --test-dir build`: 12 tests)

| Test | What it checks |
|---|---|
| `features_{fast,orientation,brief,matcher}_test` | the shared feature modules (built from `../tests/unit`): segment test, score, NMS, border, cap, determinism; orientation; BRIEF pattern, smoothing, **rotation invariance**, **Hamming distance**; matching and the filter |
| `test_png` | CRC-32 and Adler-32 check values; stored, fixed and back-reference DEFLATE streams from an independent encoder; a dynamic-Huffman dataset image; RGB and depth samples equal to an independent decoder's; 202/202 depth values equal to the baseline export's |
| `test_frontend` | frames 0→1: counts, back-projection, geometric quality of the matches against the true motion, determinism |
| `test_essential` | exact recovery, (s, s, 0), the four candidates, cheirality, 8-point minimum, noise, Sampson distance |
| `test_ransac` | 30 % gross outliers: linear PnP alone 28° off, RANSAC 0.47° (as good as the solver on the true inliers only); mask quality; seeds; failures; RANSAC essential on a wide scene |
| `test_pipeline_0_1` | the whole pipeline on frames 0→1: pose within 3° / 3 cm (2.31° / 0.020 m), inliers, reprojection, proper R, mask, comparison without RANSAC, determinism |
| `milestone1/2/3_*` | the three programs exit 0 |

## Figures and the comparison with the reference pipeline

```bash
python3 pnp_from_scratch/tools/make_figures.py       # after build/scratch_pipeline; NumPy + Matplotlib
python3 pnp_from_scratch/tools/check_trajectory.py   # frame-by-frame transform-chain check
```

The tools use NumPy, Matplotlib and `tools/png_reader.py`, a PNG reader built on the
Python standard library (`zlib`, `struct`). It is checked against an independent decoder:
identical RGB and 16-bit depth, and 202/202 baseline depth values.
- The tools load no vision library (`cv2` is never imported). Matplotlib imports Pillow
  internally for writing images; the tools never decode images with it.
- They run no reference program. The reference data is frozen in
  `../docs/migration/baseline/`, including the ORB keypoint lists of frames 0 and 1 in
  `reference_features/`, regenerated from the `baseline-pre-migration` tag.

**Outputs** (`results/figures/`, PNG + PDF, generated):

| Figure | Shows |
|---|---|
| `bunny_correspondences_0_1` | scratch correspondences on the real frames 0 and 1, with the same colour for the same match in both frames: RANSAC inliers (solid), outliers (magenta dashed), matches without depth (grey), other keypoints (white) |
| `bunny_correspondences_0_1_inliers` | the 57 RANSAC inliers only |
| `bunny_correspondences_0_1_outliers` | the 19 rejected matches, numbered. Several ear-tip features of frame 0 are matched to the same frame-1 keypoint (nearest-neighbour matching is many-to-one); RANSAC rejects them |
| `fig03_orb_keypoints` | the reference ORB keypoints (366 / 367, with orientation) |
| `feature_comparison_0_1` | scratch (FAST → orientation → rotated BRIEF → Hamming → scratch RANSAC) vs reference (ORB → ORB descriptor → Hamming → library RANSAC): keypoints with orientation and the correspondence maps, on the same frame pair. The pipelines use different keypoints and matches |
| `pnp_comparison` (+ `pnp_comparison.md`) | per-pair counts and rotation errors, and accumulated position/rotation error per frame: scratch PnP, reference PnP, reference ICP |

**Scratch vs reference PnP** (all 35 pairs; different features in each):

| | Scratch | Reference (baseline) |
|---|---|---|
| keypoints / frame (mean) | 131.4 | 430.4 |
| filtered matches / valid 3D→2D / RANSAC inliers (means) | 94.2 / 83.6 / 56.5 | 191.7 / 142.5 / 132.0 |
| inlier reprojection error (mean) | 2.42 px | 1.94 px |
| per-pair rotation error, mean / median / max | 4.74° / 2.75° / 29.66° | 1.39° / 1.17° / 3.28° |
| per-pair translation error, mean / median | 0.042 m / 0.024 m | 0.012 m / 0.011 m |
| trajectory error, mean over 36 frames | 0.187 m / 21.0° | 0.180 m / 21.4° |
| final error (frame 35) | 0.095 m / 9.9° | 0.277 m / 32.1° (reference ICP 0.525 m / 64.9°) |

**How to read it.**
- The reference is more accurate per pair, with more features, a nonlinear PnP
  refinement and a larger inlier set.
- Over the whole trajectory the two are about equal.
- The scratch result's smaller frame-35 error is not uniform accuracy:
  - its per-pair errors have mixed signs (19 of 35 pairs over-rotate), so they partly cancel
  - one bad pair (10→11, 29.7°) puts it above the reference for frames 11–28
- The reference's small errors are one-signed (under-rotation), so they accumulate steadily.

## Blender scene

`visualization/scripts/build_scene.py` reads **`results/pipeline/trajectory.csv`** (milestone 3)
and adds `ScratchPnP_Animated_Camera` (36 keyframes) and `ScratchPnP_Trajectory` (36 points),
in cyan. The HUD label is "Scratch PnP", with the note "own features + RANSAC + linear PnP".

The pose goes through the same `trajectory_to_blender_pose()` (R_FIX) and `WorldRoot`
path as the GT, PnP and ICP cameras. `scripts/check_blender_consistency.py` verifies,
from the saved scene:
- the camera matches its data: position within 3e-8 m, rotation exactly
- it looks at the bunny: maximum look-at error 5.61°, equal to the value
  computed from the data to within 8e-4°

Until this was fixed, the scene loaded the milestone-2 file
`results/scratch_pnp_trajectory.csv` (no RANSAC), whose camera looks up to 141.7° away from
the bunny. That trajectory is still available by pointing `SCRATCH_TRAJECTORY_CSV_PATH` at it.
`tools/check_trajectory.py` compares both files with GT and the reference frame by frame.

Before drawing, the build checks:
- the frame-0 anchor
- the quaternion columns against the matrix columns
- the stored errors against errors recomputed from the poses
- that no frame equals the reference PnP trajectory

The scene supports partial trajectories: a prefix of frames 0..k−1 is shown, and the
camera is hidden after it.

## Reproduce everything

From the repository root:

```bash
cd pnp_from_scratch
cmake -S . -B build && cmake --build build -j
ctest --test-dir build              # 12 tests
./build/scratch_pipeline            # milestone 3: images -> results/pipeline/
./build/pnp_from_scratch            # milestone 1
./build/pnp_full_sequence           # milestone 2 -> results/scratch_pnp_trajectory.csv
python3 tools/check_trajectory.py   # transform-chain check of the trajectories (NumPy)
cd .. && blender --python visualization/scripts/build_scene.py
```

## Limitations

- **Milestones 1–2 have no outlier rejection** (by design). Milestone 3 adds RANSAC.
- **The PnP is linear** (DLT, algebraic error). There is no nonlinear refinement of the
  reprojection error yet.
- **The essential matrix is a diagnostic only** on this dataset; see "Essential-matrix stage".
- Algebraic least squares only; there is no nonlinear (Gauss-Newton/LM) refinement.
- Milestones 1–2 read correspondences exported by the reference pipeline. Milestone 3
  computes its own from the images.
