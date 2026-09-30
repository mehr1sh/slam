# Chapter 6/7 test reference

Validation reference for the 8 executables built from slambook2's `ch6`
(nonlinear optimization) and `ch7` (visual odometry) source. Not a tutorial.

## Build

```
pixi run build
```

Builds these 8 executables (among the project's other programs) in `build/`. Run everything from `build/`.

## `two_view_pose_test`

```
./two_view_pose_test ../data/tum_sample/1.png ../data/tum_sample/2.png
```

**Verifies**: `find_feature_matches()` (`src/features/features.cpp`) and
`pose_estimation_2d2d()` (`src/geometry/geometry.cpp`) -- both book-verbatim
(book: `pose_estimation_2d2d.cpp`). No depth used; monocular 2D-2D only.

| Output | Meaning |
|---|---|
| `matches: N` | ORB matches surviving the `distance <= max(2*min_dist, 30)` filter. **N = 79** on this dataset (verified). |
| `fundamental_matrix` | `F` from the 8-point algorithm (`findFundamentalMat`, pixel coordinates). Encodes epipolar geometry without needing `K`. |
| `essential_matrix` | `E = K^T F K`, computed directly via `findEssentialMat`. Encodes only rotation+translation (calibrated coordinates). |
| `homography_matrix` | `H` from `findHomography` -- only meaningful if the scene is planar. Printed for reference; this scene (a cluttered desk) is not planar, so `H` is not further used. |
| `R`, `t` | Recovered relative pose from `recoverPose` (SVD decomposition of `E`, chirality check picks the one physically valid solution of 4). `t` is unit-norm (see scale-ambiguity note below). |
| `t^R` | Skew-symmetric `[t]_x` times `R`, should be proportional to `E` (checks `E = [t]_x R` holds for the recovered pose, up to the same scale ambiguity). |
| `epipolar constraint = [x]` | `y2^T [t]_x R y1` for every match, in normalized camera coordinates. Should be **~0** if `R,t` are correct -- **exact algebraic identity** for a perfect two-view pose. |

**Expected magnitude**: epipolar constraint residuals cluster around **1e-3**
(min/max observed here: ~1e-6 to ~7e-3). This is noise from pixel-level
feature localization, not from a bug -- matches the book's own reported
order of magnitude on this exact dataset.

**Variation**: `R`, `t`, and the exact match count can shift slightly across
OpenCV versions (ORB/RANSAC/`recoverPose` internals aren't bit-identical
across releases), but should stay close to the reported values on the same
dataset. The *order of magnitude* of the epipolar constraint is the
important invariant, not its exact per-match value.

**Added reporting (not book code, `tests/report_utils.hpp`)**: translation
norm, rotation angle (degrees) of `R`, and full-sample epipolar-residual
statistics (mean/median/rms/max absolute value) plus a pass count against a
`1e-2` threshold, printed under `=== TWO-VIEW GEOMETRY SUMMARY ===`. Also
saves `output/two_view_pose/epipolar_lines_img{1,2}.png` -- the epipolar
line for each match, drawn via `cv::computeCorrespondEpilines` on
`F = K^-T [t]_x R K^-1`, with the matching point marked on its own line --
a visual (not just numeric) demonstration that the recovered `R,t` are
geometrically consistent with the matches.

## `pnp_test`

```
./pnp_test ../data/tum_sample/1.png ../data/tum_sample/2.png \
           ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
```

**Verifies**: 3D-2D pose estimation three ways -- OpenCV `solvePnP`, a
manual Gauss-Newton bundle adjustment, and a g2o bundle adjustment (all
book-verbatim, book: `pose_estimation_3d2d.cpp`, `src/tracking/pnp.cpp`).

| Output | Meaning |
|---|---|
| `matches: N` | Same ORB matching as above (79). |
| `3d-2d pairs: M` | Matches where frame 1's depth map has a valid (nonzero) reading at the keypoint -- these become 3D points (via `pixel2cam()` + depth) paired with frame 2's 2D keypoint. **M = 75** on this dataset (4 matches dropped to zero depth). |
| `solve pnp in opencv cost time` | Wall-clock time for `cv::solvePnP` (EPnP by default) -- not a correctness metric. |
| `R=`, `t=` (OpenCV) | Pose of camera 2 recovered directly by `solvePnP` + `Rodrigues()`. This `t` **is metric** (metres), because the 3D points came from real depth, not from unit-norm 2D-2D triangulation. |
| `iteration N cost=X` (Gauss-Newton) | Sum of squared reprojection errors (pixels²) at each GN iteration. Should **monotonically decrease** and converge in a handful of iterations; GN stops early once `‖dx‖ < 1e-6` or cost stops improving. |
| `pose by g-n:` | 4x4 `SE(3)` matrix from the manual GN refinement. Should closely match OpenCV's `R,t` above (same physical pose, different solver). |
| `iteration= N chi2=X ... edges=75` (g2o) | g2o's own verbose optimizer log (`optimizer.setVerbose(true)`) -- `chi2` is the same sum-of-squared-reprojection-error cost as GN's, `edges=75` = one edge per 3D-2D pair. Should converge to the **same chi2** GN converged to. |
| `pose estimated by g2o =` | Same `SE(3)` matrix, should closely match both prior results. |

**Expected numerical result** (this dataset): all three methods agree to
~6 significant figures (e.g. GN converges to cost **≈299.7636**, g2o
converges to **chi2 ≈299.7636** by iteration 2-3 and stays flat for the
remaining fixed 10 iterations). This three-way agreement is the actual
correctness check -- it means the closed-form OpenCV solver and both
from-scratch optimizers found the same optimum.

**Variation**: solver iteration counts/timings are platform-dependent;
the converged cost/pose should not be.

**Added reporting (not book code)**: rotation-angle and translation-norm
differences between all three pose estimates (solvePnP/GN/g2o -- all
~0, confirming agreement), and per-method reprojection-error statistics
(mean/median/rms/max, pixels) over the 75 used correspondences, plus the
used/rejected point count, under `=== PNP SUMMARY ===`.

## `icp_test`

```
./icp_test ../data/tum_sample/1.png ../data/tum_sample/2.png \
           ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
```

**Verifies**: 3D-3D pose estimation two ways -- closed-form SVD (Kabsch/Umeyama-style)
and g2o bundle adjustment (book-verbatim, book: `pose_estimation_3d3d.cpp`,
`src/tracking/icp.cpp`).

| Output | Meaning |
|---|---|
| `3d-3d pairs: M` | Matches with valid depth in **both** frames (both become 3D points via depth backprojection). **M = 72**. |
| `W=` | Cross-covariance matrix `sum(q1_i * q2_i^T)` of the two mean-centered point sets -- the core quantity the closed-form ICP solves via SVD. |
| `U=`, `V=` | Left/right singular vectors of `W` (`W = U*Sigma*V^T`). |
| `R = U*V^T` (with a determinant-sign fix) | Closed-form rotation. `t` follows from the centroids: `t = centroid1 - R*centroid2`. **Metric** (real depth in metres, not unit-norm). |
| `R_inv`, `t_inv` | The inverse transform (`R^T`, `-R^T t`) -- maps cloud 2's frame back into cloud 1's, a sanity-check quantity, not a separate computation. |
| `chi2` (g2o) | Sum of squared 3D point-to-point errors (metres²) for the BA refinement, using the closed-form SVD result as its initial value. |
| `T=` (after optimization) | 4x4 `SE(3)` from the g2o refinement -- should closely match the closed-form `R,t`. |
| `p1 = ...`, `p2 = ...`, `(R*p2+t) = ...` | Explicit check on 5 sample points: `R*p2+t` should be close to `p1`. This *is* the ICP equation the solver satisfies, printed out for visual confirmation. |

**Expected numerical result**: SVD and g2o results should match closely
(this dataset: g2o chi2 converges to **≈1.8155** essentially immediately,
i.e. the closed-form solution is already at/near the optimum -- expected,
since ICP with known correspondences has an exact closed-form solution and
BA has little left to improve). `R*p2+t` should match `p1` to within
~0.01-0.02 (real depth-sensor + pixel-localization noise, not error).

**Added reporting (not book code)**: the book's `bundleAdjustment(pts1, pts2,
R, t)` call overwrites `R,t` in place with the g2o result, so `R_svd`/`t_svd`
are cloned immediately after the SVD result is printed and before that call,
enabling an explicit before/after comparison. Reports rotation/translation
diff SVD-vs-g2o, and point-to-point alignment-error statistics
(mean/median/rms/max, metres) both before (SVD) and after (g2o) refinement,
over all 72 correspondences (not just the 5 samples the book prints), under
`=== ICP SUMMARY ===`.

## `feature_matching_test`

```
./feature_matching_test ../data/tum_sample/1.png ../data/tum_sample/2.png
```

**Verifies**: ORB extraction + BF-Hamming matching only (book: `orb_cv.cpp`
Sec. 6.2.1, via the shared `find_feature_matches()`, extended with an
optional out-param so this test can see the pre-filter matches too).

- `all matches: N` -- raw BF-Hamming matches before filtering (**500**, one
  per ORB keypoint in image 1 -- OpenCV's `ORB::create()` default `nfeatures`).
- `matches: N` -- surviving the `distance <= max(2*min_dist, 30)` filter (**79**, unchanged from before).
- Saves all three of `orb_cv.cpp`'s own visualizations (path relative to
  the working directory, i.e. relative to `build/` when run as shown above):
  - `output/feature_matching/keypoints.png` -- detected ORB keypoints (image 1 only, book's "ORB features" window).
  - `output/feature_matching/all_matches.png` -- all matches before filtering.
  - `output/feature_matching/good_matches.png` -- after filtering.

**Added reporting (not book code)**: keypoint counts per image, and
Hamming-distance statistics (min/max/mean/median) over the retained good
matches plus the retention percentage, under
`=== FEATURE MATCHING SUMMARY ===`.

## `triangulation_test`

```
./triangulation_test ../data/tum_sample/1.png ../data/tum_sample/2.png
```

**Verifies**: the full 2D-2D-pose -> triangulation pipeline (book:
`triangulation.cpp`). No depth maps used -- 3D points come purely from
triangulating the two-view geometry, so depths are in the same **arbitrary,
unit-norm-`t` scale** as `two_view_pose_test`, not metres.

- `matches: N` -- **79**; then the same `fundamental_matrix`/`essential_matrix`/
  `homography_matrix`/`R`/`t` block as `two_view_pose_test` (same function).
- `depth: X` printed once per match -- the triangulated point's Z coordinate
  in camera 1's frame (arbitrary scale).
- Saves two depth-colored reprojection images (book's `get_color()`: red =
  near `low_th=10`, blue = near `up_th=50`, in this arbitrary scale) to
  `output/triangulation/img1_reprojection.png` and `img2_reprojection.png`.

**Added reporting (not book code)**: cheirality-check counts/percentage
(positive depth in both cameras -- the book's `triangulation()` returns all
points unconditionally, so this classification is done in the test using
the `depth1`/`depth2` values the book loop already computes) and depth
statistics (min/max/mean/median) over the valid points, plus an explicit
arbitrary-scale disclaimer, under `=== TRIANGULATION SUMMARY ===`. Also
saves `output/triangulation/points_and_cameras.ply` -- an ASCII PLY of the
valid triangulated points (grey) plus the two camera centres (red = camera
1 at the origin, green = camera 2 at `-R^T t`), viewable in any point-cloud
viewer (e.g. MeshLab, CloudCompare).

## `orb_from_scratch_test`

```
./orb_from_scratch_test
```

No arguments -- paths to `1.png`/`2.png` are hardcoded (book source), fixed
here to `../data/tum_sample/{1,2}.png` (see file header comment) so it runs
from `build/`. Independent from-scratch FAST+BRIEF-style implementation
(book: `orb_self.cpp`, Sec. 6.2.2) -- does **not** call any `src/` module;
included for comparison against OpenCV's ORB, not as production code.

| Output | Meaning |
|---|---|
| `bad/total: 43/638`, `bad/total: 8/595` | Keypoints too close to the image border (within `half_boundary=16px`) to compute a descriptor for, per image. **Exact/deterministic** given the same OpenCV `cv::FAST(threshold=40)` implementation. |
| `matches: N` | Brute-force Hamming matches under a fixed distance threshold (`d_max=40`) -- **not directly comparable to the 79 above**: different keypoint detector (plain FAST, not ORB's oriented+pyramided variant), different descriptor (this file's own fixed-pattern BRIEF, not OpenCV's), different matching filter (fixed threshold vs. `2*min_dist`). Observed: **41** on this dataset. |

Saves `output/orb/matches.png` (book's own `imwrite` call, redirected from
the working directory into `output/orb/`; the book's additional `imshow`/
`waitKey` were removed so this runs headless too).

## `gauss_newton_curve_fit_test` / `g2o_curve_fit_test`

```
./gauss_newton_curve_fit_test
./g2o_curve_fit_test
```

No arguments. Not part of the VO pipeline -- these fit `y = exp(ax^2+bx+c)`
to synthetic noisy data to demonstrate the optimization machinery
(book: `ch6/gaussNewton.cpp`, `ch6/g2oCurveFitting.cpp`) later reused inside
`pnp_test`'s/`icp_test`'s bundle adjustment.

Both use `cv::RNG rng;` with **no explicit seed** -- OpenCV's `RNG()`
default constructor uses a fixed internal seed, so the generated
`y_data` (and therefore the converged result) is **exactly reproducible**
across runs and identical between the two executables (same data, same
underlying least-squares problem, two different solvers).

- Ground truth: `a=1.0, b=2.0, c=1.0`; initial guess: `a=2.0, b=-1.0, c=5.0`.
- `total cost:`/`chi2=` -- sum of squared residuals; should monotonically
  decrease (GN breaks early on convergence; g2o always runs its fixed 10
  iterations, flat-lining once converged).
- Both converge to **`a≈0.890912, b≈2.1719, c≈0.943629`** -- not exactly
  the ground truth (1,2,1), because the noisy data itself doesn't have its
  minimum exactly there; the check is that **both solvers agree with each
  other**, confirming the manual normal-equations math and the g2o
  vertex/edge Jacobians are consistent.

**Added reporting (not book code)**: both files now have explicit
`// Book code starts/ends` markers (previously implicit, verbatim book
copies with no markers). `g2o_curve_fit_test` splits the book's single
`optimizer.optimize(10)` into 10x `optimize(1)` to record `chi2` after every
iteration -- behaviorally identical to a single `optimize(10)` call for
`OptimizationAlgorithmGaussNewton` (no adaptive damping state carried
across `optimize()` calls), so the final estimate is unchanged. Both write
a per-iteration cost CSV (`output/optimization/gauss_newton_cost.csv`,
`g2o_cost.csv`) and print an initial/final cost, iteration count, converged
status, and final parameters under `=== GAUSS-NEWTON CURVE FIT SUMMARY ===`
/ `=== G2O CURVE FIT SUMMARY ===`.

## Visualization: what and why

`feature_matching_test`, `triangulation_test`, and `orb_from_scratch_test`
are the only tests that produce images -- exactly the three book files
(`orb_cv.cpp`, `triangulation.cpp`, `orb_self.cpp`) that call `imshow` in
the first place (verified by grepping every file under `tests/`, `src/`,
`include/`, and slambook2's `ch6`+`ch7`; no other book file in scope
calls any GUI function). Each `imshow` was a quick visual sanity check
while developing (are the matches sane? does triangulated depth look
physically plausible? does the from-scratch ORB match anything at all?),
never load-bearing for the algorithm -- every one of these files already
prints every number the visualization would show, so headless output loses
no validation information, only the human-eyeball convenience, which
`output/*.png` restores non-interactively.

The other 5 tests never call `imshow` because their corresponding book
examples (`pose_estimation_2d2d.cpp`, `pose_estimation_3d2d.cpp`,
`pose_estimation_3d3d.cpp`, `gaussNewton.cpp`, `g2oCurveFitting.cpp`) don't
either -- confirmed by the same grep, not assumed from file naming.

**Saved output**:
- `output/feature_matching/keypoints.png`, `all_matches.png`, `good_matches.png`
- `output/triangulation/img1_reprojection.png`, `img2_reprojection.png`, `points_and_cameras.ply`
- `output/orb/matches.png`
- `output/two_view_pose/epipolar_lines_img1.png`, `epipolar_lines_img2.png` (new,
  not from a book `imshow` call -- an added visualization, see `two_view_pose_test` above)
- `output/optimization/gauss_newton_cost.csv`, `g2o_cost.csv` (new, per-iteration
  cost curves, see `gauss_newton_curve_fit_test`/`g2o_curve_fit_test` above)

Paths are relative to the working directory (i.e. relative to `build/` per
the run commands above), not `--output`-configurable -- kept to the exact
argument counts the book's own examples use. `output/` is created
automatically if missing.

## Monocular scale ambiguity (2D-2D)

`recoverPose` normalizes `t` to unit length -- this is fundamental to
monocular two-view geometry, not a limitation of this code: `E = [t]_x R`
is invariant to scaling `t` by any nonzero constant, so two views alone
cannot recover the true baseline length. `two_view_pose_test`'s and
`triangulation_test`'s `t` and triangulated depths are therefore in an
**arbitrary, self-consistent scale**, not metres.

## Metric translation (PnP/ICP)

`pnp_test` and `icp_test` recover **metric** `t` because their 3D points
come from the TUM depth maps (`depth_value / 5000.0` -> metres, TUM's
documented encoding), not from unit-norm 2D-2D triangulation. This is why
their `t` is directly comparable in magnitude to the physical scene
(desk-scale, ~0.03-0.15 units here) while the 2D-2D `t` is not.

## Platform-dependent variation summary

| Category | Examples | Expected variation |
|---|---|---|
| Deterministic/exact | curve-fit converged params (fixed RNG seed), `orb_from_scratch_test`'s bad/total counts | Should reproduce exactly on the same OpenCV/Eigen/g2o versions |
| Small numeric drift | `two_view_pose_test`/`pnp_test`/`icp_test` matrices, match counts | OpenCV ORB/RANSAC/solver internals can differ slightly across versions; order of magnitude and cross-method agreement (PnP's 3 methods, ICP's 2 methods) are the real invariants |
| Timing only | all `cost time`/`time=` fields | Hardware-dependent, not a correctness signal |
