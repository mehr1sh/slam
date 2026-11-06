# RGB-D ICP (experiments/rgbd_icp)

## 1. Purpose and status

`experiments/rgbd_icp/` holds the teammate's dense RGB-D registration: library `rgbd_icp`, sequence
program `icp_sequence` and unit test `rgbd_icp_test`. For each consecutive frame pair it estimates the
rigid motion with feature-based 3D RANSAC, then point-to-point ICP, then point-to-plane ICP. After the
sequence it searches for loop closures and runs a small pose graph.

Status: **provisional experiment, not production.** No final ICP decision has been made
(`experiments/rgbd_icp/CMakeLists.txt:1`). The production pipeline (`apps/pipeline_main.cpp`,
`scratch_core`) is frame-to-frame PnP and does not use this code. Two other ICP variants exist and are
never linked into the same executable as each other:

- `reference/opencv_slambook2/src/tracking/icp.cpp` is the teammate's compatibility wrapper.
  `pose_estimation_3d3d` calls `rgbd_icp::rigidSVD` (`icp.cpp:8-24`), and `bundleAdjustment` only calls
  `pose_estimation_3d3d`; it no longer runs g2o (`icp.cpp:29-33`).
- `reference/icp_main_variant/` keeps main's ICP unchanged: book SVD, g2o `bundleAdjustment`, `FitRigid`
  and `EstimateRigidRansac`. It is built as `tracking_main_icp` and used by `tum_trajectory_eval`,
  `icp_rigid_test` and the `*_main_icp` tests (`reference/opencv_slambook2/CMakeLists.txt:146-149`).

PnP and ICP produce two independent trajectories. PnP does not initialise ICP, and ICP does not correct
PnP. Ground truth is never used for estimation (section 4).

## 2. Build and run

OpenCV is needed only for image I/O: `cv::imread` and `cv::Mat` (`experiments/rgbd_icp/src/rgbd_icp.cpp:14,501-511`). Feature
extraction and matching use the project's `feature_core` (`experiments/rgbd_icp/src/rgbd_icp.cpp:2-3,512-526`). SVD, RANSAC,
KD-tree search, both ICP solvers and the pose graph are written in C++ with Eigen. The code includes no
Open3D, no `solvePnP`, no g2o and no Python (`experiments/rgbd_icp/src/rgbd_icp.cpp:1-18`). There are two ways to build.

**Root build.** The root `CMakeLists.txt` adds `experiments/rgbd_icp` only when `-DSLAM_WITH_OPENCV=ON`
(`CMakeLists.txt:16,125-129`). That option also builds the g2o/Sophus reference library, so g2o and
Sophus must be installed (pixi provides them). Executables are written to `build/`
(`CMakeLists.txt:11`).

```bash
pixi run build            # = cmake -B build -S . -DSLAM_WITH_OPENCV=ON && cmake --build build -j (pixi.toml:11-12)
ctest --test-dir build -R rgbd_icp_geometry --output-on-failure
./build/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp
```

**Standalone build.** `experiments/rgbd_icp/icp_cpp/` builds only OpenCV core and imgcodecs, Eigen,
`feature_core`, `rgbd_icp`, `icp_sequence` and `rgbd_icp_test`, and does not need g2o or Sophus
(`experiments/rgbd_icp/icp_cpp/CMakeLists.txt:5-23`). It also registers three tests:

- `rgbd_icp_geometry`;
- `blender_data_bridge`, which runs `visualization/tests/test_blender_data.py` when Python 3 is found
  (`experiments/rgbd_icp/icp_cpp/CMakeLists.txt:27-31`);
- `orb_benchmark_smoke` from `cmake/orb_benchmark.cmake`, which is ON by default
  (`cmake/orb_benchmark.cmake:1-9`). Turn it off with `-DBUILD_ORB_BENCHMARK=OFF`.

In this build, `rgbd_icp_test` also compiles the reference compatibility wrapper `icp.cpp`
(`experiments/rgbd_icp/icp_cpp/CMakeLists.txt:21-23`). The root build does not compile it into this test
(`experiments/rgbd_icp/CMakeLists.txt:11-13`).

```bash
pixi run icp-build        # cmake -S experiments/rgbd_icp/icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release; cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure
pixi run icp              # ./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp (pixi.toml:14-16)
```

**Command line.** `icp_sequence [DATASET_DIR] [OUTPUT_DIR]`. The defaults are `data/synthetic_bunny` and
`results/runs/rgbd_icp` (disposable), both relative to the working directory, so run it from the repository root
(`experiments/rgbd_icp/apps/icp_sequence.cpp:66-67`). Exit codes are 0 when every pair is accepted, 2 when some pairs fail (all
outputs are still written), and 1 on an input or runtime error (`experiments/rgbd_icp/apps/icp_sequence.cpp:230-233`).

`results/benchmarks/rgbd_icp/` (moved from `icp_cpp_results/` on 2026-11-06) is tracked and is the Blender scene's ICP
input. It is written only when passed explicitly (`pixi run icp`, `visualization/rebuild_blender.sh`,
`tools/figures/run_gn_orb.sh`); the default output never overwrites it.

## 3. Implementation

| File | Contents |
| --- | --- |
| `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp` | `Pose`, `Camera`, `Config` (all parameters), feature, diagnostic and result structs, API |
| `experiments/rgbd_icp/src/rgbd_icp.cpp` | SVD, RANSAC, KD-tree, point and plane Gauss–Newton ICP, depth lifting, I/O, loop search, pose graph |
| `experiments/rgbd_icp/apps/icp_sequence.cpp` | sequence loop, fallback, loop and graph call, evaluation, CSV/trajectory export |
| `experiments/rgbd_icp/tests/rgbd_icp_test.cpp` | numerical and rejection tests (section 8) |
| `experiments/rgbd_icp/CMakeLists.txt`, `experiments/rgbd_icp/icp_cpp/CMakeLists.txt` | root sub-build and standalone build |

### 3.1 Parameters (`Config`, `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp:23-33`)

| Field | Default | Meaning |
| --- | --- | --- |
| `depth_scale` | 5000 | metric depth = raw uint16 / 5000 (`:24`) |
| `ratio` | 0.87 | descriptor nearest / second-nearest ratio, applied in both directions (`:24`) |
| `depth_edge` | 0.012 m | reject a pixel whose 3×3 depth range is ≥ 12 mm (`:24`) |
| `ransac_threshold` | 0.009 m | 3D inlier distance (`:25`) |
| `features` | 1400 | total feature budget over the pyramid (`:26`) |
| `fast_threshold` | 10 | FAST intensity threshold (`:26`) |
| `ransac_iterations` | 650 | 3-point hypotheses (`:26`) |
| `min_inliers` | 12 | minimum RANSAC consensus (`:27`) |
| `stride` | 3 | dense-cloud pixel stride (`:28`) |
| `max_points` | 2800 | dense-cloud cap (`:28`) |
| `min_pairs` | 70 | minimum nearest-neighbour pairs (`:28`) |
| `point_iterations` / `plane_iterations` | 18 / 16 | outer ICP iteration limits (`:28-29`) |
| `max_distance` | 0.024 m | NN rejection distance and the acceptance RMSE bound (`:30`) |
| `trim` | 0.83 | fraction of NN pairs kept, closest first (`:30`) |
| `huber` | 0.004 m | Huber transition on point-to-plane residuals (`:30`) |
| `max_angle_deg` / `max_translation` | 13° / 0.07 m | motion gate relative to the stage's initial pose (`:30-31`) |
| `seed` | 42 | RNG seed; pair *i* uses `seed + i` (`:32`; `experiments/rgbd_icp/apps/icp_sequence.cpp:111-112`) |

The values are fixed in code. `icp_sequence` has no flags apart from the two directory arguments.

### 3.2 Stages for one pair (`registerFrames` → `registerFramesWithFeatures`, `experiments/rgbd_icp/src/rgbd_icp.cpp:527-566`)

1. **Features** (`extractFeatures`, `:512-526`). The RGB image is converted to grey with
   `features::GrayFromInterleaved`. Then `features::ExtractMultiscale(BuildPyramid(gray), fast, {true, 1400})`
   runs. The pyramid has 8 levels with scale 1.2 (`include/slam/features/pyramid.hpp:25-26`). FAST uses
   threshold 10 and border 16 (`include/slam/features/fast.hpp:17`). Candidates are ranked by the Sobel
   Harris response (`src/features/multiscale.cpp:13-33`), which is computed once per candidate and then
   stable-sorted (`:73-85`). Per-level quotas follow a geometric split of 1400
   (`LevelQuotas`, `:34-55`). A quota is an upper limit, and unused capacity is not redistributed
   (`:86-88`). Each keypoint then gets an intensity-centroid orientation and a rotated BRIEF descriptor
   computed on the smoothed level image (`:89-92`).
2. **Matching** (`matchedFeaturePoints`, `experiments/rgbd_icp/src/rgbd_icp.cpp:288-308`). This uses
   `features::MatchMutualRatio(…, 0.87)` (`src/features/matcher.cpp:55-85`). The forward and reverse
   nearest-neighbour searches must each pass `d1 < 0.87·d2`, and the match must be mutual.
3. **Depth lifting** (`stableDepth`, `:237-251`; `project`, `:234-236`). The keypoint position is rounded
   to the nearest pixel. The pixel is rejected if it is within 2 px of the image border, if any of its 3×3
   depths is 0, or if the 3×3 depth range is ≥ 12 mm. Otherwise
   `z = raw/5000`, `x = (u−cx)z/fx` and `y = (v−cy)z/fy`, using the sub-pixel `(u, v)`. Each surviving
   match becomes a 3D–3D pair: target is frame *i*, source is frame *i+1*.
4. **Baseline**: one `rigidSVD` over all valid pairs (`:543`). This is evaluation only.
   `rigidSVD` (`:44-65`) needs at least 3 pairs of equal count with finite values, otherwise it throws.
   It uses the Kabsch/Umeyama cross-covariance and corrects reflections by flipping only the last
   singular axis (`D(2,2) = sign det(VUᵀ)`, `:60-63`).
5. **RANSAC** (`ransac`, `:309-355`). Each of the 650 iterations samples 3 distinct pairs. The sample is
   skipped if either triplet is collinear (cross-product norm ≤ 1e-7, `:67-69`). A hypothesis is fitted
   with SVD and counts pairs with `‖R p + t − q‖ < 9 mm`. The largest consensus wins; ties go to the
   smaller median residual (`:337-342`). If fewer than 12 inliers remain, the pair fails with
   `RANSAC_FAILED`. Otherwise the inliers are refitted and re-scored up to 4 times until the set stops
   changing, followed by a final refit (`:346-353`). This RANSAC pose is the only initialiser.
6. **Dense clouds** (`cloud`, `:252-286`). Every third pixel with stable depth is used (same 3×3 test).
   For the target cloud only, the normal at each point is the normalised cross product of the central
   differences of the neighbouring back-projected pixels. A point is dropped if this cross product has
   norm ≤ 1e-10 (`:259-268`). A cloud with more than 2800 points is reduced to 2800 by an `mt19937(seed)`
   shuffle, so sampling is deterministic (`:271-285`). If either cloud has fewer than 70 points, the
   status is `INSUFFICIENT_CLOUD` (`:557-560`).
7. **Point-to-point ICP** (`pointICP`, `:356-420`), started from the RANSAC pose. See section 3.3.
8. **Point-to-plane ICP** (`planeICP`, `:421-492`), started from the point-to-point result.
9. **Acceptance**: the pair is accepted only if both ICP stages are accepted (`:563-564`). The status is
   then one of `OK`, `ICP_REJECTED`, `RANSAC_FAILED`, `INSUFFICIENT_CLOUD` or `INSUFFICIENT_FEATURES` (the
   last means fewer than 3 valid 3D pairs; it is the default in `rgbd_icp.hpp:67`).

### 3.3 Dense ICP iteration (both stages)

Each outer iteration does the following.

- **Nearest neighbours** (`nearestPairs`, `:149-171`). These are recomputed at the current estimate in
  every iteration. A search uses an exact balanced 3D KD-tree built once per target cloud (`:99-144`).
  Pairs with distance ≥ 24 mm are rejected. Pairs are sorted by distance and the closest
  `max(3, ⌊0.83·n⌋)` are kept when n > 3. Each target point is used at most once, closest first.
- **Too few pairs**: if fewer than 70 pairs remain, the stage stops with `INSUFFICIENT_SUPPORT`.
- **Linear system**: H and b are built with the correspondences held fixed.
  - Point-to-point: `r = x − q`, `J = [−[x]×, I]` (`pointJacobian`, `:25-29`), `H = Σ JᵀJ`,
    `b = −Σ Jᵀr` (`:373-379`).
  - Point-to-plane: `e = nᵀ(x − q)` with the target normal, `j = Jᵀn = [(x × n)ᵀ, nᵀ]`, and Huber weight
    `w = 1` if `|e| ≤ 4 mm`, otherwise `4 mm/|e|`. Then `H = Σ w j jᵀ` and `b = −Σ w j e` (`:439-447`).
- **Damping and solve**: `H += (1e-5·trace(H)/6 + 1e-9)·I`, solved with LDLT. A failed or non-finite
  solve stops the stage with `NUMERICAL_FAILURE` (`:381-387`, `:448-455`). The damping is fixed: there is
  no adaptive Levenberg–Marquardt schedule in these two functions.
- **Step bound**: the whole 6-vector is scaled down so that ‖ω‖ ≤ 0.1 rad and ‖v‖ ≤ 0.03 m. This scales
  the full vector norm, not each component (`:389-392`, `:456-459`).
- **Backtracking**: step factors α ∈ {1, ½, ¼, ⅛, 1/16} are tried in order (`:395-409`, `:462-478`).
  - A candidate is skipped if it fails the motion gate (`safeMotion`, `:172-175`): the change from the
    stage's initial pose must be ≤ 13° and ≤ 7 cm.
  - The first remaining α whose fixed-pair cost does not exceed the old cost (+1e-14·n) is accepted.
    For point-to-point the cost is Σ‖r‖², for point-to-plane it is Σ Huber(e).
  - If no α qualifies, the stage stops with `NO_DESCENT`.
- **Pose update**: `leftIncrement` gives `(R, t) ← (Exp(αω) R, Exp(αω) t + αv)` (`:30-32`). This is a
  left SO(3)×R³ retraction, not the SE(3) exponential (`rgbd_icp.hpp:86-88`).
- **Convergence**: if both ‖αω‖ and ‖αv‖ are below 1e-7, the stage is marked `converged` with reason
  `CONVERGED`. Otherwise it continues until the iteration limit (`ITERATION_LIMIT`).

**After the loop** (`finish`, `:176-184`): the nearest neighbours are recomputed once more. The stage is
accepted if all of these hold:

- at least one update was performed;
- at least 70 pairs remain;
- the Euclidean NN RMSE is finite and below 24 mm.

ICP's job is to choose correspondences and re-estimate the pose. Gauss–Newton is only the numerical
method that performs each pose update. The binary-descriptor (Hamming) matches serve only for the RANSAC
initialisation and loop verification. The dense stages use Euclidean 3D nearest neighbours. The
point-to-plane stage is a Huber-weighted (IRLS) Gauss–Newton.

The before/after values in one record are computed on the same fixed pairs. The next iteration searches
the neighbours again, so the objective can rise from one iteration to the next when the correspondence
membership changes.

`accepted` and `converged` are separate fields. A stage that reaches its iteration limit can still be
accepted. The reported RMSE is the Euclidean NN distance in both stages, but the plane stage minimises
the normal-distance objective. Its Euclidean RMSE can therefore rise while its own objective falls.

**Per-update record** (`recordIteration`, `:189-233`). For every accepted update the code stores:

- the number of pairs;
- the Euclidean and normal RMSE before and after the update, over the same fixed pairs;
- the mean objective before and after;
- the step size;
- an FNV-1a hash of the sorted (source, target) assignment set, which ignores distance ordering
  (`:224-231`).

A new hash between updates shows that the correspondences changed.

### 3.4 Sequence, fallback, loops and pose graph (`experiments/rgbd_icp/apps/icp_sequence.cpp`)

**Frame count.** Frames are counted by the existence of `NNNNNN.png`, not by ground truth
(`:79-88`). At least 2 frames are required.

**Accepted pairs.** For an accepted pair *i→i+1*, all four stage trajectories (`baseline`, `robust`,
`point`, `plane`) compose `W_{i+1} = W_i · T_{i←i+1}`, starting from `W_0 = I` (`:91-92`, `:132-137`).
The plane pose is added as an odometry edge (`:124-127`).

**Rejected pairs.** Each trajectory instead reuses its last accepted relative motion, or the identity
if none exists yet. The pair is flagged in `pair_metrics.csv` and counted in `failed_pairs`. It is not
added as a graph edge (`:124-137`). These fallback motions are part of the evaluated trajectories, so
ATE and RPE include them.

**Loop search** (`findLoops`, `experiments/rgbd_icp/src/rgbd_icp.cpp:567-595`). Candidates are built from the plane
trajectory:

- frame pairs (i, j) with j ≥ i + 25 whose odometry relative pose is < 0.12 m and < 32°;
- sorted by that distance, with at most 8 tried.

Each candidate is registered again from scratch with `registerFrames` (seed `seed + i`). It becomes a
loop edge only if all of these hold:

- the registration is accepted;
- RANSAC has ≥ 35 inliers and an inlier ratio ≥ 0.65;
- the measured pose agrees with the odometry prediction within 8° and 4.5 cm.

This search is specific to this short orbit. It is not general place recognition.

**Pose graph** (`optimizeGraph`, `:625-743`). It runs only when at least one loop edge exists.

- **Residual**: each edge has `Log(M⁻¹ W_i⁻¹ W_j)`, scaled by 0.015 rad / 0.003 m for loop edges and
  0.025 rad / 0.005 m for odometry edges (`:598-604`).
- **Loss**: component-wise soft-L1 `4(√(1 + r²/4) − 1)` (`:606-615`).
- **Gauge**: one pose per connected component is held fixed, using union-find (`:640-663`).
- **Solver**: iteratively reweighted Gauss–Newton with numerical central-difference Jacobians (step
  1e-6) and adaptive damping. λ starts at 1e-5, is multiplied by 10 on rejection and divided by 3 on
  acceptance, with up to 8 attempts per iteration (`:664-732`). There are at most 25 iterations.
- **When the result is used**: only if it converged and the final cost is ≤ the initial cost
  (`experiments/rgbd_icp/apps/icp_sequence.cpp:163-166`). Otherwise the odometry trajectory is kept.

### 3.5 Outputs (the tracked set in `results/benchmarks/rgbd_icp/`; default `results/runs/rgbd_icp/`)

| File | Writer and content |
| --- | --- |
| `pair_metrics.csv` | per pair: status, matches, valid 3D pairs, RANSAC inliers and ratio, iterations, final pairs, RMSE, `converged`, stop reason per stage (`icp_sequence.cpp:104-108,138-144`) |
| `iteration_history.csv` | one row per accepted ICP update (section 3.3) (`:99-103,114-123`) |
| `loop_edges.csv` | accepted loop pairs `i,j` (`:150-155`) |
| `{baseline,robust,point,plane,pose_graph}_trajectory.txt` | `frame tx ty tz qx qy qz qw`, T_wc. The first field is the frame index, not a timestamp. Left-multiplied by GT pose 0 when `groundtruth.txt` exists (`:183-193`). `pose_graph` exists only if the graph result was used |
| `summary_metrics.csv` | per method: ATE RMSE, mean RPE rotation and translation, final position and rotation error, accepted and failed pairs; written only when GT exists (`:177-211`) |
| `solver_metadata.txt` | fixed text `point_solver=damped_gauss_newton`, `plane_solver=huber_damped_gauss_newton`, `initialization=feature_3d_ransac_svd`, `update=left_SO3_R3_retraction` (`:95-97`) |
| `selected_trajectory.txt` | `pose_graph` if the graph result was used, else `plane` (`:215-221`) |
| `run_complete.txt` | `frames=N failed_pairs=K`, written last (`:222-227`) |
| `audit_pair_15_16/{clouds.csv,stage_poses.csv}` | sampled source and target clouds with target normals, and the five stage poses for pair 15→16, written only if that pair is accepted (`:11-43,112-113`) |

These files in `results/benchmarks/rgbd_icp/` are not written by `icp_sequence`:

- `blender_trajectory.csv` and `blender_source.json` are written by `visualization/scripts/trajectory_data.py`
  (`:88-92`).
- `blender_consistency.csv` is written by `visualization/check_blender_consistency.py`.

`results/reports/figures/icp/` holds only the PNG figures and `trajectory_metrics.csv`, which
`tools/figures/render_outputs.py` (`icp()`, `:164-256`) draws or derives from `results/benchmarks/rgbd_icp/`. The
former copies of the ICP CSVs and of `audit_pair_15_16/` were removed as byte-identical duplicates.

## 4. Conventions

- **Pair solvers** return `T_i←j`: source frame *j* (the newer one) mapped into target frame *i*,
  `x_i = R x_j + t` (`registerFramesWithFeatures(target, source, …)`, `rgbd_icp.hpp:106-112`).
- **Camera poses** compose as `T_wc[j] = T_wc[i] · T_i←j` (`experiments/rgbd_icp/apps/icp_sequence.cpp:136`).
- **`Pose`** stores `(R, t)` with `apply(p) = R p + t`, `inverse() = (Rᵀ, −Rᵀt)` and
  `a*b = (R_a R_b, R_a t_b + t_a)` (`rgbd_icp.hpp:12-18`).
- **Depth PNG** is uint16. Metric depth = raw / 5000, and 0 means invalid (`rgbd_icp.hpp:24`,
  `experiments/rgbd_icp/src/rgbd_icp.cpp:243-249`).
- **Intrinsics** are read from `<dataset>/intrinsics.txt` as `width height fx fy cx cy`
  (`experiments/rgbd_icp/src/rgbd_icp.cpp:493-500`).
- **Ground truth** is never read during estimation. The file is opened only after all pair registration,
  the loop search and the pose graph have finished (`experiments/rgbd_icp/apps/icp_sequence.cpp:168-176`). If present:
  - GT pose 0 is used as the world anchor for the written trajectories;
  - GT supplies the metrics;
  - its row count must equal the frame count, otherwise the program fails.

  Ground truth is optional. Without it, trajectories are written in the frame-0 world and
  `summary_metrics.csv` has only a header. GT never supplies an initial pose, a correspondence filter or
  a loop acceptance test. The Bunny mesh is not an input to estimation either.
- **Metrics** (`experiments/rgbd_icp/apps/icp_sequence.cpp:186-211`):
  - ATE RMSE: `√(mean ‖t_est,i − t_GT,i‖²)` after the fixed frame-0 anchor, with no fitted alignment or
    scale.
  - RPE: mean over all consecutive pairs, fallbacks included, of the rotation angle and translation
    norm of `(GT_{i-1}⁻¹GT_i)⁻¹ (W_{i-1}⁻¹W_i)`.
  - Final rotation error: the angle of `R_est R_GTᵀ`.
  - A low pair-level error does not guarantee a low trajectory error, so report both, together with the
    failed pairs.

## 5. Safeguards

| Safeguard | Code |
| --- | --- |
| Proper rotation from SVD (det = +1, single-axis correction) | `experiments/rgbd_icp/src/rgbd_icp.cpp:59-63`; test `rgbd_icp_test.cpp:28-33` |
| Input validation (size ≥ 3, equal sizes, finite values) | `experiments/rgbd_icp/src/rgbd_icp.cpp:45-50` |
| Collinear RANSAC samples skipped | `:67-69,333-334` |
| Depth validity and 3×3 depth-edge rejection for features and clouds | `:237-251` |
| Distance gate, trimming and one-to-one target use for NN pairs | `:149-171` |
| Fixed damping, step-norm bound, motion gate, monotone backtracking | `:381-409,448-478` |
| Separate `accepted` / `converged` with an explicit stop reason | `:176-184`; `pair_metrics.csv` |
| Fallback motion flagged and counted in metrics, excluded from the graph | `experiments/rgbd_icp/apps/icp_sequence.cpp:124-137` |
| Loop edges need independent registration plus consistency | `experiments/rgbd_icp/src/rgbd_icp.cpp:584-593` |
| One anchor per graph component (no free gauge) | `:640-663` |
| Graph result used only if converged and not worse | `experiments/rgbd_icp/apps/icp_sequence.cpp:163-166` |
| Stale-output protection: `run_complete.txt`, `selected_trajectory.txt` and `pose_graph_trajectory.txt` deleted at start, the first two rewritten at the end | `:73-77,213-227` |

None of these checks proves that an accepted pose is globally correct. Ambiguous or planar geometry, low
overlap and depth errors can still produce a wrong but accepted estimate.

## 6. Measurements (synthetic Bunny, 36 frames, 35 pairs)

The source is `results/benchmarks/rgbd_icp/summary_metrics.csv`, `pair_metrics.csv`, `loop_edges.csv` and
`iteration_history.csv` as tracked. The results apply to this clean, noise-free synthetic sequence
only.

| Method (trajectory file) | ATE RMSE | mean RPE rot | mean RPE trans | final position error | final rotation error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature-pair SVD (`baseline`) | 245.49 mm | 1.569° | 12.86 mm | 370.77 mm | 41.41° |
| RANSAC + SVD (`robust`) | 156.31 mm | 0.872° | 7.61 mm | 236.33 mm | 27.05° |
| + point-to-point GN (`point`) | 45.82 mm | 0.706° | 3.62 mm | 57.29 mm | 8.74° |
| + point-to-plane Huber GN (`plane`) | 1.688 mm | 0.0280° | 0.204 mm | 1.187 mm | 0.183° |
| + verified loop + pose graph (`pose_graph`) | 1.200 mm | 0.0274° | 0.205 mm | 0.103 mm | 0.020° |

- **Pairs**: 35/35 accepted, 0 failed (`pair_metrics.csv`, `run_complete.txt`).
- **Loops**: one loop edge, 0→35 (`loop_edges.csv`). `selected_trajectory.txt` is `pose_graph`. The
  number of graph iterations is printed to stdout only and is not stored in a tracked file.
- **Feature support per pair** (means): 275.6 mutual matches, 214.5 valid 3D pairs, 208.5 RANSAC
  inliers.
- **Stop reasons**:
  - Point stage: `CONVERGED` in 16 pairs, `ITERATION_LIMIT` (18) in 19 pairs.
  - Plane stage: `CONVERGED` in 33 pairs, `ITERATION_LIMIT` (16) in 2 pairs.
- **Pair 15→16** (`iteration_history.csv`):
  - point stage: 16 updates, 13 changes of the assignment set;
  - plane stage: 6 updates, 3 changes.

The rows are cumulative stages: each one starts from the previous one. They are not isolated ablations
of single changes, and they do not compare against an equal-budget older frontend.

Earlier numbers that are no longer current:

- Point-ICP ATE 47.31 mm came from the run with incremental SVD updates, before Gauss–Newton.
- 0.2648 / 0.1772 / 0.0666 m for SVD / RANSAC / point came from the earlier frontend, before Harris.

`results/reports/figures/00_pipeline_flowchart.png` draws the implemented flow: the ICP stages, loop and
graph, plus the separate PnP branch (`tools/figures/render_outputs.py:266-288`). The current figures are
in `results/reports/figures/icp/`:

- `01_trajectory_stages.png`: stage trajectories plus the independent scratch PnP;
- `02_position_errors.png`;
- `03_iteration_updates.png`;
- `04_nearest_neighbour_refresh.png`;
- `05_cloud_alignment_before_after.png`;
- `06_pnp_refinement_before_after.png`: scratch PnP, not ICP;
- `trajectory_metrics.csv`.

The independent scratch PnP trajectory is plotted next to the ICP stages. Its numbers belong to the
production pipeline and are documented in `docs/components/pipeline.md`.

## 7. Limits

- **Sequence.** Only the 36-frame synthetic Bunny orbit has been measured. Real RGB-D data (noise,
  lens distortion, dynamic objects, larger scenes) has not been validated with this code.
- **Scope.** There is no mapping, no keyframes, no relocalisation, no bundle adjustment and no fusion
  with PnP. Each ICP solve is a pose-only update of one relative pose with fixed cloud coordinates.
- **Loop search.** It is tailored to this short orbit: gap ≥ 25 frames, at most 8 candidates chosen by
  odometry proximity. It is not a place-recognition system.
- **Point stage.** It frequently stops at its iteration limit (19 of 35 pairs). It is accepted by the
  support and RMSE gates, not by convergence.
- **Weak constraints.** Flat or symmetric geometry can leave directions weakly constrained. The damping
  stabilises the solve but adds no information.
- **Timing.** The code is single-threaded. No timing figures are recorded for `icp_sequence`.

### Open questions (no final ICP decision has been made)

Two ICP variants exist: the teammate's (this experiment; `reference/opencv_slambook2/src/tracking/icp.cpp`
wraps it) and main's book SVD + g2o variant with RANSAC (`reference/icp_main_variant/`). No target links both:
both define `pose_estimation_3d3d` and `bundleAdjustment`.
1. The comments in `reference/opencv_slambook2/tests/synthetic/{slam_trajectory,trajectory_validation}_test.cpp`
   still say "existing book code, unmodified", but `bundleAdjustment` is no longer book code. The tracked
   `data/synthetic_bunny/icp_trajectory.txt` was produced by the book ICP, so the current build cannot
   reproduce it, and `slam_trajectory_test` without `--out-dir` overwrites files in `data/synthetic_bunny/`
   (only main's variant has `--out-dir`).
2. Does `rigidSVD` handle reflections correctly where the book's `R = −R` did not? Test it directly;
   main's `icp_rigid_test` covers its own `FitRigid`.
3. Is `icp_sequence`'s pose-graph trajectory comparable with the sparse 3D-3D ICP evaluated on TUM? They solve
   different problems (dense RGB-D registration vs sparse feature correspondences).
4. Both `experiments/rgbd_icp/icp_cpp/` and the root build (`-DSLAM_WITH_OPENCV=ON`) build `rgbd_icp`; the
   `icp_cpp` build also compiles `reference/opencv_slambook2/src/tracking/icp.cpp` into `rgbd_icp_test` and is
   the only build that registers `blender_data_bridge`. Keep both until the ICP decision.
5. Which ICP should the TUM evaluator and the dashboard use for their "ICP" column? The archived runs come
   from main's variant.

## 8. Verification

`rgbd_icp_test` (`experiments/rgbd_icp/tests/rgbd_icp_test.cpp`) checks the following:

- SVD recovery of a known transform (`:23-26`);
- inverse composition (`:27`);
- `det R = +1` under a reflected input (`:28-33`);
- an empty input throws (`:34-40`);
- RANSAC with 30 % outliers recovers the pose and exactly 70 inliers (`:41-51`);
- an aligned input stays fixed and every update is logged with a non-increasing objective (`:52-63`);
- the analytic point and plane Jacobians match central finite differences in all six columns
  (`:64-76`);
- point GN and plane GN recover a small non-zero motion (`:77-96`);
- partial overlap with 15 outliers still gives ≥ 80 pairs (`:97-101`);
- a motion gate that is too tight is rejected with `NO_DESCENT` (`:102-106`);
- zero iterations is rejected (`:107-110`);
- no overlap is rejected (`:112-116`);
- a blank frame gives `INSUFFICIENT_FEATURES` (`:117-122`);
- a 4-node pose graph with one loop is corrected with anchor 0 fixed (`:123-136`);
- a disconnected graph does not get worse (`:137-141`).

Other checks:

- `tools/figures/validate_outputs.py` checks the ICP records against the figures:
  - every iteration row has ≥ 70 pairs, finite values and a non-increasing fixed-pair objective
    (`:16-22`);
  - the iteration counts match `pair_metrics.csv` (`:23-25`);
  - pair 15 has more than one assignment hash in both stages (`:26-28`);
  - the plotted ATE and final errors agree with `summary_metrics.csv` within 2e-8 (`:29-33`);
  - the stage rotations in `audit_pair_15_16/stage_poses.csv` are proper (`:34-36`).

  It also checks the PNG set and hashes against `image_manifest.json` (`:11`). **Known, pre-existing
  failures** of the tracked manifest (both visible, neither hidden):
  - it lists 43 PNGs, but 55 are on disk: the 12 `orb_benchmark/keypoint_visuals/*.png` are missing from it;
  - `orb_benchmark/01_benchmark_summary.png` and `orb_benchmark/02_counts_and_timing.png` do not match their
    recorded hashes, although both files are byte-identical to the versions before the repository
    reorganisation.

  The fix is a deliberate regeneration of the manifest (`python3 tools/figures/render_outputs.py`, which
  rewrites it), reviewed as a change to a tracked report. With those two checks disabled, the ICP checks
  pass.
- `visualization/tests/test_blender_data.py` (CTest `blender_data_bridge` in the standalone build)
  checks the C++ to Blender data bridge.
- The scene-versus-data comparison of the Blender consistency checker is
  `results/benchmarks/rgbd_icp/blender_consistency.csv` (section 9).

Reported by the teammate and not re-run here:

- an independent Release build with GCC 13.3, OpenCV 4.6.0 and Eigen 3.4.0. The pixi environment pins
  Eigen ≥ 5 and OpenCV 4.x, so that environment differs.
- a second full run without `groundtruth.txt` that gave identical pair diagnostics, loop indices and all
  five trajectories once the GT anchor was undone.
- the original g2o/Sophus suite was not built in that environment.

## 9. Blender integration

`bash visualization/rebuild_blender.sh` (or `pixi run blender-rebuild`, `pixi.toml:17`) runs from the
repository root and performs these steps:

1. Checks the Python plotting dependencies (`:6`).
2. Builds the root project without OpenCV into `build-scratch/`, runs its CTest, and runs
   `scratch_pipeline` (`:7-10`).
3. Runs `feature_audit` plus its render and validate scripts for pairs 0→1 and 15→16 (`:12-17`).
4. Builds and tests `experiments/rgbd_icp/icp_cpp` into `build-icp/` (`:18-20`).
5. Runs `icp_sequence`, accepting exit code 2 with a warning (`:21-29`).
6. Runs `orb_benchmark` and both ORB renderers (`:30-32`).
7. Runs `render_outputs.py` and `validate_outputs.py` (`:33-34`).
8. Builds the scene headless and runs `visualization/check_blender_consistency.py`, which writes
   `results/benchmarks/rgbd_icp/blender_consistency.csv` (`:35-37`).

The scene is `results/blender/bunny_slam_demo.blend`. Open it with
`blender -y results/blender/bunny_slam_demo.blend`. Space plays the animation, Numpad 0 looks through
the active camera, and the N-panel "Trajectories" tab toggles methods and cameras.

| Method | Colour | Data file |
| --- | --- | --- |
| Ground truth | green | `data/synthetic_bunny/groundtruth.txt` |
| Corrected ICP | magenta | `results/benchmarks/rgbd_icp/<selected>_trajectory.txt` |
| Scratch PnP | cyan | `tests/fixtures/bunny_t0_scene/trajectory.csv` (frozen T0 run of the production pipeline) |
| Legacy reference PnP | orange | `data/synthetic_bunny/pnp_trajectory.txt` (OpenCV `solvePnPRansac`, frozen; a labelled baseline, not a same-correspondence comparison with scratch PnP) |

Colours are from `visualization/scripts/build_scene.py:151-158`.

How the data reaches Blender (`visualization/scripts/trajectory_data.py`):

- It refuses to run without `results/benchmarks/rgbd_icp/run_complete.txt` (`:47-48`).
- It reads only the trajectory named in `selected_trajectory.txt`, never a leftover graph file
  (`:49-51`).
- It requires the GT, legacy PnP and ICP frame counts to agree, and frame 0 of each to equal GT frame 0
  (`:55-62`).
- It requires the scratch trajectory to exist (`:79-81`).
- It writes `blender_trajectory.csv` and `blender_source.json` with a SHA-256 fingerprint over the
  intrinsics, mesh, GT, legacy PnP, selected ICP trajectory, `pair_metrics.csv`,
  `selected_trajectory.txt` and the scratch trajectory (`:84-92`).

`visualization/check_blender_consistency.py` compares the scene against that fingerprint. It also
checks the camera poses, curves, HUD values and intrinsics.

Final image rendering is not part of these checks. The repository contains a scene, not a rendered
video.

## 10. References

- Point-to-point and point-to-plane ICP objectives:
  https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
- Robust (Huber/IRLS) weighting for ICP:
  https://www.open3d.org/docs/0.19.0/tutorial/pipelines/robust_kernels.html
