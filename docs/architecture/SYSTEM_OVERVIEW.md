# System overview

This document describes what the repository implements today, as read from the code and the CMake files
(root `CMakeLists.txt`, `cmake/*.cmake`, `reference/opencv_slambook2/CMakeLists.txt`,
`reference/icp_main_variant/CMakeLists.txt`, `experiments/rgbd_icp/CMakeLists.txt`,
`experiments/rgbd_icp/icp_cpp/CMakeLists.txt`). All paths are relative to the repository root.
Where any document disagrees with the code, the code is the authority.

For the per-target details (headers, functions, tests, dependency graph) see `MODULE_REFERENCE.md`.

---

## 1. What the system implements

The production system is **frame-to-frame RGB-D visual odometry written from scratch** in C++17 with Eigen
and the C++ standard library only. For every consecutive frame pair i -> i+1 it estimates the metric relative
pose T_{i+1<-i} (convention X_{i+1} = R X_i + t, `include/slam/pipeline/pipeline.hpp:18`) and chains these
poses into a camera trajectory.

The per-pair chain, as executed by `scratch::ProcessPair` (`src/pipeline/pipeline.cpp:56-193`) after
`scratch::ExtractFeatures` has been run on both frames (`apps/pipeline_main.cpp:307,315`):

| Step | What happens | Code |
|---|---|---|
| Image I/O | PNG decoded by a project-owned decoder (zlib/DEFLATE, CRC-32, Adler-32, scanline filters) | `include/slam/pipeline/png.hpp`, `src/pipeline/png.cpp` |
| Grayscale | BT.601 luma, 14-bit fixed point | `include/slam/features/types.hpp` (`GrayFromRGB`) |
| Pyramid | 8 levels x 1.2, bilinear, pixel-centre aligned | `include/slam/features/pyramid.hpp:25-26` |
| Detection | FAST-9 with score, 3x3 NMS, border 16; optional Harris ranking and a total feature budget across levels; optional per-cell threshold fallback | `include/slam/features/fast.hpp`, `include/slam/features/multiscale.hpp:39-42`, `src/features/fast.cpp:54-69` |
| Orientation | intensity centroid, radius 15 | `include/slam/features/orientation.hpp` |
| Description | Gaussian-smoothed rotated BRIEF, 256 bits, fixed seeded pattern | `include/slam/features/brief.hpp` |
| Matching | brute-force Hamming + distance filter `max(2 d_min, 30)`; or mutual + ratio (`MatchMutualRatio`); or the ranked matcher (`MatchRanked`) | `src/pipeline/frontend.cpp:77-101` |
| Depth association | depth of frame i at the truncated keypoint pixel, back-projected with K; optional 3x3 depth-edge rejection | `src/pipeline/frontend.cpp:115-142` |
| Two-view stage (diagnostic) | 8-point fundamental matrix in pixels; RANSAC 8-point essential matrix with Sampson distance, decomposition, cheirality; scale-free, **reported only, never accumulated** | `src/pipeline/pipeline.cpp:76-99`, `include/slam/pipeline/pipeline.hpp:18-20` |
| PnP + RANSAC | linear DLT PnP (6-point minimal sample) inside RANSAC with a deterministic `std::mt19937` sampler; refit on inliers | `include/slam/pipeline/ransac.hpp`, `include/slam/pipeline/pnp.hpp`, `src/pipeline/pipeline.cpp:101-128` |
| LM refinement | Levenberg-Marquardt on SE(3) (left perturbation) minimising pixel reprojection error over the fixed RANSAC inliers; optional Huber IRLS weight | `include/slam/pipeline/refine.hpp`, `src/pipeline/refine.cpp:80-85`, `src/pipeline/pipeline.cpp:129-146` |
| Inlier re-selection | (T0 only) re-threshold residuals under the refined pose, refine once more, require >= 6 inliers | `src/pipeline/pipeline.cpp:147-169` |
| Pose-validity gate | always evaluated; decides acceptance only when enabled | `include/slam/pipeline/validity.hpp`, `src/pipeline/pipeline.cpp:178-191` |
| Trajectory accumulation | T_wc[0] = anchor; T_wc[k+1] = T_wc[k] * T_{k+1<-k}^-1 for accepted pairs; otherwise the previous pose is held | `include/slam/pipeline/accumulate.hpp:4-5,39-49` |

Evaluation is separate from estimation:

* `apps/pipeline_main.cpp:339-365` computes per-pair and absolute errors against ground truth after all pairs
  have been estimated and writes them into `pairs.csv`, `trajectory.csv` and `summary.txt`.
* `scratch_baseline_lib` / `scratch_experiment_lib` (`include/slam/evaluation/`) are a C++ regression and
  experiment harness. `include/slam/evaluation/baseline.hpp:14-17` states the isolation rule: the estimation
  entry point `MeasureFixture` receives no ground truth; `ScoreMeasurement` adds errors afterwards.
* Python tooling under `tools/evaluation/` computes TUM-style metrics and dashboards from the files the C++
  programs write.

**Use of ground truth.** No estimation step reads ground truth (`src/pipeline/pipeline.cpp` and
`src/pipeline/frontend.cpp` take no pose input). There is one non-estimation use: the world frame of the
output trajectory is anchored at the first ground-truth pose, `AccumulateTrajectory(Twc_gt[0], steps)`
(`apps/pipeline_main.cpp:335`; the frame-0 source is labelled `"gt_anchor"`, `include/slam/pipeline/accumulate.hpp:45`).

**Datasets.** `scratch_pipeline` reads the synthetic Bunny sequence `data/synthetic_bunny/` by default
(`apps/pipeline_main.cpp:195-204`), or a TUM RGB-D sequence with `--tum-frames CSV --tum-sequence DIR
--intrinsics fx,fy,cx,cy` (`apps/pipeline_main.cpp:42-45,205-239`). The TUM mode needs a `frames.csv` that is
written by the reference program `tum_trajectory_eval` (see section 7, inconsistencies).

## 2. Not implemented (production code)

A grep of `include/slam`, `src/` and `apps/` for `keyframe`, `loop clos*`, `bundle`, `reloc*`, `mapping`,
`map point`, `landmark` and `pose graph` returns no match except one unrelated comment
(`include/slam/features/pyramid.hpp:13`, "Coordinate mapping"). Reading the code confirms:

* **No map.** No 3D landmarks persist beyond a pair; the 3D points are re-created per pair from frame-i depth
  (`src/pipeline/frontend.cpp:138-142`).
* **No keyframes.** Every consecutive pair i -> i+1 is processed (`apps/pipeline_main.cpp:310-321`).
* **No loop closure, no pose-graph optimisation.**
* **No bundle adjustment.** Refinement is pose-only over one pair (`include/slam/pipeline/refine.hpp`).
* **No relocalisation / tracking-failure recovery.** A failed or rejected pair simply holds the previous pose
  (`include/slam/pipeline/accumulate.hpp:4-5`).
* **No multi-frame feature tracking.** Matching is between two frames only.
* **No use of the two-view essential-matrix estimate** in the trajectory (diagnostic only).
* **No lens distortion model** (`include/slam/pipeline/projection.hpp`: "no lens distortion").
* **No dense / ICP alignment** in production.
* **No real-time or online interface**: programs run offline over image files.

Outside production, the experimental library `rgbd_icp` does contain a loop-candidate search and a pose-graph
optimiser (`experiments/rgbd_icp/include/tracking/rgbd_icp.hpp:113-116`, used in
`experiments/rgbd_icp/apps/icp_sequence.cpp:149-166`). These are not part of the production pipeline.

## 3. The four code families

| Family | Directories | Language | Role |
|---|---|---|---|
| **Production** | `include/slam/`, `src/`, `apps/`, `tests/unit/`, `tests/pipeline/`, `tests/fixtures/`, `cmake/feature_core.cmake` | C++17, Eigen + std only | the estimator, its CLI, its tests and the C++ evaluation harness |
| **Reference** | `reference/opencv_slambook2/`, `reference/icp_main_variant/` | C++17, OpenCV + g2o + Sophus (+ Eigen) | slambook2-derived reference pipeline, demos, TUM evaluator; comparison only |
| **Experiments** | `experiments/rgbd_icp/`, `experiments/orb_opencv/`, `experiments/orb_extractor_benchmark/`, `experiments/depth_edge/` | C++17 (+ Python in the standalone benchmark) | provisional or one-off studies |
| **Tooling** | `tools/`, `visualization/` | Python, shell, Blender Python | analysis, figures, dashboards, Blender scenes; reads C++ outputs, never estimates geometry |

### Component table

| Directory | Language | Dependencies | Build target(s) | Status |
|---|---|---|---|---|
| `include/slam/features/`, `src/features/` | C++17 | std only | `feature_core` (`cmake/feature_core.cmake:7-15`) | production |
| `include/slam/geometry/fundamental.hpp` | C++17 header-only | Eigen | none (header; included by `scratch_core` and reference `geometry`) | production |
| `include/slam/pipeline/`, `src/pipeline/` | C++17 | Eigen, `feature_core` | `scratch_core` (`CMakeLists.txt:27-32`) | production |
| `apps/pipeline_main.cpp` | C++17 | `scratch_core` | `scratch_pipeline` (`CMakeLists.txt:47-49`) | **the production application** |
| `apps/milestones/` | C++17 | Eigen (compiles `pnp.cpp`, `projection.cpp` directly) | `pnp_from_scratch`, `pnp_full_sequence` (`CMakeLists.txt:35-44`) | production history (self-checking milestones) |
| `apps/feature_audit.cpp` | C++17 | `scratch_core` | `feature_audit` (`CMakeLists.txt:125-127`) | production diagnostic |
| `include/slam/evaluation/`, `src/evaluation/` | C++17 | `scratch_core` | `scratch_baseline_lib`, `scratch_experiment_lib` (`CMakeLists.txt:59-66`) | evaluation harness (not in the per-frame path) |
| `apps/baseline_main.cpp`, `apps/experiments_main.cpp` | C++17 | the two evaluation libraries, Threads | `scratch_baseline`, `scratch_experiments` (`CMakeLists.txt:67-73`) | evaluation |
| `tests/unit/`, `tests/pipeline/`, `tests/fixtures/` | C++17 | `feature_core` / `scratch_core` / evaluation libs | `features_*_test`, `test_*` (`CMakeLists.txt:81-122`) | production tests |
| `experiments/depth_edge/` | C++17 | `scratch_core` | `depth_edge_experiment` (`CMakeLists.txt:53-55`) | experiment, **built in the default configuration** |
| `reference/opencv_slambook2/` | C++17 | OpenCV, Eigen, g2o, Sophus; `feature_core`; `rgbd_icp` | `camera`, `features`, `geometry`, `optimization`, `tracking`, `render`, `dataset` + demos/tests | reference (OpenCV build only) |
| `reference/icp_main_variant/` | C++17 | as above, without `rgbd_icp` | `tracking_main_icp`, `icp_rigid_test`, `tum_trajectory_eval`, `*_main_icp` | reference (OpenCV build only) |
| `experiments/rgbd_icp/` | C++17 | OpenCV (core, imgcodecs), Eigen, `feature_core` | `rgbd_icp`, `icp_sequence`, `rgbd_icp_test` | experimental, provisional |
| `experiments/rgbd_icp/icp_cpp/` | CMake | same | standalone build of the above + `orb_benchmark` | experimental |
| `experiments/orb_opencv/` | C++17 | `rgbd_icp`, `opencv_features2d` | `orb_benchmark` (`cmake/orb_benchmark.cmake:1-10`) | experimental benchmark |
| `experiments/orb_extractor_benchmark/` | C++17 + Python | std only (own vendored feature copy) | `feature_core` (vendored), `orb_extract` (own `CMakeLists.txt`) | historical standalone snapshot |
| `tools/evaluation/` | Python | NumPy, Matplotlib (pixi `results` env) | none | tooling |
| `tools/pipeline_analysis/`, `tools/figures/`, `tools/evaluation/run_diagnostics.py`, `tools/evaluation/run_reference_suite.sh` | Python, shell | NumPy, Matplotlib | none | tooling |
| `visualization/` | Blender Python, shell | Blender | none (Python test `blender_data_bridge` in the `icp_cpp` build) | tooling |

## 4. The production decision

`scratch_pipeline` starts from `PipelineParams{}` and immediately applies `ApplyImprovedDefaults`
(`apps/pipeline_main.cpp:98-101`). That function is the T0 configuration (`src/pipeline/pipeline.cpp:39-48`):

| Setting | T0 value (CLI default) | Struct default `PipelineParams{}` (L0 / `--legacy`) |
|---|---|---|
| Harris ranking | on (`selection = {true, 1400}`, `pipeline.cpp:40`) | off (`multiscale.hpp:39-40`) |
| Feature budget | 1400 total across levels (`pipeline.cpp:40`) | 500 per level (`fast.hpp:18`) |
| FAST threshold | 10 (`pipeline.cpp:41`) | 20 (`frontend.hpp:47`; `FastParams` itself defaults to 30, `fast.hpp:15`) |
| Matcher | mutual + ratio 0.87, `MatchMutualRatio` (`pipeline.cpp:42`, `frontend.hpp:36-37`, `frontend.cpp:81-83`) | NN + `max(2 d_min, 30)` |
| Depth-edge rejection | 12 mm, 3x3 window (`pipeline.cpp:43`, `frontend.cpp:120-137`) | off (`frontend.hpp:38`) |
| LM loss | Huber, delta 2 px (`pipeline.cpp:44`) | least squares (`refine.hpp:26`) |
| Inlier re-selection | on (`pipeline.cpp:45`) | off (`pipeline.hpp:40`) |
| PnP RANSAC | 650 iterations, 3 px (`pipeline.cpp:46-47`) | 300 iterations, 8 px, seed 12345 (`ransac.hpp:32-34`) |
| Essential RANSAC | 2000 it, 2 px Sampson, seed 54321 (`pipeline.hpp:48-51`) | same |
| Pose-validity gate | **off** (`pipeline.hpp:47`; enable with `--pose-gate`, `pipeline_main.cpp:168-169`) | off |

Alternatives retained in the code (none is the default):

* `--stage2-frontend` -> `ApplyStage2Frontend` (`src/pipeline/pipeline.cpp:50-54`): main's earlier Stage-1/2
  front end (M0): `PipelineParams{}` plus FAST per-cell fallback 30 px / threshold 7.
* `--legacy` -> `PipelineParams{}` (`apps/pipeline_main.cpp:112-113`), the L0 configuration.
* T0-ransac (`--pnp-iterations 300 --pnp-threshold 8`): an experimental alternative; defined as
  integration configuration `T0-ransac` in `src/evaluation/experiment.cpp:140-141`.
* `MatchRanked` (Stage 3): opt-in with `--ranked-matcher` and the `--match-*` options
  (`apps/pipeline_main.cpp:152-165`); takes precedence over mutual/ratio (`include/slam/pipeline/frontend.hpp:43-46`).
* FAST per-cell fallback: opt-in with `--fast-cell C --fast-min-threshold T2` (`apps/pipeline_main.cpp:134-137`).
* Every T0 component can be switched off individually for ablations (`--no-harris`, `--feature-budget`,
  `--no-mutual-ratio`, `--ratio`, `--depth-edge-m`, `--huber-delta`, `--no-reselect`;
  `apps/pipeline_main.cpp:138-151`).
* The full catalogue of named configurations (M0..M3L, L0, T0, T0G, T0R, T0M, T0Q, T0GR, T0E, T0H, the T0
  leave-one-outs, T0C, T0CG) lives in `ApplyIntegrationConfig` (`src/evaluation/experiment.cpp:96-150`) and is
  used by `scratch_experiments` and `scratch_baseline --config`; it is not used by `scratch_pipeline`.

Note that the C++ regression lock `scratch_baseline` does **not** run T0: it uses main's Stage-2
configuration with the gate on (`apps/baseline_main.cpp:48-52`).

## 5. The two ICP variants

ICP (3D-3D alignment) is not part of the production pipeline. Two reference variants exist, and both are
kept because no final ICP decision has been made (`experiments/rgbd_icp/CMakeLists.txt:1`: "provisional ICP;
no final ICP decision yet").

| | Teammate's variant | Main's variant |
|---|---|---|
| Source | `reference/opencv_slambook2/src/tracking/icp.cpp` (in library `tracking`) | `reference/icp_main_variant/src/icp.cpp` (in library `tracking_main_icp`) |
| `pose_estimation_3d3d` | wraps `rgbd_icp::rigidSVD` (`icp.cpp:8-24`) | the book's SVD |
| `bundleAdjustment` | **no g2o**: calls `pose_estimation_3d3d` again, because the fixed unweighted objective has a closed-form solution (`icp.cpp:26-33`) | g2o Levenberg-Marquardt with `EdgeProjectXYZRGBDPoseOnly` (`icp_main_variant/src/icp.cpp:66-101`) |
| Extra API | none (points to `rgbd_icp` for dense ICP) | `FitRigid`, `EstimateRigidRansac` (`reference/icp_main_variant/include/tracking/icp.hpp:30-44`) |
| Links | `rgbd_icp` (PRIVATE, `reference/opencv_slambook2/CMakeLists.txt:58-60`) | no `rgbd_icp` (`reference/icp_main_variant/CMakeLists.txt:23-25`) |
| Consumers | the 11 `add_test_exe` demos, `essential_matrix_test`, `fast_test`, `brief_test`, `vo_pipeline_test` | `icp_rigid_test`, `tum_trajectory_eval`, `slam_trajectory_test_main_icp`, `trajectory_validation_test_main_icp` |

Both define the same symbols `pose_estimation_3d3d` and `bundleAdjustment` and both have a header named
`tracking/icp.hpp`; they are therefore never linked into the same executable, and every main-variant target
puts `reference/icp_main_variant/include` first with `BEFORE` (`reference/icp_main_variant/CMakeLists.txt:1-12,21,30`).
The TUM evaluator used by the Python tooling (`tum_trajectory_eval`) uses main's variant
(`EstimateRigidRansac` + g2o, `reference/opencv_slambook2/tests/tum/tum_trajectory_eval.cpp:5-8`).

Separately, the teammate's dense RGB-D ICP library `rgbd_icp` (`experiments/rgbd_icp/`) provides point-to-point
and point-to-plane ICP, feature RANSAC, loop candidates and pose-graph optimisation, run by `icp_sequence`.
Its output in `results/benchmarks/rgbd_icp/` is what the Blender Bunny scene shows as the "corrected ICP" trajectory.

## 6. Build configurations

| Configuration | Command (from the repository root) | What is built |
|---|---|---|
| Production only (default) | `cmake -S . -B build && cmake --build build` (`SLAM_WITH_OPENCV` defaults to OFF, `CMakeLists.txt:16`) | `feature_core`, `scratch_core`, `scratch_baseline_lib`, `scratch_experiment_lib`, `scratch_pipeline`, `scratch_baseline`, `scratch_experiments`, `feature_audit`, `pnp_from_scratch`, `pnp_full_sequence`, `depth_edge_experiment`, and all production tests. Requires Eigen3 and Threads only. |
| With OpenCV | `cmake -S . -B build -DSLAM_WITH_OPENCV=ON` (also the pixi `configure` task, `pixi.toml:11`) | additionally `experiments/rgbd_icp` (`rgbd_icp`, `icp_sequence`, `rgbd_icp_test`), `reference/opencv_slambook2` (libraries and demos) and, by default (`BUILD_MAIN_ICP_VARIANT=ON`, `reference/opencv_slambook2/CMakeLists.txt:147-150`), `reference/icp_main_variant`, and `orb_benchmark` (`BUILD_ORB_BENCHMARK=ON`, `cmake/orb_benchmark.cmake:1`). Requires OpenCV, g2o, Sophus (`CMakeLists.txt:130-134`). |
| `icp_cpp` standalone | `cmake -S experiments/rgbd_icp/icp_cpp -B build-icp` (pixi `icp-configure`, `pixi.toml:14`) | its own `feature_core` (same `cmake/feature_core.cmake`), `rgbd_icp`, `icp_sequence`, `rgbd_icp_test` (also compiles the teammate's `reference/opencv_slambook2/src/tracking/icp.cpp`, `icp_cpp/CMakeLists.txt:21-22`), `orb_benchmark`, and the Python test `blender_data_bridge`. Needs only OpenCV core + imgcodecs (+ features2d for `orb_benchmark`) and Eigen; no g2o/Sophus. |
| `orb_extractor_benchmark` standalone | `cmake -S experiments/orb_extractor_benchmark -B <dir>` | its own vendored `feature_core` (a snapshot whose `matcher` and `multiscale` differ from the production files) and `orb_extract`. No dependency on the rest of the repository. |

All executables of the root build land directly in the build directory (`CMAKE_RUNTIME_OUTPUT_DIRECTORY`,
`CMakeLists.txt:11`). Production tests run from the repository root (`WORKING_DIRECTORY ${_repo}`,
`CMakeLists.txt:79,92`); programs find the root by walking up to `data/synthetic_bunny/intrinsics.txt`
(`FindRepo`, `include/slam/pipeline/common.hpp:73-81`). The reference demos are built but not registered
with CTest (no `add_test` in `reference/`).

## 7. Where outputs go

| Program | Default output | Source |
|---|---|---|
| `scratch_pipeline` | `results/runs/pipeline/` (ignored, `.gitignore:44`): `pairs.csv`, `trajectory.csv`, `trajectory.txt`, `correspondences/`, `keypoints/`, `summary.txt`, plus `pose_validity.csv` (with `--pose-gate`), `matcher.csv`/`matches/` (ranked matcher), `frames.csv`/`command.txt` (TUM mode) | `apps/pipeline_main.cpp:16-26,262-273` |
| `pnp_from_scratch`, `pnp_full_sequence` | `results/runs/milestones/` | `apps/milestones/main.cpp:255`, `apps/milestones/sequence_main.cpp:195` |
| `depth_edge_experiment` | `results/runs/depth_edge/` | `experiments/depth_edge/depth_edge_experiment.cpp:173` |
| `scratch_baseline` | `results/runs/scratch_baseline/stage1`, `results/runs/scratch_baseline/stage3_seeds` (ignored) | `apps/baseline_main.cpp:135,146` |
| `scratch_experiments` | `--out DIR` (required; must not exist) | `apps/experiments_main.cpp:1-8` |
| `feature_audit` | `results/reports/figures/frontend/pair_<i>_<j>/` (inside the source tree) | `apps/feature_audit.cpp:47` |
| `icp_sequence` | `results/runs/rgbd_icp/` (default second argument); the tracked `results/benchmarks/rgbd_icp/` (read by the Blender scene) only when passed explicitly | `experiments/rgbd_icp/apps/icp_sequence.cpp:66-67` |
| ctest `milestone3_scratch_pipeline`, `scratch_baseline_lock` | build tree | `CMakeLists.txt:98-112` |
| Python tooling | `results/runs/...` (runtime), `results/reports/...` (tracked reports) | e.g. `tools/evaluation/diagnostics/make_diagnostics.py` header |

The Blender scene reads no runtime output: `results/benchmarks/rgbd_icp/` (tracked) and the frozen T0 trajectory
`tests/fixtures/bunny_t0_scene/trajectory.csv` (`visualization/scripts/trajectory_data.py:15-16`).

## 8. Component diagram

```mermaid
graph TD
  subgraph Production["Production (Eigen + std, no OpenCV)"]
    FC["feature_core<br/>FAST, orientation, BRIEF,<br/>matchers, pyramid, multiscale"]
    FH["geometry/fundamental.hpp<br/>(header-only)"]
    SC["scratch_core<br/>PNG, frontend, PnP, essential,<br/>RANSAC, LM refine, validity, ProcessPair"]
    APP["scratch_pipeline<br/>(production app)"]
    EVAL["scratch_baseline_lib<br/>scratch_experiment_lib"]
    EVA["scratch_baseline<br/>scratch_experiments"]
    AUD["feature_audit, milestones,<br/>depth_edge_experiment"]
    FC --> SC
    FH -.include.-> SC
    SC --> APP
    SC --> EVAL --> EVA
    SC --> AUD
  end
  subgraph Experiments["Experiments (OpenCV)"]
    ICP["rgbd_icp<br/>(dense RGB-D ICP, loops, pose graph)"]
    ICPS["icp_sequence"]
    ORB["orb_benchmark"]
    ICP --> ICPS
    ICP --> ORB
  end
  subgraph Reference["Reference (OpenCV + g2o + Sophus)"]
    REF["camera, features, geometry,<br/>optimization, tracking, render, dataset"]
    MICP["tracking_main_icp<br/>tum_trajectory_eval"]
    REF --> MICP
  end
  FC --> ICP
  FC --> REF
  FH -.include.-> REF
  ICP --> REF
  subgraph Tooling["Tooling (Python / Blender)"]
    TOOLS["tools/evaluation, pipeline_analysis,<br/>figures, diagnostics"]
    VIS["visualization (Blender)"]
  end
  APP -. files .-> TOOLS
  APP -. trajectory.csv .-> VIS
  ICPS -. results/benchmarks/rgbd_icp .-> VIS
  MICP -. frames.csv / pairs.csv .-> TOOLS
  MICP -. frames.csv .-> APP
```

Solid arrows are CMake link dependencies; dotted arrows are file (data) dependencies or header-only includes.

## 9. Architectural inconsistencies observed

1. The reference library `tracking` links the experimental `rgbd_icp`
   (`reference/opencv_slambook2/CMakeLists.txt:58-60`; used by `reference/opencv_slambook2/src/tracking/icp.cpp:2,16`).
   A reference component therefore depends on an experiment, and the reference build cannot exist without
   `experiments/rgbd_icp` (the root adds it first, `CMakeLists.txt:131`).
2. The standalone `icp_cpp` build compiles a reference source file into an experiment test:
   `experiments/rgbd_icp/icp_cpp/CMakeLists.txt:21-22` adds `reference/opencv_slambook2/src/tracking/icp.cpp`
   to `rgbd_icp_test`, although `experiments/rgbd_icp/tests/rgbd_icp_test.cpp` calls only `rgbd_icp::rigidSVD`
   and never `pose_estimation_3d3d`/`bundleAdjustment`. The root build of the same test does not
   (`experiments/rgbd_icp/CMakeLists.txt:11-12`).
3. `scratch_core` includes `geometry/fundamental.hpp` (`src/pipeline/pipeline.cpp:2`) but declares no include
   path for it; it resolves only because `feature_core` publishes `include/slam` (`cmake/feature_core.cmake:15`).
4. `SLAM_WITH_OPENCV=OFF` is described as "the production pipeline only" (`CMakeLists.txt:13`), but it also
   builds the experiment `depth_edge_experiment` from `experiments/depth_edge/` (`CMakeLists.txt:53-55`).
5. Production TUM runs depend on a reference program's output: `scratch_pipeline --tum-frames` reads a
   `frames.csv` "written by tum_trajectory_eval" (`apps/pipeline_main.cpp:42-45`), which exists only in the
   OpenCV build.
6. Stale comments: `include/slam/evaluation/baseline.hpp:6-8` says the harness runs "ProcessPair with default
   PipelineParams, i.e. exactly what scratch_pipeline does", but `scratch_baseline` uses
   `ApplyStage2Frontend` + gate on (`apps/baseline_main.cpp:50-52`) and `scratch_pipeline` uses T0
   (`apps/pipeline_main.cpp:101`). `experiments/depth_edge/depth_edge_experiment.cpp:172` labels
   `PipelineParams{}` as "the production defaults"; it is L0. The `PipelineParams` field comments
   (`include/slam/pipeline/pipeline.hpp:35-37`: FAST 20, 500, 300 it / 8 px) describe the struct defaults, not
   the production CLI defaults.
7. CLI order dependence: `--stage2-frontend` and `--legacy` reset all parameters, including the gate
   (`src/pipeline/pipeline.cpp:51`, `apps/pipeline_main.cpp:113`), so `--pose-gate --stage2-frontend` runs
   without the gate while `--stage2-frontend --pose-gate` runs with it.
8. A ctest writes into the source tree: `feature_audit_pair_0_1` (`CMakeLists.txt:127`) writes
   `results/reports/figures/frontend/pair_0_1/` (`apps/feature_audit.cpp:47`); the milestone tests write into
   the ignored `results/runs/milestones/` (`apps/milestones/main.cpp:255`, `apps/milestones/sequence_main.cpp:195`).
   The comment at `CMakeLists.txt:97` only guarantees this for `scratch_pipeline`.
9. Framing in `visualization/README.md:27`: the production PnP trajectory is labelled the "legacy
   comparison" and the experimental ICP the "corrected" trajectory, which does not match the production
   decision (section 4).
10. The milestone-1 and -2 tests read the frozen reference-pipeline export
    `tests/fixtures/reference_baseline/correspondences/` (`apps/milestones/main.cpp:132`), not the current reference build.
