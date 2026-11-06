# Learning path

A study sequence from "what is in this repository" to "I understand every step of the RGB-D visual odometry".
Each stage lists what to read, in order, what to be able to explain afterwards, and a few checks you can
answer from the code. Keep `docs/FILE_GUIDE.md` open: every file named here has an entry there.

Paths are relative to the repository root. "T0" is the production configuration
(`ApplyImprovedDefaults`, `src/pipeline/pipeline.cpp:39-48`).

---

## Stage 0 — Orientation (1–2 hours)

Read:
1. `README.md`: what is production, what is experimental, how to build.
2. `docs/architecture/SYSTEM_OVERVIEW.md` §1–4: the per-pair chain, what is *not* implemented, the four code
   families, the production decision.
3. `docs/architecture/MODULE_REFERENCE.md` §13–14: the CMake target graph and the layering rules.
4. The root `CMakeLists.txt`: which target is built from which files; `SLAM_WITH_OPENCV`.

Be able to explain:
- why the production code (`include/slam`, `src/`, `apps/`) builds without OpenCV, and what needs OpenCV;
- that the system is frame-to-frame odometry: no map, no loop closure, no bundle adjustment in production;
- the difference between `scratch_pipeline` (production), `scratch_baseline` / `scratch_experiments`
  (evaluation harness) and the reference demos.

Check: which target compiles `src/pipeline/ransac.cpp`? Which executables link `rgbd_icp`?

## Stage 1 — Coordinate conventions (2 hours; do not skip)

Read `docs/architecture/COORDINATE_CONVENTIONS.md` completely, then the code it cites:
1. `include/slam/pipeline/common.hpp`: `Pose` (X_b = R X_a + t), `ReadGroundtruthTwc` (TUM T_wc), quaternion
   handling.
2. `include/slam/pipeline/projection.hpp`, `src/pipeline/projection.cpp`: `Intrinsics`, `projectPoint`.
3. `include/slam/pipeline/accumulate.hpp`: `Compose`, `Inverse`, `AccumulateTrajectory`.

Be able to explain:
- every estimator in production returns T_{i+1←i} (X_{i+1} = R X_i + t);
- the trajectory is T_wc[k+1] = T_wc[k] · T_{k+1←k}⁻¹, anchored at the ground-truth first pose
  (`apps/pipeline_main.cpp:335`), with the previous pose held when a pair is not accepted;
- the ICP variants return the opposite direction (T_{i←j}) — §10 and §12 of the conventions document.

Check: derive the accumulation formula yourself from X_{k+1} = R X_k + t and X_w = R_wc X_c + t_wc.

## Stage 2 — Data input (2 hours)

Read `docs/architecture/DATA_FLOW.md` §0–2.3, then:
1. `data/synthetic_bunny/README.md` and the files in `data/synthetic_bunny/`.
2. `include/slam/pipeline/png.hpp`, `src/pipeline/png.cpp`: the project's own PNG decoder (DEFLATE,
   filters, 16-bit depth).
3. `apps/pipeline_main.cpp`: argument parsing, the Bunny path vs `--tum-frames` / `--tum-sequence`,
   intrinsics, depth scale.
4. `include/slam/features/types.hpp`: `GrayImage`, `GrayFromRGB`.

Be able to explain: the depth units and scale (Bunny vs TUM), what an invalid depth is, where the intrinsics
come from, why TUM mode needs a `frames.csv`.

Tests: `tests/pipeline/test_png.cpp`.

## Stage 3 — Feature extraction (half a day)

Read `DATA_FLOW.md` §2.4–2.8 and the `feature_core` section of the file guide, then in this order:
1. `include/slam/features/pyramid.hpp`, `src/features/pyramid.cpp` (8 levels × 1.2, pixel-centre alignment).
2. `include/slam/features/fast.hpp`, `src/features/fast.cpp` (FAST-9 segment test, score, 3×3 NMS, border).
3. `include/slam/features/multiscale.hpp`, `src/features/multiscale.cpp` (per-level detection, Harris
   ranking, the 1400-feature budget and per-level quotas, the opt-in per-cell FAST fallback).
4. `include/slam/features/orientation.hpp`, `src/features/orientation.cpp` (intensity centroid).
5. `include/slam/features/brief.hpp`, `src/features/brief.cpp` (smoothed, rotated 256-bit BRIEF).

Be able to explain: how T0 changes detection (Harris ranking, budget 1400, FAST threshold 10) relative to the
struct defaults (FAST 20, 500 per level); why the comment in `multiscale.hpp:5` is not true under T0
(see the file guide).

Tests: `tests/unit/features_{pyramid,fast,orientation,brief}_test.cpp`, `tests/pipeline/test_improvements.cpp`.

## Stage 4 — Matching (2–3 hours)

Read `DATA_FLOW.md` §2.9, then:
1. `include/slam/features/matcher.hpp`, `src/features/matcher.cpp`: brute-force Hamming, distance filter,
   `MatchMutualRatio` (ratio 0.87, T0), `MatchRanked` (optional).
2. `include/slam/pipeline/frontend.hpp`, `src/pipeline/frontend.cpp`: `MatchFrames` (from line 72; matching up to line 101).

Be able to explain: why, under T0, the brute-force match set is computed but replaced by the mutual + ratio
set (and which `pairs.csv` columns still come from the brute-force pass).

Tests: `tests/unit/features_matcher_test.cpp`, `features_matcher_ranked_test.cpp`,
`tests/pipeline/test_frontend.cpp`, `test_ranked_frontend.cpp`.

## Stage 5 — Depth association (2 hours)

Read `DATA_FLOW.md` §2.10 and §2.12, then the depth-association part of `MatchFrames`, `src/pipeline/frontend.cpp` lines 115–142.

Be able to explain: which pixel's depth is used (truncation, not rounding), the 3×3 depth-edge test and its
12 mm threshold (T0), back-projection X = ((u − cx) z / fx, (v − cy) z / fy, z), and why only frame i needs
depth (3D–2D correspondences).

## Stage 6 — PnP (half a day)

Read `include/slam/pipeline/pnp.hpp`, `src/pipeline/pnp.cpp`: the linear (DLT) PnP, its normalisation, and the
recovery of a proper rotation. Then the milestone program `apps/milestones/main.cpp`, which runs it on one pair
from frozen correspondences.

Be able to explain: why at least 6 points, what the SVD solves, how R is projected onto SO(3).

Tests: milestone CTests `milestone1_pnp_from_scratch`, `milestone2_pnp_full_sequence`.

## Stage 7 — RANSAC (half a day)

Read `DATA_FLOW.md` §2.13 and `include/slam/pipeline/ransac.hpp`, `src/pipeline/ransac.cpp`.
Also the diagnostic two-view stage: `include/slam/pipeline/essential.hpp`, `src/pipeline/essential.cpp`,
`include/slam/geometry/fundamental.hpp` (computed for every pair, never used for the pose).

Be able to explain: sampling (seeded `std::mt19937`, partial Fisher–Yates), scoring (inlier count, then
residual sum), the fixed iteration count (650 at 3 px in T0; 300 at 8 px in T0-ransac and the struct
defaults), and the unguarded PnP refit vs the guarded essential refit.

Tests: `tests/pipeline/test_ransac.cpp`, `test_essential.cpp`.

## Stage 8 — Refinement (half a day)

Read `DATA_FLOW.md` §2.14–2.16, then:
1. `include/slam/pipeline/refine.hpp`, `src/pipeline/refine.cpp`: Levenberg–Marquardt on SE(3), left
   perturbation, analytic Jacobian, Huber IRLS (δ = 2 px in T0).
2. `src/pipeline/pipeline.cpp` lines 101–193 (`ProcessPair`): refinement, inlier re-selection, the gate.
3. `include/slam/pipeline/validity.hpp`, `src/pipeline/validity.cpp`: the Stage-2 pose-validity gate
   (evaluated always, decisive only with `--pose-gate`).

Be able to explain: what is optimised (one relative pose, no structure — not bundle adjustment), what
re-selection changes, what happens to a rejected pair, and why `--stage2-frontend` / `--legacy` must come
before `--pose-gate` on the command line.

Tests: `tests/pipeline/test_refine.cpp`, `test_validity.cpp`, `test_pipeline_0_1.cpp`.

## Stage 9 — Trajectory accumulation and outputs (2 hours)

Read `DATA_FLOW.md` §2.17–2.18 and `apps/pipeline_main.cpp` from the accumulation to the output writers.

Be able to explain: every file `scratch_pipeline` writes (`pairs.csv`, `trajectory.csv`, `trajectory.txt`,
`summary.txt`, `correspondences/`, `keypoints/`, `pose_validity.csv`), what each column means (note
`pnp_ok` holds `pose_accepted`), and that `trajectory.txt` uses frame indices, not TUM timestamps.

Exercise (when you are allowed to run things): `build/scratch_pipeline --out /tmp/run` and read the outputs.

## Stage 10 — Evaluation (half a day)

1. `COORDINATE_CONVENTIONS.md` §9: rotation/translation error, ATE (raw vs aligned), RPE.
2. `include/slam/evaluation/baseline.hpp`, `src/evaluation/baseline.cpp`: fixtures, the frozen locks,
   `MeasureFixture` (no ground truth) vs `ScoreMeasurement` (adds errors afterwards).
3. `include/slam/evaluation/experiment.hpp`, `src/evaluation/experiment.cpp`, `apps/experiments_main.cpp`: the
   configuration ids (M0, L0, T0, T0G, T0-ransac, …) and the multi-seed harness.
4. `tests/fixtures/` (each file's entry in the file guide) and `tools/pipeline_analysis/run_baseline.sh`.
5. `results/archive/integration_2026-11/README.md` and `analysis_final2.md`: the evidence behind the T0 decision.
6. `tools/evaluation/tum_metrics.py` and the dashboard (`tools/evaluation/dashboard.py`).

Be able to explain: how a frozen lock detects a behaviour change, why ground truth never enters estimation,
and how the T0 decision was reached (`docs/components/pipeline.md`, "Production defaults (T0) and why").

## Stage 11 — ICP (half a day; experimental, no final decision)

1. `COORDINATE_CONVENTIONS.md` §10 and `SYSTEM_OVERVIEW.md` §5.
2. `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp`, `experiments/rgbd_icp/src/rgbd_icp.cpp`, `experiments/rgbd_icp/apps/icp_sequence.cpp`: the
   teammate's dense RGB-D ICP (feature SVD → RANSAC → point-to-point → point-to-plane → loop verification and
   pose graph); `docs/components/rgbd_icp.md`.
3. `reference/opencv_slambook2/src/tracking/icp.cpp`: the teammate's wrappers (SVD only).
4. `reference/icp_main_variant/src/icp.cpp`: main's book SVD + g2o, `FitRigid`, `EstimateRigidRansac`.

Be able to explain: why the two `icp.cpp` files can never be linked together, the `R = −R` reflection issue
in the book code versus the diag(1, 1, −1) correction, and the open questions in
`docs/components/rgbd_icp.md` (§7, "Open questions").

## Stage 12 — Reference library (optional, as needed)

`reference/opencv_slambook2/`: the OpenCV/g2o book-derived pipeline that the production code replaced
(camera, geometry, tracking/pnp, render for the synthetic Bunny, the TUM evaluator). Read it to compare
against production, not to learn the production behaviour. File-guide section "Reference implementation".

## Stage 13 — Visualization (2 hours)

1. `visualization/README.md`.
2. `visualization/scripts/slam_coords.py` (the single SLAM → Blender pose conversion),
   `trajectory_data.py`, `build_scene.py`.
3. The TUM scene scripts (`export_tum_scene.py`, `tum_reconstruction.py`, `build_tum_scene.py`).

Be able to explain: that Blender only reads C++ outputs, the camera-axis conversion, and why
`results/blender/bunny_slam_demo.blend` must stay where it is.

---

## Suggested pace

About two weeks part-time: stages 0–2 in the first days, stages 3–9 (the production core) over the first
week, stages 10–13 in the second. Re-read `COORDINATE_CONVENTIONS.md` whenever a sign or a transform
direction looks wrong; most confusion in this codebase comes from mixing T_{i+1←i} with T_{i←j}.
