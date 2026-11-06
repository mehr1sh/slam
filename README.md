# RGB-D visual odometry from scratch

A C++17 implementation of frame-to-frame RGB-D visual odometry. Features, matching, depth association,
PnP, RANSAC and pose refinement are written from scratch on Eigen and the C++ standard library, with no
OpenCV in the production code. An OpenCV/g2o reference implementation (slambook2-derived), experimental
ICP and benchmark code, evaluation tools and Blender visualization sit beside it, clearly separated.

## What is implemented today

For every consecutive RGB-D frame pair, `scratch_pipeline`:
1. decodes the PNG colour and depth images (own PNG decoder);
2. detects FAST corners on an 8-level pyramid, ranks them by Harris score under a 1400-feature budget,
   and computes orientation and rotated BRIEF descriptors;
3. matches descriptors (mutual nearest neighbour + ratio test 0.87);
4. back-projects frame-i keypoints with their depth (rejecting depth edges > 12 mm) to form 3D–2D
   correspondences;
5. estimates the relative pose with linear (DLT) PnP inside RANSAC (650 iterations, 3 px);
6. refines it with Levenberg–Marquardt on SE(3) with a Huber loss (2 px), re-selects inliers and refines again;
7. chains the relative poses into a camera trajectory (T_wc), anchored at the first ground-truth pose.

This production configuration is called **T0** (`ApplyImprovedDefaults`, `src/pipeline/pipeline.cpp:39-48`).
Ground truth is used only for evaluation and for that initial anchor; no estimation step reads it.
It runs on the synthetic Bunny sequence (`data/synthetic_bunny`) and on TUM RGB-D sequences.

**Not implemented** in the production pipeline: mapping, keyframes, loop closure, bundle adjustment, global
optimisation, relocalisation, lens-distortion correction. (The experimental ICP in `experiments/rgbd_icp`
contains loop verification and a pose graph; it is not part of production.)

## Production, reference, experiments, tooling

| family | where | dependencies | status |
|---|---|---|---|
| **Production** | `include/slam/`, `src/`, `apps/`, `tests/` | Eigen, C++17 | `scratch_pipeline` with T0 defaults |
| Options of production (opt-in) | flags of `scratch_pipeline` | — | `--pose-gate` (Stage-2 pose-validity gate, off by default), `--stage2-frontend` (earlier M0 configuration), `--pnp-iterations 300 --pnp-threshold 8` (T0-ransac, experimental), `--ranked-matcher` (`MatchRanked`), `--fast-cell` (FAST per-cell fallback) |
| Evaluation harness (C++) | `src/evaluation/`, `apps/{baseline,experiments}_main.cpp`, `tests/fixtures/` | Eigen | frozen regression locks, multi-seed experiments |
| Reference | `reference/opencv_slambook2/`, `reference/icp_main_variant/` | OpenCV, g2o, Sophus | not production; TUM evaluator, book demos, main's ICP |
| Experiments | `experiments/` | OpenCV (most) | RGB-D ICP (provisional, no final decision), ORB benchmarks, depth-edge study |
| Tooling | `tools/`, `visualization/` | Python, Blender | read C++ outputs only; never estimate poses |
| Evidence | `results/`, `tests/fixtures/` | — | tracked results and frozen test data; ignored runs are archived (`docs/DATA_AND_ARCHIVE.md`) |

## Build

```bash
pixi install                                             # toolchain and libraries from pixi.lock
pixi run cmake -S . -B build && pixi run cmake --build build -j   # production only (Eigen)
pixi run build                                           # + OpenCV reference library and experiments
```

All executables are written to `build/`. See `COMMANDS.md` §1 for options and a build without pixi.

## Run

```bash
./build/scratch_pipeline --out results/runs/run_bunny          # synthetic Bunny, T0
./build/scratch_pipeline --tum-frames tests/fixtures/stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4/frames.csv \
  --tum-sequence "$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz" --intrinsics 517.3,516.5,318.6,255.3 --out results/runs/run_tum
```

The TUM sequences are an external dataset: download them once and point `SLAM_TUM_ROOT` at the directory that
contains them (`docs/DATA_AND_ARCHIVE.md`). Everything else, including all tests, runs without them.

Outputs: `pairs.csv`, `trajectory.csv` / `trajectory.txt`, `summary.txt`, `correspondences/`, `keypoints/`.
The other applications (`scratch_baseline`, `scratch_experiments`, `feature_audit`, the reference programs,
`icp_sequence`, `orb_benchmark`, the dashboard, Blender scenes) are in `COMMANDS.md`.

## Test

```bash
ctest --test-dir build --output-on-failure               # 24 tests (26 with the OpenCV build)
bash tools/pipeline_analysis/run_baseline.sh --build build # frozen Stage-1/2 regression locks (--build is required)
# --rerun-stride4/--rerun-full also need the external archive: --archive DIR or SLAM_ARCHIVE_DIR (docs/DATA_AND_ARCHIVE.md)
```

## Documentation

| read | for |
|---|---|
| `docs/architecture/LEARNING_PATH.md` | the order in which to study the code |
| `docs/architecture/SYSTEM_OVERVIEW.md` | what the system is, its components and decisions |
| `docs/architecture/DATA_FLOW.md` | one frame pair through the pipeline, step by step |
| `docs/architecture/COORDINATE_CONVENTIONS.md` | frames, transform directions, trajectory composition |
| `docs/architecture/MODULE_REFERENCE.md` | each library target and the dependency graph |
| `docs/FILE_GUIDE.md` | every source, header, test, script and build file |
| `docs/components/` | the production pipeline in detail (including why T0 is the default), RGB-D ICP, ORB benchmark, frontend visuals |
| `docs/DATA_AND_ARCHIVE.md` | datasets (including the TUM setup), tracked evidence, the external archive |
| `tests/fixtures/README.md` | what each frozen fixture is, where it came from and what reads it |

## Current limitations

- Frame-to-frame odometry only: drift accumulates; a failed pair holds the previous pose.
- No lens-distortion model (TUM freiburg1 images are distorted).
- `trajectory.txt` uses frame indices, not TUM timestamps.
- `--stage2-frontend` and `--legacy` reset all other options: give them first.
- The two ICP variants (teammate's dense RGB-D ICP; main's book SVD + g2o with RANSAC) are both kept; no final
  ICP decision has been made. The tracked reference trajectories in `data/synthetic_bunny/`
  (`pnp_trajectory.txt`, `icp_trajectory.txt`, `slam_trajectory.csv`; book ORB and ICP) are not reproduced by the
  current reference build (35 of 36 rows of each differ, checked 2026-11-06), so
  `tools/pipeline_analysis/export_correspondences.sh` aborts and the byte checks of `run_diagnostics.py` fail.
- At larger inter-frame motion T0 is seed-dependent (TUM stride 4: 1 gross pose in 495 pair-runs; pair 73→74 bad
  in 5 of 20 seeds); RANSAC 300 iterations at 8 px is seed-invariant at +16 % full-sequence ATE
  (`docs/components/pipeline.md`, "Production defaults (T0) and why").
- Test coverage of T0 is thin: `test_frontend`, `test_pipeline_0_1`, `test_multiscale_0_1` and
  `test_ranked_frontend` exercise the legacy / Stage-2 configuration; T0 itself runs end to end only in the
  milestone-3 smoke test, `test_experiment` checks its parameter values, and `test_improvements` its new
  components (budget, Harris, mutual + ratio matching, Huber refinement). No test checks the behaviour of
  the 12 mm depth-edge rejection.

## Future work (not implemented)

Keyframes and a local map, bundle adjustment, loop closure and global optimisation for the production
pipeline; a distortion model; a final ICP decision. None of these exist in the production code today.

## Licence

`LICENSE`; third-party material and the Stanford Bunny terms: `THIRD_PARTY_NOTICES.md`.
