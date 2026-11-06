# Commands

Copy-pasteable commands, grouped by component. Run everything from the repository root unless a block
starts with `cd`. Architecture and conventions: `docs/architecture/`. Every file: `docs/FILE_GUIDE.md`.

> **Commands that overwrite tracked files** (run them only when you mean to regenerate that evidence):
> - `slam_trajectory_test` on `data/synthetic_bunny` → `pnp_trajectory.txt`, `icp_trajectory.txt`, `slam_trajectory.csv` (always run it on a copy, as below)
> - `icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp` (`pixi run icp`) → `results/benchmarks/rgbd_icp/` (the default output without that argument is the disposable `results/runs/rgbd_icp/`)
> - `orb_benchmark` with its default output, `tools/figures/render_*.py` → `results/reports/figures/`
> - `feature_audit` (also the CTest `feature_audit_pair_0_1`) → `results/reports/figures/frontend/pair_<i>_<j>/` (byte-identical when the code is unchanged)
> - `visualization/rebuild_blender.sh` → `results/blender/bunny_slam_demo.blend` and everything above
>
> `scratch_pipeline` without `--out` writes `results/runs/pipeline/` (ignored, disposable; read by the analysis tools,
> never by the Blender scene). The frozen Stage 1 export is tracked in `tests/fixtures/stage1_bunny_export/`; the
> frozen T0 trajectory that the Blender scene shows is `tests/fixtures/bunny_t0_scene/`.

## 1. Environment and build

```bash
pixi install                                  # environments from pixi.lock (C++ toolchain, OpenCV, Eigen, g2o, Sophus)

# production only: Eigen + standard library, no OpenCV
pixi run cmake -S . -B build && pixi run cmake --build build -j

# everything: production + OpenCV reference library + OpenCV experiments + main's ICP variant
pixi run build                                # = cmake -B build -S . -DSLAM_WITH_OPENCV=ON && cmake --build build -j

# standalone ICP build (rgbd_icp, icp_sequence, orb_benchmark; no g2o/Sophus)
pixi run icp-build                            # = cmake -S experiments/rgbd_icp/icp_cpp -B build-icp ... && build
```

All executables are written to `build/` (or `build-icp/`). Options: `-DSLAM_WITH_OPENCV=ON|OFF` (default OFF),
`-DBUILD_MAIN_ICP_VARIANT=ON|OFF` (default ON, with OpenCV), `-DBUILD_ORB_BENCHMARK=ON|OFF` (default ON, with OpenCV).

Without pixi (Ubuntu): `sudo apt install build-essential cmake libeigen3-dev` (production);
add `libopencv-dev` and g2o/Sophus for the OpenCV parts; `blender` for the scenes;
`python3-numpy python3-matplotlib python3-pil` for the tools.

## 2. Tests

```bash
ctest --test-dir build --output-on-failure    # 24 production tests (26 with -DSLAM_WITH_OPENCV=ON)
ctest --test-dir build -R scratch_baseline_lock --output-on-failure   # the frozen Stage-1/2 locks only
ctest --test-dir build-icp --output-on-failure                        # rgbd_icp_geometry, blender_data_bridge, orb_benchmark_smoke

# Stage-1/2 regression harness: Bunny byte identity vs the SHA-256 manifests, ctest, fixture locks, ATE locks
bash tools/pipeline_analysis/run_baseline.sh --build build   # [--rerun-stride4] [--rerun-full] [--archive DIR]
# --build DIR (or SLAM_BUILD_DIR) is required: a build tree of THIS repository that registers scratch_baseline_lock,
# or a new directory; anything else (e.g. the stale pre-reorganisation build/) is refused with exit 2, untouched.
# A differing re-run is explained by tools/pipeline_analysis/describe_run_diff.py (FORMAT / NUMERICAL per column).
# the re-runs compare with the complete frozen runs in the external archive (never a partial copy); without the
# archive they stop with exit status 2. Give its root (the directory holding MANIFEST.sha256) once per shell:
export SLAM_ARCHIVE_DIR=/path/to/archive                     # or: run_baseline.sh --rerun-stride4 --archive /path/to/archive
ctest --test-dir build -R fixture_checksums                  # the frozen fixture directories against their SHA-256 manifests

# OpenCV build only: unit tests of the TUM adapter and main's ICP (not registered with CTest)
./build/tum_dataset_test
./build/icp_rigid_test

# Python self-tests (results environment; no dataset needed)
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/test_tum_metrics.py
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/test_plot_tum_results.py
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard_app/test_methods.py
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/test_tum_reconstruction.py
```

`scratch_baseline_lock` reads the frozen TUM export from `tests/fixtures/stage1_tum_exports/` (tracked) and needs the
TUM images for the two TUM fixtures (otherwise those checks are skipped); with `SLAM_TUM_ROOT` set they are read
from there; see `docs/DATA_AND_ARCHIVE.md`.

## 3. Production pipeline (`scratch_pipeline`)

```bash
# synthetic Bunny, production configuration T0 (default) -> results/runs/pipeline/
./build/scratch_pipeline
./build/scratch_pipeline --out results/runs/pipeline_bunny_T0          # anywhere else

# alternatives (put --stage2-frontend / --legacy FIRST: they reset all other options)
./build/scratch_pipeline --stage2-frontend --out results/runs/pipeline_M0              # main's Stage-1 configuration (M0)
./build/scratch_pipeline --stage2-frontend --pose-gate --out results/runs/pipeline_M0G # Stage-2 configuration with the gate
./build/scratch_pipeline --pose-gate --out results/runs/pipeline_T0G                   # T0 with the pose-validity gate
./build/scratch_pipeline --pnp-iterations 300 --pnp-threshold 8 --out results/runs/pipeline_T0_ransac   # T0-ransac (experimental)
./build/scratch_pipeline --legacy --out results/runs/pipeline_L0                       # the historical L0 settings

# TUM RGB-D (frames.csv from a tum_trajectory_eval run; see docs/DATA_AND_ARCHIVE.md)
./build/scratch_pipeline \
  --tum-frames tests/fixtures/stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4/frames.csv \
  --tum-sequence data/rgbd_dataset_freiburg1_xyz --intrinsics 517.3,516.5,318.6,255.3 \
  --seed 12345 --out results/runs/pipeline_tum_stride4_T0

# figures and checks of the default run (results environment)
python3 tools/pipeline_analysis/check_trajectory.py
python3 tools/pipeline_analysis/make_figures.py
python3 tools/pipeline_analysis/trajectory_figures.py
```

The full option list, output files and columns: `docs/FILE_GUIDE.md` → `apps/pipeline_main.cpp`.

## 4. Evaluation harness and integration analysis

```bash
# frozen fixture locks (also the CTest scratch_baseline_lock)
./build/scratch_baseline                      # report -> results/runs/scratch_baseline/stage1/

# multi-seed experiments (configuration ids: M0, L0, T0, T0G, T0-ransac, ... — FILE_GUIDE, experiment.hpp)
./build/scratch_experiments --fixtures tests/fixtures/stage3_pairs.csv --configs M0,T0 --seed-count 20 \
  --out results/runs/experiments/fixtures_M0_T0
./build/scratch_experiments --sequence-frames tests/fixtures/stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4/frames.csv \
  --tum-sequence data/rgbd_dataset_freiburg1_xyz --intrinsics 517.3,516.5,318.6,255.3 \
  --configs M0,T0 --seed-count 5 --out results/runs/experiments/stride4_M0_T0

# regenerate the tracked integration tables (byte-identical): see results/archive/integration_2026-11/README.md
```

## 5. Reference library (OpenCV/g2o; `pixi run build` first)

```bash
# self-checking reference programs and the book demos, with logs (copies the dataset; nothing tracked is written)
bash tools/evaluation/run_reference_suite.sh results/runs/reference_suite

cd build
./fast_test && ./brief_test && ./essential_matrix_test && ./vo_pipeline_test
./feature_matching_test ../data/tum_sample/1.png ../data/tum_sample/2.png
./two_view_pose_test    ../data/tum_sample/1.png ../data/tum_sample/2.png
./triangulation_test    ../data/tum_sample/1.png ../data/tum_sample/2.png
./pnp_test  ../data/tum_sample/1.png ../data/tum_sample/2.png ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
./icp_test  ../data/tum_sample/1.png ../data/tum_sample/2.png ../data/tum_sample/1_depth.png ../data/tum_sample/2_depth.png
./orb_from_scratch_test
./gauss_newton_curve_fit_test && ./g2o_curve_fit_test
./pnp_synthetic_test       ../data/synthetic_bunny 0 1
./icp_synthetic_test       ../data/synthetic_bunny 0 1
./pnp_multiframe_test      ../data/synthetic_bunny 0 10
./synthetic_pose_eval_test ../data/synthetic_bunny/groundtruth.txt 0 1
./trajectory_validation_test ../data/synthetic_bunny
cd ..

# full reference trajectories: ALWAYS on a copy (the program writes into the dataset directory)
rm -rf /tmp/bunny_copy && cp -r data/synthetic_bunny /tmp/bunny_copy
(cd build && ./slam_trajectory_test /tmp/bunny_copy)
# main's ICP variant (book SVD + g2o, RANSAC) with an explicit output directory
(cd build && ./slam_trajectory_test_main_icp /tmp/bunny_copy --out-dir ../results/runs/reference_trajectory_main_icp)

# regenerate the synthetic dataset into a scratch directory (compare it; never overwrite data/)
(cd build && ./render_bunny_test ../data/meshes/bunny/bun_zipper.ply /tmp/bunny_render --num-frames 36 --radius-scale 4 --height-scale 1)
```

Results figures and diagnostics of the reference pipeline (results environment):

```bash
pixi run -e results results                   # tools/figures/generate_results.py -> results/{data,figures,tables}
pixi run -e results correspondence-map        # [--pair K]
pixi run -e results correspondences           # [--pair K]
pixi run -e results diagnostics               # tools/evaluation/run_diagnostics.py -> results/reports/diagnostics/
```

`generate_results.py` checks that a fresh run reproduces the tracked `data/synthetic_bunny` trajectories, which
were made with the book ICP. With the teammate's ICP in the reference `tracking` library that check is expected
to fail (open ICP question; not run during the reorganisation).

## 6. TUM RGB-D evaluator and dashboard (OpenCV build; ported from main)

```bash
./build/tum_trajectory_eval data/rgbd_dataset_freiburg1_xyz --frames 20 --out-dir results/runs/tum/fr1_xyz_n20   # default: results/runs/tum/<sequence>
#   --frames N (0 = all)  --stride S  --max-dt 0.02  --max-gt-gap 0.05  --out-dir DIR (never inside data/)
#   --export-pairs LIST (per-pair diagnostics)  --pnp-min-distinct-inliers N (default 6)
# linked to main's ICP variant (tracking_main_icp)

# stress runs and figures (results environment)
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/run_tum_stress.py \
  --dataset data/rgbd_dataset_freiburg1_xyz --output results/runs/stress/xyz_new_session      # --force, --figures
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/plot_tum_results.py --input results/runs/stress/xyz_new_session/n100

# dashboard over every run in results/runs/ (http://127.0.0.1:8765/)
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard.py            # [--port 8765] [--open]
PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard.py --check    # load every run, no server

# OpenCV ORB correspondences on the pairs of a TUM run (diagnostic only)
A=/home/mehirizz/IIITH/RRC/slam/archive/2026-11-04_main/output      # frozen archived runs: read-only inputs
./build/tum_orb_correspondences data/rgbd_dataset_freiburg1_xyz \
  --frames-csv $A/stress/xyz_session3_pnp_validity/n100_stride4/frames.csv --out results/runs/orb_tum/n100_stride4

# visual diagnostics (results environment)
diag() { PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/diagnostics/"$@"; }
diag make_diagnostics.py --dataset data/rgbd_dataset_freiburg1_xyz --frames 100 --stride 4 --export-pairs 66-80 \
  --failure 73 --windows 66-70,71-75 --baseline $A/stress/xyz_session1/n100_stride4
diag test_diagnostics.py
```

Archived runs are frozen and stay in the external archive (`$A` above): read them as inputs, never write
into them. To browse one in the dashboard, **copy** it into `results/runs/` (never link it). See `docs/DATA_AND_ARCHIVE.md`.

## 7. RGB-D ICP experiment (`experiments/rgbd_icp`)

```bash
pixi run icp-build && ctest --test-dir build-icp --output-on-failure
./build-icp/icp_sequence data/synthetic_bunny results/runs/icp_run   # to a scratch directory
pixi run icp                                  # = ./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp (overwrites tracked output)
```

Details, stages and results: `docs/components/rgbd_icp.md`.

## 8. ORB benchmark (`experiments/orb_opencv`)

```bash
./build-icp/orb_benchmark data/synthetic_bunny results/runs/orb_benchmark --repeats 3     # to a scratch directory
python3 tools/figures/render_orb_benchmark.py --results results/runs/orb_benchmark
python3 tools/figures/render_orb_keypoints.py --results results/runs/orb_benchmark
bash tools/figures/run_gn_orb.sh              # build-icp, tests, ICP, benchmark, figures into the tracked results
```

The historical standalone benchmark `experiments/orb_extractor_benchmark/` has its own README and build.
Details: `docs/components/orb_benchmark.md`.

## 9. Frontend audit and teammate figures

```bash
bash tools/figures/visualize_frontend.sh 15 16      # feature_audit + render + validate for one pair
python3 tools/figures/render_outputs.py             # redraw results/reports/figures and image_manifest.json
python3 tools/figures/validate_outputs.py           # run after render_outputs.py
xdg-open results/reports/figures
```

Details: `docs/components/frontend_visuals.md`.

## 10. Visualization (Blender)

```bash
bash visualization/rebuild_blender.sh               # full rebuild (also: pixi run blender-rebuild)
blender -y results/blender/bunny_slam_demo.blend    # open the tracked scene (-y lets its scripts register)
blender -b results/blender/bunny_slam_demo.blend --python visualization/check_blender_consistency.py -- /tmp/blender_check.csv

# TUM scene (results environment + Blender)
PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/export_tum_scene.py \
  --run results/runs/stress/xyz_session3_pnp_validity/n100_stride4     # a run under results/runs/ (copy an archived one first)
blender --background --python visualization/scripts/build_tum_scene.py -- \
  --export results/runs/blender/rgbd_dataset_freiburg1_xyz/stress__xyz_session3_pnp_validity__n100_stride4 \
  --density preview --frame 74 --check --render /tmp/tum_preview.png
```

Details: `visualization/README.md`.
