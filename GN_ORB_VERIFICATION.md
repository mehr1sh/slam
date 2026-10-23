# Verification: GN ICP and ORB benchmark

The supplied run uses 36 synthetic-bunny frames and all 35 consecutive pairs. This archive includes the updated source, generated measurements, refreshed PNGs and a rebuilt Blender scene. Build directories and downloaded development dependencies are excluded.

## Results

| Method | Solved pairs | ATE RMSE |
| --- | ---: | ---: |
| Project ORB-style + GN ICP, no loop closure | 35/35 | 1.688436 mm |
| OpenCV ORB + identical GN ICP, no loop closure | 35/35 | 1.691829 mm |
| Production GN ICP + verified loop + pose graph | 35/35 | 1.200057 mm |

OpenCV 4.6.0 was used. Median feature extraction was 81.345 ms (project) and 4.405 ms (OpenCV) on this runtime. The matcher is identical project code in both methods. Timings are local measurements; extraction speed does not imply the same end-to-end speed ratio.

## Checks performed

- Independent `icp_cpp/` Release build, with ORB benchmark enabled: passed.
- Three CTest entries: GN geometry/unit checks, Blender data bridge, and two-frame ORB smoke run: passed.
- Seventeen scratch-PnP/feature CTest entries: passed. One PNG test was rerun after fixing a build-artifact execute permission in this runtime; no source change was needed.
- Actual `bash scripts/run_gn_orb.sh` run: exit 0; 35/35 production ICP pairs and 35/35 pairs in each benchmark method.
- Analytic point and plane Jacobians checked against central finite differences in all six columns.
- Known nonzero pose recovery, aligned pose, partial overlap, outlier RANSAC, motion gate and no-overlap rejection: passed.
- All logged fixed-pair GN objectives are nonincreasing, and both stages show refreshed assignment hashes.
- Benchmark coverage: 72 frame-method rows and 70 pair-method rows. Pooled GT precision and summary means agree with individual rows.
- Benchmark project trajectory equals the production plane trajectory after the same frame-zero world transform (maximum position discrepancy below 1e-10 m).
- PNG exporter and validator: 43 PNG files passed dimensions/hashes, plotted metrics, NN refresh, rigid-pose and copied-data checks.
- Both frontend audit pairs (0->1 and 15->16): passed exact pipeline and matching checks.
- Benchmark summary plot and pipeline flowchart visually inspected for layout and method labels.
- Blender 4.0.2 scene rebuilt from new results; all 36 frames, four camera poses, curves, calibration, frame range, freshness fingerprint and HUD values passed. Maximum projection discrepancy: 0.000033 pixels. No final photorealistic animation render was needed for these data/geometry checks.
- Run-script syntax and Python plot-script compilation: passed.

## Source and documentation changes

- `CMakeLists.txt`
- `COMMANDS.md`
- `FINAL_WORKFLOW.md`
- `GN_ORB_GUIDE.md`
- `ICP_OPTIMIZATIONS.md`
- `README.md`
- `START_HERE.md`
- `apps/icp_sequence.cpp`
- `apps/orb_benchmark.cpp`
- `cmake/orb_benchmark.cmake`
- `icp_cpp/CMakeLists.txt`
- `include/tracking/rgbd_icp.hpp`
- `scripts/rebuild_blender.sh`
- `scripts/render_orb_benchmark.py`
- `scripts/render_outputs.py`
- `scripts/run_gn_orb.sh`
- `src/tracking/rgbd_icp.cpp`
- `tests/unit/rgbd_icp_test.cpp`

New measurements: `outputs/orb_benchmark/`. Production solver identity: `icp_cpp_results/solver_metadata.txt`. Commands and maths: `GN_ORB_GUIDE.md`. Raw test and build logs: `outputs/verification/` (files containing `gn` or `gn_orb`). Historical reports and baseline logs are retained for context.

Current visual export audit and code review: see `ORB_VISUALS_AND_GN.md`.
