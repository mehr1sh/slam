# ORB benchmark (experiments/orb_opencv)

## 1. What it measures

`experiments/orb_opencv/orb_benchmark.cpp` builds the executable `orb_benchmark`. It compares two
feature extractors on every consecutive pair of an RGB-D sequence:

- the project's ORB-style extractor, `rgbd_icp::extractFeatures`, the same one used by
  `experiments/rgbd_icp`;
- OpenCV `cv::ORB`.

Everything after extraction is the same project code for both methods:

- the mutual ratio Hamming matcher;
- the depth gates;
- the RANSAC seeds;
- the dense clouds;
- the Gauss–Newton point and plane ICP of `experiments/rgbd_icp`.

It is a comparison of extractors at matched public settings. It is not an ORB-SLAM or ORB-SLAM3
comparison. It is not a component-level test, and the two implementations are not byte-identical.
There is no loop closure in either benchmark trajectory. This is an experiment, not part of the
production pipeline. The Blender scene has no camera for either benchmark method.

Reference for the OpenCV parameters: https://docs.opencv.org/4.13.0/db/d95/classcv_1_1ORB.html

## 2. How to run

**Build.** `orb_benchmark` is defined by `cmake/orb_benchmark.cmake`, which is controlled by the option
`BUILD_ORB_BENCHMARK` (default ON). It links `rgbd_icp` and `opencv_features2d`
(`cmake/orb_benchmark.cmake:1-7`). The file is included in two places:

- the root build when `-DSLAM_WITH_OPENCV=ON` (`CMakeLists.txt:130-134`);
- the standalone ICP build (`experiments/rgbd_icp/icp_cpp/CMakeLists.txt:33`).

Pass `-DBUILD_ORB_BENCHMARK=OFF` to leave it out. The ICP library itself never links `features2d`.

```bash
# standalone (also builds icp_sequence and rgbd_icp_test)
pixi run icp-build
# or: cmake -S experiments/rgbd_icp/icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release -DBUILD_ORB_BENCHMARK=ON
#     cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure     # includes orb_benchmark_smoke (2 frames, 1 repeat)

./build-icp/orb_benchmark data/synthetic_bunny results/reports/figures/orb_benchmark --repeats 3
python3 tools/figures/render_orb_benchmark.py      # 01_benchmark_summary.png, 02_counts_and_timing.png, RESULTS.md
python3 tools/figures/render_orb_keypoints.py      # keypoint_visuals/ (frames 0 and 15, plus frame 15 level 0)
```

With the root build the executable is `./build/orb_benchmark`. `bash tools/figures/run_gn_orb.sh` runs
the following in order: the standalone build, CTest, `icp_sequence`, `orb_benchmark` and both renderers
(`tools/figures/run_gn_orb.sh:10-21`). `bash tools/figures/run_gn_orb.sh --blender` instead runs
`visualization/rebuild_blender.sh` (`:6-8`).

**Command line.** `orb_benchmark [DATASET] [OUTPUT] [--repeats N] [--frames N]`
(`orb_benchmark.cpp:92-102`):

- The defaults are `data/synthetic_bunny`, `results/reports/figures/orb_benchmark`, 3 repeats, and all
  frames (`--frames 0`), relative to the working directory, so run it from the repository root.
- The program exits 0 when it completes, even if some pairs fail. Failures appear in `pairs.csv` and
  `summary.csv`. It exits 1 on an error.
- `results/reports/figures/orb_benchmark/` is tracked. Rerunning with the default output overwrites it.

The Python scripts need NumPy, matplotlib and Pillow. They only plot the C++ exports and estimate
nothing.

## 3. Controls (verified in code)

**Common settings**

| Item | Project extractor | OpenCV extractor |
| --- | --- | --- |
| Call | `extractFeatures` (`experiments/rgbd_icp/src/rgbd_icp.cpp:512-526`) | `cv::ORB::create(1400, 1.2f, 8, 16, 0, 2, HARRIS_SCORE, 31, 10)` (`orb_benchmark.cpp:39`) |
| Grey conversion | `features::GrayFromInterleaved` | the same function, then wrapped as `cv::Mat` (`:36-38`) |
| Budget / levels / scale | 1400 / 8 / 1.2 (`rgbd_icp.hpp:26`, `include/slam/features/pyramid.hpp:25-26`) | 1400 / 8 / 1.2 |
| FAST threshold / border | 10 / 16 (`include/slam/features/fast.hpp:17`) | 10 / edgeThreshold 16 |
| Ranking | project Sobel Harris + geometric per-level quotas | OpenCV `HARRIS_SCORE` |
| Descriptor | project rotated BRIEF, 256 bits | OpenCV ORB 256 bits (WTA_K 2, patch 31), converted bit by bit into `features::BriefDescriptor` (`:48-52`) |

Actual feature counts differ and are reported. The remaining differences are real: detector scores,
pyramid sampling, smoothing and the BRIEF sampling pattern. Descriptors are never compared across
methods.

**Pipeline after extraction.** Matching is `features::MatchMutualRatio` with ratio 0.87 in both
directions plus the mutual check (`:144`). Registration calls `registerFramesWithFeatures` with seed
`42 + i`, so the depth gates, RANSAC, dense clouds and both GN stages are identical (`:152`).

**Timing** (`:103,119-153`):

- OpenCV runs with one thread (`cv::setNumThreads(1)`) and RNG seed 42.
- Method order alternates by frame and by pair.
- Extraction and matching each get one untimed warm-up, then N timed repeats, and the median is kept.
  Extraction time includes grey conversion, detector creation and descriptor conversion. Image loading
  is excluded.
- Keypoint CSV export happens after the timed section.
- Registration is timed once. It already includes matching, depth lifting, RANSAC and both ICP stages,
  so do not add the matching time to it.

**Ground truth** (`:160-181`). GT is read only after every pair of both methods has been estimated. A
valid-depth feature match is GT-correct if `‖T_GT·p_source − p_target‖ < 9 mm`. `gt_precision_3d` pools
the correct pairs over all valid-depth pairs. It is not all-feature precision, not recall, and not a
descriptor-equality test.

**Trajectory** (`:154-156,194-206`):

- A failed pair reuses the last accepted relative motion (constant-velocity fallback, identity if none
  yet) and is counted as failed.
- ATE uses the frame-0 GT anchor with no fitted alignment or scale, and includes those fallbacks.
- `mean_accepted_*_error` averages accepted pairs only.
- The trajectory files are written in the frame-0 world, without the GT anchor.

## 4. Outputs (`results/reports/figures/orb_benchmark/`)

| File | Content |
| --- | --- |
| `frames.csv` | method, frame, retained keypoints, median extraction ms |
| `pairs.csv` | method, pair, status, matches, valid 3D, RANSAC inliers, GT-correct, GT precision, matching ms, registration ms, accepted-pair rotation/translation error |
| `summary.csv` | per method: frames, pairs, solved/failed, mean keypoints/matches/valid 3D/inliers, pooled GT precision, median extraction ms, mean matching and registration ms, ATE, accepted-pair mean errors |
| `{project_orb_style,opencv_orb}_trajectory.txt` | `frame tx ty tz qx qy qz qw`, frame-0 world |
| `keypoints/frame_NNNNNN_{project_orb_style,opencv_orb}.csv` | every retained keypoint from the timed extraction: `index,x,y,level,angle_rad`. x, y are original-image coordinates (`:56-70`). 72 files for 36 frames |
| `protocol.txt` | OpenCV and C++ version and all controls above (`:211-223`) |
| `run_complete.txt` | `frames=N`. Removed at start, written last (`:112,226`) |
| `01_benchmark_summary.png`, `02_counts_and_timing.png`, `RESULTS.md` | `tools/figures/render_orb_benchmark.py`, which refuses to run without `run_complete.txt` (`:22-23`) |
| `keypoint_visuals/` | `tools/figures/render_orb_keypoints.py` (section 6) |

## 5. Results (synthetic Bunny, 36 frames, 35 pairs; `summary.csv`, `protocol.txt`)

The run used OpenCV 4.6.0 and C++17 with 3 repeats.

| Method | Solved | Mean keypoints | Mean matches | Mean valid 3D | Mean inliers | GT precision | ATE RMSE | Median extraction | Mean matching | Mean registration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Project ORB-style | 35/35 | 1013.9 | 275.6 | 214.5 | 208.5 | 97.00 % | 1.688 mm | 81.35 ms | 170.3 ms | 190.2 ms |
| OpenCV ORB | 35/35 | 993.9 | 286.3 | 213.2 | 207.7 | 97.07 % | 1.692 mm | 4.40 ms | 213.8 ms | 236.3 ms |

Accepted-pair mean errors are 0.0280° / 0.204 mm (project) and 0.0280° / 0.204 mm (OpenCV).

The project row matches the production ICP plane stage:

- its ATE (0.00168843638096 m) and mean errors equal the `plane` row of
  `results/benchmarks/rgbd_icp/summary_metrics.csv`;
- its mean matches, valid 3D pairs and inliers equal the means of `results/benchmarks/rgbd_icp/pair_metrics.csv`.

So the benchmark's project path reproduces `icp_sequence` before the loop and pose graph. With the
verified loop and pose graph, `icp_sequence` reaches 1.200 mm ATE (see `rgbd_icp.md`).

The timings are local measurements and depend on hardware and load. The extraction speed ratio does not
carry over to end-to-end time. The data is a single clean synthetic sequence: these results are a local
correctness and behaviour check, not evidence about real scenes.

## 6. Keypoint overlays (`keypoint_visuals/`)

`tools/figures/render_orb_keypoints.py` reads the keypoint CSVs. It detects nothing itself. Default
frames are 0 and 15 (`:43`), and for frame 15 it adds a level-0-only view (`:49`). For each view it
writes:

- `*_project.png`: all retained project points (blue circles);
- `*_opencv.png`: all retained OpenCV points (orange);
- `*_difference.png`: spatial pairs as green midpoints, unpaired project points as blue circles,
  unpaired OpenCV points as orange triangles;
- `*_comparison.png`: the three panels side by side, with the same crop and scale;
- `*_spatial_pairs.csv`: project and OpenCV index, distance, level;
- `*_unpaired.csv`;
- an entry in `summary.json`.

No points are subsampled.

**How points are paired** (`compare`, `:24-39`). Pairs are greedy and one-to-one, in ascending image
distance, then by array index. Two points can pair only if they are within 2 original-image pixels
(`--tolerance`) and on the same pyramid level. The script asserts that pairs plus unpaired points add up
to each method's count (`:100`).

Results (`keypoint_visuals/summary.json`):

| View | Project | OpenCV | Spatial pairs | Unpaired project / OpenCV |
| --- | ---: | ---: | ---: | ---: |
| Frame 0, all levels | 838 | 789 | 627 | 211 / 162 |
| Frame 15, all levels | 1313 | 1292 | 932 | 381 / 360 |
| Frame 15, level 0 | 304 | 304 | 251 | 53 / 53 |

At level 0 of frame 15 both extractors keep 304 points, yet 53 of each have no partner within 2 px. An
equal count does not imply the same locations.

The overlap measures only how similar the selected positions look. It does not test descriptor
equality, physical correspondence or correctness.


## 7. Solver identity

`registerFramesWithFeatures` is the shared back end (`rgbd_icp.cpp:532-566`). Production `registerFrames`
is just `extractFeatures` followed by that call (`:527-531`). The benchmark therefore exercises exactly
the dense solver documented in `rgbd_icp.md`:

- RANSAC + SVD initialisation;
- point-to-point damped Gauss–Newton with `J = [−[x]×, I]`, LDLT and damping `1e-5·trace(H)/6 + 1e-9`,
  up to 18 iterations;
- point-to-plane Huber (4 mm) damped Gauss–Newton, up to 16 iterations;
- left SO(3)×R³ updates with step-norm bounds and 5-step backtracking.

`results/benchmarks/rgbd_icp/solver_metadata.txt` records this solver identity for the production `icp_sequence`
run.

## 8. Verification

- **CTest `orb_benchmark_smoke`** runs the executable on 2 frames with 1 repeat into
  `<build>/orb-smoke` (`cmake/orb_benchmark.cmake:8-9`). It checks only that the program completes.
- **Built-in checks.**
  - `orb_benchmark` throws if the keypoint metadata is incomplete (`:58-60`) or if a match count exceeds
    the query count (`:149`).
  - `render_orb_keypoints.py` asserts that pairs and unpaired points reconcile (`:100`).
- **Agreement with production.** As shown in section 5, the project-method ATE and means equal the
  production `plane` stage values in `results/benchmarks/rgbd_icp/`.

Reported by the teammate and not re-run here:

- both methods solved 35/35 pairs;
- the benchmark project trajectory equals the production plane trajectory after the same frame-0
  transform, with a maximum difference below 1e-10 m;
- the 72 keypoint CSVs agree with frame counts, image bounds, levels and finite orientations.

No script in the repository performs that last CSV check. It was a one-off check whose code is not part
of the repository.

## 9. Historical standalone: experiments/orb_extractor_benchmark

`experiments/orb_extractor_benchmark/` is a self-contained, historical snapshot with its own CMake
project, LICENSE and notices. It vendors `feature_core` as of commit `fb23599`
(`experiments/orb_extractor_benchmark/CMakeLists.txt:8-16`). That snapshot predates Harris ranking,
per-level quotas, `MatchMutualRatio` and `MatchRanked`.

It contains:

- the C++ adapter `orb_extract`, which turns a PGM into a feature CSV;
- Python scripts (`benchmark.py`, `run_all.py`, `extract_image.py`, `build_report.py`) that compare it
  with OpenCV ORB on image datasets (Oxford Affine, HPatches, TUM, KITTI, any image sequence), using
  homography-based repeatability and precision where ground-truth homographies exist;
- sample results in `sample_results/`.

It is not built by the root CMake. Its README is the reference for its use. Note that the README lists a
precompiled `bin/orb_extract` that is not present in the repository; build it from source as the README
describes.
