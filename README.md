# Visual SLAM from scratch (C++)

An ongoing, incremental implementation of a **visual SLAM pipeline in C++**,
built component by component with the goal of owning the whole pipeline,
rather than relying on OpenCV's feature, PnP, ICP or SLAM implementations.
The geometric and computer-vision components come first; mapping,
optimization and loop closure follow.

C++ is the source of truth for every camera, geometry, pose and trajectory
computation. Python and Blender are used only for dataset handling, plotting,
visualization and debugging.

## Roadmap and status

| Stage | Status | Where |
|---|---|---|
| Feature detection / description | **from scratch**: image pyramid, FAST, intensity-centroid orientation, rotated BRIEF | `src/features/` (`feature_core`, no OpenCV) |
| Feature matching | **from scratch**: brute-force Hamming, distance filter | `src/features/matcher.cpp` |
| Camera geometry | **from scratch** in the scratch pipeline (pinhole projection, back-projection); the book-derived `camera/` module uses OpenCV types | `pnp_from_scratch/src/projection.*`, `src/camera/` |
| Relative pose (2D–2D) | from-scratch 8-point essential matrix + decomposition (diagnostic); the book-derived module uses OpenCV | `pnp_from_scratch/src/essential.*`, `src/tracking/essential_matrix.cpp`, `src/geometry/` |
| PnP | **from scratch**: linear DLT, RANSAC, Levenberg–Marquardt refinement | `pnp_from_scratch/src/{pnp,ransac,refine}.*` |
| 3D–3D alignment (ICP) | reference: closed-form SVD + g2o refinement on feature correspondences | `src/tracking/icp.cpp` |
| Triangulation | reference only (OpenCV) | `src/geometry/` |
| Motion estimation / tracking | frame-to-frame odometry with trajectory chaining (no keyframes) | `pnp_from_scratch/src/pipeline*`, `tests/synthetic/slam_trajectory_test.cpp` |
| Landmark management, local mapping | not started | — |
| Pose graph optimization, loop closure | not started | — |
| Bundle adjustment | not started (g2o is used only for single-pose refinement in the book-derived PnP/ICP) | — |
| Full visual SLAM system | not started | — |

The repository currently has two tracks:

- **Scratch pipeline** (`pnp_from_scratch/` + `feature_core`): the
  from-scratch implementation, with no OpenCV, which the SLAM system will be
  built on. It currently runs RGB-D frames → features → matching → RANSAC PnP →
  refinement → trajectory. See
  [Self-contained scratch pipeline](#self-contained-scratch-pipeline).
- **Reference pipeline** (the book-derived modules in `src/` and the programs
  in `tests/`): code derived from *14 Lectures on Visual SLAM* (slambook2),
  using OpenCV, g2o and Sophus. Its results with the original OpenCV ORB
  features are frozen as the **baseline** the scratch implementation is
  measured against (`docs/migration/baseline/`, tag `baseline-pre-migration`,
  and the committed `data/synthetic_bunny/slam_trajectory.csv`). Its feature
  stage has since been switched to `feature_core` (checkpoint 01); the rest
  still uses OpenCV. It also contains the book's two-view frontend on a real
  TUM RGB-D frame pair.

**Evaluation environment.** Development and debugging use a controlled
synthetic RGB-D sequence: a CPU rasterizer renders a test object (the
Stanford Bunny mesh) from a known camera orbit, giving RGB, depth and exact
ground-truth poses. The scene is a test fixture, not the purpose of the
project. A Blender scene animates the estimated cameras against ground truth
for visual inspection.

> **Current scope.** This is *not yet* a full SLAM system. There is no map, no keyframes,
> no loop closure, no global optimization, and **no fused PnP+ICP backend**.
> PnP and ICP each produce an independent dead-reckoning trajectory. The file
> `slam_trajectory.csv` and the program `slam_trajectory_test` are named after
> the experiment. They contain and produce only the GT / PnP / ICP comparison,
> not a fused SLAM estimate.

## Repository layout

Roles: **[scratch]** = from-scratch implementation (no OpenCV), the basis of
the SLAM system; **[reference]** = book-derived baseline using OpenCV/g2o;
**[prototype]** = earlier from-scratch component, superseded but kept with its
tests; **[evaluation]** = test data, rendering, visualization and analysis tools.

```
include/<module>/        public headers
src/<module>/            implementation
  camera/                [reference] PinholeCamera, pixel2cam() / cam2pixel()
  features/              [scratch] feature_core: types, image pyramid, FAST, orientation,
                         rotated BRIEF, Hamming matcher, multiscale extraction
                         ([reference] features.cpp: book interface, OpenCV types in/out,
                         feature_core inside; transitional)
  geometry/              [reference] 2D-2D pose (F/E/H + recoverPose), triangulation
  optimization/          [reference] g2o SE(3) pose vertex (header-only, include/ only)
  tracking/              [reference] PnP & ICP (+ Gauss-Newton / g2o refinement);
                         [prototype] fast, brief, hamming_matching, essential_matrix
  render/                [evaluation] PLY I/O, CPU rasterizer, orbit trajectory, PLY scene export
tests/                   [reference] programs, one per component
  unit/                  self-contained (synthetic inputs)
  integration/           on data/tum_sample/ (real TUM RGB-D frame pair)
  synthetic/             on data/synthetic_bunny/, incl. the dataset generator
                         (render_bunny_test) and the trajectory pipeline (slam_trajectory_test)
data/                    [evaluation]
  meshes/bunny/          test-scene mesh (Stanford Bunny, bun_zipper.ply)
  synthetic_bunny/       synthetic evaluation sequence + reference trajectories (see its README)
  tum_sample/            one real TUM RGB-D frame pair (+ depth)
pnp_from_scratch/        [scratch] the from-scratch pipeline: PNG decoding, projection, essential
                         matrix, RANSAC, PnP, refinement, frame-to-frame tracking (own CMake
                         project and tests); experiments/ and tools/ are [evaluation]
cmake/                   feature_core.cmake: the feature library shared by both builds
visualization/           [evaluation] Blender scene builder (see visualization/README.md)
scripts/                 generate_results.py (results/), run_diagnostics.py (diagnostics/),
                         Blender checks and renders; migration/: regression suite and metrics
results/                 README.md (tracked) + generated figures/, data/, tables/
diagnostics/             drift diagnosis reports (.md tracked; CSVs and figures generated)
presentation/            progress slides (LaTeX source + PDF)
docs/
  chapter6_7_tests.md    reference numbers for the chapter 6/7 programs
  library_removal_audit.md   where the reference pipeline uses OpenCV
  migration/             checkpoint reports 01–05 with evidence; baseline/ = frozen
                         reference outputs (tag baseline-pre-migration)
COMMANDS.md              every command in one place
```

Generated artifacts are not tracked: `build/`, `output/` (images, PLYs and
CSVs written by the programs), `pnp_from_scratch/build/` and
`pnp_from_scratch/results/`, and the Blender scene
`visualization/scenes/bunny_slam_demo.blend`.

## Dependencies

- [pixi](https://pixi.sh) provides CMake, a C++ compiler, OpenCV 4.x, Eigen,
  g2o and Sophus from conda-forge (`pixi.toml`, locked in `pixi.lock`).
  Linux x86-64.
- [Blender](https://www.blender.org) **5.x**, for the visualization only
  (tested with 5.2). It is not managed by pixi.
- A second pixi environment, `results` (Python, NumPy, matplotlib), runs only
  the figure pipeline. It is kept separate from the C++ environment.

## Build

```bash
pixi run build          # = cmake -B build -S .  &&  cmake --build build -j
```

All executables land in `build/`. **Run them from `build/`**: several use
paths relative to it (`../data/...`, `../output/...`). They run from a plain
shell, because the pixi library path is embedded in the binaries.

## Evaluation environment: synthetic RGB-D sequence

`render_bunny_test` loads `bun_zipper.ply` (35,947 vertices, 69,451
triangles), takes its bounding box (centre *c*, radius r_b = 0.1251 m), and
places N = 36 cameras on a horizontal circle (world is Y-up):

- camera centre: C_i = c + h·up + r·(cos θ_i·a + sin θ_i·b), with θ_i = i·360°/36
  - 10° steps; the 360° endpoint is not repeated
- orbit radius: r = 4·r_b = 0.5005 m
- orbit height: h = 1·r_b = 0.1251 m
- orientation: every camera looks at *c* (look-at construction)

Each pose is rendered with a z-buffer rasterizer (flat Lambert shading,
back-face culling) into:

- `NNNNNN.png`: 640×480 RGB
- `NNNNNN_depth.png`: 16-bit depth, value = depth_m × 5000, 0 = no surface
- `intrinsics.txt`: fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7
- `groundtruth.txt`: exact camera poses (T_wc, TUM format)

The committed dataset is reproduced **byte-for-byte** by:

```bash
cd build
./render_bunny_test ../data/meshes/bunny/bun_zipper.ply ../data/synthetic_bunny \
    --num-frames 36 --radius-scale 4 --height-scale 1
```

The orbit flags matter: the program's defaults (3.0 / 0.5) give a different
dataset.

## Reference pipeline (OpenCV baseline)

For each consecutive pair (i, i+1), the reference program
`slam_trajectory_test` does the following. The description and numbers are
those of the frozen baseline (OpenCV ORB features; the committed
`data/synthetic_bunny/slam_trajectory.csv`). The current build uses
`feature_core` for the feature step (`docs/migration/01_features/`). The
scratch pipeline replaces all of these steps with its own implementations
(see below).

- **Features.** ORB keypoints (OpenCV defaults) are matched by brute-force
  Hamming distance, keeping matches with distance ≤ max(2·d_min, 30). For pair
  0→1 that is 202 matches.
- **PnP.** Each kept match with valid depth in frame *i* is back-projected:
  X = ((u−cx)/fx·Z, (v−cy)/fy·Z, Z), in camera-*i* coordinates. It is paired
  with the 2D keypoint in frame *i+1*. OpenCV `solvePnPRansac` runs with 100
  iterations, an 8 px threshold and 0.99 confidence, and minimizes
  reprojection error. The result is **T_{i+1←i}**. For pair 0→1: 136
  correspondences, 121 inliers.
  The repo's own Gauss-Newton and g2o PnP refinements (`src/tracking/pnp.cpp`)
  are exercised by `pnp_test` on the TUM pair, not by the trajectory pipeline.
- **"ICP".** The same matches, now with valid depth in *both* frames, give
  3D–3D pairs (119 for pair 0→1). Alignment is closed-form: centroids, the
  cross-covariance W = Σ q₁q₂ᵀ, SVD, R = UVᵀ and t = p̄₁ − R·p̄₂. It is then
  refined with g2o Levenberg–Marquardt (10 iterations, started from identity),
  minimizing Σ‖p₁ − (R·p₂ + t)‖². The native result is T_{i←i+1} and is
  inverted to T_{i+1←i}. This is correspondence-based registration: there is
  **no dense point cloud and no nearest-neighbour search**, unlike classic ICP.

## Trajectory accumulation

Frame 0 is anchored to ground truth. Then, for i = 0 … 34:

```
T_wc[0]   = T_gt[0]
T_wc[i+1] = T_wc[i] · T_rel(i, i+1)⁻¹
```

The estimators return T_{i+1←i}, but chaining needs T_{i←i+1}, because
T_{w←i+1} = T_{w←i} · T_{i←i+1}. A failed pair would keep the previous pose
(none fail: 35/35 for both methods).

There is no smoothing, no scale estimation (depth makes the scale metric) and
no global correction, so errors accumulate:

| Method | Final position error | Final rotation error | Mean position error |
|---|---|---|---|
| PnP (RANSAC) | 0.277 m | 32.1° | 0.180 m |
| ICP | 0.525 m | 64.9° | 0.270 m |

Outputs, reproduced byte-for-byte from the dataset above:

- `pnp_trajectory.txt`, `icp_trajectory.txt`: TUM format, T_wc
- `slam_trajectory.csv`: GT/PnP/ICP poses per frame, plus each estimate's
  absolute error:
  - translation error: ‖C_gt − C_est‖ (metres)
  - rotation error: angle of R_gtᵀ·R_est (degrees)

```bash
cd build && ./slam_trajectory_test ../data/synthetic_bunny
```

## Results and figures

```bash
pixi run -e results results
```

This re-runs the trajectory program (unchanged outputs) with export flags,
cross-checks every number, and writes 15 figures plus per-frame and per-pair
CSVs and a summary table to `results/`. Examples: the experiment setup, ORB
matches, the correspondence funnel, the 3D trajectories, the error per frame,
local vs global error, and a one-slide summary. See `results/README.md`.

## Self-contained scratch pipeline

`pnp_from_scratch/` is a separate CMake project that needs only a C++17
compiler and Eigen:

```
RGB-D PNG (own decoder) → grayscale → 8-level pyramid (scale 1.2)
  → FAST (threshold 20) → intensity-centroid orientation → rotated BRIEF
  → Hamming matching, distance ≤ max(2·d_min, 30) → depth-backed 3D→2D
  → RANSAC (6-point linear PnP, 300 iterations, 8 px) → LM refinement
  → trajectory T_wc[i+1] = T_wc[i]·T_rel⁻¹ → Blender (ScratchPnP_Animated_Camera)
```

```bash
cd pnp_from_scratch
cmake -S . -B build && cmake --build build -j
ctest --test-dir build          # 15 self-checking tests
./build/scratch_pipeline        # -> results/pipeline/
```

Measured on the 35 pairs (no ranking; details and figures in
`docs/migration/05_figures_depth_edges/REPORT.md`):

| | Scratch | Reference PnP | Reference ICP |
|---|---|---|---|
| RANSAC inliers per pair, mean | 240.3 | 132.0 | — |
| mean pair rotation / translation error | 1.16° / 0.011 m | 1.39° / 0.012 m | 2.81° / 0.023 m |
| mean trajectory error | 0.147 m / 18.2° | 0.180 m / 21.4° | 0.270 m / 30.3° |
| final error (frame 35) | 0.284 m / 31.9° | 0.277 m / 32.1° | 0.525 m / 64.9° |

The remaining drift is a systematic under-rotation traced to depth-edge
(silhouette) correspondences; see the checkpoint-05 report.

## Coordinate conventions

- **Pose files** store **T_wc** (camera → world). The translation is the
  camera centre in world coordinates; the rotation maps camera axes to world
  axes.
- **Camera axes** follow OpenCV: +X right, +Y down, +Z forward.
- **Rendering** uses T_cw = T_wc⁻¹, i.e. X_cam = R·X_world + t.
- **Relative poses** T_{i+1←i} satisfy x_{i+1} = R·x_i + t.
- **Blender** applies only two visualization-side conversions:
  - a fixed Y-up → Z-up rotation of the whole scene
  - a 180° rotation about each camera's own X axis, because Blender cameras
    look down −Z with +Y up

## Visualization (Blender)

```bash
blender --python visualization/scripts/build_scene.py
```

This builds and opens the scene, and saves it to
`visualization/scenes/bunny_slam_demo.blend` (generated, not tracked).

The scene contains:

- the test object (Stanford Bunny mesh)
- the three full trajectory lines: GT green, PnP orange, ICP magenta
- `GT_Animated_Camera`, `PnP_Animated_Camera`, `ICP_Animated_Camera`: real
  Blender cameras, keyframed so that frame *i* = pose *i*
- `ScratchPnP_Animated_Camera` and `ScratchPnP_Trajectory` (cyan): the from-scratch
  pipeline of `pnp_from_scratch/` (own features, pyramid, RANSAC, linear PnP + LM
  refinement); see `pnp_from_scratch/README.md`
- a HUD with the per-frame errors

To look through one camera, run this in Blender's Python console and then
press Numpad 0:

```python
bpy.context.scene.camera = bpy.data.objects["PnP_Animated_Camera"]
```

See `visualization/README.md` for details.

## Tests

`CMakeLists.txt` builds each program as a separate executable.

- **Self-checking tests** (print PASS/FAIL, exit non-zero on failure):
  `brief_test`, `fast_test`, `essential_matrix_test`, `vo_pipeline_test`,
  `trajectory_validation_test ../data/synthetic_bunny`.
- **Demonstration and diagnostic programs** (print results for inspection):
  the remaining integration and synthetic programs, and the chapter-6
  curve-fitting demos.

Expected numbers are documented in `docs/chapter6_7_tests.md`. All commands
are listed in `COMMANDS.md`.

## Known limitations

- Frame-to-frame odometry only. There is no mapping, bundle adjustment over
  windows, loop closure, relocalization or PnP+ICP fusion.
- ICP uses feature correspondences with no outlier rejection, so mismatches
  enter the least-squares fit. This is why it drifts more than PnP.
- The synthetic data is noise-free, with fixed lighting and a single object.
- The programs under `tests/` use paths relative to `build/`, and most of them
  print results rather than assert them; `scripts/migration/run_test_suite.sh`
  runs all of them and records exit codes. (`pnp_from_scratch/` has CTest.)
- `orb_from_scratch_test` (book demo) crashes intermittently: its border check
  is smaller than its sampling pattern (see `docs/migration/03_scratch_pyramid/REPORT.md`).
- Linux x86-64 only (as pinned in `pixi.toml`). `-msse4` is required by the
  from-scratch Hamming matcher.

## Credits and license

- **Code:** the project's own code is under the MIT license (`LICENSE`).
- **Book-derived code:** several algorithm blocks and example programs are
  derived from **slambook2** by Xiang Gao
  (<https://github.com/gaoxiang12/slambook2>, MIT), the companion code of
  *14 Lectures on Visual SLAM*. They are marked `// Book code starts/ends`,
  and each file is listed in `THIRD_PARTY_NOTICES.md`.
- **Stanford Bunny:** courtesy of the Stanford 3D Scanning Repository
  (non-commercial use; this also applies to the rendered synthetic images).
- **TUM RGB-D sample frames:** from the TUM RGB-D benchmark (CC BY 4.0;
  Sturm et al., IROS 2012).

See `THIRD_PARTY_NOTICES.md` for the full notices.
