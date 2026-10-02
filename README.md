# Stanford Bunny RGB-D: PnP & ICP trajectory estimation

A C++17 visual-odometry study built step by step from *14 Lectures on Visual
SLAM* (slambook2). It contains:

1. **A synthetic RGB-D dataset generator.** A CPU rasterizer renders the
   Stanford Bunny from a known circular camera orbit, producing RGB, depth and
   exact ground-truth poses.
2. **Frame-to-frame pose estimation** on that sequence:
   - **PnP:** 3D points from depth in frame *i*, matched to 2D points in frame *i+1*.
   - **ICP:** 3D–3D alignment of matched points.
3. **Trajectory accumulation** of the 35 relative motions into global
   camera trajectories, compared with ground truth.
4. **A Blender visualization** in which real, animated Blender cameras
   (GT, PnP, ICP, and the from-scratch linear PnP) move along their trajectories around the bunny.

The feature-based two-view frontend from the book (ORB matching, 2D-2D pose,
triangulation, PnP/ICP on a real TUM RGB-D frame pair) and some from-scratch
components (FAST, BRIEF, Hamming matching, 8-point essential matrix) are
included as well.

> **Scope.** This is *not* a full SLAM system. There is no map, no keyframes,
> no loop closure, no global optimization, and **no fused PnP+ICP backend**.
> PnP and ICP each produce an independent dead-reckoning trajectory. The file
> `slam_trajectory.csv` and the program `slam_trajectory_test` are named after
> the experiment. They contain and produce only the GT / PnP / ICP comparison,
> not a fused SLAM estimate.

## Repository layout

```
include/<module>/        public headers
src/<module>/            implementation
  camera/                PinholeCamera, pixel2cam() / cam2pixel()
  features/              ORB extraction + brute-force Hamming matching + distance filter
  geometry/              2D-2D pose (F/E/H + recoverPose), triangulation
  optimization/          g2o SE(3) pose vertex (header-only, include/ only)
  tracking/              PnP & ICP (+ Gauss-Newton / g2o refinement),
                         from-scratch FAST, BRIEF, Hamming matching, 8-point E
  render/                PLY I/O, CPU rasterizer, orbit trajectory, PLY scene export
tests/
  unit/                  self-contained (synthetic inputs)
  integration/           on data/tum_sample/ (real TUM RGB-D frame pair)
  synthetic/             on data/synthetic_bunny/, incl. the dataset generator
                         (render_bunny_test) and the trajectory pipeline (slam_trajectory_test)
data/
  meshes/bunny/          Stanford Bunny mesh (bun_zipper.ply)
  synthetic_bunny/       generated dataset + reference trajectories (see its README)
  tum_sample/            one real TUM RGB-D frame pair (+ depth)
visualization/           Blender scene builder (see visualization/README.md)
scripts/                 generate_results.py: figures, CSVs and tables for results/
results/                 README.md (tracked) + generated figures/, data/, tables/
presentation/            progress slides (LaTeX source + PDF)
docs/                    reference numbers for the chapter 6/7 programs
COMMANDS.md              every command in one place
```

Generated artifacts are not tracked: `build/`, `output/` (images, PLYs and
CSVs written by the programs) and the Blender scene
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

## The synthetic dataset

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

## Pose estimation

For each consecutive pair (i, i+1), `slam_trajectory_test` does the following.

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

- the bunny
- the three full trajectory lines: GT green, PnP orange, ICP magenta
- `GT_Animated_Camera`, `PnP_Animated_Camera`, `ICP_Animated_Camera`: real
  Blender cameras, keyframed so that frame *i* = pose *i*
- `ScratchPnP_Animated_Camera` and `ScratchPnP_Trajectory` (cyan): the
  from-scratch linear PnP of `pnp_from_scratch/`. It runs on raw matches with no
  RANSAC, so it drifts badly by design; see `pnp_from_scratch/README.md`
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
  print results rather than assert them. There is no CTest integration.
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
