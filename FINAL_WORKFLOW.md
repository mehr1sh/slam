# Updated: Gauss–Newton ICP + ORB benchmark

From this directory, run `bash scripts/run_gn_orb.sh`. For the complete Blender rebuild, run `bash scripts/run_gn_orb.sh --blender`.

See [GN_ORB_GUIDE.md](GN_ORB_GUIDE.md) for dependency installation, precise changes, benchmark controls, and the complete maths. Current benchmark measurements are in `outputs/orb_benchmark/RESULTS.md`; dense ICP metadata is in `icp_cpp_results/solver_metadata.txt`.

The production dense point and plane stages both use damped Gauss–Newton. SVD remains for robust feature initialization. Historical measurements and older method descriptions below refer to earlier revisions unless explicitly updated.

---

# Direct images and Blender output

All current visual outputs are in `outputs/`. Open the folder directly:

```bash
xdg-open outputs
```

- `outputs/frontend/pair_15_16/`: PNG before/after comparisons for Harris ranking,
  eight-level quotas and mutual ratio matching, with complete CSV evidence.
- `outputs/frontend/pair_0_1/`: the same comparisons for a pair where quotas do
  not remove any candidates.
- `outputs/icp/`: cloud alignment, iteration updates, trajectory/error PNGs and CSVs.
- `outputs/00_pipeline_flowchart.png`: actual ICP and independent PnP flow.
- `outputs/blender/bunny_slam_demo.blend`: the Blender animation scene.

No HTML gallery or browser is required. PNGs and the Blender scene are already
included; rebuilding reproduces them. See `AUDIT_REPORT.md`, `FRONTEND_VISUALS.md`
and `FINAL_WORKFLOW.md` for the verification, interpretation and commands.

---

# Final C++ pipelines and Blender comparison

Run commands from the extracted `slam-main` folder. Keep the directory tree intact.

## 1. Install dependencies (Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev libopencv-dev blender python3 python3-numpy python3-matplotlib python3-pil unzip
```

If Blender is already installed separately, use that installation's `blender`
command. The viewer was checked with Blender 4.0.2. Version-dependent EEVEE,
compositor and animation-action APIs have compatibility branches.

## 2. Rebuild, test, estimate and create the scene

```bash
bash scripts/rebuild_blender.sh
```

This builds the scratch pipeline, runs its 17 tests, estimates all 36 poses,
builds and tests corrected ICP, estimates its 36 poses and loop closure, then
writes PNG comparisons to outputs/, then builds and validates the Blender scene. Allow several minutes for a first build.
No Python notebook is involved. Python scripts are display-data and Blender glue;
estimation, feature extraction and optimization are C++.

## 3. Open the generated scene

```bash
blender -y outputs/blender/bunny_slam_demo.blend
```

Press Space to animate. Press Numpad 0 to look through the active ICP camera.
Press N and use the Trajectories sidebar to toggle methods and select cameras.
An external overview shows the stationary bunny and four camera trajectories:

- Green: ground truth.
- Magenta: corrected ICP with verified loop closure when available.
- Cyan: improved from-scratch PnP (independent PnP trajectory).
- Orange: legacy reference PnP, retained as an explicitly labelled baseline.

Magenta should nearly overlap green. Cyan should follow the orbit much more
closely than the old scratch run, but still has residual drift. Orange retains
the older baseline errors; it is not silently relabelled as a new algorithm.

The ZIP includes generated outputs and a ready-to-open scene. You may skip step 2
if you only want to view the supplied results; step 2 reproduces them from source.

## Measurements

The exact current measurements are in `icp_cpp_results/summary_metrics.csv` and
`pnp_from_scratch/results/pipeline/summary.txt`. Approximate expected bunny results:

| Measurement | Result |
| --- | ---: |
| Consecutive pairs solved, scratch PnP | 35 / 35 |
| Consecutive pairs accepted, ICP | 35 / 35 |
| ICP verified loop closures | 1 |
| Point-to-plane ICP ATE RMSE | 1.69 mm |
| ICP plus pose-graph ATE RMSE | 1.20 mm |
| Improved scratch PnP ATE RMSE | 23.86 mm |
| Improved scratch PnP final position error | 40.46 mm |
| Improved scratch PnP final rotation error | 3.86 degrees |
| Improved scratch PnP mean relative rotation error | 0.484 degrees |
| Improved scratch PnP mean relative translation error | 4.3 mm |

These results combine frontend, filtering and optimizer changes; they are not an isolated Harris-ranking ablation or a controlled comparison with the old baseline feature budget.

The slides' old scratch final error was 284 mm / 31.9 degrees; that is historical,
not the expected output of the improved default. RMSE is an aggregate metric;
individual frames' HUD values vary.

## Implemented changes

1. Project-owned Sobel-gradient Harris corner response, used to rank FAST candidates.
2. A single feature budget distributed geometrically across eight pyramid levels.
3. Project-owned mutual Hamming nearest/second-nearest ratio matching.
4. Source-depth 3x3 validity/discontinuity filtering for scratch PnP.
5. Huber-weighted reprojection LM and post-refinement inlier reselection.
6. Eigen-only normalized eight-point fundamental estimator with degeneracy and
   rank-two checks. Used in scratch diagnostics and replaces the reference F call.
7. Corrected ICP now uses the same project-owned feature extraction/matching.
   OpenCV there supplies image decoding/storage only; its ORB/BFMatcher are gone.
8. The all-in-one rebuild generates the scratch trajectory and requires it in
   Blender. Source hashes include the scratch trajectory, intrinsics and mesh.
9. Both C++ trajectories, all four Blender cameras, curves, errors and calibration
   are checked. Source changes invalidate the scene freshness check.

The scratch-PnP executable links no OpenCV or image/compression library; it uses
its own PNG reader plus Eigen/standard C++. The ICP executable still links OpenCV
for RGB-D image I/O. Do not claim the entire repository has no OpenCV dependencies.

## Limits

Residual odometry drift is reduced, not eliminated. The tiny bunny sequence can
be ill-conditioned for essential-matrix recovery, so that result remains a
scale-free diagnostic rather than a metric pose source. A correct fundamental
estimator does not make noisy or degenerate correspondences well-conditioned.
Loop search is designed for this short sequence. Real scenes, dynamic objects,
lens distortion, mapping and relocalization need additional work and validation.

Camera/animation/HUD/projection checks are run inside Blender; final image
rendering could not be validated in this headless environment. Earlier reports
and ablations are retained for history and may describe the old frontend.
