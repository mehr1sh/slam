# Results: reference pipeline on the synthetic evaluation sequence

Figures, CSVs and tables comparing **ground truth**, **PnP-based trajectory
estimation**, and the project's **feature-correspondence-based 3D→3D ICP**
(SVD alignment + g2o refinement). Each method estimates frame-to-frame
relative motion on 36 rendered RGB-D frames; the motions are then chained into
trajectories.

This is frame-to-frame odometry only. The following are **not** part of it:
- full SLAM
- mapping
- loop closure
- global optimization
- dense nearest-neighbour ICP

## Regenerate

```bash
pixi run build              # C++ (once)
pixi run -e results results # everything below; about 30 s
```

`scripts/generate_results.py` does the following:
1. Runs `build/slam_trajectory_test` on a temporary copy of
   `data/synthetic_bunny/`, with the export-only flags
   `--export-dir results/data/cpp_export --export-pair 0`.
2. Checks that this run reproduces the committed `pnp_trajectory.txt`,
   `icp_trajectory.txt` and `slam_trajectory.csv` byte for byte.
3. Reads those files, the dataset images and the bunny mesh.
4. Cross-checks the numbers and aborts if any check fails:
   - CSV poses against the TUM files
   - errors re-derived from the poses
   - relative motions re-chained with `T_wc[i+1] = T_wc[i]·T_rel⁻¹`, which must reproduce the trajectories
   - one keypoint back-projected by hand, compared with the C++ 3D point
5. Writes everything below.

`--skip-cpp` reuses the existing `results/data/cpp_export/`.

No figure contains hand-typed numbers. Everything is read from these outputs.
The generated directories (`figures/`, `data/`, `tables/`) are gitignored.

## Where the numbers come from

| Data | Produced by |
|---|---|
| GT / PnP / ICP poses and absolute errors | `slam_trajectory_test` → `data/synthetic_bunny/slam_trajectory.csv` |
| Per-pair counts, relative errors and relative poses | `slam_trajectory_test --export-dir` → `cpp_export/pair_metrics.csv` (the same values it prints in its PER-PAIR table) |
| Keypoints, matches and 3D correspondences of pair 0→1 | `slam_trajectory_test --export-dir` → `cpp_export/pair_0_1_*.csv` |
| Images, depth, intrinsics, mesh | `data/synthetic_bunny/`, `data/meshes/bunny/bun_zipper.ply` |

## Metrics

- **Translation error (per frame):** ‖C_gt − C_est‖, the distance between the
  ground-truth and estimated camera centres, in metres.
- **Rotation error (per frame):** the angle of R_gtᵀ·R_est, in degrees.
- Both are **absolute and accumulated**: measured against ground truth, with
  every trajectory starting from the ground-truth pose 0 and no other alignment.
  The Blender HUD shows exactly these values.
- **Relative (local) error of pair i→i+1:** the same two measures, applied to
  the estimated motion T_{i+1←i} versus the true motion. This is the error of
  one step before accumulation.
- **Correspondences:**
  - PnP 3D→2D pairs = filtered matches with valid depth in frame *i*
  - ICP 3D→3D pairs = filtered matches with valid depth in *both* frames
- **PnP inlier ratio:** RANSAC inliers / PnP correspondences. ICP has no
  outlier rejection, so it has no inlier ratio.

## Files

| File | Content |
|---|---|
| `data/trajectory_metrics.csv` | one row per frame: GT / PnP / ICP camera centres, absolute translation and rotation errors |
| `data/pair_metrics.csv` | one row per pair: keypoints, raw and filtered matches, PnP correspondences and inliers, inlier ratio, ICP correspondences, relative errors |
| `data/cpp_export/` | the raw C++ export (see above) plus the program log |
| `tables/summary.md`, `tables/summary.json` | dataset facts, PnP vs ICP statistics, the pair 0→1 funnel |

## Figures (`figures/*.png`, plus a `.pdf` of each)

| Figure | Shows |
|---|---|
| `fig01_experiment_setup` | the orbit seen from above and the side: 36 poses 10° apart, 35 relative motions, radius, height and camera spacing; the camera moves, the bunny stays fixed |
| `fig02_rgbd_backprojection` | frame 0 RGB and depth, and one real PnP inlier turned from (u, v, Z) into (X, Y, Z) using intrinsics.txt |
| `fig03_orb_keypoints` | the ORB keypoints the C++ run detected in frames 0 and 1 |
| `fig04_feature_matches` | pair 0→1 matches: kept (green) and rejected by the distance filter (red), with the filter rule |
| `fig05_pnp_vs_icp_concept` | PnP (3D→2D, RANSAC) next to *our ICP implementation* (3D→3D feature pairs, SVD + g2o) |
| `fig06_correspondence_funnel` | keypoints → raw → filtered → PnP pairs → inliers, and ICP pairs, for pair 0→1, plus the spread over all pairs |
| `fig07_correspondence_points` | the actual PnP and ICP 3D points of pair 0→1, and a table separating mesh vertices, pixels, keypoints, correspondences and poses |
| `fig08_trajectories_3d` | GT / PnP / ICP trajectories around the bunny, with start, end and final errors |
| `fig09_translation_error`, `fig10_rotation_error` | accumulated error per frame (unsmoothed) |
| `fig11_local_pair_errors` | relative error of every pair; the worst pair is annotated |
| `fig12_correspondence_counts` | matches and correspondences per pair |
| `fig13_pnp_inlier_ratio` | PnP inlier ratio per pair against its rotation error |
| `fig14_local_vs_global_error` | how chaining turns small local errors into large global drift, using the measured values |
| `fig15_summary` | a one-slide summary: setup, matching, trajectories and a PnP vs ICP table |

## Correspondence figures

`scripts/generate_correspondence_map.py` (`pixi run -e results correspondence-map [--pair K]`)
re-runs the C++ export for pair K → K+1 and checks every keypoint, match, depth value,
3D point and RANSAC flag against the dataset and the C++ counts. It then writes these
maps for visual auditing:

- `figures/bunny_correspondences_<i>_<j>.{png,pdf,svg}`: all matches, with three zooms
- `figures/bunny_correspondences_<i>_<j>_inliers.*`: RANSAC inliers only
- `figures/bunny_correspondences_<i>_<j>_outliers.*`: rejected matches only, numbered

Colour is the correspondence ID: it is fixed by the match's frame-i position, and the
same colour marks its other end in frame i+1. Marker size encodes frame-i depth.

`scripts/generate_correspondence_figure.py` (`... correspondences`) makes the annotated,
explanatory versions (`bunny_correspondences_annotated_*`, `bunny_pnp_icp_correspondences*`).

## Reading the main result

**Local errors are small; they add up over the circle.**
- PnP's local rotation error is about 1.4° per pair on average; ICP's is about 2.8°.
- Summed over the circle, they leave PnP 32° and ICP 65° off at frame 35
  (see `tables/summary.md` for the exact values).

**PnP drifts gradually.** Its largest single pair (24→25) is about 7% of the
sum of its local errors.

**ICP has a few larger steps.**
- Pair 24→25 alone contributes 13.2°, about 13% of its sum, and is also a dip
  in correspondences.
- ICP keeps mismatched feature pairs, because it has no outlier rejection.
  That makes it more sensitive than RANSAC PnP.

**Correspondence quality doesn't predict PnP's step error.** The PnP inlier
ratio stays at 0.86–0.98 throughout, and it is essentially uncorrelated with
PnP's per-pair rotation error (Pearson ≈ −0.08).
