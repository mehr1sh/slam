# Audit of the uploaded ZIP and the two screenshots

The screenshot requirements are implemented in the current C++ pipelines.
This is an RGB-D odometry experiment with verified loop correction in ICP,
not a full mapping/relocalization system.

| Requirement | Verified code path | Before/after evidence in outputs/ |
| --- | --- | --- |
| Harris scoring and ranking | `HarrisResponse`, `ExtractMultiscale` in `src/features/multiscale.cpp`; both active CLIs enable it | `frontend/pair_15_16/*01_harris_ranking_before_after.png`, `*identifier_changes.png`, `*03_ranking_same_quota.png` |
| Quotas across an eight-level pyramid | `LevelQuotas`, `ExtractMultiscale`; total budget 1400, scale 1.2 | `*02_quotas_before_after.png`, `04_pyramid_quotas.png`, complete per-level CSVs |
| Mutual ratio matching | `MatchMutualRatio` in `src/features/matcher.cpp`; ratio 0.87 in both directions and mutual best | `05_matching_before_after.png`, `06_matching_filter_counts.png`, all proposal decisions in CSV |
| Iterative nearest-neighbour search | `pointICP` and `planeICP` call `nearestPairs` inside each iteration; custom exact KD tree | `icp/03_iteration_updates.png`, `04_nearest_neighbour_refresh.png`, `iteration_history.csv` |
| Point-to-plane constraints | Target normals, residual n dot (transformed source - target), robust normal-equation update and backtracking in `planeICP` | `icp/05_cloud_alignment_before_after.png`, `03_iteration_updates.png`, trajectory and error plots |

The eight-level pyramid existed before this diagnostic update. Harris and quotas
are selection steps on that pyramid. The screenshots do not imply that removing
more keypoints is always better. Frame 0->1 is below all quota limits, and the
figures truthfully show no membership reduction there.

## Issues fixed in this update

1. The supplied frontend diagnostics were SVG/HTML, with no primary PNG output.
   They now export directly viewable PNGs and CSVs under `outputs/frontend/`.
   The new workflow neither creates nor opens an HTML gallery.
2. ICP lacked direct cloud/iteration/stage comparison images. Actual C++ update
   records and sampled clouds now drive the plots in `outputs/icp/`.
3. Harris responses were recomputed during sorting. They are now computed once
   per candidate; stable sorting preserves equal-score ordering.
4. Stale README status entries described fixed-pair SVD/g2o and unimplemented
   loops. The current status, paths, plotting dependencies and commands are updated.
5. The Blender scene and its optional render destination are now under `outputs/`;
   relative script loading is adjusted for the new scene location.

## Verification and expected results

- All 17 scratch tests and both ICP/Blender-data tests pass. One scratch executable
  lost its executable bit in this hosted workspace; restoring that local bit let
  its test pass. Build artifacts are excluded from the ZIP.
- Exact estimator agreement is checked for both diagnostic pairs: candidate IDs,
  Harris order, quota limits, descriptors and final matcher correspondences.
- All 36 current keypoint CSVs, the scratch trajectory, all ICP stage trajectories
  and ICP summary metrics are byte-identical to the uploaded version. The caching
  change and diagnostic logging do not improve or alter the trajectory estimates.
- PNG integrity/hashes, plotted metric agreement, rigid stage transforms, per-pair
  iteration counts and fixed-pair objective descent are checked by
  `scripts/validate_outputs.py`. Actual pair 15->16 assignment sets change in both
  ICP stages. Point-to-plane minimizes a different objective from Euclidean NN RMSE.
- Blender checks verify source fingerprint, all camera poses, curves, HUD and
  intrinsics after saving at its new location. Logs are in `outputs/verification/`.

Expected bundled sequence: 36 poses, 35/35 ICP pairs accepted, 35/35 PnP pairs
solved, one verified loop. Position ATE RMSE: feature-pair SVD 245.49 mm, RANSAC
156.31 mm, point ICP 47.31 mm, plane ICP 1.69 mm, pose graph 1.20 mm; independent
scratch PnP 23.86 mm. These are cumulative method stages, not controlled ablations
of individual optimizations. Scratch PnP still has about 40.46 mm final position
error and 3.86 degrees final rotation error; that residual drift is not fixed by
this visualization update.

The orange Blender curve is a frozen historical PnP baseline. It is not a fair
same-correspondence comparison between OpenCV's PnP solver and scratch PnP.
Accepted registration is not a guarantee of convergence or a globally correct
pose; planar geometry, ambiguity, low overlap and depth errors remain limitations.
Blender animation is delivered as a scene. No pre-rendered video is claimed.
