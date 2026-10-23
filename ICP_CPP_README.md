# Current final workflow

See [FINAL_WORKFLOW.md](FINAL_WORKFLOW.md) for the authoritative build/view commands, new
scratch frontend, robust PnP, own fundamental estimator, current measurements
and four-camera Blender scene. The default scratch executable now uses the
improved settings; `--legacy` selects historical settings. The notes and old
measurements below are preserved for history and earlier ablations.

---

# Corrected C++ RGB-D ICP

The corrected implementation is C++17. Run `icp_sequence` for the complete pipeline;
it does not invoke Python or a notebook.

## Build and run on Ubuntu

```bash
sudo apt install build-essential cmake libopencv-dev libeigen3-dev
cmake -S icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release
cmake --build build-icp -j2
ctest --test-dir build-icp --output-on-failure
./build-icp/icp_sequence data/synthetic_bunny icp_cpp_results
```

Run these commands from the extracted `slam-main` directory. The executable takes
`DATASET_DIRECTORY OUTPUT_DIRECTORY`. It returns 0 if every pair is accepted,
2 if some pairs fail (results still exported), and 1 on an input/runtime error.

The same `rgbd_icp`, `icp_sequence`, and `rgbd_icp_test` targets are registered in
the main CMake project. The standalone `icp_cpp/` build avoids the g2o/Sophus
requirements of unrelated original PnP/render targets. The rest of the original
repository is retained.

## Implementation

- `include/tracking/rgbd_icp.hpp`: camera, configuration, poses, results and API.
- `src/tracking/rgbd_icp.cpp`: ORB frontend, geometry and optimizers.
- `apps/icp_sequence.cpp`: full sequence, graph and CSV/trajectory export.
- `src/tracking/icp.cpp`: repaired compatibility API for old callers.
- `tests/unit/rgbd_icp_test.cpp`: numerical and rejection tests.

Pipeline: eight-level project-owned Harris-ranked FAST/rotated BRIEF -> bidirectional ratio tests and mutual
matches -> 3x3 depth-edge filtering -> proper-rotation rigid SVD -> 3D RANSAC ->
changing nearest-neighbour point-to-point ICP -> robust point-to-plane ICP ->
trajectory -> geometrically verified loop closure -> robust pose graph.

The current frontend uses project-owned FAST, Harris selection, rotated BRIEF and mutual Hamming matching. OpenCV supplies image I/O only. SVD, RANSAC, exact
3D KD-tree search, ICP iterations, point-to-plane Gauss-Newton, backtracking and
the pose-graph solver are explicit C++ with Eigen. No Open3D, solvePnP, Python,
or g2o is used by this pipeline.

### Conventions and acceptance

All pair solvers map source frame j into target frame i: `T_i_from_j`.
Camera poses compose as `T_wc[j] = T_wc[i] * T_i_from_j`.
PNG depth is uint16 with metric depth = raw / 5000.

Inference starts at identity. Ground truth is opened only after inference and
loop optimization finish; if present, its first pose supplies the evaluation
world anchor. Ground truth is optional and never supplies an initializer,
correspondence filter or loop acceptance criterion. Image availability determines
the sequence length.

Pair acceptance requires both ICP refinements to perform an update, retain at
least 70 final pairs and yield a finite final NN RMSE below 0.024 m. Acceptance
and convergence are separate diagnostics; hitting an iteration limit can still
produce an accepted estimate. Motion and support gates do not guarantee a global
optimum. Point-to-plane updates minimize Huber-weighted normal residuals; reported
solver RMSE is the Euclidean nearest-neighbour distance.

Rejected pairs use the last accepted relative motion, explicitly flagged in CSV.
ATE and RPE include those fallback transitions. Thus failures cannot silently
vanish from trajectory error metrics. The graph fixes one anchor per connected
component, avoiding free gauges after rejected odometry edges.

Loop search is deliberately specific to this short sequence: temporal gap >=25,
rough odometry overlap, independent feature/RANSAC support, refinement acceptance
and measurement consistency. It is not a general place-recognition system.

### Historical API repair

`pose_estimation_3d3d(pts1,pts2,R,t)` retains the original `pts1 ~= R*pts2+t`
direction. It now uses the proper SVD reflection correction and validates paired
input size/finite values. `bundleAdjustment` remains as a compatibility name;
for fixed unweighted 3D pairs it uses the exact SVD optimum instead of redundantly
optimizing the same objective with g2o from identity. Old callers of this API
still operate on fixed feature pairs. Use the new `icp_sequence` executable for
actual dense iterative correspondence updates and point-to-plane refinement.

The C++ port uses std::mt19937 sampling and a damped robust graph solver with
numerical edge Jacobians. These solve the same objectives as the reference;
random samples and intermediate baseline/point results need not be bit-identical.

## Verification

Built with GCC 13.3, OpenCV 4.6.0 and Eigen 3.4.0 in Release mode.
The standalone build compiles both the new implementation and legacy ICP wrappers.
The whole original g2o/Sophus application suite was not built or certified here.
A second full run with groundtruth.txt absent produced identical pair diagnostics,
loop indices and all five trajectories after undoing the evaluation world anchor.

Tests cover known-transform SVD, proper rotations under reflection, inverse
composition, invalid inputs, RANSAC outlier rejection, aligned point ICP, recovery
of a nonzero point-to-plane transform, skipped refinements, no overlap,
textureless/invalid depth rejection, pose-graph correction and disconnected gauges.

Actual 36-frame bunny run: 35/35 pairs accepted, one 0-to-35 closure, pose graph
converged in 2 iterations. Trajectory ATE RMSE:

| Method | ATE RMSE (m) |
| --- | ---: |
| Feature SVD | 0.264781523 |
| RANSAC SVD | 0.177241149 |
| Point-to-point ICP | 0.066551773 |
| Point-to-plane ICP | 0.001691827 |
| After pose graph | 0.001204402 |

`icp_cpp_results/` includes pair diagnostics, summary metrics, all trajectories,
loop indices and verification logs from the actual C++ run. These measurements
apply to this clean synthetic sequence; real RGB-D scenes need separate validation.

## Blender integration

Run `bash scripts/rebuild_blender.sh` from the repository root to build/test C++,
estimate all frames, rebuild the Blender scene, and check every displayed camera
pose/curve/HUD value plus camera intrinsics. The new scene and results are included.
The executable writes selected_trajectory.txt and run_complete.txt, invalidating
these at run start so an interrupted run cannot be silently reused.
Blender reads this corrected selection directly, never the historical ICP CSV.
