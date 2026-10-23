# Gauss–Newton ICP and ORB frontend benchmark

## Run from the extracted slam-main folder

Ubuntu dependencies (one-time):

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev libopencv-dev python3-numpy python3-matplotlib python3-pil
```

Build, test, run all 36 RGB-D frames and compare the two frontends:

```bash
bash scripts/run_gn_orb.sh
```

For the complete scratch-PnP + frontend audits + GN ICP + ORB benchmark + regenerated Blender scene, install Blender and run:

```bash
sudo apt install -y blender
bash scripts/run_gn_orb.sh --blender
blender -y outputs/blender/bunny_slam_demo.blend
```

The ORB comparison is saved as CSVs and PNGs in `outputs/orb_benchmark/`. The existing Blender scene compares GT, GN ICP, scratch PnP and legacy PnP; it does not add an ORB-SLAM camera.

Run only the benchmark after building:

```bash
./build-icp/orb_benchmark data/synthetic_bunny outputs/orb_benchmark --repeats 3
python3 scripts/render_orb_benchmark.py
```

Run only ICP after building:

```bash
./build-icp/icp_sequence data/synthetic_bunny icp_cpp_results
```

Build without the optional OpenCV ORB comparison using `-DBUILD_ORB_BENCHMARK=OFF` when configuring `icp_cpp/`. The normal ICP library still uses project-owned features and OpenCV image I/O only. OpenCV `features2d` is linked by the separate benchmark executable.

## What changed

- Dense point-to-point ICP now uses analytic damped Gauss–Newton, replacing incremental SVD updates.
- Dense point-to-plane ICP uses Huber-weighted damped Gauss–Newton. Its step bounding now scales the entire direction, and both solvers share the same tested left-update Jacobian.
- Both stages refresh spatial nearest-neighbour assignments every iteration. Each proposed update and line-search trial uses fixed assignments; it must decrease that fixed-pair objective.
- Small diagonal damping, bounded steps, motion gates, minimum support, iteration limits and failure diagnostics remain explicit.
- SVD is retained for the all-feature baseline, minimal RANSAC hypotheses and the RANSAC inlier refit. Initializing robustly is separate from refining dense ICP.
- `registerFramesWithFeatures()` provides the same geometric backend to both benchmark frontends. Production `registerFrames()` calls it with project-owned features.
- `orb_benchmark` compares the project ORB-style extractor and OpenCV ORB on every consecutive pair. It reports actual feature counts, timings, RANSAC support, GT-verified match precision, failed pairs and trajectory error without loop closure.
- Numerical central-difference tests verify all six columns of both ICP Jacobians. Nonzero motion, partial overlap, aligned input, motion-gate rejection and no-overlap cases exercise the actual solver.

## Benchmark interpretation

Both frontends request 1400 features, eight levels, scale factor 1.2, FAST threshold 10, Harris ranking, a 16-pixel image border and 256-bit descriptors. Grayscale conversion is identical. Actual counts are reported and may differ. The production project code uses its own detector scores, pyramid sampling, smoothing and BRIEF implementation; OpenCV uses its implementation. This compares the complete extractors at matched public settings, not one isolated component or byte-identical implementations.

Matching uses the exact same project Hamming distance, a 0.87 nearest/second-nearest ratio in both directions and mutual agreement. Descriptors are only compared within a method. Both methods then use the exact same depth gates, RANSAC seeds, dense point clouds and GN refinement.

Extraction and matching each have one unmeasured warmup followed by repeated measurements (default 3; median retained). Method order alternates. OpenCV uses one thread. Image loading is excluded. Extraction includes grayscale conversion, detector construction and descriptor conversion into the common representation. Registration is timed once and includes matching, depth lifting, RANSAC and both ICP stages; do not add matching time to it again.

Ground truth is read only after all inference. A valid-depth feature match is labelled GT-correct if its transformed 3D distance is below 9 mm. `gt_precision_3d` pools correct/valid-depth matches; it does not measure all-feature precision, recall or descriptor equality. ATE uses frame-zero anchoring, metric scale and no fitted alignment. Benchmark ATE includes the explicitly counted constant-velocity fallback on failed pairs; accepted-pair relative-error means exclude failures. Neither benchmark trajectory uses loop closure. Production `icp_sequence` separately reports both odometry and verified pose-graph results.

This is an ORB frontend benchmark, not an ORB-SLAM/ORB-SLAM3 comparison. Results on a single synthetic bunny are a local correctness/behavior check, not evidence of broad real-scene performance. Timing is hardware dependent.

## Maths, in order

For two consecutive frames, call the older frame target and the newer frame source. We estimate `T_target_source`: it moves source-camera coordinates into target-camera coordinates. A rigid pose is `x = R p + t`, with three rotational and three translational degrees of freedom.

1. **Find image features.** FAST finds corners. Harris ranks them. A geometric budget distributes features over eight image scales. The intensity centroid sets the patch angle; rotated BRIEF forms 256 binary intensity comparisons. Hamming distance counts unequal bits. Best/second-best ratio and mutual tests produce tentative image matches.
2. **Lift with depth.** For pixel `(u,v)` and raw depth `d`, let `z=d/5000`, `x=(u-cx)z/fx`, `y=(v-cy)z/fy`. This makes a 3D point in that camera's coordinates. Both frames have depth, so a feature match becomes a 3D–3D pair. Reject missing depth and a 3x3 neighborhood with a range >=12 mm.
3. **Find a robust initial pose.** RANSAC samples three noncollinear feature pairs, estimates a rigid pose with SVD, and counts pairs closer than 9 mm. Keep the strongest consensus, then refit its inliers. The inlier test is `||R p + t - q|| < 0.009`. This pose starts ICP near the correct alignment.
4. **Build dense clouds.** Sample stable-depth pixels (stride 3, at most 2800 points). Transform each source point by the current pose. Search the target KD-tree for its closest 3D point by Euclidean distance. Keep distances below 24 mm, trim the worst fraction (retain 83%), and retain at most one source per target. These are spatial assignments, not Hamming matches.
5. **Define the point objective.** With assignments fixed during one update, `r_i=R p_i+t-q_i`, and `E=sum ||r_i||^2`. Each residual has three components. The target assignment can change after the pose changes.
6. **Linearize a small pose correction.** Let `delta=[omega_x,omega_y,omega_z,v_x,v_y,v_z]`. For `x=R p+t`, the actual left update is `x_new=Exp(omega)x+v`. For small angles, `Exp(omega)x ~= x + omega cross x`. Thus `r_new ~= r + J delta`, where `J=[-[x]_cross | I]`, a 3x6 matrix. The skew matrix satisfies `[x]_cross y = x cross y`. A Jacobian column tells how a tiny change in one pose variable changes the residual.
7. **Solve Gauss–Newton.** Form `H=sum J^T J` and `b=-sum J^T r`. Solve `(H+lambda I)delta=b`, a 6x6 linear system, with LDLT. Do not explicitly invert H. Damping is `lambda=1e-5 trace(H)/6+1e-9`. This is a regularized GN step with backtracking, not an adaptive Levenberg–Marquardt schedule.
8. **Apply only a descending step.** Scale the direction if its rotation exceeds 0.1 radians or translation exceeds 0.03 m. Try step multipliers 1, 1/2, 1/4, 1/8 and 1/16. Keep a safe step that reduces the current fixed-pair error. Set `R_new=Exp(alpha omega)R`, `t_new=Exp(alpha omega)t+alpha v`. This is an SO(3) x R^3 left retraction. It is not the full SE(3) exponential, which would have a translation Jacobian too. Refresh the 3D neighbours and repeat, up to 18 times or a sufficiently small step.
9. **Refine point-to-plane.** The target normal is the normalized cross product of local depth tangents. Now `e_i=n_i^T(Rp_i+t-q_i)` is one scalar: distance along the surface normal. Its Jacobian is `j_i=n_i^T J_i=[(x_i cross n_i)^T | n_i^T]`. Huber weighting reduces large residuals: `w=1` when `|e|<=0.004m`, otherwise `w=0.004/|e|`. Build `H=sum w j^T j`, `b=-sum w j^T e`, and use the same damped solve and left update, with fixed-pair Huber backtracking. Up to 16 iterations. This robust weighted least-squares procedure is often called IRLS Gauss–Newton. Flat geometry may leave motions weakly constrained; damping does not create missing information.
10. **Accumulate a camera trajectory.** Let `W_i=T_world_camera_i`, `M_i=T_camera_i_camera_(i+1)`. Then `W_(i+1)=W_i M_i`. Initialize `W_0=I`. Ground truth is only used to choose the displayed world anchor and measure error, after estimation.
11. **Use verified loops in the production ICP run.** Candidate revisits are registered and checked. A graph edge compares measured relative motion M with predicted `W_i^-1 W_j`; its pose error is `M^-1 W_i^-1 W_j`. Optimize all poses together with a fixed anchor in each connected component. The existing graph solver uses numerical Jacobians, soft-L1 residual weights and adaptive damping; it is separate from the new analytic dense ICP solve.
12. **Evaluate.** Position ATE is `sqrt(mean ||t_est_i-t_GT_i||^2)` after the fixed frame-zero anchor. Rotation error uses the angle of `R_est R_GT^T`. A low match error does not guarantee low trajectory error; report both, plus failed pairs.

ICP chooses correspondences and repeats pose estimation. Gauss–Newton is the numerical method used to update that pose. Initial Hamming matches find a plausible starting motion; dense ICP then uses Euclidean neighbours in 3D.

Primary references:

- OpenCV ORB parameters: https://docs.opencv.org/4.13.0/db/d95/classcv_1_1ORB.html
- Point/plane ICP objectives: https://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html
- Robust weighted GN for ICP: https://www.open3d.org/docs/0.19.0/tutorial/pipelines/robust_kernels.html

## Actual keypoint overlays

The run commands also export every detected keypoint and render side-by-side OpenCV/project overlays. See `ORB_VISUALS_AND_GN.md` and `outputs/orb_benchmark/keypoint_visuals/`.
