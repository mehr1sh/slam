# Actual ORB keypoints and basic Gauss-Newton ICP

The added comparisons show points exported by the exact C++ extractors used in `orb_benchmark`, on the same RGB image and with the same crop and magnification.

## Run

From the `slam-main` directory:

```bash
bash scripts/run_gn_orb.sh
```

This now also exports every retained keypoint's original-image position, pyramid level and orientation for all 36 frames, then renders the keypoint comparisons. The export runs outside the timed extraction section.

For a full scratch-PnP, ICP and Blender rebuild:

```bash
bash scripts/run_gn_orb.sh --blender
blender -y outputs/blender/bunny_slam_demo.blend
```

To redraw only the visual evidence from existing C++ exports:

```bash
python3 scripts/render_orb_keypoints.py
```

## Where to look

- `outputs/orb_benchmark/keypoints/`: all 72 frame-method CSVs.
- `outputs/orb_benchmark/keypoint_visuals/`: same-image overlays, comparison PNGs, pairing CSVs and unpaired-point CSVs.
- `presentation/SLAM_Final_Slides_ORB_Visuals_GN.pdf`: revised 13-slide PDF. All eight original slides remain. Earlier point-ICP figures are updated, actual ORB point overlays occupy slides 9-10, basic GN explanation occupies slides 11-12, and the full benchmark/equations slide remains as slide 13.

## How to interpret the difference overlay

The main example is frame 15. The complete feature sets contain 1313 project points and 1292 OpenCV ORB points. Greedy one-to-one spatial pairing within two original-image pixels at the same pyramid level finds 932 pairs, leaving 381 project points and 360 OpenCV points unpaired.

At pyramid level 0 both extractors retain 304 points. There are 251 spatial pairs and 53 unpaired points in each method. Thus the same count does not imply identical locations.

The third panel uses green for paired nearby locations, blue circles for unpaired project points and orange triangles for unpaired OpenCV points. The first two panels show all retained points for their method. No points are subsampled. Green markers are pair midpoints; raw positions remain in the CSVs.

This tolerance-based location overlap measures visual similarity of the selected positions. It does not test descriptor equality, physical correspondence or correctness. Spatial pairs are formed only within the same pyramid level. The overlay does not compare descriptors from the two implementations.

## Code review and solver identity

`extractFeatures()` uses the project's FAST detector, Sobel Harris ranking, geometrically distributed quotas, intensity-centroid orientation and smoothed rotated BRIEF. The OpenCV benchmark uses `cv::ORB::create(1400,1.2,8,16,0,2,HARRIS_SCORE,31,10)`. Both use identical project grayscale conversion, Hamming matching, ratio/mutual checks, depth gates, RANSAC and dense GN ICP.

`pointJacobian()` returns `[-[x]_cross, I]`. `pointICP()` accumulates `H=sum J^T J` and `b=-sum J^T r`, adds diagonal damping `1e-5*trace(H)/6+1e-9`, and solves the six-variable system with LDLT. It bounds a step's rotation/translation, backtracks over five scales, and checks that the current fixed-pair objective decreases. Neighbours refresh on the next iteration. The cap is 18 point iterations.

`planeICP()` projects each residual and Jacobian onto the target normal and applies Huber weights with a 4 mm transition. It uses the same normal-equation solve and pose update, with a cap of 16 iterations. It checks descent using the fixed-pair Huber objective.

`leftIncrement()` uses `(R,t) <- (Exp(omega)R, Exp(omega)t+v)`, a left SO(3) x R^3 retraction. This is basic damped pose-only Gauss-Newton with backtracking. There is no adaptive Levenberg-Marquardt damping schedule in these dense ICP functions. The separate existing PnP/pose-graph optimizers have their own refinement behavior.

Feature 3D RANSAC still initializes the pose with SVD. A GN solve updates one relative pose while keeping cloud coordinates fixed. This is not joint bundle adjustment of poses and landmarks.

## Current measurements

The new keypoint metadata/export does not change the estimated poses. The full 36-frame benchmark was rerun and both methods still solve all 35 pairs. Trajectory RMSE is 1.688436 mm for the project frontend and 1.691829 mm for OpenCV ORB with the same GN backend and no loop closure.

Current production ICP stages: 45.823969 mm after point GN, 1.688436 mm after plane GN, and 1.200057 mm after the verified loop/pose graph. The old slides' 47.31 mm was from the earlier SVD-update run. Frame pair 15-16 now has 16 point updates (13 retained-assignment-set changes) and six plane updates (three set changes).

Latest measured median feature extraction: 81.345 ms for the project and 4.405 ms for OpenCV. Timings vary with hardware and load. Registration includes matching, so those columns should not be added together. The corresponding accuracy/count measurements are unchanged.

## Verification

The independent C++ Release build and all three ICP CTest entries pass. The benchmark's 72 keypoint CSVs agree with frame counts, image bounds, pyramid levels and finite orientations. Spatial pair indices are unique, distances satisfy the declared tolerance and unpaired counts reconcile. All 35 pairs are accepted in both methods. No GN solver update was introduced for this visualization change.
