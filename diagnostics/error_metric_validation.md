# Error metric validation

## Definitions used by the C++ code (`tests/report_utils.hpp`)

- **Rotation error:** `RotationAngleDeg(R1, R2) = acos(clamp((trace(R1ᵀ R2) − 1) / 2, −1, 1))`
  in degrees, i.e. the geodesic angle between two rotations.
- **Translation error:** `TranslationDiff(t1, t2) = ‖t1 − t2‖`.
  - **Absolute (per frame)** errors are applied to T_wc, so t is the **camera
    centre** in world coordinates. This is the physically meaningful choice.
  - **Relative (per pair)** errors are applied to T_{j←i}, so t is the
    translation of the relative motion expressed in camera j. Its norm equals
    the camera-centre distance (0.0872 m for every GT pair; see
    `gt_relative_motion.csv`).

## Checks (`scripts/run_diagnostics.py`)

| Check | Result |
|---|---|
| Absolute errors recomputed in Python (NumPy, own quaternion→R) from the poses in `slam_trajectory.csv` vs the C++ error columns, 2 methods × 36 frames | max difference **8.3e-7 m** and **7.0e-5°**. This is the 6-decimal rounding of the CSV (`error_metric_recomputation.csv`). |
| Relative errors recomputed in Python from the exported relative poses vs the C++ values, 2 methods × 35 pairs | agree within 1e-4° / 1e-7 m. The script aborts otherwise. |
| Known-answer tests: B = A·Rot(axis, θ) and B = Rot(axis, θ)·A for random A and axis, θ ∈ {0, 0.5, 1, 10, 45, 90, 135, 179}° | the metric returns θ within **2.3e-12°**. It is correct for both right and left perturbations and has no wrap-around problem below 180°. |
| Quaternion sign: R(q) vs R(−q) | **0.0°**. The metric works on rotation matrices, so q/−q ambiguity cannot inflate it. |
| Camera centre vs T_cw translation | The absolute translation error uses T_wc.translation() (the centre). Using T_cw's translation instead gives different numbers (column `translation_of_T_cw_diff_m`), which confirms the metric does not mix up the two. |
| Relative-error inputs | GT relative motion = 10.000° ± 0.0001° and 0.08724 m for all 35 pairs (`gt_relative_motion.csv`), as the 10° orbit requires: 2·0.5005·sin 5° = 0.08724 m. |

## Interpretation caveat (not an error)

The absolute errors are **accumulated** errors. Frame 0 is anchored to GT and
there is no alignment step (no Umeyama/SE(3) alignment, no ATE/RPE
normalisation). Because the camera orbits the object, a heading error at frame
k also shows up as a position error at every later frame. That is intended,
and it is why a ~10% under-estimate of the 10° steps becomes 0.28 m (PnP) and
0.53 m (ICP) by frame 35.

**Conclusion:** the translation-error and rotation-error calculations are correct.
