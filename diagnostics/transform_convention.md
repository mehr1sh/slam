# Transform convention: derivation and verification

This page states every pose convention the pipeline uses and gives the test
that confirms each one. The numbers come from `scripts/run_diagnostics.py`,
which runs `tests/synthetic/trajectory_diagnostics_test.cpp`.

## 1. Stored poses

| Quantity | Meaning | Source |
|---|---|---|
| `groundtruth.txt`, `pnp_trajectory.txt`, `icp_trajectory.txt` | **T_wc** (camera → world): `p_world = R_wc · p_cam + t_wc`. `t_wc` is the camera centre. | `slam_trajectory_test.cpp` header; `WriteTrajectoryTUM` |
| Camera axes | OpenCV: +X right, +Y down, +Z forward | `src/render/trajectory.cpp` look-at |
| Rendering | T_cw = T_wc⁻¹, `X_cam = R_cw · X_world + t_cw` | `render_bunny_test`, `RenderMesh` |

## 2. Relative motion of a pair (i, j = i+1)

All three sources are expressed as **T_{j←i}**, where `x_j = R · x_i + t` maps
a point written in camera i into camera j:

| Source | Native output | Conversion | Code |
|---|---|---|---|
| Ground truth | — | `T_gt[j].inverse() * T_gt[i]` = T_cw[j]·T_wc[i] = T_{j←i} | `RunPnpPair`, `RunIcpPair`, `WritePairMetrics` |
| PnP (`solvePnPRansac`, 3D in camera i, 2D in image j) | `x_j = R x_i + t` = **T_{j←i}** | none | `RunPnpPair` |
| ICP (`pose_estimation_3d3d(pts1 = camera i, pts2 = camera j)` + `bundleAdjustment`) | `p_i = R p_j + t` = **T_{i←j}** | `R' = Rᵀ`, `t' = −Rᵀ t` → T_{j←i} | `RunIcpPair` |

## 3. Accumulation

    T_wc[0]   = T_gt[0]
    T_wc[i+1] = T_wc[i] · T_{i+1←i}⁻¹

Derivation: T_{w←i+1} = T_{w←i} · T_{i←i+1}, and T_{i←i+1} = (T_{i+1←i})⁻¹.

## 4. Verification

| Test | What it would catch | Result |
|---|---|---|
| **GT replay**: feed the GT relative motions (built exactly as the pipeline builds them) through the pipeline's accumulation formula | a wrong composition order, a missing or extra inverse, a wrong GT relative-motion convention | max error over 36 frames **5.2e-16 m, 0.0°** (`gt_replay.csv`) → **PASS** |
| Negative control A: `T = T · T_rel` (no inverse) | shows the test is sensitive | 0.174 m / 20.0° already at frame 1, up to 1.00 m / 180° (`gt_replay_controls.csv`) |
| Negative control B: `T = T_rel⁻¹ · T` (wrong side) | as above | 0.269 m / 19.85° at frame 1, up to 1.67 m / 178.3° |
| **Perfect-correspondence PnP**: same 3D points, 2D = their exact projection under the GT T_{j←i}, same `solvePnPRansac` call | PnP returning T_{i←j}, a wrong K, a wrong back-projection or projection model | max rotation error **5.1e-6°**, max translation error **4.6e-8 m** over 35 pairs (`perfect_correspondence_test.csv`) → **PASS** |
| **Perfect-correspondence ICP**: same p_i, q = T_{j←i}·p_i, same `pose_estimation_3d3d` + `bundleAdjustment` calls, then the same R′ = Rᵀ, t′ = −Rᵀt conversion | ICP's native direction or the inversion being wrong | max rotation error **5.4e-6°** (SVD and g2o), max translation error **4.6e-8 m** → **PASS** |
| Re-running the solvers on the same data and re-accumulating | non-determinism, a mismatch between the exported motions and the files | max **7.8e-7 m** from `pnp_trajectory.txt` / `icp_trajectory.txt`, which store 6 decimals (`cpp/reaccumulation_check.csv`) → **PASS** |
| Python re-accumulation of the exported relative motions | an independent implementation of the same maths | final errors 0.27673 m / 32.1212° (PnP) and 0.52528 m / 64.9186° (ICP), equal to the C++ values → **PASS** |

**Half-pixel note (not a convention error that affects the result).** The
rasterizer samples pixel (x, y) at the continuous point (x+0.5, y+0.5)
(`src/render/rasterizer.cpp:77`). OpenCV places pixel centres at integer
coordinates. The exactly consistent intrinsics for back-projection and PnP are
therefore cx − 0.5, cy − 0.5. The pipeline uses cx, cy. Re-solving PnP with the
shifted principal point changes the mean estimated rotation from 9.132° to
9.138° and the mean rotation error from 1.391° to 1.371° (`solver_sensitivity.csv`).
The effect is **negligible** and does not explain the errors.

**Conclusion:** the transform conventions, the ICP inversion and the
accumulation are correct.
