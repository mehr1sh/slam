# Checkpoint 02: Scratch PnP in Blender, library-free scratch figures

## Root cause of the Scratch PnP camera problem

The Blender scene loaded the **wrong scratch trajectory file**, not a wrongly converted pose.

- `visualization/scripts/build_scene.py` (`SCRATCH_TRAJECTORY_CSV_PATH`) and
  `scripts/check_blender_consistency.py` read
  `pnp_from_scratch/results/scratch_pnp_trajectory.csv`.
- That file is **milestone 2**: linear PnP on raw matches **without RANSAC**.
  - 5 of its pairs are physically invalid (most points behind the camera).
  - It ends 0.477 m / 143.8° from GT.
  - Its camera looks up to **141.7°** away from the bunny (90.7° at frame 10).
- The self-contained RANSAC pipeline writes `results/pipeline/trajectory.csv`
  (0.095 m / 9.9°). It was never loaded.

The transform chain was already correct, and it is unchanged:

    relative pose  T_{i+1<-i}:  x_{i+1} = R x_i + t           (scratch PnP, pairs.csv)
    trajectory     T_wc[0] = T_gt[0];  T_wc[i+1] = T_wc[i] T_{i+1<-i}^-1
    CSV            tx ty tz = camera centre C_w;  r.. / q.. = R_wc (camera -> world), CV camera axes
    Blender        trajectory_to_blender_pose(): location = C_w, rotation = R_wc · R_FIX
                   (180° about the camera's own X: CV +Z forward / +Y down -> Blender -Z forward / +Y up),
                   parented under WorldRoot (+90° about X: Y-up -> Z-up). This is the same path as GT/PnP/ICP.

**Classification:** H (something else), i.e. a data-source selection.
- A (relative pose): verified correct.
- B (accumulation): verified correct.
- C (camera centre): verified correct.
- D (rotation transpose/inverse): verified correct.
- E (Blender conversion): verified correct.
- F (keyframing): verified correct.
- G (forward axis): verified correct.

## Evidence

**Numerical** (`trajectory_check.txt`, from `pnp_from_scratch/tools/check_trajectory.py`):
- The pipeline trajectory satisfies:
  - frame 0 = GT frame 0
  - quaternion = matrix
  - T_wc[i+1]⁻¹·T_wc[i] = the exported T_{i+1←i} for all 35 pairs (entries to 1.6e-9)
  - C = −R_cwᵀ·t_cw (7.5e-10 m)
- The explicit 0 → 1 chain (T_rel, T_rel⁻¹, T_wc[1], C_w[1]) recomputed independently
  equals the CSV:
  - C_w[1] = (−0.107916, 0.216838, −0.502289); GT is (−0.103750, 0.235277, −0.494425)
  - 0.0205 m / 2.31° off, look-at error 0.05°
- **Look-at error** is the angle between the optical axis and the direction to the bunny
  centre, which is recovered independently as the intersection of the GT optical axes.

| | final position / rotation error | look-at error, max |
|---|---|---|
| GT | 0 / 0° | 0.00° |
| reference PnP | 0.277 m / 32.1° | 2.29° |
| reference ICP | 0.525 m / 64.9° | 3.18° |
| scratch milestone 2 (before) | 0.477 m / 143.8° | **141.69°** |
| scratch pipeline (after) | 0.095 m / 9.9° | **5.61°** |

**Blender** (`blender_check_before.txt`, `blender_check_after.txt`, from the saved scene):

- **Before and after:** every camera equals its data, within 3e-8 m and 0°. Blender's
  look-at equals the data's look-at within 8e-4°.
- **Before:** the scratch camera's look-at was up to 141.69° (mean 42.50°).
- **After:** at most 5.61° (mean 1.80°).
- **GT / PnP / ICP:** identical before and after (look-at maximum 0.00 / 2.29 / 3.18°).

Requested frames, scratch camera after the fix (Blender world, metres):

| frame | position | look-at |
|---|---|---|
| 0 | (+0.0000, +0.5005, +0.2023), the GT anchor (same as all cameras) | 0.00° |
| 1 | (−0.0911, +0.5008, +0.1839), GT (−0.0869, +0.4929, +0.2023) | 0.05° |
| 2 | (−0.1800, +0.4690, +0.1858) | 0.09° |
| 10 | (−0.4909, −0.1305, +0.1295) | 0.89° |
| 11 | (−0.2939, −0.3973, +0.1876), after the 29.7° pair 10→11 | 1.23° |
| 35 | (−0.0049, +0.4863, +0.2267), GT (+0.0869, +0.4929, +0.2023) | 5.61° |

## Files changed in this checkpoint

| Commit | Files | Change |
|---|---|---|
| `bd0ca2d` | `pnp_from_scratch/tools/check_trajectory.py` (new), `pnp_from_scratch/src/pipeline_main.cpp` | frame-by-frame transform-chain check; `pairs.csv` also exports the relative rotation matrix |
| `49b439c` | `visualization/scripts/build_scene.py` | scratch source → `results/pipeline/trajectory.csv`; label "Scratch PnP"; records `scene["slam_scratch_source"]`; reference wording |
| | `scripts/check_blender_consistency.py` | checks the recorded source; look-at test; requested-frame report |
| | `pnp_from_scratch/src/pipeline_main.cpp` | `# note:` / `# hud_note:` provenance lines in `trajectory.csv` |
| | `README.md`, `COMMANDS.md`, `visualization/README.md`, `pnp_from_scratch/README.md`, `docs/migration/02_scratch_blender/*.txt` | documentation and evidence |
| `48e2c8d` | `pnp_from_scratch/tools/png_reader.py`, `make_figures.py` (new) | library-free PNG reader and figure generation |
| | `pnp_from_scratch/src/pipeline_main.cpp` | per-frame keypoint export |
| | `docs/migration/baseline/reference_features/` (new) | frozen ORB keypoints and raw matches of pair 0 → 1 |
| | `pnp_from_scratch/README.md` | figures and comparison |
| this commit | `docs/migration/02_scratch_blender/REPORT.md` | this report |

Not changed:
- the scratch solver, RANSAC and features (numerical results identical)
- the reference PnP and ICP
- the dataset
- `trajectory_to_blender_pose()` / `R_FIX` / `WorldRoot`

## Dependency scan (scratch boundary)

- **Text** (`pnp_from_scratch/`, `cmake/feature_core.cmake`, feature-core sources):
  0 matches for `opencv`, `cv::`, `cv2`, `find_package(OpenCV)`, `imread`, `imwrite`,
  or Pillow imports.
- **Binaries:** all 12 scratch binaries have 0 vision, image or compression libraries in
  `ldd`; they need only the C++ runtime.
- **Python tools:** `cv2` is never loaded. Images are decoded by `tools/png_reader.py`
  (standard library + NumPy). Matplotlib imports Pillow internally for writing figures;
  that is not a vision library and is not used for decoding.

## Tests (final run, clean builds)

- **Scratch:** `ctest` 12/12 pass (features ×4, PNG, front end, essential matrix,
  RANSAC, end-to-end 0→1, milestones 1–3).
- **Normal pipeline:**
  - 24/24 programs exit 0
  - Bunny metrics (PnP and ICP) identical to checkpoint 01
  - the dataset regenerates byte-identically (74 files)
  - no file under `data/` changed
- **Blender:**
  - the scene builds with exactly 4 animated cameras and 4 trajectory lines
  - the consistency check passes (positions 3e-8 m, rotations 0°, HUD values exact)

## Figures (generated: `pnp_from_scratch/results/figures/`, PNG + PDF)

- `bunny_correspondences_0_1`
- `bunny_correspondences_0_1_inliers`
- `bunny_correspondences_0_1_outliers`
- `fig03_orb_keypoints`
- `feature_comparison_0_1`
- `pnp_comparison`, plus `pnp_comparison.md`

## Remaining issues

- **Scratch accuracy is worse per pair** than the reference: median 2.75° against 1.17°.
  - The weak pairs (9→11, 26→30) have only 88–112 keypoints and 30–36 inliers.
  - The PnP has no nonlinear refinement.
  - Over all frames the two trajectories are about equally accurate (0.187 m against
    0.180 m mean). The scratch result's smaller frame-35 error comes partly from
    errors cancelling.
- **The essential matrix is a diagnostic only:** the orbit makes two-view estimation
  near-degenerate here (`pnp_from_scratch/README.md`).
- **The milestone-2 trajectory** (`results/scratch_pnp_trajectory.csv`) still exists. It is
  shown only if `SCRATCH_TRAJECTORY_CSV_PATH` is pointed at it.
- **Outside the scratch boundary (not changed here):**
  - The normal pipeline still uses the library: its PnP/RANSAC, two-view geometry, image
    I/O, core types and build/package configuration.
  - Its features have used `feature_core` with threshold 30 since checkpoint 01.
  - The normal figure scripts (`scripts/generate_correspondence_*`) run the
    library-linked executable.
