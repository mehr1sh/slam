# Frontend audit visuals (feature_audit)

## 1. Purpose

`feature_audit` (`apps/feature_audit.cpp`) is a diagnostic program. It shows, for one frame pair, what
each production frontend selection step does:

- Harris ranking of FAST candidates;
- per-level quotas over the eight-level pyramid;
- bidirectional ratio and mutual matching.

It calls the estimator's own C++ functions:

- `scratch::ExtractFeatures` (`src/pipeline/frontend.cpp`);
- `features::ExtractMultiscale`, `features::HarrisResponse`, `features::LevelQuotas` and
  `features::MatchMutualRatio` (`feature_core`).

It is not part of the per-frame pipeline and estimates no pose. The Python scripts only draw and check
its CSV exports. No HTML is produced.

The output now goes to `results/reports/figures/frontend/pair_<i>_<j>/`. Tracked examples exist for
pairs 0→1 and 15→16.

## 2. What feature_audit writes

**Command line.** `feature_audit [--repo DIR] [--i N] [--j N]`. The defaults are the current directory
and frames 0 and 1. The two frame indices must be different and non-negative
(`apps/feature_audit.cpp:44-46`). The output directory is
`<repo>/results/reports/figures/frontend/pair_<i>_<j>/` (`:47`).

**Four extractions per frame.** The program loads the frame from `data/synthetic_bunny` with the
project PNG reader. It runs `ExtractFeatures` four times with FAST threshold 10 and no per-level FAST cap
(`max_keypoints = 0`) (`:23-24`):

| Stage file suffix | `SelectionParams` | Meaning |
| --- | --- | --- |
| `00_fast_all` | `{harris=false, total=0}` | all FAST candidates, FAST order |
| `01_harris_all` | `{true, 0}` | the same candidates, Harris order: membership unchanged |
| `02_fast_quotas` | `{false, 1400}` | FAST order + quotas (control) |
| `03_harris_quotas` | `{true, 1400}` | Harris order + quotas = the production T0 selection |

The pyramid uses the `FrontendParams` default: 8 levels, scale 1.2 (`include/slam/pipeline/frontend.hpp:33`;
`include/slam/features/pyramid.hpp:25-26`).

**Files written by the C++ program:**

- `frame_NNNNNN_<stage>.csv`, one per stage and frame (`:34-36`). Columns:
  - `stable_id`, `index`, `level`;
  - `x`, `y` (original image) and `level_x`, `level_y`;
  - `fast_score`, `harris_response`;
  - `fast_order` and `harris_order` (the index in stages 00 and 01);
  - `kept_fast_quota`, `kept_harris_quota` (membership in stages 02 and 03);
  - `descriptor_hex` (256 bits).
- `frame_NNNNNN_levels.csv`, per level: `quota` from `LevelQuotas(1400, 8, 1.2)`, plus the count of each
  stage (`:37-38`).
- `matching_decisions.csv`, one row per query keypoint of frame *i* (stage 03), matched against frame *j*
  (`:41-52`). Columns:
  - best train index, d1, d2, and the forward ratio pass (`d1 < 0.87·d2`);
  - the reverse d1, d2 and reverse ratio pass;
  - `mutual_best`, `accepted`.

  The program recomputes these decisions on its own and aborts with "Audit diverged from actual matcher"
  if any `accepted` flag differs from `features::MatchMutualRatio` (`:49-51`).
- `config.txt`: settings and interpretation notes (`:53`).

**Stable IDs.** A stable ID such as `f000015_L0_x315_y170` means frame 15, pyramid level 0, integer
level pixel (315, 170) (`:17`). It stays fixed through sorting and selection, so the same candidate can
be followed across stages. It is local to one frame and is not a physical track across frames. Array
order is level-first (level 0, then level 1, and so on), so indices are not a global Harris ranking
across scales.

**CTest.** `feature_audit_pair_0_1` runs `feature_audit --repo <source dir>` (`CMakeLists.txt:124-127`).
It therefore rewrites the tracked CSVs and `config.txt` in
`results/reports/figures/frontend/pair_0_1/` whenever `ctest` runs.

## 3. Render and validate

All commands run from the repository root.

```bash
# one pair: build feature_audit (root, no OpenCV needed) into build-scratch/, export, render, validate
bash tools/figures/visualize_frontend.sh 15 16     # default pair 15 16; e.g. "0 1" for the other example

# step by step (from an existing build/ or build-scratch/)
./build/feature_audit --repo "$PWD" --i 15 --j 16
python3 tools/figures/render_frontend_audit.py --i 15 --j 16
python3 tools/figures/validate_frontend_audit.py --i 15 --j 16

# redraw the frontend figures for every pair_*/ present (also the ICP figures, flowchart and manifest)
python3 tools/figures/render_outputs.py            # --frontend-only skips the ICP figures

# open the PNGs directly
xdg-open results/reports/figures/frontend/pair_15_16
```

`visualize_frontend.sh` does the following (`tools/figures/visualize_frontend.sh:5-13`):

1. Checks the plotting dependencies.
2. Configures the root project into `build-scratch/` and builds only `feature_audit`.
3. Runs the audit, the renderer and the validator.

The Python scripts need NumPy, matplotlib and Pillow (`tools/figures/render_outputs.py:12-21`). On
Ubuntu: `sudo apt install -y python3-matplotlib python3-numpy python3-pil`.

**Validator prerequisite.** `validate_frontend_audit.py` compares against the production run's exports
in `results/runs/pipeline/`, so run `build/scratch_pipeline` in its default T0 configuration
first. `visualization/rebuild_blender.sh` does this before auditing pairs 0→1 and 15→16
(`visualization/rebuild_blender.sh:10-17`).

**Files written by the renderer.** `render_frontend_audit.py` calls `frontend(i, j)` and `manifest()`
from `render_outputs.py` (`:68-159,258-264`). It writes:

| File | What it compares |
| --- | --- |
| `frame_NNNNNN_00_feature_stages.png` | the four stages side by side; colours are pyramid levels, labels are the first 12 array indices |
| `frame_NNNNNN_01_harris_ranking_before_after.png` | the same candidates in FAST order versus Harris order; nothing is removed |
| `frame_NNNNNN_identifier_changes.png` | for the union of the first 10 FAST and first 10 Harris candidates: stable ID, FAST index, Harris index, final index or "removed" |
| `frame_NNNNNN_02_quotas_before_after.png` | Harris order on both sides, quotas off versus on |
| `frame_NNNNNN_03_ranking_same_quota.png` | the same quotas, FAST-selected versus Harris-selected, plus the membership difference (both / FAST only / Harris only) |
| `04_pyramid_quotas.png` | per level, both frames: candidates, quota limit, retained count |
| `05_matching_before_after.png` | raw nearest proposals versus final mutual-ratio survivors |
| `matches_{raw_nearest,forward_ratio,mutual_final,rejected}.png` | individual match views |
| `06_matching_filter_counts.png` | counts (raw, forward ratio, final mutual) and a d1-versus-d2 scatter with the line d1 = 0.87·d2 |
| `identifier_mapping.csv` | every candidate: FAST order, Harris order, quota indices, Harris response |
| `summary.json` | per-stage counts, matching counts, ratio 0.87, line limit 100 |

It also rewrites `results/reports/figures/image_manifest.json`.

All views of one frame use the same crop: the bounding box of all candidates plus 16 px (`:48-52`).

Match images draw at most 100 lines, chosen with a fixed stride (`:135`). Every proposal is listed in
`matching_decisions.csv`.

**Checks done by `validate_frontend_audit.py`** (`tools/figures/validate_frontend_audit.py:9-46`):

- Stages 00 and 01 have the same ID set, and IDs are unique.
- The quota stages are subsets of the full candidate set.
- Harris responses decrease within each level in stage 01.
- Descriptors are unchanged by ranking.
- The quotas sum to 1400, and each stage-03 level count is within its quota.
- Stage 03 equals `results/runs/pipeline/keypoints/frame_NNNNNN.csv` in count, descriptors
  and positions (to 0.001 px).
- Every forward, reverse and mutual decision is recomputed, and the accepted set equals the (query,
  train) pairs in `results/runs/pipeline/correspondences/pair_<i>_<j>.csv`.
- `summary.json` agrees.
- All PNGs verify.

These checks compare the audit with the estimator's own exports. They do not compare it with an
independently written detector.

## 4. Reading the results

- **Harris ranking** changes order, not membership or positions. Only through the quotas does it change
  which corners survive. Stage 02 is the control that isolates this effect: the same quotas with FAST
  order.
- **Quotas** are upper limits. A level with fewer candidates keeps all of them, and unused capacity is
  not given to other levels (`src/features/multiscale.cpp:86-88`). The final count can therefore be
  below 1400.
- **Descriptor checks** do not guarantee a correct physical match. The depth-edge filter and geometric
  RANSAC come after these images.

Tracked results (`summary.json`, `frame_NNNNNN_levels.csv`):

| Pair | FAST candidates i / j | After Harris + quotas i / j | Raw nearest | Forward ratio | Final mutual |
| --- | --- | --- | ---: | ---: | ---: |
| 0 → 1 | 838 / 815 | 838 / 815 | 838 | 468 | 246 |
| 15 → 16 | 1573 / 1624 | 1313 / 1324 | 1313 | 730 | 407 |

For frames 0 and 1 no level exceeds its quota, so the quotas remove nothing. That is expected, not a
fault in the figures.

For frame 15, 1573 candidates are reduced to 1313 retained features. Under the same quotas, 1192 are
selected by both FAST and Harris order and 121 by only one of them.

The 246 final mutual matches for pair 0→1 equal the `matches` value of pair 0 in
`results/benchmarks/rgbd_icp/pair_metrics.csv`, and 407 equals pair 15 there. This is because `rgbd_icp` uses the
same `feature_core` selection and matcher.

The ICP registration figures, the pipeline flowchart and the Blender scene are documented in
`rgbd_icp.md`. `results/reports/figures/README.md` maps every image in that folder.
