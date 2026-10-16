# Checkpoint 01: project-owned feature pipeline

The body of `find_feature_matches()` now runs the project-owned pipeline. The
inherited detector, descriptor and matcher are no longer called. The
distance-filter rule is unchanged, so feature extraction is the main variable
that changed.

    grayscale (BT.601, 14-bit fixed point)
    -> FAST-9, threshold 30, corner score, 3x3 NMS, border 16, strongest 500
    -> intensity-centroid orientation, radius 15
    -> Gaussian smoothing (sigma 2, 9x9) -> rotated BRIEF
       (256 tests, own seeded Gaussian pattern, seed 0x5B21EF)
    -> brute-force Hamming nearest neighbour
    -> distance <= max(2 d_min, 30)   (unchanged)

PnP (still the library RANSAC call), ICP, the dataset and the accumulation are
unchanged.

## Files

| | Files |
|---|---|
| new modules (library target `feature_core`, standard library only) | `include/features/{types,fast,orientation,brief,matcher}.hpp`, `src/features/{types,fast,orientation,brief,matcher}.cpp` |
| new tests (link only `feature_core`) | `tests/unit/features_{fast,orientation,brief,matcher}_test.cpp`, `tests/unit/feature_test_utils.hpp` |
| changed | `src/features/features.cpp` (body replaced), `include/features/features.hpp` (comment), `CMakeLists.txt` (`feature_core` + 4 tests), `scripts/migration/run_test_suite.sh` (4 tests added), `scripts/migration/bunny_metrics.py` (print fix) |
| untouched, still present | `src/tracking/{fast,brief,hamming_matching}.cpp` and their tests. They are removed only in a later stage, after their callers move to `feature_core` |

`find_feature_matches()` keeps its signature (library `Mat`/`KeyPoint`/`DMatch`),
so its 11 callers are unchanged. This is the one transitional boundary
recorded in `docs/library_removal_audit.md`; it goes away in the types stage.

## Tests

- **New unit tests:** all 4 pass, with 49 checks between them.
  - **FAST:** segment test, including the arc wrap and the exact threshold;
    the score value; corners of a square; no responses on edges or flat regions;
    3×3 NMS; the border; the strongest-500 cap and ordering; determinism.
  - **Orientation:** 24 known directions, maximum error 0.59°; the image-axis
    convention; an exact 90° image rotation gives angle − 90° to 1e-5°.
  - **BRIEF:**
    - the pattern is fixed: 256 tests, all inside the radius-15 disc, no degenerate tests, fingerprint `24bb0ef5a0a96263`
    - smoothing: symmetric, 9×9 support
    - balanced bits; unrelated descriptors are about 124 bits apart
    - an exact 90° rotation gives distance **0.00** with steering and **130** without
  - **Matcher:** nearest neighbour, the lowest-index tie rule, the filter rule.
    On a shifted image, 99 of 101 filtered matches have exactly the true shift.
- The 4 test binaries link no vision library (`ldd`).
- **Full suite** (`tests/summary.tsv`): all 24 programs exit 0 (20 existing + 4 new).
  `render_bunny_test` still reproduces the dataset byte for byte.

## Bunny metrics (`metrics.json`, `comparison.md`)

| | baseline (library ORB) | **checkpoint 01** |
|---|---|---|
| keypoints / frame | 430 (max 500) | **70 (max 99)** |
| raw / filtered matches | 433 / 192 | 70 / 53 |
| 3D→2D correspondences | 142.5 | 42.4 |
| RANSAC inliers | 132 | 26.5 |
| PnP inlier reprojection error | 1.94 px | 1.90 px |
| PnP relative rotation error, mean / median / max | 1.39° / 1.17° / 3.28° | 3.49° / 1.43° / **56.9°** (pair 24→25) |
| ICP relative rotation error, mean | 2.81° | 13.86° |
| failed pairs, PnP / ICP | 0 / 0 | 0 / 0 |
| final PnP error | 0.277 m / 32.1° | 0.357 m / 56.2° |
| final ICP error | 0.525 m / 64.9° | 0.940 m / 158.9° |

**Interpretation.** The pipeline works:
- the match pattern is coherent (`output/feature_matching/good_matches.png`)
- the inlier reprojection error equals the baseline's
- the median PnP pair error is close to the baseline's

But with threshold 30 at a single scale it finds only about 70 corners per frame
on the small, flat-shaded bunny (on the textured TUM pair it finds the full 500).
With only 10–20 inliers in a few pairs, PnP occasionally fails badly:
- 24→25: 10 inliers, 56.9°
- 9→10: 20 inliers, 7.2°

ICP, which has no outlier rejection, suffers more from the smaller sets.

**Sensitivity (diagnostic only, not committed; the default stays 30).** The same
code with only `FastParams::threshold` changed:

| threshold | keypoints / frame | inliers | PnP pair error, mean | final PnP | final ICP |
|---|---|---|---|---|---|
| 30 (this checkpoint) | 70 | 26 | 3.49° | 0.357 m / 56.2° | 0.940 m / 158.9° |
| 20 | 131 | 58 | 1.06° | 0.135 m / 16.1° | 0.717 m / 99.6° |
| 12 | 275 | 137 | 0.95° | 0.122 m / 15.2° | 0.589 m / 67.7° |

The threshold, not the algorithm, limits the result on this dataset. Changing
the default is a decision for review; it is not made in this checkpoint.

## Remaining library use after this checkpoint

`python3 scripts/migration/audit_library_refs.py` reports 41 source files.

- **No longer used:** the detector and descriptor-extractor/matcher calls in
  `src/features/features.cpp`. Category E has no remaining files; category D is
  down to 3 files (`fast_test`, `vo_pipeline_test`, `orb_from_scratch_test`).
- **Still used:**
  - image I/O (A, B)
  - PnP/RANSAC (G, H)
  - two-view geometry (J)
  - core types (K)
  - drawing (L)
  - the build/package dependency (M)
  - the older `src/tracking/{fast,brief,hamming_matching}` modules, which use library types
