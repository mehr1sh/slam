# File guide

A file-by-file guide to the repository after the 2026-11-04 reorganisation, written from the code itself
(224 entries). Line references (`path:line`) are to the code at that date; where a comment and the code
disagree, the entry says so and follows the code.

**How to use it.** Read `docs/architecture/LEARNING_PATH.md` for the order of study; this guide is the
reference you keep open beside each file. Entries are grouped by architectural role:

| category | meaning |
|---|---|
| production | C++ that `scratch_pipeline` runs (Eigen + standard library only; no OpenCV) |
| evaluation harness / application | C++ programs and libraries for regression locks, multi-seed evaluation, diagnostics |
| reference | OpenCV/g2o slambook2-derived code; not production (`-DSLAM_WITH_OPENCV=ON`) |
| experiment / benchmark | experimental implementations and comparisons (ICP, ORB, depth edges) |
| tooling / visualization | Python, shell, Blender: read C++ outputs, never estimate poses |
| test / fixture / build | tests, frozen locks, CMake and environment files |

Inside an entry, short paths such as `src/tracking/icp.cpp:29` or `include/tracking/rgbd_icp.hpp:23`
are relative to that entry's component directory (`reference/opencv_slambook2/`, `reference/icp_main_variant/`,
`experiments/rgbd_icp/`, ...); `:N` alone means line N of the file in the entry heading.

Each entry gives: category, status (built/used or not), purpose, responsibility, key types and functions,
inputs and outputs, dependencies, users, tests, and what to understand first. The frozen Bunny export
(`tests/fixtures/stage1_bunny_export/`, 75 files) and `experiments/orb_extractor_benchmark/` (a historical
standalone snapshot) are described as one entry each, as is the frozen reference-pipeline baseline
`tests/fixtures/reference_baseline/` (see `tests/fixtures/README.md`).

## Contents
- [Production feature extraction and matching (feature_core)](#production-feature-extraction-and-matching-feature_core)
- [Production pipeline library (scratch_core)](#production-pipeline-library-scratch_core)
- [Applications, evaluation harness, tests and fixtures](#applications-evaluation-harness-tests-and-fixtures)
- [Reference implementation (OpenCV/g2o, not production)](#reference-implementation-opencvg2o-not-production)
- [`reference/opencv_slambook2/`](#referenceopencv_slambook2)
- [`reference/icp_main_variant/`](#referenceicp_main_variant)
- [Experiments, tooling, visualization and configuration](#experiments-tooling-visualization-and-configuration)

## Production feature extraction and matching (feature_core)

This section covers the project-owned feature module (library `feature_core`), the header-only
fundamental-matrix estimator `include/slam/geometry/fundamental.hpp`, the unit tests in `tests/unit/`,
and the build file `cmake/feature_core.cmake`. All of it is C++17 with the standard library only
(plus Eigen for `fundamental.hpp`); there is no OpenCV in any of these files.

Namespaces: `features` for everything under `include/slam/features/` and `src/features/`;
`geometry_scratch` for `fundamental.hpp`; `ftest` for the test helpers.

Include convention: `cmake/feature_core.cmake:15` puts `include/slam` on the PUBLIC include path of
`feature_core`, so code writes `#include "features/fast.hpp"`. The same include root also makes
`#include "geometry/fundamental.hpp"` resolve for every target that links `feature_core`
(directly or through `scratch_core`), even though `fundamental.hpp` is not part of the library's sources.

Coordinates throughout: pixels, x = column (right), y = row (down), origin at the top-left pixel.
Angles are radians in these image axes, so a positive angle turns from +x toward +y (clockwise on
screen).

---

### `cmake/feature_core.cmake`
- **Category:** build
- **Status:** used. It is included by the root `CMakeLists.txt:23` and by the standalone experiment build
  `experiments/rgbd_icp/icp_cpp/CMakeLists.txt:14`. It is the only definition of `feature_core` in the
  main build. (`experiments/orb_extractor_benchmark/CMakeLists.txt:8` defines its own, separate
  `feature_core` from vendored copies. That is a historical snapshot and does not use this file.)
- **Purpose and why it exists:** Defines the static library `feature_core` once, so the root build and the
  standalone `icp_cpp` build compile the same feature sources with the same include root. It finds the
  repository root from its own location (`get_filename_component(... "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)`,
  line 6). Because of that, it works when another CMake project includes it.
- **Responsibility in the architecture:** The build boundary of the production feature module. Everything
  that extracts or matches features in production goes through this target.
- **Key types and functions:**
  - `add_library(feature_core ...)` (lines 7–14): the seven sources `types.cpp`, `fast.cpp`,
    `orientation.cpp`, `brief.cpp`, `matcher.cpp`, `pyramid.cpp`, `multiscale.cpp`.
  - `target_include_directories(feature_core PUBLIC <root>/include/slam)` (line 15).
  - The file sets no compile options. Unlike `scratch_core` (`CMakeLists.txt:32`), `feature_core` is not built with `-Wall -Wextra`.
    It also has no `target_link_libraries`: no Eigen, no OpenCV.
- **Inputs → outputs:** CMake source list → static library target `feature_core`.
- **Dependencies:** CMake ≥ 3.16 (root `CMakeLists.txt:1`); no packages.
- **Used by:** `CMakeLists.txt:23` (include), `CMakeLists.txt:31` (`scratch_core PUBLIC ... feature_core`),
  `CMakeLists.txt:84` (the six `features_*_test` executables), `experiments/rgbd_icp/CMakeLists.txt:8`
  (`rgbd_icp`, only with `-DSLAM_WITH_OPENCV=ON`), `experiments/rgbd_icp/icp_cpp/CMakeLists.txt:14,17`,
  `reference/opencv_slambook2/CMakeLists.txt:27` (`features PRIVATE feature_core`, OpenCV build only).
  Through `scratch_core`, every production and evaluation executable links it: `scratch_pipeline`,
  `scratch_baseline`, `scratch_experiments`, `feature_audit`, `depth_edge_experiment` and the
  `tests/pipeline/test_*` programs.
- **Covered by tests:** indirectly: every test in `tests/unit/` links it (`CMakeLists.txt:81-86`).
- **Before studying it, understand:** CMake targets, PUBLIC vs PRIVATE include directories, `include()` vs
  `add_subdirectory()`.

---

### `include/slam/features/types.hpp`
- **Category:** public header
- **Status:** used. It is included by every other feature header and is compiled into `feature_core`.
  Through `scratch_core` it reaches `scratch_pipeline` and all other production and evaluation targets.
- **Purpose and why it exists:** Declares the small, project-owned data types that the whole feature chain
  passes around: a grey image, a keypoint, a 256-bit binary descriptor and a match. It also declares the
  RGB/BGR→grey conversion. The types exist so that the production path needs no third-party vision type
  (no `cv::Mat`, `cv::KeyPoint`, `cv::DMatch`).
- **Responsibility in the architecture:** The shared vocabulary of the feature module. The pipeline's
  `FrameFeatures` (`include/slam/pipeline/frontend.hpp:63-75`) and `Correspondence` are built from these types.
- **Key types and functions:**
  - `GrayImage` (lines 14–23): an 8-bit single-channel image, row-major and tightly packed (`pixels` holds
    `width*height` values). `at(x, y)` performs no bounds check. `contains(x, y)` is the bounds test.
  - `GrayFromInterleaved(data, w, h, row_step, channels)` (line 29): converts B,G,R(,A) order to grey.
  - `GrayFromRGB(...)` (line 32): converts R,G,B(,A) order, for example pixels decoded from PNG.
    Both use the ITU-R BT.601 luma in 14-bit fixed point, `Y = (1868 B + 9617 G + 4899 R + 8192) >> 14`
    (comment at lines 27–28, code at `src/features/types.cpp:18`).
  - `Keypoint` (lines 34–38): `x, y` (float pixels), `score` (FAST score, int) and `angle`
    (radians, `atan2(m01, m10)`, 0 until assigned).
  - `BriefDescriptor` (lines 41–45): `std::array<uint64_t,4>`. `bit(i)` and `set(i)` address bit `i` in word `i>>6`.
  - `Match` (lines 47–51): `query`, `train` (indices into the two lists) and `distance` (Hamming, 0..256).
- **Inputs → outputs:** interleaved 8-bit pixels → `GrayImage` (grey levels 0..255).
- **Dependencies:** `<array>`, `<cstddef>`, `<cstdint>`, `<vector>`.
- **Used by:** all `include/slam/features/*.hpp`; `include/slam/pipeline/frontend.hpp:24`;
  `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp:3`; `tests/unit/feature_test_utils.hpp:11`.
  The conversion functions are called from `src/pipeline/frontend.cpp:48` (`GrayFromRGB`, production),
  `reference/opencv_slambook2/src/features/features.cpp:29`, `experiments/rgbd_icp/src/rgbd_icp.cpp:513` and
  `experiments/orb_opencv/orb_benchmark.cpp:37` (`GrayFromInterleaved`).
- **Covered by tests:** the types are used by every unit test. No unit test checks the grey conversion formula
  directly. It is exercised end-to-end by `tests/pipeline/test_frontend.cpp` and the pipeline tests.
- **Before studying it, understand:** row-major image storage; bit packing; luma weights (BT.601).

### `src/features/types.cpp`
- **Category:** production source
- **Status:** used. It is part of `feature_core` (`cmake/feature_core.cmake:8`). In production its code runs
  in `scratch_pipeline` through `ExtractFeatures` (`src/pipeline/frontend.cpp:47-49`).
- **Purpose and why it exists:** Implements the grey conversion. Both public functions call one private
  `ToGray` with different byte offsets for blue and red (lines 10–22, 26–28 with `b=0, r=2`, 30–32 with
  `b=2, r=0`). A 1-channel input is copied unchanged. Any channel count other than 1, 3 or 4 throws
  `std::invalid_argument` (lines 11–12).
- **Responsibility in the architecture:** The first step of the production front end: decoded PNG RGB → `GrayImage`.
- **Key types and functions:** `ToGray` (anonymous namespace). It uses integer fixed-point weights that sum to
  16384, adds the rounding constant 8192 and shifts right by 14 (line 18). The alpha channel is ignored.
- **Inputs → outputs:** `const uint8_t*` interleaved 8-bit samples, `row_step` in bytes → `GrayImage`.
- **Dependencies:** `features/types.hpp`, `<stdexcept>`.
- **Used by:** see `types.hpp`.
- **Covered by tests:** no direct unit test. Indirectly covered by `tests/pipeline/test_frontend.cpp` and
  `tests/pipeline/test_multiscale_0_1.cpp`, which run real Bunny frames.
- **Before studying it, understand:** `types.hpp`; interleaved pixel layout and row stride.

---

### `include/slam/features/fast.hpp`
- **Category:** public header
- **Status:** used (`feature_core`; production through `ExtractMultiscale`).
- **Purpose and why it exists:** Declares the FAST corner detector (Rosten & Drummond). It covers the
  segment test on the 16-pixel Bresenham circle of radius 3, the corner score, 3×3 non-maximum suppression,
  the border margin, the strongest-N cap, and an optional per-cell low-threshold fallback. The detector is
  deterministic: ties break in raster order and the sort order is total.
- **Responsibility in the architecture:** Keypoint detection on each pyramid level.
- **Key types and functions:**
  - `FastParams` (lines 14–25), with these defaults:
    `threshold = 30` (line 15). Production overrides it: `FrontendParams()` sets 20 (`include/slam/pipeline/frontend.hpp:47`),
    and `ApplyImprovedDefaults` sets 10 (`src/pipeline/pipeline.cpp:41`).
    `arc_length = 9` (FAST-9, line 16); `border = 16` (line 17); `max_keypoints = 500` (line 18, `<= 0` keeps all);
    `cell_size = 0` and `min_threshold = 0` (lines 23–24), so the fallback is off by default.
  - `kFastCircle[16][2]` (line 28): the circle offsets, clockwise from 12 o'clock.
  - `IsFastCorner(img, x, y, threshold, arc_length)` (line 31): the segment test. It requires `3 <= x < W-3`
    and `3 <= y < H-3`.
  - `FastScore(img, x, y, threshold)` (line 36): score "V" = max(Σ_brighter (I(p)−I(c)−t), Σ_darker (I(c)−I(p)−t)).
  - `DetectFast(img, params)` (line 43): corners → scores → [per-cell fallback] → 3×3 NMS → sort by
    (score desc, y asc, x asc) → first `max_keypoints`. Returned keypoints have `angle = 0`.
- **Inputs → outputs:** `GrayImage` (one pyramid level) → `std::vector<Keypoint>` in integer pixel coordinates
  of that level, with `score` > 0.
- **Dependencies:** `features/types.hpp`, `<vector>`.
- **Used by:** `include/slam/features/multiscale.hpp:16`; `include/slam/pipeline/frontend.hpp:20`
  (`FastParams` is a member of `FrontendParams`); `reference/opencv_slambook2/src/features/features.cpp:8,34`;
  `experiments/rgbd_icp/src/rgbd_icp.cpp:515-516` (`FastParams`); `tests/unit/features_fast_test.cpp`,
  `features_orientation_test.cpp`, `features_brief_test.cpp`, `features_matcher_test.cpp`,
  `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_fast_test.cpp` (all functions);
  `tests/pipeline/test_frontend.cpp:78-100` (per-cell fallback on a real frame).
- **Before studying it, understand:** the FAST segment test and why a score is needed for NMS; Bresenham circles.

### `src/features/fast.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:9`; production through `ExtractMultiscale`).
- **Purpose and why it exists:** Implements FAST-N detection with deterministic selection, without OpenCV.
- **Responsibility in the architecture:** Produces the candidate keypoints on each level. In production they
  are then re-ranked by Harris and cut to the per-level quota (see `multiscale.cpp`).
- **Key types and functions:**
  - `kFastCircle` (lines 7–8).
  - `IsFastCorner` (lines 10–26) classifies each circle pixel as brighter (`v > c + t`), darker (`v < c − t`)
    or similar (line 15). The comparison is strict, so a difference exactly equal to `t` does not count.
    The function then looks for a run of at least `arc_length` equal non-zero classes. It scans 32 positions
    (two laps of the circle) so that runs wrapping from pixel 15 to pixel 0 are found (lines 18–24).
  - `FastScore` (lines 28–37): the sum-of-absolute-excess score described in the header.
  - `DetectFast` (lines 39–106):
    - The effective border is `b = max(3, params.border)` (line 40). An image no larger than `2b` in either
      dimension yields no keypoints (line 43).
    - It builds a dense score map, where a corner's score is `max(1, FastScore)` so that 0 means "not a
      corner" (lines 46–50).
    - Per-cell fallback (lines 54–71) runs only if `cell_size > 0` and `0 < min_threshold < threshold`.
      Cells of `cell_size` px start at the border. A cell with no corner at the main threshold is
      re-searched at `min_threshold`, and those corners are scored at `min_threshold`.
    - 3×3 NMS (lines 75–97): a neighbour with a higher score suppresses the centre. A neighbour with an equal
      score suppresses it only if it comes earlier in raster order.
    - Sort by (score desc, y asc, x asc) (lines 99–103), then cap at `max_keypoints` (line 104).
- **Inputs → outputs:** `GrayImage` → `std::vector<Keypoint>` (level pixels).
- **Dependencies:** `features/fast.hpp`, `<algorithm>`.
- **Used by:** `src/features/multiscale.cpp:72` (production). Direct calls: `reference/opencv_slambook2/src/features/features.cpp:34`
  and the unit tests.
- **Covered by tests:** `tests/unit/features_fast_test.cpp`. The test checks the 9-arc yes/no cases
  (lines 34–43), the score formula (46–49), the square corners and that edges give no keypoints (52–74),
  NMS, border, cap, order and determinism on noise (77–115), and the per-cell fallback (118–156).
- **Before studying it, understand:** `fast.hpp`; non-maximum suppression; why tie-breaking matters for reproducibility.

---

### `include/slam/features/orientation.hpp`
- **Category:** public header
- **Status:** used (`feature_core`; production through `ExtractMultiscale`).
- **Purpose and why it exists:** Declares keypoint orientation by the intensity centroid (Rosin 1999, as in
  oriented FAST). Over a circular patch, `m10 = Σ x I` and `m01 = Σ y I`, and `angle = atan2(m01, m10)`.
  The orientation is what lets BRIEF be steered, i.e. made rotation-aware.
- **Responsibility in the architecture:** Sits between detection and description on each level.
- **Key types and functions:**
  - `kOrientationRadius = 15` (line 14).
  - `IntensityCentroidAngle(img, x, y, radius = 15)` (line 17). It requires the whole patch inside the image.
  - `AssignOrientations(img, keypoints, radius = 15)` (line 20) sets `angle` on every keypoint.
- **Inputs → outputs:** `GrayImage` and integer keypoint position → angle in radians, in (−π, π], image axes.
- **Dependencies:** `features/types.hpp`, `<vector>`.
- **Used by:** `src/features/multiscale.cpp:4,89`; `reference/opencv_slambook2/src/features/features.cpp:10,35`;
  `tests/unit/features_orientation_test.cpp`, `features_brief_test.cpp`, `features_matcher_test.cpp`,
  `features_pyramid_test.cpp`, `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_orientation_test.cpp`.
- **Before studying it, understand:** image moments; `atan2` quadrant conventions; the y-down image axis.

### `src/features/orientation.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:10`).
- **Purpose and why it exists:** Implements the intensity-centroid angle.
- **Responsibility in the architecture:** Orientation for each level's keypoints in `ExtractMultiscale`.
- **Key types and functions:**
  - `IntensityCentroidAngle` (lines 8–21) throws `std::out_of_range` if the square `[x−r, x+r]×[y−r, y+r]`
    leaves the image (lines 9–10). It sums `dx·I` and `dy·I` over the disc `dx²+dy² ≤ r²` as `long long`
    (lines 11–19) and returns `atan2(m01, m10)` (line 20). A uniform patch gives `atan2(0,0) = 0`.
    The moments use raw intensities, not intensities minus the patch mean. Because the patch is symmetric,
    the result is the same.
  - `AssignOrientations` (lines 23–25) truncates `k.x`, `k.y` to int.
- **Inputs → outputs:** see header.
- **Dependencies:** `features/orientation.hpp`, `<cmath>`, `<stdexcept>`.
- **Used by:** `src/features/multiscale.cpp:89` (production); reference and tests as above.
- **Covered by tests:** `tests/unit/features_orientation_test.cpp` checks a blob at 24 directions
  (error < 3°, lines 28–37), "blob below → +90°" (39–46), an exact 90° image rotation that lowers the angle
  by 90° (48–65), and a uniform patch giving 0 (67–72). The DetectFast border (16) guarantees that the
  radius-15 patch is inside the image; a smaller `border` would make this function throw.
- **Before studying it, understand:** `orientation.hpp`.

---

### `include/slam/features/brief.hpp`
- **Category:** public header
- **Status:** used (`feature_core`; production through `ExtractMultiscale`).
- **Purpose and why it exists:** Declares the rotated BRIEF descriptor. The method has four steps:
  1. Smooth the image with a Gaussian (σ = 2, 9×9 kernel).
  2. Run 256 binary tests `I(p_i) < I(q_i)` on the smoothed image.
  3. Take the test pairs from a fixed, project-owned pattern. It is drawn once from an isotropic Gaussian
     (σ = 31/5, BRIEF "G II") with a fixed seed and restricted to a disc of radius 15.
  4. Rotate the pattern by the keypoint angle before sampling.

  The pattern is the project's own. It is not ORB's learned pattern.
- **Responsibility in the architecture:** Produces the binary descriptor on each level.
- **Key types and functions:**
  - Constants: `kBriefTests = 256` (line 20), `kBriefPatchRadius = 15` (line 21), `kBriefSeed = 0x5B21EF`
    (line 22), `kBriefSmoothingSigma = 2.0` (line 23), `kBriefSmoothingRadius = 4`, i.e. 9×9 (line 24).
  - `BriefTest {x1,y1,x2,y2}` (lines 26–28).
  - `BriefPattern()` (line 31).
  - `GaussianSmooth(img, sigma = 2.0, radius = 4)` (lines 35–36): separable blur with a mirrored border, rounded.
  - `ComputeBriefDescriptor(smoothed, kp, out)` (line 40): returns false and a zero descriptor if any rotated
    test point leaves the image.
  - `ComputeBrief(smoothed, keypoints, valid*)` (lines 44–45).
  - `HammingDistance(a, b)` (line 47).
- **Inputs → outputs:** an already smoothed `GrayImage` and `Keypoint` (level pixels, angle in radians) →
  `BriefDescriptor` (256 bits) and a validity flag. The Hamming distance is an int in 0..256.
- **Dependencies:** `features/types.hpp`, `<cstdint>`, `<vector>`.
- **Used by:** `src/features/multiscale.cpp:3`, `src/features/matcher.cpp:8` (for `HammingDistance`),
  `apps/feature_audit.cpp:4,42` (`HammingDistance`), `reference/opencv_slambook2/src/features/features.cpp:7,36,46`,
  and the tests `features_brief_test.cpp`, `features_matcher_test.cpp`, `features_pyramid_test.cpp`,
  `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_brief_test.cpp`.
- **Before studying it, understand:** BRIEF (Calonder et al. 2010) and steered BRIEF in ORB; Gaussian smoothing;
  the Hamming distance on bit strings.

### `src/features/brief.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:11`).
- **Purpose and why it exists:** Implements the pattern, the smoothing, the descriptor and the Hamming distance
  in a way that gives the same result with every standard library.
- **Responsibility in the architecture:** Description step in `ExtractMultiscale`. `HammingDistance` is the
  metric of all three matchers.
- **Key types and functions:**
  - `Gaussian(rng)` (lines 15–19) is Box–Muller on two `std::mt19937` outputs. `std::normal_distribution` is
    avoided on purpose because its algorithm is implementation-defined (comment, lines 11–14).
  - `BriefPattern()` (lines 23–46) is a function-local static, so it is built once. It uses `sigma = 31/5 = 6.2`
    (line 26) and rejection-samples each point into the disc `x²+y² ≤ 225` (lines 28–33). A test whose two
    points coincide is skipped (line 40).
  - `GaussianSmooth` (lines 48–74):
    - normalised 1-D kernel (lines 49–52);
    - mirror without repeating the edge pixel (`…cba|abc…`, lines 54–58);
    - horizontal pass into `double` (59–65), then vertical pass;
    - rounds with `lround` and clamps to 0..255 (line 71).
  - `ComputeBriefDescriptor` (lines 76–93): the keypoint centre is truncated to int (line 79). Each test point
    is rotated by `(c·x − s·y, s·x + c·y)` and rounded (lines 84–85). The descriptor is invalid if any point is
    outside the image (86–89). Bit i = 1 iff `I(p) < I(q)` (line 90).
  - `ComputeBrief` (lines 95–104).
  - `HammingDistance` (lines 106–116): XOR of the four words plus a Kernighan bit count. It does not use a
    popcount intrinsic.
- **Inputs → outputs:** see header.
- **Dependencies:** `features/brief.hpp`, `<algorithm>`, `<cmath>`, `<random>`. It uses `M_PI` (POSIX).
- **Used by:** see header. In production: `src/features/multiscale.cpp:91-92` (`ComputeBrief(GaussianSmooth(img), ...)`),
  `src/features/matcher.cpp` (all matchers).
- **Covered by tests:** `tests/unit/features_brief_test.cpp`:
  - the pattern has 256 tests, all inside the disc, with no self-tests, at least 250 distinct, built once, and
    an FNV-1a fingerprint equal to `0x24bb0ef5a0a96263` (lines 19–35); the spread is about σ (36–39);
  - smoothing leaves a constant image unchanged and has a symmetric impulse response with 9×9 support (42–53);
  - every border-16 keypoint has a valid descriptor and the bits are balanced (56–69);
  - on noise the mean distance is about 128 (70–82); a keypoint 3 px from the edge is invalid (84–90);
  - after a 90° rotation the steered descriptor stays within mean < 8 bits, while the unsteered one is > 60
    bits away (92–117);
  - a Hamming-distance check (119–124).
  The matcher tests also exercise `HammingDistance`.
- **Before studying it, understand:** `brief.hpp`; separable convolution; Box–Muller.

---

### `include/slam/features/matcher.hpp`
- **Category:** public header
- **Status:** used (`feature_core`). Each matcher is used as follows:
  - `MatchMutualRatio` is the production matcher. `ApplyImprovedDefaults` sets `frontend.mutual_ratio = true`
    (`src/pipeline/pipeline.cpp:42`) and `MatchFrames` then calls it (`src/pipeline/frontend.cpp:81-83`).
  - `MatchBruteForce` and `FilterMatchesByDistance` always run in the non-ranked branch
    (`src/pipeline/frontend.cpp:78-80`). With `mutual_ratio` they supply only `PairMatches::raw`, `d_min` and
    `d_max`; the filtered list is replaced. They are the actual matcher for `--legacy`, `--stage2-frontend` and `--no-mutual-ratio`.
  - `MatchRanked` is opt-in (`FrontendParams::ranked_matcher`, default false, `include/slam/pipeline/frontend.hpp:45`).
- **Purpose and why it exists:** Declares three matching strategies over BRIEF descriptors, all brute force:
  1. the original nearest neighbour plus the distance filter inherited from slambook2's `find_feature_matches()`;
  2. the teammate's bidirectional ratio test with mutual consistency;
  3. main's instrumented "Stage 3" matcher, which records the fate of each match, uses a location-aware second
     best and computes a quality ranking.
- **Responsibility in the architecture:** The 2D–2D association step between consecutive frames in `MatchFrames`.
- **Key types and functions:**
  - `MatchBruteForce(query, train)` (lines 15–16): the nearest train for every query; ties go to the lowest train index.
  - `FilterMatchesByDistance(matches, floor_distance = 30, d_min_out, d_max_out)` (lines 22–25): keeps
    `distance <= max(2·d_min, floor_distance)`.
  - `MatchMutualRatio(a, b, ratio = .87)` (lines 28–30): the ratio test in both directions plus the mutual
    nearest-neighbour test. Caveat: its second-best neighbour is **naive**, so another pyramid level's copy of the
    same corner can be the second best (29–54 % of naive second bests on the earlier front end; fewer with Harris
    ranking and the budget, but not removed); 0.87 is also permissive (Lowe uses 0.8). `MatchRanked`'s
    location-aware second best avoids this but is not the default (`docs/components/pipeline.md`,
    "Production defaults (T0) and why").
  - The `MatchRanked` contract (comment, lines 32–61):
    - **best:** the nearest valid train.
    - **second:** the smallest distance among valid trains more than `same_location_radius_px` from the best
      train's position, so that another pyramid level's copy of the same corner is never the second best.
    - **ratio:** an exact fraction, with 0 when there is no second and 1 when both distances are 0.
    - **mutual / mutual_location:** the reverse nearest neighbour is q itself, or lies within the radius of q.
    - **filters, in order:** distance → mutual → ratio → level. A removed match is counted once, by the first
      filter it fails.
    - **quality order:** ratio, then distance, then query index.
  - `MatcherParams` (lines 63–71), with these defaults: `floor_distance = 30`, `mutual = false`,
    `mutual_location_aware = false`, `ratio = 0.0` (off), `same_location_radius_px = 3.0f`,
    `max_level_diff = -1` (off), `rank = false`.
  - `MatchFate` and `MatchFateName` (lines 73–74); `RankedMatch` (lines 76–90; `ratio()`, `match()`);
    `MatchCounters` (92–99); `MatchSet {raw, kept, counters}` (101–105).
  - `MatchRanked(query, train, train_x, train_y, query_level, train_level, params, query_valid, train_valid,
    query_x, query_y)` (lines 111–116). Positions are level-0 pixels.
  - `QualityBefore(a, b)` (line 119).
- **Inputs → outputs:** descriptor lists (plus, for `MatchRanked`, train/query positions in original-image
  pixels, pyramid levels and validity flags) → `std::vector<Match>` or `MatchSet`. Distances are Hamming bits.
- **Dependencies:** `features/types.hpp`, `<string>`, `<vector>`.
- **Used by:**
  - `include/slam/pipeline/frontend.hpp:21` and `src/pipeline/frontend.cpp:5,78-91` (production);
  - `apps/pipeline_main.cpp:513` (`MatchFateName`) and `apps/feature_audit.cpp:5,49` (`MatchMutualRatio`, 0.87);
  - `experiments/rgbd_icp/src/rgbd_icp.cpp:2,295` and `experiments/orb_opencv/orb_benchmark.cpp:3,144`
    (`MatchMutualRatio`);
  - `reference/opencv_slambook2/src/features/features.cpp:9,68,70` (`MatchBruteForce`, `FilterMatchesByDistance`);
  - `tests/pipeline/test_improvements.cpp:1,33-38`;
  - `tests/unit/features_matcher_test.cpp` and `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_matcher_test.cpp` (`MatchBruteForce`, `FilterMatchesByDistance`);
  `tests/unit/features_matcher_ranked_test.cpp` (`MatchRanked`, `QualityBefore`, `MatchFateName`);
  `tests/pipeline/test_improvements.cpp:30-38` (`MatchMutualRatio`: exact mutual matching, and rejection of an
  ambiguous duplicate); `tests/pipeline/test_ranked_frontend.cpp` (`MatchRanked` inside the front end).
  **No test in `tests/unit/` exercises `MatchMutualRatio`.**
- **Before studying it, understand:** nearest-neighbour matching; Lowe's ratio test; mutual (cross-check)
  matching; why multiscale extraction produces near-duplicate keypoints at the same location.

### `src/features/matcher.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:12`).
- **Purpose and why it exists:** Implements the three matchers, all O(N_q·N_t) Hamming distances with no index structure.
- **Responsibility in the architecture:** Association step of `MatchFrames`.
- **Key types and functions:**
  - `MatchBruteForce` (lines 12–33) starts from distance 257 (line 21). A strict `<` keeps the lowest index
    among ties (line 24). It returns one match per query, so several queries may share a train (many-to-one).
  - `FilterMatchesByDistance` (lines 35–53) computes `d_min` and `d_max` over all input matches. The limit is
    `max(2·d_min, floor)` (line 48), applied inclusively. Query order is preserved.
  - `MatchMutualRatio` (lines 55–85):
    - It returns an empty list unless `0 < ratio < 1` (lines 58–59).
    - In each direction it tracks the best and second-best distance. A tie with the best becomes the second
      best (`else if (d < second)`), so an exact tie fails the ratio test.
    - A query is accepted in that direction only if `t.size() >= 2` and `first < ratio * second`, a strict
      inequality (line 74).
    - The function computes both directions a→b and b→a, then keeps an a→b match only if b's accepted best is
      the same a (lines 79–83). The result is one-to-one, in query order, with `distance` = Hamming distance.
    - There is no distance floor, no location-aware second best and no validity mask: descriptors flagged
      invalid by `ComputeBrief` would take part. In practice the border of 16 keeps every BRIEF test inside
      each level, so all are valid; `features_matcher_ranked_test.cpp:100-101` checks this for an 8-level extraction.
  - `MatchFateName` (lines 87–95).
  - `RatioFraction` (lines 99–103) and `RankedMatch::ratio()` (106–110).
  - `QualityBefore` (lines 112–119) compares the ratios by cross-multiplication in integers (`long`), then the
    distance, then the query index.
  - `MatchRanked` (lines 121–204):
    - **Argument checks** (lines 128–137) throw `std::invalid_argument` on size mismatches, or when
      `mutual_location_aware` is set without query positions. If every train is invalid it returns an empty
      result (line 145).
    - **Forward pass** (lines 151–174): one forward pass computes, for each valid query, the best train and a
      column-wise reverse nearest neighbour (`col_best_q`, lowest query index on ties). It then finds the
      location-aware second best. A train within `same_location_radius_px` of the best train is skipped
      (line 168); among equal second distances the lowest index wins (line 166).
    - **Distance limit:** `d_min` and the limit come from the raw matches (lines 176–177).
    - **Fates** (lines 179–194): each match is assigned exactly one fate.
    - **Ranking** (lines 195–201) assigns a rank to every kept match. `kept` is in query order unless `params.rank` is set.
- **Inputs → outputs:** see header.
- **Dependencies:** `features/matcher.hpp`, `features/brief.hpp`, `<algorithm>`, `<stdexcept>`, `<limits>`, `<cmath>`.
- **Used by:** see header.
- **Covered by tests:**
  - `tests/unit/features_matcher_test.cpp`: nearest neighbour, tie and empty inputs (lines 36–44); the filter
    limit (46–55); the whole chain on a shifted copy, where ≥ 90 % of more than 100 filtered matches have
    exactly the true shift (7, −4) (57–79).
  - `tests/unit/features_matcher_ranked_test.cpp`:
    - the defaults reproduce `MatchBruteForce` + `FilterMatchesByDistance` on random descriptors with ties and
      on an 8-level extraction (67–104);
    - many-to-one vs mutual (106–127); location-aware mutual (129–165); tie-breaking (167–181);
    - location-aware ratio vs naive ratio (radius 0) and genuine ambiguity (183–212);
    - zero distance and a missing second (214–230); invalid descriptors and argument errors (232–250);
    - fates and counters (252–269); the quality order (271–308).
  - `tests/pipeline/test_improvements.cpp:30-38` covers `MatchMutualRatio`.
- **Before studying it, understand:** `matcher.hpp`; integer comparison of fractions.

---

### `include/slam/features/pyramid.hpp`
- **Category:** public header
- **Status:** used (`feature_core`; production `ExtractFeatures` calls `BuildPyramid`).
- **Purpose and why it exists:** Declares the image pyramid. Level 0 is the input image. Level l (l ≥ 1) has
  size `(round(W0/s^l), round(H0/s^l))` and is resampled from level l−1 by centre-aligned bilinear interpolation:
  `dst(x,y) = src((x+0.5)·rx − 0.5, (y+0.5)·ry − 0.5)`. Resampling from the previous level smooths
  progressively. Coordinates map back as `x0 = (x+0.5)·scale_x[l] − 0.5`, with `scale_x[l] = W0/W_l`.
- **Responsibility in the architecture:** Gives the multiscale structure (8 levels × 1.2, like ORB) on which
  FAST, orientation and BRIEF run independently.
- **Key types and functions:**
  - `PyramidParams` (lines 24–27): `levels = 8` (line 25), `scale_factor = 1.2` (line 26).
    Production does not change these; `ApplyImprovedDefaults` (`src/pipeline/pipeline.cpp:39-48`) does not touch them.
  - `ImagePyramid {levels, scale_x, scale_y}` (lines 29–32). `scale_*` = level-0 size / level size per axis.
    This is the ratio of the actually rounded sizes, not exactly `1.2^l`.
  - `ResizeBilinear(src, w, h)` (line 35); `BuildPyramid(base, params)` (line 37).
  - `LevelToBaseX/Y(p, level, x)` (lines 40–41): level pixel → level-0 pixel.
- **Inputs → outputs:** `GrayImage` → `ImagePyramid` (8 grey images; scale factors are dimensionless, ≥ 1).
- **Dependencies:** `features/types.hpp`, `<vector>`.
- **Used by:**
  - `include/slam/features/multiscale.hpp:17` and `include/slam/pipeline/frontend.hpp:23,33`
    (`PyramidParams` is part of `FrontendParams`);
  - `src/pipeline/frontend.cpp:53` (production `BuildPyramid`);
  - `apps/feature_audit.cpp:19,25`; `experiments/rgbd_icp/src/rgbd_icp.cpp:517`;
  - `tests/unit/features_pyramid_test.cpp` and `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_pyramid_test.cpp`.
- **Before studying it, understand:** bilinear interpolation; pixel-centre vs pixel-corner conventions; aliasing.

### `src/features/pyramid.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:13`).
- **Purpose and why it exists:** Implements the resampling and the pyramid construction.
- **Responsibility in the architecture:** First step of `ExtractFeatures` after the grey conversion.
- **Key types and functions:**
  - `ResizeBilinear` (lines 9–29) throws on empty sizes (lines 10–11) and returns a copy if the size is
    unchanged (line 12). It computes centre-aligned source coordinates, clamped to `[0, size−1]` (lines 16, 20),
    takes the four-neighbour weighted sum, and rounds with `lround` and a clamp (line 25).
  - `BuildPyramid` (lines 31–45) throws if `levels < 1` (line 32). The size of level l is computed from the
    base, `round(W0 / s^l)` with a minimum of 1 (lines 38–39), but the pixels are resampled from the previous
    level (line 40). `scale_x/y = W0/w`, `H0/h` (lines 41–42).
- **Inputs → outputs:** see header. For 640×480 the level sizes are 640×480, 533×400, 444×333, 370×278,
  309×231, 257×193, 214×161, 179×134 (asserted in `tests/unit/features_pyramid_test.cpp:21-25`).
- **Dependencies:** `features/pyramid.hpp`, `<algorithm>`, `<cmath>`, `<stdexcept>`.
- **Used by:** see header.
- **Covered by tests:** `tests/unit/features_pyramid_test.cpp`:
  - level sizes and scales (lines 18–31); level 0 equals the input and the build is deterministic (32–33);
    `levels = 1` gives the input only (34–39);
  - a constant image is preserved (42–48); a linear ramp is reproduced within 2 grey levels at the mapped
    coordinates (49–68);
  - a blob centroid maps back within 0.5 level pixels (69–91).
- **Before studying it, understand:** `pyramid.hpp`.

---

### `include/slam/features/multiscale.hpp`
- **Category:** public header
- **Status:** used (`feature_core`; production through `ExtractFeatures`, `src/pipeline/frontend.cpp:52-54`).
- **Purpose and why it exists:** Declares multiscale extraction. On every pyramid level, independently, it runs
  DetectFast → intensity-centroid orientation on that level → Gaussian smoothing of that level → rotated BRIEF.
  It also maps each keypoint to level-0 coordinates. `SelectionParams` adds the teammate's Harris ranking and a
  geometric total-feature budget across levels. With the default `SelectionParams{}`, the historical extraction
  (FAST order, 500 per level) is kept for ablations and regression locks.
- **Responsibility in the architecture:** The complete per-frame feature extractor. `ExtractFeatures` is a thin
  wrapper around it.
- **Key types and functions:**
  - `MultiscaleKeypoint` (lines 22–27): `level_kp` (in level coordinates), `level`, `scale`
    (`scale_x[level]`), and `x, y` (level-0 pixels).
  - `MultiscaleFeatures` (lines 29–36): `keypoints`, `descriptors`, `per_level_count`, `valid`.
  - `SelectionParams` (lines 38–43): `harris = false` (lines 39–40) and `total_features = 0` (lines 41–42;
    0 = legacy per-level cap `FastParams::max_keypoints`). Production uses `{true, 1400}`
    (`src/pipeline/pipeline.cpp:40`).
  - `HarrisResponse(image, x, y)` (line 44); `LevelQuotas(total, levels, scale_factor)` (line 45);
    `ExtractMultiscale(pyramid, fast, selection = SelectionParams())` (lines 46–48).
- **Inputs → outputs:** `ImagePyramid` + `FastParams` + `SelectionParams` → `MultiscaleFeatures`. Keypoints are
  concatenated level by level, level 0 first. Positions are level-0 pixels (float). `scale` is level-0 size ÷ level size.
- **Dependencies:** `features/fast.hpp`, `features/pyramid.hpp`, `features/types.hpp`, `<vector>`.
- **Used by:**
  - `include/slam/pipeline/frontend.hpp:22,35` and `src/pipeline/frontend.cpp:6,52` (production);
  - `apps/feature_audit.cpp:6,24,37` (four selection stages; `LevelQuotas(1400, ...)`);
  - `experiments/rgbd_icp/src/rgbd_icp.cpp:3,517` (`ExtractMultiscale` with `{true, c.features}`);
  - `tests/pipeline/test_improvements.cpp:2,15-29` (`LevelQuotas`, `HarrisResponse`);
  - `tests/unit/features_pyramid_test.cpp` and `features_matcher_ranked_test.cpp`.
- **Covered by tests:** `tests/unit/features_pyramid_test.cpp:93-137` (`ExtractMultiscale` with default
  selection); `tests/pipeline/test_improvements.cpp:15-29` (`LevelQuotas`: the sum is 1400 and level 0 gets more
  than level 7; `HarrisResponse`: corner > edge). No unit test runs `ExtractMultiscale` with `harris = true`
  or `total_features > 0`. That path is exercised only end-to-end, through `scratch_pipeline`, the
  `milestone3_scratch_pipeline` ctest (`CMakeLists.txt:98`), `feature_audit_pair_0_1` (`CMakeLists.txt:127`)
  and the T0 configuration checks in `tests/pipeline/test_experiment.cpp:82-87`.
- **Before studying it, understand:** `fast.hpp`, `orientation.hpp`, `brief.hpp`, `pyramid.hpp`; the Harris
  corner measure; ORB's per-level feature allocation.

### `src/features/multiscale.cpp`
- **Category:** production source
- **Status:** used (`cmake/feature_core.cmake:14`).
- **Purpose and why it exists:** Implements multiscale extraction, Harris re-ranking and the per-level budget.
- **Responsibility in the architecture:** Per-frame extraction in production.
- **Key types and functions:**
  - `HarrisResponse` (lines 13–33) returns 0 within 4 px of the border (lines 14–15). Over a 7×7 unweighted
    (box) window (lines 17–18) it computes 3×3 Sobel gradients scaled by 1/1020, so they lie in [−1, 1]
    (lines 20–27). It accumulates `a = Σgx²`, `b = Σgy²`, `c = Σgx·gy` and returns
    `det − k·trace² = a·b − c² − 0.04·(a+b)²`, with k = 0.04 at line 32. It runs on the raw (unsmoothed) level image.
  - `LevelQuotas` (lines 34–56) throws for `total < 0`, `levels < 1` or `scale < 1` (lines 35–36). Level l
    receives `total·s^−l / Σ_k s^−k` (lines 40–48), floored. The leftover units go to the levels with the
    largest fractional parts (a stable sort, so ties go to the lower level), and the quotas sum exactly to `total` (lines 49–54).
  - `ExtractMultiscale` (lines 57–107):
    - If `total_features > 0`, the quotas are computed with `scale = pyramid.scale_x[1]` (the rounded ratio, for
      example 640/533 ≈ 1.2008), or 1.2 if there is only one level (lines 62–66).
    - On each level, if Harris ranking or quotas are active, DetectFast runs with no cap (`max_keypoints = 0`,
      lines 69–72).
    - With `harris`, the response of each candidate is computed once into a `std::map` keyed by integer
      position. The candidates are then stable-sorted by descending response, so equal responses keep FAST
      order (lines 73–85).
    - The list is cut to `quotas[l]`, or to `fast.max_keypoints` if there are no quotas (lines 86–88). With
      quotas the cut always applies, even when the quota is 0.
    - Then `AssignOrientations` (line 89) and `ComputeBrief(GaussianSmooth(img), ...)` with validity flags
      (lines 90–92). Each keypoint is mapped to level 0 with `LevelToBaseX/Y` (lines 94–104).
- **Inputs → outputs:** see header.
- **Dependencies:** `features/multiscale.hpp`, `features/brief.hpp`, `features/orientation.hpp`, `<algorithm>`,
  `<cmath>`, `<numeric>`, `<map>`, `<stdexcept>`.
- **Used by:** see header.
- **Covered by tests:** `tests/unit/features_pyramid_test.cpp:93-137` checks that every level has keypoints,
  that the level/scale/coordinate fields are consistent, that orientation and descriptor are computed on the
  keypoint's own level, and that a single level reproduces the single-scale path bit for bit.
  `tests/pipeline/test_improvements.cpp:15-29` checks quotas and Harris. `tests/pipeline/test_multiscale_0_1.cpp`
  covers the front end on Bunny pair 0→1 with `FrontendParams{}` (threshold 20, no Harris, no budget).
- **Before studying it, understand:** `multiscale.hpp`; Sobel operator; the Harris–Stephens response.

---

### `include/slam/geometry/fundamental.hpp`
- **Category:** public header (header-only implementation, Eigen)
- **Status:** used in production, but its result is not consumed. `src/pipeline/pipeline.cpp:2,87-88` calls it
  for every pair on the filtered pixel correspondences and stores `PairResult::fundamental_ok` and `PairResult::F`
  (`include/slam/pipeline/pipeline.hpp:66-67`). A repository-wide grep finds no reader of either field in
  `apps/`, `src/`, `include/slam/` or `tests/pipeline/`, so F is computed but affects no estimate or output.
  It is also compiled into the reference library `geometry` (`reference/opencv_slambook2/src/geometry/geometry.cpp:2,35`;
  the include path is added at `reference/opencv_slambook2/CMakeLists.txt:35-36`). It is not a source of
  `feature_core`. It is reached through `feature_core`'s PUBLIC include root `include/slam`.
- **Purpose and why it exists:** A project-owned normalised 8-point fundamental-matrix estimator in pixel
  coordinates (`x2ᵀ F x1 = 0`) using Eigen only. It replaces OpenCV's `findFundamentalMat` in the reference demo
  (the OpenCV build). The reference demo throws if the estimate fails.
- **Responsibility in the architecture:** A diagnostic two-view quantity. Pose comes from PnP. The two-view
  stage that does run is the essential-matrix RANSAC (`src/pipeline/pipeline.cpp:89-90`), not this function.
- **Key types and functions:**
  - `geometry_scratch::EstimateFundamental8Point(x1, x2, F)` (lines 9–61):
    1. **Size check:** it fails if the sizes differ or there are fewer than 8 points (lines 12–13).
    2. **Hartley normalisation** of each point set (lines 14–34): translate to the centroid and scale so that
       the mean distance is √2. It fails on a non-finite point or a mean distance below 1e-12.
    3. **Design matrix:** rows `[x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]` (lines 38–44).
    4. **Null-space solve:** a full-V Jacobi SVD (line 45). The rank-eight check fails if
       `σ8 < 1e-10·σ1` (lines 46–47); otherwise the null vector is the last column of V (lines 48–50).
    5. **Rank-2 projection:** the smallest singular value is set to zero (lines 51–54).
    6. **Denormalisation:** `F = T2ᵀ · F̂ · T1` (lines 55–56). F is then scaled to unit Frobenius norm
       (lines 57–60). It fails if F is non-finite or nearly zero.

    With no outlier handling (no RANSAC), all input correspondences are used.
- **Inputs → outputs:** `std::vector<Eigen::Vector2d>` pixel points (x1 in image 1, x2 in image 2) → `Eigen::Matrix3d F`
  (unit Frobenius norm, rank 2) and `bool` success.
- **Dependencies:** `<Eigen/Core>`, `<Eigen/SVD>`, `<cmath>`, `<vector>`.
- **Used by:** `src/pipeline/pipeline.cpp:2,88`; `reference/opencv_slambook2/src/geometry/geometry.cpp:2,35`;
  `tests/pipeline/test_improvements.cpp:3,61,70`.
- **Covered by tests:** `tests/pipeline/test_improvements.cpp:39-71` checks recovery on 100 synthetic noiseless
  points ("Fundamental recovery"), `|det F| < 1e-10` ("Fundamental rank two"), `|x2ᵀ F x1| < 1e-7` for all points
  ("Epipolar consistency"), and that 10 identical points are rejected ("Degenerate fundamental rejection").
  No test is in `tests/unit/`.
- **Before studying it, understand:** epipolar geometry, the 8-point algorithm and Hartley normalisation (Hartley &
  Zisserman §11.1–11.2); SVD null spaces; rank-2 enforcement.

---

### `tests/unit/feature_test_utils.hpp`
- **Category:** test (header-only helper)
- **Status:** used by the six `features_*_test` executables (`CMakeLists.txt:81-86`). It is found because each
  test source includes it from its own directory.
- **Purpose and why it exists:** Gives the feature unit tests a common failure-counting check macro and
  deterministic synthetic images, so the tests need no data files and no third-party code.
- **Responsibility in the architecture:** Test infrastructure for `feature_core` only. The pipeline tests use
  `tests/pipeline/check.hpp` instead.
- **Key types and functions:**
  - `ftest::Failures()` (lines 15–18): a static counter.
  - `FCHECK(cond, fmt, ...)` (lines 20–27) prints `[PASS]` or `[FAIL]` and counts failures. It does not abort.
  - `Finish(name)` (lines 29–33) prints a summary and returns exit code 0 or 1.
  - `Noise(w, h, seed)` (lines 36–41): uniform 8-bit noise from `std::mt19937`.
  - `Texture(w, h, seed)` (lines 45–69): six random oriented sinusoids around 128, plus 40 sharp-edged
    rectangles that create real corners.
  - `Rotate90(in)` (lines 74–79): an exact 90° rotation, `out(x', y') = in(W−1−y', x')`, so a point (x, y) moves
    to (y, W−1−x). The header calls it counter-clockwise; in y-down image coordinates the angle decreases by 90°,
    which is what `features_orientation_test.cpp:48-65` checks.
- **Inputs → outputs:** seeds and sizes → `features::GrayImage`.
- **Dependencies:** `features/types.hpp`, `<cmath>`, `<cstdint>`, `<cstdio>`, `<random>`.
- **Used by:** `tests/unit/features_fast_test.cpp`, `features_orientation_test.cpp`, `features_brief_test.cpp`,
  `features_matcher_test.cpp`, `features_pyramid_test.cpp`, `features_matcher_ranked_test.cpp`.
- **Covered by tests:** n/a (test helper).
- **Before studying it, understand:** `types.hpp`; `std::mt19937` determinism.

### `tests/unit/features_fast_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_fast_test` (`CMakeLists.txt:81-85`, links `feature_core`).
- **Purpose and why it exists:** A self-checking test of `features/fast` (exit 0 = pass). It locks the
  detector's exact behaviour: segment test, score, NMS, border, cap, order, determinism and the per-cell fallback.
- **Responsibility in the architecture:** Unit-level regression guard for detection.
- **Key types and functions:**
  - helpers `Ring` (7×7 image with chosen circle pixels) and `Arc` (lines 16–26);
  - segment-test checks, including "difference exactly at the threshold does not count" and "arc wrapping 15 → 0"
    (lines 34–43);
  - the score equals 9·(200−100−30) (line 46);
  - square corners detected and none on edges (52–74);
  - on a noise image: more than 500 corners, the 500 cap, border 16, NMS, order and prefix property,
    determinism, and fewer corners at threshold 60 (77–115);
  - per-cell fallback with threshold 20/7 and 32-px cells: the weak square in an empty cell is found, the one
    sharing a cell with a corner is not added, `min_threshold >= threshold` disables it, and it is off by default
    (118–156).
- **Inputs → outputs:** synthetic images → PASS/FAIL lines and exit status.
- **Dependencies:** `features/fast.hpp`, `feature_test_utils.hpp`; `feature_core`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `include/slam/features/fast.hpp` and `src/features/fast.cpp`.
- **Before studying it, understand:** `fast.hpp`.

### `tests/unit/features_orientation_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_orientation_test` (`CMakeLists.txt:81-85`).
- **Purpose and why it exists:** A self-checking test of the intensity-centroid angle and its sign convention.
- **Responsibility in the architecture:** Unit guard for orientation, on which steered BRIEF depends.
- **Key types and functions:**
  - `AngleDiffDeg` (lines 13–18);
  - a blob at 24 directions with maximum error < 3° (lines 28–37);
  - a blob below the keypoint gives +90° (39–46);
  - `Rotate90` on a texture: more than 50 keypoints, and each angle after rotation equals the original angle
    −90° within 1e-3° (48–65);
  - a uniform patch gives 0 (67–72).
- **Inputs → outputs:** synthetic images → PASS/FAIL.
- **Dependencies:** `features/fast.hpp`, `features/orientation.hpp`, `feature_test_utils.hpp`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `orientation.hpp/.cpp` and uses `DetectFast`.
- **Before studying it, understand:** `orientation.hpp`; the y-down angle convention.

### `tests/unit/features_brief_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_brief_test` (`CMakeLists.txt:81-85`).
- **Purpose and why it exists:** A self-checking test of the BRIEF pattern, the smoothing, the descriptor
  validity and rotation behaviour, and the Hamming distance. The pattern fingerprint check (line 35) makes any
  change to the pattern, seed or sampling visible.
- **Responsibility in the architecture:** Unit guard for description.
- **Key types and functions:** see the coverage list under `src/features/brief.cpp` (lines 19–124 of this test).
- **Inputs → outputs:** synthetic images → PASS/FAIL; it also prints the pattern fingerprint.
- **Dependencies:** `features/brief.hpp`, `features/fast.hpp`, `features/orientation.hpp`, `feature_test_utils.hpp`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `brief.hpp/.cpp`.
- **Before studying it, understand:** `brief.hpp`; FNV-1a hashing (used only as a fingerprint).

### `tests/unit/features_matcher_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_matcher_test` (`CMakeLists.txt:81-85`).
- **Purpose and why it exists:** A self-checking test of the original matcher (`MatchBruteForce`,
  `FilterMatchesByDistance`) and of the whole single-scale chain on an image and its shifted copy.
- **Responsibility in the architecture:** Unit guard for the legacy / M0 matcher. It does not test `MatchMutualRatio`.
- **Key types and functions:**
  - helpers `WithBits` and `M` (lines 17–29);
  - nearest neighbour, tie → lowest index, empty inputs (36–44);
  - filter limits `max(20, 30) = 30` and `max(40, 30) = 40` (46–55);
  - shifted copy (7, −4): more than 100 filtered matches, ≥ 90 % with exactly the true shift (57–79).
- **Inputs → outputs:** synthetic descriptors and images → PASS/FAIL.
- **Dependencies:** `features/brief.hpp`, `fast.hpp`, `matcher.hpp`, `orientation.hpp`, `feature_test_utils.hpp`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `matcher.hpp/.cpp` (two functions) and the FAST → orientation → BRIEF chain.
- **Before studying it, understand:** `matcher.hpp`.

### `tests/unit/features_pyramid_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_pyramid_test` (`CMakeLists.txt:81-85`).
- **Purpose and why it exists:** A self-checking test of `features/pyramid` and of `ExtractMultiscale` with the
  default `SelectionParams`.
- **Responsibility in the architecture:** Unit guard for the scale structure and the level → original coordinate mapping.
- **Key types and functions:** level sizes and scales for 640×480 (lines 18–31); identity, determinism and the
  single-level case (32–39); constant and ramp reproduction (42–68); blob-centroid mapping (69–91); multiscale
  extraction at threshold 20: every level has keypoints, the fields are consistent, the descriptor and angle come
  from the keypoint's own level, and one level equals the single-scale path (93–137).
- **Inputs → outputs:** synthetic images → PASS/FAIL.
- **Dependencies:** `features/brief.hpp`, `multiscale.hpp`, `orientation.hpp`, `pyramid.hpp`, `feature_test_utils.hpp`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `pyramid.hpp/.cpp` and `multiscale.hpp/.cpp` (legacy selection only; Harris
  and the budget are not tested here).
- **Before studying it, understand:** `pyramid.hpp`, `multiscale.hpp`.

### `tests/unit/features_matcher_ranked_test.cpp`
- **Category:** test
- **Status:** used; ctest `features_matcher_ranked_test` (`CMakeLists.txt:81-85`).
- **Purpose and why it exists:** A self-checking test of `MatchRanked` (Stage 3, ported from main). Above all, it
  shows that with every option off `MatchRanked` is identical to `FilterMatchesByDistance(MatchBruteForce(...))`.
  It then checks each option separately.
- **Responsibility in the architecture:** Unit guard for the optional ranked matcher.
- **Key types and functions:**
  - helpers `Bits` (a descriptor whose Hamming distances are differences of run lengths), `SameMatches`,
    `CountersAddUp` and `Equivalent` (lines 25–59);
  - test blocks listed under `src/features/matcher.cpp`, "Covered by tests" (lines 67–308);
  - the real-data equivalence block (lines 85–103) runs `ExtractMultiscale` on an 8-level pyramid with
    `FastParams{}` (threshold 30), with `fa.valid` and `fb.valid`.
- **Inputs → outputs:** synthetic descriptors and images → PASS/FAIL.
- **Dependencies:** `features/brief.hpp`, `fast.hpp`, `matcher.hpp`, `multiscale.hpp`, `orientation.hpp`,
  `pyramid.hpp`, `feature_test_utils.hpp`.
- **Used by:** CTest only.
- **Covered by tests:** n/a. It tests `MatchRanked`, `QualityBefore`, `MatchFateName`, `RankedMatch::ratio`.
- **Before studying it, understand:** `matcher.hpp` (the `MatchRanked` contract, lines 32–61).

---

### How the feature module fits together

**Call order, image → matches (production `scratch_pipeline`):**

1. `scratch::ExtractFeatures` (`src/pipeline/frontend.cpp:42-70`): decoded PNG → `features::GrayFromRGB` (line 48)
   → `features::BuildPyramid(gray, params.pyramid)` → `features::ExtractMultiscale(pyramid, params.fast,
   params.selection)` (lines 52–54).
2. Inside `ExtractMultiscale` (`src/features/multiscale.cpp:57-107`), for each level l = 0..7:
   1. `DetectFast` (uncapped when Harris or a budget is active);
   2. optional Harris re-ranking (`HarrisResponse`);
   3. cut to `LevelQuotas[l]` or to `max_keypoints`;
   4. `AssignOrientations`;
   5. `GaussianSmooth` → `ComputeBrief`;
   6. map to level 0 with `LevelToBaseX/Y`.

   `ExtractFeatures` then overwrites each keypoint's `x, y` with the level-0 position and keeps `level`, `scale`
   and the level position (`frontend.cpp:58-68`).
3. `scratch::MatchFrames` (`src/pipeline/frontend.cpp:72-151`):
   - **Non-ranked branch:** `MatchBruteForce` → `FilterMatchesByDistance(match_floor = 30)` (lines 78–80).
     If `mutual_ratio` is set, `filtered` is replaced by `MatchMutualRatio(fi, fj, ratio)` (lines 81–83).
   - **Ranked branch:** `MatchRanked(...)` with level-0 positions, levels and validity (lines 85–91).

   Each filtered match becomes a `Correspondence`: pixels in both frames, depth at the truncated pixel, the
   optional 3×3 depth-edge rejection, and back-projection to `X_i` (lines 102–149).
4. `ProcessPair` also calls `geometry_scratch::EstimateFundamental8Point` on the filtered pixels
   (`src/pipeline/pipeline.cpp:87-88`). The result is stored and not used further.

**Parameter layers.** Three layers set the defaults; the production value is the last one applied.
- Library defaults: `FastParams{threshold 30, arc 9, border 16, max 500, cell 0, min 0}` (`fast.hpp:15-24`);
  `PyramidParams{8, 1.2}` (`pyramid.hpp:25-26`); `SelectionParams{false, 0}` (`multiscale.hpp:39-42`);
  `MatchMutualRatio` ratio `.87` (`matcher.hpp:30`); `FilterMatchesByDistance` floor 30 (`matcher.hpp:23`);
  `MatcherParams{floor 30, mutual off, ratio 0, radius 3 px, level diff off, rank off}` (`matcher.hpp:64-70`).
- Front-end defaults, `FrontendParams{}` (`include/slam/pipeline/frontend.hpp:30-48`): FAST threshold 20 (line 47),
  `mutual_ratio = false` (36), `ratio = .87` (37), `depth_edge_m = 0` (38), `match_floor = 30` (39–40),
  `ranked_matcher = false` (45). These are what `--legacy` gives (`apps/pipeline_main.cpp:112-113`).
- Production defaults (T0): `apps/pipeline_main.cpp:101` calls `ApplyImprovedDefaults`
  (`src/pipeline/pipeline.cpp:39-48`) before parsing options. For the feature module this sets:

| Option | Status | Where |
|---|---|---|
| Harris ranking | **production default** (`selection.harris = true`) | `pipeline.cpp:40` |
| Feature budget 1400 across 8 levels (geometric quotas; replaces 500 per level) | **production default** | `pipeline.cpp:40`; quotas `multiscale.cpp:62-66,86-88` |
| FAST threshold 10 | **production default** | `pipeline.cpp:41` |
| `MatchMutualRatio`, ratio 0.87 | **production default** (`mutual_ratio = true`; ratio left at `.87`) | `pipeline.cpp:42`; `frontend.hpp:37`; `frontend.cpp:81-83` |
| Pyramid 8 levels × 1.2, border 16, FAST-9 | unchanged library defaults | `pyramid.hpp:25-26`, `fast.hpp:16-17` |
| Distance filter `max(2·d_min, 30)` | runs, but in T0 it only fills `raw`/`d_min`/`d_max`; the filtered set comes from `MatchMutualRatio` | `frontend.cpp:78-83` |
| `MatchRanked` | **optional**: `--ranked-matcher` and the `--match-*` flags; replaces `mutual_ratio` | `frontend.hpp:43-45`; `pipeline_main.cpp:152-165`; `frontend.cpp:77,84-101` |
| FAST per-cell fallback | **opt-in**: off by default (`cell_size = 0`, `min_threshold = 0`); enabled by `--fast-cell C --fast-min-threshold T2`, or by `--stage2-frontend` (30 px / 7) | `fast.hpp:23-24`; `pipeline_main.cpp:132-137`; `pipeline.cpp:50-54` |
| Ablations | `--no-harris`, `--feature-budget N` (0 = 500 per level), `--no-mutual-ratio`, `--ratio R`, `--fast-threshold T`, `--pyramid-levels L` | `pipeline_main.cpp:118-145` |

The same T0 values are asserted in `tests/pipeline/test_experiment.cpp:82-87`: Harris as in the reference
configuration, budget 1400, threshold 10, `mutual_ratio`, ratio equal to the reference, `ranked_matcher` off,
`cell_size == 0`.

**Not implemented in this module.** There is no sub-pixel keypoint refinement (positions are integer level
pixels mapped to level 0), no scale-space blob detection, no learned ORB pattern, and no approximate or indexed
matching (all matchers are brute force). Nothing tracks features across more than two frames: matching is only
between consecutive frames, and there is no map or landmark association.

## Production pipeline library (scratch_core)

`scratch_core` is the production per-frame-pair visual odometry library. It is defined in the root
`CMakeLists.txt:27-32` from the nine sources in `src/pipeline/`, with public include directory
`include/slam/pipeline` (so code writes `#include "pipeline.hpp"`), and it links `Eigen3::Eigen` and
`feature_core` publicly (`CMakeLists.txt:31`). There is no OpenCV dependency. Two headers in the directory
(`common.hpp`, `accumulate.hpp`) are header-only and have no `.cpp`.

What the library does: two consecutive RGB-D frames → project-owned features → matches → 3D→2D
correspondences (frame-i depth) → RANSAC around a linear (DLT) PnP → Levenberg–Marquardt refinement →
(optionally) inlier re-selection and a pose-validity gate → one metric relative pose `T_{i+1<-i}`. A second,
scale-free two-view estimate (RANSAC 8-point essential matrix) is computed alongside it for comparison only.

What it does NOT do (verified by reading every file): no map, no keyframes, no local or global bundle
adjustment, no loop closure, no relocalisation, no motion model or pose prior between pairs. Each pair
`i → i+1` is solved independently; the trajectory is a plain chain of relative poses (`accumulate.hpp`). Ground
truth is never an input to estimation; the only ground-truth value the production application feeds into the
trajectory is the anchor `T_wc[0]` (`apps/pipeline_main.cpp:335`), which fixes the world frame for evaluation.

Configurations referred to below:
- **`PipelineParams{}`** (struct defaults, also called L0 / `--legacy`): FAST threshold 20, 500 keypoints per
  level, 8-level pyramid, distance-filter matching, PnP RANSAC 300 iterations / 8 px, LM least squares, no
  depth-edge rejection, no re-selection, gate off.
- **T0** = production default of `scratch_pipeline`: `PipelineParams{}` + `ApplyImprovedDefaults`
  (`apps/pipeline_main.cpp:101`, `src/pipeline/pipeline.cpp:39-48`).
- **M0** (`--stage2-frontend`): `ApplyStage2Frontend` = `PipelineParams{}` + FAST per-cell fallback 30/7
  (`src/pipeline/pipeline.cpp:50-54`); the gate is still off unless `--pose-gate` is also given.

Tests named below live in `tests/pipeline/` and are built and registered with ctest by
`CMakeLists.txt:87-93` (`test_png test_frontend test_essential test_ransac test_pipeline_0_1
test_multiscale_0_1 test_refine test_validity test_ranked_frontend`), `CMakeLists.txt:120-122`
(`test_improvements`), `CMakeLists.txt:100-107` (`test_baseline`, `test_experiment`, through the evaluation
libraries). The checks
use the `CHECK` macro of `tests/pipeline/check.hpp:18-25`.

---

### `include/slam/pipeline/common.hpp`
- **Category:** public header (header-only; not compiled into `scratch_core` as a source, but on its include path)
- **Status:** used — included by `scratch_pipeline`, `pnp_from_scratch`, `pnp_full_sequence`,
  `depth_edge_experiment`, `scratch_baseline_lib`/`scratch_experiment_lib` (via `include/slam/evaluation/*.hpp`),
  `accumulate.hpp`, and every pipeline test (via `tests/pipeline/check.hpp:9`).
- **Purpose and why it exists:** Small shared helpers that every program needs but which are not part of the
  per-pair estimation: a rigid-pose struct, quaternion ↔ rotation conversion written out by hand, the
  repository's rotation/translation error metric, finding the repository root, a minimal CSV reader, a TUM
  ground-truth reader, and reprojection statistics. It exists so that the milestone programs, the production
  application, the evaluation harness and the tests compute errors and read files identically. Its header
  comment (`common.hpp:2-4`) still says "shared by the two programs (milestone 1 and the full sequence)"; it is
  now shared much more widely.
- **Responsibility in the architecture:** Utility / evaluation layer. Nothing in the per-pair estimation
  (`src/pipeline/*.cpp`) includes it; it is used by the application and tooling around the library.
- **Key types and functions:**
  - `struct Pose { R, t }` (`common.hpp:27-30`). Convention comment: **`X_b = R * X_a + t`** (`common.hpp:27`);
    the meaning of a/b depends on the variable (e.g. `T_wc`, or `T_{k+1<-k}`).
  - `QuatToR(x, y, z, w)` (`common.hpp:33-41`): normalises the quaternion, then the standard explicit formula for
    the rotation matrix; quaternion order is (x, y, z, w) as in TUM files.
  - `RotErrDeg(A, B)` (`common.hpp:45-48`): `acos(clamp((trace(AᵀB) − 1)/2))` in **degrees**.
    `RotAngleDeg(R)` (`common.hpp:49`) = angle of R. Comment refers to `tests/report_utils.hpp`; that file now
    lives at `reference/opencv_slambook2/tests/report_utils.hpp`.
  - `RToQuat(R)` (`common.hpp:52-71`): Shepperd's method (four branches by largest diagonal term), returns
    (x, y, z, w) with canonical sign `w ≥ 0` (`common.hpp:69`) and unit norm.
  - `IsRepoRoot` / `FindRepo(start)` (`common.hpp:73-81`): walk up from `start` until
    `data/synthetic_bunny/intrinsics.txt` exists; returns an empty path if not found.
  - `struct Csv` (`common.hpp:84-108`): header row → column index map; rows split on `,`; a trailing comma
    yields an empty last cell (`common.hpp:101`). No quoting support.
  - `ReadGroundtruthTwc(path)` (`common.hpp:110-125`): TUM format `ts tx ty tz qx qy qz qw`, skipping `#` lines;
    returns **`T_wc`** (camera → world, `X_w = R X_c + t`) per the comment at `common.hpp:110`.
  - `Reproj` / `Reprojection(X, uv, T, K)` (`common.hpp:127-154`): projects each 3D point with `projectPoint`
    under `T` (so `X_cam = T.R X + T.t`), returns per-point pixel error, mean, median, max, RMSE and the
    count of points with positive depth.
- **Inputs → outputs:** rotations (3×3), translations (metres), quaternions (unitless); errors in degrees and
  metres; pixel errors in pixels; file paths.
- **Dependencies:** `<algorithm> <cmath> <filesystem> <fstream> <map> <sstream> <string> <vector>`,
  `<Eigen/Dense>`, `projection.hpp`.
- **Used by:** `apps/pipeline_main.cpp`, `apps/milestones/main.cpp`, `apps/milestones/sequence_main.cpp`,
  `apps/baseline_main.cpp`, `apps/experiments_main.cpp`, `experiments/depth_edge/depth_edge_experiment.cpp`,
  `include/slam/evaluation/baseline.hpp`, `include/slam/evaluation/experiment.hpp`, `src/evaluation/baseline.cpp`,
  `src/evaluation/experiment.cpp`, `include/slam/pipeline/accumulate.hpp`, `tests/pipeline/check.hpp` (hence all
  pipeline tests).
- **Covered by tests:** no dedicated test. Exercised indirectly: `FindRepo` by every test via
  `scratch_test::Repo()` (`check.hpp:34-41`); `QuatToR`/`ReadGroundtruthTwc` by `test_frontend`,
  `test_pipeline_0_1`, `test_multiscale_0_1`, `test_validity`, `test_baseline`; `RToQuat` and `Reprojection` by
  `test_baseline`; `RotErrDeg` by most geometry tests; `Csv` by `test_png` (depth values vs the frozen export).
- **Before studying it, understand:** rotation matrices and unit quaternions, the TUM trajectory file format,
  the meaning of `T_wc` vs `T_cw`.

### `include/slam/pipeline/png.hpp`
- **Category:** public header
- **Status:** used — part of `scratch_core`; consumed by `scratch_pipeline` and everything that loads frames
  (through `frontend.hpp:25`).
- **Purpose and why it exists:** Declares a self-written PNG decoder so that the production pipeline needs no
  image library (no OpenCV, no libpng, no zlib). It supports exactly what the RGB-D data needs: colour types 0,
  2, 4, 6; bit depth 8 or 16; no interlacing; palette and Adam7 are rejected (`png.hpp:5-8`).
- **Responsibility in the architecture:** Image I/O at the bottom of the stack.
- **Key types and functions:**
  - `struct PngImage` (`png.hpp:20-27`): width, height, channels (1–4), bit_depth (8/16), `samples` as
    `uint16_t` row-major interleaved; 16-bit samples are the big-endian value assembled into a native integer.
  - `DecodePng`, `LoadPng`, `Inflate`, `ZlibDecompress`, `Crc32`, `Adler32` (`png.hpp:30-42`); the last four are
    exposed for tests.
- **Inputs → outputs:** PNG bytes or path → `PngImage`; raw sample values (no unit conversion; depth scaling
  happens in `frontend.cpp`).
- **Dependencies:** `<cstdint> <string> <vector>`.
- **Used by:** `include/slam/pipeline/frontend.hpp`, `src/pipeline/png.cpp`, `tests/pipeline/test_png.cpp`.
- **Covered by tests:** `tests/pipeline/test_png.cpp`.
- **Before studying it, understand:** PNG chunk structure, zlib (RFC 1950) and DEFLATE (RFC 1951).

### `src/pipeline/png.cpp`
- **Category:** production source
- **Status:** used — compiled into `scratch_core` (`CMakeLists.txt:27`).
- **Purpose and why it exists:** Implements the decoder declared in `png.hpp`, written from the specifications
  (`png.cpp:1-3`). It is the reason the production pipeline can read the Bunny and TUM PNG files with only the
  C++ standard library.
- **Responsibility in the architecture:** Image I/O.
- **Key types and functions:**
  - `BitReader` (`png.cpp:23-48`): LSB-first bit reader; sets `overrun` instead of reading past the end.
  - `Huffman` (`png.cpp:54-91`): canonical Huffman code stored as count-per-length + sorted symbols; `build`
    rejects over-subscribed length sets (`png.cpp:64-68`); `decode` reads one bit at a time (`png.cpp:78-90`).
  - Length/distance base and extra-bit tables (`png.cpp:93-98`).
  - `InflateCodes` (`png.cpp:101-123`): literal / end-of-block / length+distance back-reference copy (overlapping
    copy intended, `png.cpp:120`).
  - `InflateFixed` (`png.cpp:125-141`): fixed codes of RFC 1951 §3.2.6, built once (static).
  - `InflateDynamic` (`png.cpp:143-177`): reads HLIT/HDIST/HCLEN, the code-length code (permuted order
    `png.cpp:144`), repeat codes 16/17/18, then the literal/length and distance codes.
  - `Crc32` (`png.cpp:191-205`): table-driven, polynomial `0xEDB88320`. `Adler32` (`png.cpp:207-214`): mod 65521.
  - `Inflate` (`png.cpp:216-239`): block loop over stored (type 0, LEN/NLEN check), fixed (1), dynamic (2);
    type 3 rejected.
  - `ZlibDecompress` (`png.cpp:241-252`): checks CM = 8, FCHECK `(CMF·256+FLG) % 31 == 0`, no preset dictionary,
    inflates, then verifies the Adler-32 trailer.
  - `DecodePng` (`png.cpp:254-327`): signature check (`255-256`); chunk loop with CRC check of every chunk
    (`266`); IHDR parsing, interlace rejected (`274`); IDAT concatenated; PLTE rejected (`279-280`); colour type →
    channels (`284-290`); bit depth 8/16 only (`291`); size check of the decompressed stream (`298`); per-row
    filter reversal None/Sub/Up/Average/Paeth (`301-318`, Paeth predictor `183-187`); 16-bit samples assembled
    big-endian (`323`).
  - `LoadPng` (`png.cpp:329-334`): reads the file and calls `DecodePng`.
- **Inputs → outputs:** bytes → `PngImage` samples (raw 8- or 16-bit integers); errors as strings.
- **Dependencies:** `png.hpp`, `<cstdlib> <fstream> <iterator>`.
- **Used by:** through `LoadPng`: `src/pipeline/frontend.cpp` (`LoadRgbdFiles`, `frontend.cpp:20-21`);
  `tests/pipeline/test_png.cpp`.
- **Covered by tests:** `test_png`: CRC-32 of "123456789" = `0xCBF43926` (line 37); Adler-32 of "Wikipedia"
  (38); stored block and fixed/dynamic zlib streams produced by Python zlib (46-61); corrupted stream rejected
  (65); Bunny frame 0 RGB is 640×480×3 8-bit and its sample fingerprint equals Pillow's (73-75); depth 16-bit
  single channel with matching fingerprint (77-80); the image's first DEFLATE block is dynamic Huffman (88);
  depth values equal the frozen baseline export (100).
- **Before studying it, understand:** canonical Huffman codes, LZ77 back-references, PNG filter types, CRC-32.

### `include/slam/pipeline/projection.hpp`
- **Category:** public header
- **Status:** used — part of `scratch_core`; also compiled (with `projection.cpp`) directly into
  `pnp_from_scratch` and `pnp_full_sequence` (`CMakeLists.txt:35, 41`).
- **Purpose and why it exists:** Defines the pinhole intrinsics struct and the single projection function
  used everywhere in the library, so that every residual (RANSAC, LM, validity statistics, evaluation) uses the
  same model. No lens distortion (`projection.hpp:5`).
- **Responsibility in the architecture:** Camera model (lowest geometric layer).
- **Key types and functions:**
  - `struct Intrinsics { fx, fy, cx, cy }` (`projection.hpp:6-8`), in pixels. Note: declared in the global
    namespace, not in `scratch`.
  - `projectPoint(X, R, t, K, depth*)` (`projection.hpp:15-16`): convention **`X_camera = R * X + t`**
    (`projection.hpp:10-12`), `u = fx·Xc/Zc + cx`, `v = fy·Yc/Zc + cy`; optional output `Zc` (point in front iff
    `Zc > 0`).
- **Inputs → outputs:** 3D point (metres) + pose → pixel (u, v) and depth Zc (metres).
- **Dependencies:** `<Eigen/Core>`.
- **Used by:** `common.hpp`, `essential.hpp`, `frontend.hpp`, `pnp.hpp`, `ransac.hpp`, `refine.hpp`,
  `src/pipeline/ransac.cpp`, `refine.cpp`, `pipeline.cpp`, `apps/milestones/main.cpp`,
  `apps/milestones/sequence_main.cpp`, `apps/experiments_main.cpp`, `src/evaluation/experiment.cpp`,
  `experiments/depth_edge/depth_edge_experiment.cpp`, and tests (below).
- **Covered by tests:** indirectly by `test_frontend`, `test_multiscale_0_1`, `test_ransac`, `test_refine`,
  `test_validity`, `test_experiment` (they project synthetic or ground-truth points with it).
- **Before studying it, understand:** the pinhole camera model.

### `src/pipeline/projection.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`, `pnp_from_scratch`, `pnp_full_sequence`.
- **Purpose and why it exists:** The 9-line implementation of `projectPoint`.
- **Responsibility in the architecture:** Camera model.
- **Key types and functions:** `projectPoint` (`projection.cpp:3-9`): `Xc = R*X + t` (`projection.cpp:5`), then
  perspective division and intrinsics (`projection.cpp:7-8`). No guard against `Zc = 0` (division yields inf);
  callers check `Zc > 0` themselves.
- **Inputs → outputs:** as in the header.
- **Dependencies:** `projection.hpp`.
- **Used by:** see `projection.hpp`.
- **Covered by tests:** see `projection.hpp`.
- **Before studying it, understand:** nothing beyond the header.

### `include/slam/pipeline/pnp.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`; also `pnp_from_scratch`, `pnp_full_sequence` (milestones 1 and 2).
- **Purpose and why it exists:** Declares the from-scratch linear PnP solver, the minimal model inside the
  production RANSAC. It also carries diagnostics of the linear solve (singular values, scale, rotation deviation
  before orthonormalisation, null-space ratio) that the milestone programs print.
- **Responsibility in the architecture:** Minimal / linear pose solver (model fitting for RANSAC).
- **Key types and functions:**
  - `struct PnPResult` (`pnp.hpp:10-21`): `ok`, `reason`, `R`, `t`, `sv_left_block`, `t_from_dlt`, `scale`
    (λ), `raw_rotation_deviation`, `nullspace_ratio` (σ12/σ11 of the DLT matrix).
  - `solvePnPDLT(X, uv, K)` (`pnp.hpp:27-28`): needs n ≥ 6; X in the object frame (here camera i), uv pixels in
    the target image (frame i+1). Returns **`X_camera = R * X + t`** (`pnp.hpp:26`), i.e. `T_{i+1<-i}` in the
    pipeline. No RANSAC inside.
- **Inputs → outputs:** 3D points (metres), pixels, `Intrinsics` → R, t (metres).
- **Dependencies:** `<string> <vector> <Eigen/Core>`, `projection.hpp`.
- **Used by:** `src/pipeline/pnp.cpp`, `src/pipeline/ransac.cpp`, `src/evaluation/experiment.cpp`,
  `apps/milestones/main.cpp`, `apps/milestones/sequence_main.cpp`, `tests/pipeline/test_pipeline_0_1.cpp`,
  `test_ransac.cpp`, `test_refine.cpp`. (The `tracking/pnp.hpp` includes under `reference/` are a different,
  OpenCV-based header.)
- **Covered by tests:** see `pnp.cpp`.
- **Before studying it, understand:** homogeneous coordinates, DLT, SVD null spaces.

### `src/pipeline/pnp.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`, `pnp_from_scratch`, `pnp_full_sequence`.
- **Purpose and why it exists:** Implements the linear (DLT) PnP. It is the model fitted to every 6-point
  RANSAC sample and refitted on all RANSAC inliers.
- **Responsibility in the architecture:** Linear pose solver.
- **Key types and functions:** `solvePnPDLT` (`pnp.cpp:10-118`), algorithm:
  1. Fails if n < 6 or sizes differ (`pnp.cpp:14-17`).
  2. Removes intrinsics: normalised coordinates `x = ((u−cx)/fx, (v−cy)/fy)` (`pnp.cpp:22`), so the unknown is
     `M = λ[R | t]` (12 numbers).
  3. Hartley normalisation: 2D centroid → 0, mean distance → √2; 3D centroid → 0, mean distance → √3
     (`pnp.cpp:26-45`; scales at `pnp.cpp:39`). No guard against zero spread.
  4. 2n×12 DLT matrix, two rows per point: `[X̃ᵀ 0ᵀ −x̃X̃ᵀ]`, `[0ᵀ X̃ᵀ −ỹX̃ᵀ]` (`pnp.cpp:51-58`).
  5. Solution = right singular vector of the smallest singular value (JacobiSVD, `pnp.cpp:62-67`);
     `nullspace_ratio = σ[11]/σ[10]` (`pnp.cpp:64`).
  6. Denormalise `M = T2⁻¹ Mn T3` (`pnp.cpp:71`).
  7. Sign fix by cheirality: flip M if fewer than half of the points have positive `m3·[X 1]`
     (`pnp.cpp:76-78`).
  8. Project the left 3×3 block onto SO(3): `R = U diag(1,1,det(UVᵀ)) Vᵀ`; λ = mean singular value
     (`pnp.cpp:83-90`); fails if λ is not finite and positive (`pnp.cpp:91-94`).
  9. Re-solve t by linear least squares for the orthonormalised R (2n×3 system, column-pivoted QR,
     `pnp.cpp:103-114`), rather than using `m4/λ` (kept as `t_from_dlt`, `pnp.cpp:113`).
- **Inputs → outputs:** X (metres, camera i), uv (pixels, frame i+1) → `PnPResult` with
  `X_{i+1} = R X_i + t` (t in metres). Linear algebraic error, not reprojection error, is minimised.
- **Dependencies:** `pnp.hpp`, `<cmath>`, `<Eigen/Dense>`, `<Eigen/SVD>`.
- **Used by:** `src/pipeline/ransac.cpp` (`ransac.cpp:91, 110`), `src/evaluation/experiment.cpp`,
  `apps/milestones/main.cpp`, `apps/milestones/sequence_main.cpp`.
- **Covered by tests:** `test_pipeline_0_1` ("RANSAC improves on the plain linear PnP of the same matches",
  line 53); `test_ransac` (plain DLT on outlier-contaminated data is wrong by > 5°, line 69; oracle DLT on true
  inliers, line 83); `test_refine` (linear start for the LM comparisons); milestone ctests
  `milestone1_pnp_from_scratch`, `milestone2_pnp_full_sequence` (`CMakeLists.txt:95-96`).
- **Before studying it, understand:** DLT for camera resection, Hartley normalisation, orthogonal Procrustes
  (nearest rotation via SVD), cheirality.

### `include/slam/pipeline/frontend.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`; `scratch_pipeline`, `feature_audit`, evaluation libraries, experiments.
- **Purpose and why it exists:** Declares the front end that turns RGB-D frames into features, matches and
  3D→2D correspondences, using only the project's `feature_core` modules and the scratch PNG decoder. It holds
  the frontend parameter struct whose fields T0 and the CLI switch.
- **Responsibility in the architecture:** Front end: frame loading, feature extraction, matching,
  depth lookup, back-projection, depth-edge rejection.
- **Key types and functions:**
  - `struct FrontendParams` (`frontend.hpp:30-48`). Struct defaults (= `PipelineParams{}`):
    - `fast` (`features::FastParams`): threshold forced to **20** in the constructor (`frontend.hpp:47`;
      `fast.hpp:15` default is 30); arc 9, border 16, `max_keypoints` **500** per level (`fast.hpp:16-18`);
      per-cell fallback `cell_size = 0`, `min_threshold = 0` = off (`fast.hpp:23-24`).
    - `pyramid`: 8 levels × 1.2 (`pyramid.hpp:25-26`).
    - `selection`: `harris = false`, `total_features = 0` (`multiscale.hpp:38-43`).
    - `mutual_ratio = false`, `ratio = 0.87` (`frontend.hpp:36-37`).
    - `depth_edge_m = 0` (off) (`frontend.hpp:38`).
    - `match_floor = 30`: keep matches with distance ≤ max(2·d_min, 30) (`frontend.hpp:39-40`).
    - `depth_scale = 5000.0`: depth PNG value = metres × 5000 (TUM convention) (`frontend.hpp:41`).
    - `ranked_matcher = false`, `matcher` (`features::MatcherParams`, `matcher.hpp:63-71`) (`frontend.hpp:45-46`).
  - `struct RgbdFrame { rgb, depth }` (`frontend.hpp:50-53`).
  - `LoadRgbdFrame(dataset_dir, index, frame)` (`frontend.hpp:57`): Bunny naming `<%06d>.png`,
    `<%06d>_depth.png`. `LoadRgbdFiles(rgb_path, depth_path, frame)` (`frontend.hpp:60`): explicit files (TUM).
  - `struct FrameFeatures` (`frontend.hpp:63-75`): gray image, keypoints in ORIGINAL-image coordinates,
    descriptors, `descriptor_valid`, level, scale, level coordinates, per-level counts.
  - `ExtractFeatures(rgb, params)` (`frontend.hpp:77`).
  - `struct Correspondence` (`frontend.hpp:81-98`): query/train indices, Hamming distance, levels, `uv_i`,
    `uv_j` (pixels), raw depths, `has_3d`, `X_i` = back-projected point **in camera i** (metres), plus ranked
    matcher fields.
  - `struct PairMatches` (`frontend.hpp:100-111`) and `MatchFrames(fi, fj, depth_i, depth_j, K, params)`
    (`frontend.hpp:113-115`).
- **Inputs → outputs:** PNG frames → keypoints (pixels), 256-bit descriptors, correspondences with `X_i`
  (metres, camera-i frame) and `uv_j` (pixels, frame j).
- **Dependencies:** `features/fast.hpp`, `features/matcher.hpp`, `features/multiscale.hpp`,
  `features/pyramid.hpp`, `features/types.hpp` (library `feature_core`), `png.hpp`, `projection.hpp`, Eigen.
- **Used by:** `include/slam/pipeline/pipeline.hpp`, `src/pipeline/frontend.cpp`, `apps/feature_audit.cpp`,
  `tests/pipeline/test_frontend.cpp`, `tests/pipeline/test_ransac.cpp`; functions also used by
  `apps/pipeline_main.cpp`, `apps/experiments_main.cpp`, `src/evaluation/baseline.cpp`,
  `experiments/depth_edge/depth_edge_experiment.cpp` and most pipeline tests.
- **Covered by tests:** see `frontend.cpp`.
- **Before studying it, understand:** FAST corners, image pyramids, ORB-style orientation and rotated BRIEF,
  Hamming matching, RGB-D back-projection; the `feature_core` headers in `include/slam/features/`.

### `src/pipeline/frontend.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Implements frame loading with format checks, per-frame feature extraction,
  and pair matching + correspondence construction. This is where pixel depth becomes metric 3D points and where
  T0's depth-edge rejection happens.
- **Responsibility in the architecture:** Front end.
- **Key types and functions:**
  - `LoadRgbdFrame` (`frontend.cpp:10-16`) → `LoadRgbdFiles` (`frontend.cpp:18-40`): RGB must be 8-bit with ≥ 3
    channels, depth 16-bit single channel, equal sizes (`frontend.cpp:23-38`).
  - `ExtractFeatures` (`frontend.cpp:42-70`): `GrayFromRGB` (`47-49`) → `BuildPyramid` + `ExtractMultiscale`
    (`52-54`, in `src/features/multiscale.cpp`: per level FAST → optional Harris ranking of all FAST candidates
    and geometric per-level quotas from `LevelQuotas` when `total_features > 0` → orientation → BRIEF on the
    Gaussian-smoothed level image). Keypoint x, y are replaced by original-image coordinates (`60-62`).
  - `MatchFrames` (`frontend.cpp:72-151`):
    - Default path (`77-83`): `MatchBruteForce` then `FilterMatchesByDistance(raw, match_floor)`; **if
      `mutual_ratio` is true (T0), `filtered` is overwritten by `MatchMutualRatio(desc_i, desc_j, ratio)`**
      (`81-83`): bidirectional nearest-neighbour with ratio test `first < ratio·second` and mutual consistency
      (`src/features/matcher.cpp:55-85`). In that case the distance filter (and `match_floor`) has no effect on
      the kept matches; `d_min`/`d_max` still come from the raw distance filter.
    - Ranked path (`84-101`): `features::MatchRanked` with `floor_distance = match_floor`; takes precedence
      over `mutual_ratio`.
    - For each filtered match (`102-149`): pixels `uv_i`, `uv_j` (`112-113`); depth read at the **truncated**
      integer pixel (`115-118`); `has_3d = depth_raw_i > 0` (`119`).
    - Depth-edge rejection (`120-137`, active iff `depth_edge_m > 0`): reject if the pixel is on the 1-pixel
      image border, if any of the 3×3 source depths is ≤ 0, or if max − min of the 3×3 depths (metres, raw /
      `depth_scale`) ≥ `depth_edge_m`. T0 uses 0.012 m.
    - Back-projection (`138-142`): `Z = depth_raw_i / depth_scale` (metres),
      `X_i = ((u−cx)/fx·Z, (v−cy)/fy·Z, Z)` in camera i. Only frame-i depth is used for 3D; `depth_raw_j` is
      recorded but not used for estimation.
- **Inputs → outputs:** `RgbdFrame`s and `Intrinsics` → `FrameFeatures`, `PairMatches` (pixels, metres).
- **Dependencies:** `frontend.hpp`, `features/matcher.hpp`, `features/multiscale.hpp`, `<cstdio>`.
- **Used by:** `src/pipeline/pipeline.cpp` (`MatchFrames`, `pipeline.cpp:65-66`), `apps/pipeline_main.cpp`
  (`ExtractFeatures` at 307/315, `LoadRgbdFrame`/`LoadRgbdFiles` at 241-242), `apps/feature_audit.cpp`,
  `apps/experiments_main.cpp`, `src/evaluation/baseline.cpp`, `experiments/depth_edge/depth_edge_experiment.cpp`.
- **Covered by tests:** `test_frontend` (with `FrontendParams{}` + cell fallback 30/7: one raw match per
  frame-0 keypoint; filtered = correspondences; > 50 3D→2D; "3D points are the back-projection of
  (u_i, v_i, depth/5000) with K"; > 60 % reproject within 3 px under the true motion; fallback raises grid
  occupancy; determinism); `test_multiscale_0_1` (original coordinates = (level + 0.5)·scale − 0.5; "depth is
  read ... at the truncated ORIGINAL pixel"; back-projection with depth/5000); `test_ranked_frontend` (ranked
  matcher with options off reproduces the default front end and pose exactly); `test_improvements`
  (`MatchMutualRatio`: "Mutual exact matching", "Ambiguous descriptor rejection"). The 12 mm depth-edge
  rejection is not checked by a built test in `tests/pipeline/` (it is studied by
  `experiments/depth_edge/depth_edge_experiment.cpp`); `test_experiment` checks only that T0 sets
  `depth_edge_m == 0.012` (`test_experiment.cpp:82-88`).
- **Before studying it, understand:** as for `frontend.hpp`, plus Lowe's ratio test and mutual nearest
  neighbours.

### `include/slam/pipeline/essential.hpp`
- **Category:** public header
- **Status:** used — `scratch_core` (via `ransac.hpp`).
- **Purpose and why it exists:** Declares the from-scratch two-view geometry: normalised 8-point essential
  matrix, constraint enforcement, the four-fold decomposition, triangulated cheirality, Sampson distance, and
  pixel-space epipolar helpers. In production this yields only a scale-free comparison estimate; it does not
  feed the trajectory.
- **Responsibility in the architecture:** Two-view (2D–2D) geometry.
- **Key types and functions:** convention block (`essential.hpp:7-13`): x1, x2 are NORMALISED coordinates in
  camera 1 = frame i and camera 2 = frame i+1; recovered pose satisfies **`X_2 = R X_1 + t` (T_{i+1<-i})** with
  `x2ᵀ E x1 = 0`, `E = [t]× R`, and `|t| = 1` (`essential.hpp:11-13`). `PoseRt` (`26-29`), `Skew` (`32`),
  `EstimateEssential8Point` (`39`), `EnforceEssentialConstraints` (`43`), `DecomposeEssential` (`47`),
  `TriangulateDepths` (`52`), `CountInFront` (`55`), `RecoveredPose` (`57-62`), `RecoverPoseFromEssential`
  (`66`), `SampsonDistance` (`70`, normalised units squared), `FundamentalFromEssential` (`76`,
  `F = K⁻ᵀ E K⁻¹`), `EpipolarLine` (`81`), `EpipolarDistance` (`84`).
- **Inputs → outputs:** normalised 2D points → E (unit Frobenius norm), (R, unit t); Sampson distance in
  normalised units²; epipolar distances in pixels.
- **Dependencies:** `<array> <vector> <Eigen/Core>`, `projection.hpp`.
- **Used by:** `include/slam/pipeline/ransac.hpp`, `src/pipeline/essential.cpp`,
  `tests/pipeline/test_essential.cpp`.
- **Covered by tests:** see `essential.cpp`.
- **Before studying it, understand:** epipolar geometry, the essential matrix and its SVD structure,
  Hartley's normalised 8-point algorithm, the four-solution ambiguity.

### `src/pipeline/essential.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Implements the two-view functions. Called by `RansacEssential`.
- **Responsibility in the architecture:** Two-view geometry.
- **Key types and functions:**
  - `Skew` (`essential.cpp:10-14`).
  - `HartleyNormalization` (`essential.cpp:19-32`): zero mean, mean distance √2 (scale 1 if zero spread).
  - `EnforceEssentialConstraints` (`essential.cpp:36-41`): SVD, singular values → (s, s, 0), s = (σ1+σ2)/2.
  - `EstimateEssential8Point` (`essential.cpp:43-67`): n ≥ 8; one row per correspondence of `A e = 0` with
    `[x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]` (`54`); e = last right singular vector (`56-57`); denormalise
    `T2ᵀ En T1` then enforce (`62`); scale to unit Frobenius norm (`63-65`). No explicit rank/degeneracy check of
    A (unlike `geometry/fundamental.hpp:46`).
  - `DecomposeEssential` (`essential.cpp:69-86`): makes U, V proper (det +1) by flipping the third column
    (`74-75`); `W = [[0,−1,0],[1,0,0],[0,0,1]]` (`77`); candidates `(UWVᵀ, ±u3)`, `(UWᵀVᵀ, ±u3)` (`78-85`).
  - `TriangulateDepths` (`essential.cpp:88-100`): least squares for depths (d1, d2) in
    `d2·x2 − d1·R x1 = t`; returns false if rays are nearly parallel (`95`).
  - `CountInFront` (`102-109`) and `RecoverPoseFromEssential` (`111-124`): keep the candidate with the most points
    with d1 > 0 and d2 > 0 (strictly greater, so ties keep the earlier candidate, `116`); fail if none.
  - `SampsonDistance` (`126-132`): `(x2ᵀEx1)² / ((Ex1)₁² + (Ex1)₂² + (Eᵀx2)₁² + (Eᵀx2)₂²)`.
  - `FundamentalFromEssential` (`134-138`), `EpipolarLine` (`140-144`, normalised so a² + b² = 1),
    `EpipolarDistance` (`146-148`).
- **Inputs → outputs:** as in the header.
- **Dependencies:** `essential.hpp`, `<cmath>`, `<Eigen/Dense>`, `<Eigen/SVD>`.
- **Used by:** `src/pipeline/ransac.cpp` (`EstimateEssential8Point`, `SampsonDistance`,
  `RecoverPoseFromEssential`). `FundamentalFromEssential`, `EpipolarLine`, `EpipolarDistance` are not used by any target and not
  tested (pixel-space epipolar helpers kept for diagnostics).
- **Covered by tests:** `test_essential` (synthetic scenes, known `X_2 = R X_1 + t`): E equals [t]×R up to
  scale/sign; singular values (s, s, 0); epipolar residual < 1e-12; 4 candidates proper with |t| = 1; true pose
  is one candidate; cheirality picks it with 60/60 in front; 7 points rejected; noisy recovery < 1° / < 10°;
  Sampson distance noise-level and larger across the epipolar line. `test_ransac` (RansacEssential).
- **Before studying it, understand:** as for the header; Sampson (first-order geometric) error.

### `include/slam/pipeline/ransac.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Declares the deterministic RANSAC wrappers around the linear PnP and the
  8-point essential matrix, and the parameter struct whose iteration/threshold T0 changes.
- **Responsibility in the architecture:** Robust estimation (outlier rejection) for both stages.
- **Key types and functions:**
  - Algorithm summary (`ransac.hpp:6-17`); randomness: `std::mt19937` with partial Fisher–Yates using
    `rng() % remaining`, avoiding `std::uniform_int_distribution` for cross-platform reproducibility.
  - `struct RansacParams` (`ransac.hpp:31-35`): **iterations = 300, threshold_px = 8.0, seed = 12345**.
    (T0 sets PnP to 650 / 3.0, `pipeline.cpp:46-47`; the essential stage uses 2000 / 2.0 / seed 54321,
    `pipeline.hpp:48-51`.)
  - `SampleIndices` (`ransac.hpp:38`).
  - `struct RansacPnPResult` (`ransac.hpp:40-51`): pose **`X_{i+1} = R X_i + t` (T_{i+1<-i})** (`ransac.hpp:43`);
    `inlier_mask` and `residual_px` under the refit pose (inf behind the camera), `num_inliers`,
    `best_sample_inliers`, `models_tried`, `refit_used`.
  - `RansacPnP(X, uv, K, params)` (`ransac.hpp:54-55`).
  - `struct RansacEssentialResult` (`ransac.hpp:57-71`) and `RansacEssential(x1, x2, focal_px, params)`
    (`ransac.hpp:75-76`); the pixel threshold is applied to `sqrt(Sampson)·focal_px`.
- **Inputs → outputs:** X (metres, camera i), uv (pixels) → R, t (metres) and masks; or normalised points →
  E and a unit-translation pose.
- **Dependencies:** `<cstdint> <random> <string> <vector> <Eigen/Core>`, `essential.hpp`, `projection.hpp`.
- **Used by:** `include/slam/pipeline/pipeline.hpp`, `src/pipeline/ransac.cpp`, `src/evaluation/experiment.cpp`,
  `tests/pipeline/test_ransac.cpp`, `test_validity.cpp`, `test_experiment.cpp`.
- **Covered by tests:** see `ransac.cpp`.
- **Before studying it, understand:** RANSAC (minimal samples, consensus scoring), Mersenne Twister seeding.

### `src/pipeline/ransac.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Implements both RANSAC loops. `RansacPnP` produces the linear metric pose that
  the refinement starts from.
- **Responsibility in the architecture:** Robust estimation.
- **Key types and functions:**
  - `SampleIndices` (`ransac.cpp:11-20`): partial Fisher–Yates, `j = i + rng() % (n − i)`.
  - `Score::better_than` (`ransac.cpp:27-31`): more inliers wins; tie → smaller sum of inlier residuals.
  - `Evaluate` (`ransac.cpp:33-45`): inlier iff residual **strictly <** threshold.
  - `ReprojectionResiduals` (`ransac.cpp:47-56`): pixel distance under `projectPoint`; inf if `Zc ≤ 0`.
  - `RansacPnP` (`ransac.cpp:75-118`): minimal sample 6 (`77`); `std::mt19937 rng(params.seed)` (`84`); fixed
    number of iterations, **no adaptive early termination** (`89-102`); a sample counts only if
    `solvePnPDLT` succeeds; fail if the best model has < 6 inliers (`103-106`); refit DLT on all inliers of the
    best model (`109-110`); the refit is used **whenever it succeeds, without an inlier-count guard**
    (`111-113`); final residuals and mask under the refit pose (`114-115`).
  - `RansacEssential` (`ransac.cpp:120-174`): minimal sample 8 (`122`); residual = √Sampson × focal (`131`);
    same loop/scoring; refit on inliers kept only if it has ≥ the best sample's inlier count (guard,
    `158-164`); final mask (`166`); pose recovered by cheirality on the final inliers (`168`).
- **Inputs → outputs:** as in the header; residuals in pixels.
- **Dependencies:** `ransac.hpp`, `pnp.hpp`, `<cmath> <limits> <numeric>`.
- **Used by:** `src/pipeline/pipeline.cpp` (`pipeline.cpp:89-90, 110`), `src/evaluation/experiment.cpp`
  (`RansacPnP`, `SampleIndices`), `apps/experiments_main.cpp`, `experiments/depth_edge/depth_edge_experiment.cpp`.
- **Covered by tests:** `test_ransac`: `SampleIndices` distinct and reproducible (line 60); plain DLT fails on
  outliers but RANSAC (300 it, 8 px, seed 12345) recovers < 1° / < 1 cm (69-81); close to an oracle fit (83);
  inlier mask quality and `refit_used` (91-94); `models_tried > 250` (96); "deterministic: same seed →
  identical result" (99); another seed also works (103); 0 iterations and 5 points report failure (107, 110);
  RansacEssential recovers the pose, refit guard holds (139-145), and on the real Bunny pair a refit that loses
  inliers is rejected (171-174). `test_validity` section 6 uses `RansacPnP` on 400 synthetic points.
- **Before studying it, understand:** RANSAC; the DLT PnP; Sampson distance.

### `include/slam/pipeline/refine.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Declares the from-scratch nonlinear PnP refinement (Levenberg–Marquardt on
  SE(3)) applied to the RANSAC pose on a fixed inlier set, with an optional Huber loss (T0: δ = 2 px).
- **Responsibility in the architecture:** Nonlinear pose refinement (the only nonlinear optimisation in the
  production pipeline; it optimises one relative pose, not structure — not bundle adjustment).
- **Key types and functions:**
  - Header derivation (`refine.hpp:3-15`): cost `½ Σ‖π(K(R X_k + t)) − uv_k‖²` in pixels; left perturbation
    `R ← Exp(φ) R`, `t ← Exp(φ) t + ρ`, ξ = (ρ, φ); Jacobian `∂π/∂Xc · [I | −[Xc]×]`; accept → λ/3,
    reject → λ×4; "the inlier set is not changed".
  - `struct RefineParams` (`refine.hpp:25-34`): **`huber_delta_px = 0`** (least squares; T0 sets 2.0 at
    `pipeline.cpp:44`), **`max_iterations = 50`**, **`lambda0 = 1e-3`** (relative to mean diag of JᵀJ),
    **`min_step = 1e-10`**, **`min_rel_decrease = 1e-12`**.
  - `struct RefineResult` (`refine.hpp:36-46`): `ok`, R, t, accepted iterations, initial/final cost, initial/
    final RMS (pixels).
  - `RefinePnP(X, uv, use, K, R0, t0, params)` (`refine.hpp:49-53`): X in camera i, uv in frame i+1; the pose
    keeps the convention **`X_{i+1} = R X_i + t`** (it is applied as `Xc = R X + t`, `refine.cpp:69`).
  - `ExpSO3(φ)` (`refine.hpp:56`): Rodrigues.
- **Inputs → outputs:** metres, pixels → refined R, t (metres); costs in pixels².
- **Dependencies:** `<vector> <Eigen/Core>`, `projection.hpp`.
- **Used by:** `include/slam/pipeline/pipeline.hpp`, `src/pipeline/refine.cpp`,
  `experiments/depth_edge/depth_edge_experiment.cpp`, `tests/pipeline/test_refine.cpp`,
  `tests/pipeline/test_improvements.cpp`.
- **Covered by tests:** see `refine.cpp`.
- **Before studying it, understand:** Gauss–Newton/LM, Lie-group perturbations on SO(3)/SE(3), the projection
  Jacobian, Huber loss and IRLS weights.

### `src/pipeline/refine.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Implements the LM refinement.
- **Responsibility in the architecture:** Nonlinear pose refinement.
- **Key types and functions:**
  - `ExpSO3` (`refine.cpp:9-17`): `I + sin a/a·W + (1−cos a)/a²·W²`; first-order `I + W` for a < 1e-12.
  - `Cost` (`refine.cpp:23-38`): per point `½e²` or, with Huber δ > 0 and e > δ, `δ(e − δ/2)` (`34-35`); `½e²`
    is computed as `.5 * norm * norm` from the residual's norm rather than from `squaredNorm()`. This is numerically
    equivalent but not byte-equivalent to the former implementation: on one TUM stride-4 pair it changes
    `refine_iterations` from 5 to 4 and a residual by 1e-4 px (the known stride-4 rerun difference,
    `docs/DATA_AND_ARCHIVE.md`);
    returns false if any selected point has `Zc ≤ 0`.
  - `RefinePnP` (`refine.cpp:42-130`):
    - Select points with `use[k]`; return `ok = false` if fewer than 3 or a point is behind the camera at the
      start (`52-60`).
    - Normal equations per iteration (`66-86`): residual e (pixels), `Jp` (2×3 projection Jacobian, `73-75`),
      `Jx = [I | −[Xc]×]` (`76-78`), IRLS Huber weight `δ/‖e‖` when ‖e‖ > δ (`80-83`), `H += w JᵀJ`,
      `g += w Jᵀe`.
    - λ initialised once to `lambda0 · mean(diag H)` (`87-88`).
    - Up to 10 damping tries per iteration (`90-113`): solve `(H + λI) ξ = −g` (LDLT), update
      `R ← Exp(φ)R`, `t ← Exp(φ)t + ρ` (`96-98`); accept if the (Huber) cost decreases → λ = max(λ/3, 1e-12)
      (`105`); else λ ×= 4 (`111`). Stop on small step or small relative decrease (`108-109`) or when no try
      decreases the cost (`114-115`).
    - Re-orthonormalise R by SVD (`118-120`); report final cost and RMS of plain (non-Huber) pixel residuals
      before and after (`122-127`).
- **Inputs → outputs:** as in the header.
- **Dependencies:** `refine.hpp`, `<cmath>`, `<Eigen/Dense>`.
- **Used by:** `src/pipeline/pipeline.cpp` (`pipeline.cpp:130-131, 150-151`),
  `experiments/depth_edge/depth_edge_experiment.cpp`.
- **Covered by tests:** `test_refine`: "ExpSO3 equals the axis-angle rotation" (63); one LM step check (77);
  exact convergence on noise-free data (87); proper rotation (91); error decreases in every noisy trial and
  beats the linear PnP (108-109); only selected points are used (127); fewer than 3 points or a point behind the
  camera → not refined (137); determinism (147). `test_improvements`: "Huber resistance to contaminated inliers"
  (Huber δ = 2 vs least squares with 10 corrupted points, lines 72-81). `test_pipeline_0_1` (refinement lowers
  inlier reprojection error on Bunny 0→1, line 64).
- **Before studying it, understand:** as for the header.

### `include/slam/pipeline/validity.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`. The gate is always **evaluated** by `ProcessPair`, but it only **decides**
  acceptance when `ValidityParams::enabled` is true, which `PipelineParams` turns off by default
  (`pipeline.hpp:47`); `--pose-gate` enables it (`apps/pipeline_main.cpp:168-169`).
- **Purpose and why it exists:** The Stage-2 pose-validity gate ported from main: decides from correspondences
  and the estimated pose only (no ground truth) whether a RANSAC PnP pose may enter the trajectory, so that a
  wrong pose supported by a few repeated or clustered correspondences is held instead of accumulated.
- **Responsibility in the architecture:** Post-estimation acceptance test between `ProcessPair` and trajectory
  accumulation.
- **Key types and functions:**
  - Criteria list (`validity.hpp:12-22`) and states solver_failed / rejected / accepted (`validity.hpp:7-10`).
  - `struct ValidityParams` (`validity.hpp:34-46`), defaults: `enabled = true` (overridden to false by
    `PipelineParams`), `min_inlier_rows = 20`, `min_distinct_keypoints = 20`, `min_distinct_locations = 20`,
    `location_radius_px = 2.0`, `min_inlier_ratio = 0.0` (recorded only), `coverage_cell_px = 40`,
    `min_coverage_cells = 7`, `max_reproj_median_factor = 0.5`, `max_reproj_p90_factor = 1.0`,
    `rotation_tolerance = 1e-6`.
  - `struct ValidityRow` (`validity.hpp:49-54`), `enum PoseStatus` + `PoseStatusName` (`56-57`),
    `struct PoseValidity` (`59-71`), `EvaluatePose(...)` (`74-76`), helpers `CountDistinctIds`,
    `CountLocationGroups`, `CountOccupiedCells`, `Percentile` (`79-82`).
- **Inputs → outputs:** R, t (any rigid pose; in the pipeline `T_{i+1<-i}`), rows with pixels and residuals
  (pixels), the RANSAC threshold (pixels) → a status and the evidence.
- **Dependencies:** `<cmath> <string> <vector> <Eigen/Core>`.
- **Used by:** `include/slam/pipeline/pipeline.hpp`, `src/pipeline/validity.cpp`,
  `tests/pipeline/test_validity.cpp`; helpers also used by `apps/experiments_main.cpp`,
  `src/evaluation/experiment.cpp`, `src/evaluation/baseline.cpp`; `PoseStatusName` by `apps/pipeline_main.cpp`,
  `apps/baseline_main.cpp`.
- **Covered by tests:** see `validity.cpp`.
- **Before studying it, understand:** why RANSAC can return a confident wrong pose (degenerate or duplicated
  support), percentiles, single-linkage clustering.

### `src/pipeline/validity.cpp`
- **Category:** production source
- **Status:** used — `scratch_core` (evaluated every pair; decisive only with `--pose-gate`).
- **Purpose and why it exists:** Implements the gate's evidence and decision.
- **Responsibility in the architecture:** Pose acceptance.
- **Key types and functions:**
  - `CountDistinctIds` (`validity.cpp:23-26`): sort + unique.
  - `CountLocationGroups` (`validity.cpp:28-49`): single-linkage union–find over points within `radius`, with
    an x-sorted sweep that stops when the x gap exceeds the radius.
  - `CountOccupiedCells` (`validity.cpp:51-56`): distinct `floor(x/cell), floor(y/cell)` pairs.
  - `Percentile` (`validity.cpp:58-67`): linear interpolation between order statistics; NaN for empty input;
    handles inf without producing NaN.
  - `EvaluatePose` (`validity.cpp:69-137`): solver failed → `kSolverFailed` with the solver reason (`74-78`);
    finiteness and SO(3) tests `‖RᵀR − I‖_F ≤ tol` and `|det R − 1| ≤ tol` (`80-85`); evidence over inlier
    rows only (`89-102`): row count, distinct src/dst keypoint ids, distinct src/dst locations, inlier ratio,
    occupied cells in each frame, median and 90th percentile of inlier residuals; decision (`112-132`): every
    failed criterion appended to `reason`; reprojection limits are `0.5 × threshold` (median) and
    `1.0 × threshold` (p90) of the PnP RANSAC threshold (`129-132`); status accepted iff no failures (`135`).
- **Inputs → outputs:** as in the header.
- **Dependencies:** `validity.hpp`, `<algorithm> <cmath> <cstdio> <limits> <numeric> <set> <utility>`,
  `<Eigen/Dense>`.
- **Used by:** `src/pipeline/pipeline.cpp` (`pipeline.cpp:114-116, 188-191`); helpers as listed for the header.
- **Covered by tests:** `test_validity` sections 1–8: NaN/inf/non-orthogonal/reflection rejected; solver
  failure reported; duplicate rows are not independent support; insufficient distinct src or dst keypoints;
  20 accepted / 19 rejected boundary; poor coverage; reprojection median/p90 limits; `Percentile` edge cases;
  inlier ratio recorded but not enforced by default; a valid synthetic RANSAC pose accepted and a 12-true-of-400
  pose not; determinism and row-order invariance; and the gate inside `ProcessPair` on Bunny 0→1 with
  `ApplyStage2Frontend` + gate on (accepted, 217 inlier rows, gate does not change the pose) (lines 280-301).
  `test_experiment` uses `EvaluatePose` through the evaluation library.
- **Before studying it, understand:** as for the header.

### `include/slam/pipeline/pipeline.hpp`
- **Category:** public header
- **Status:** used — `scratch_core`; `scratch_pipeline`, `scratch_baseline_lib`, `scratch_experiment_lib`,
  `depth_edge_experiment`, tests.
- **Purpose and why it exists:** Declares the per-pair entry point `ProcessPair`, the full parameter set
  `PipelineParams`, the per-pair result `PairResult`, and the two named configurations `ApplyImprovedDefaults`
  (T0) and `ApplyStage2Frontend` (M0). It is the library's top-level API.
- **Responsibility in the architecture:** Orchestration of one frame pair.
- **Key types and functions:**
  - Pipeline overview (`pipeline.hpp:3-20`): both estimates use **`X_{i+1} = R X_i + t`**; only the metric PnP
    pose is accumulated; the essential estimate is scale-free and reported for comparison (`pipeline.hpp:18-20`).
  - `struct PipelineParams` (`pipeline.hpp:34-53`), struct defaults: `frontend` (see `FrontendParams`),
    `pnp_ransac` (300 it, 8 px, seed 12345), **`refine = true`** (`38`), `refine_params` (Huber 0),
    **`reselect_inliers = false`** (`40`), `essential_ransac` = **2000 it, 2.0 px, seed 54321** (`48-51`),
    `validity` with **`enabled = false`** (`47`).
  - `struct PairResult` (`pipeline.hpp:55-103`): counts; correspondences; ranked-matcher details; pixel
    fundamental matrix `F` and `fundamental_ok`; essential results (`R_essential`, unit `t_essential_dir`,
    mask); PnP results: `pnp_index`, `pnp_ok`, `pnp_reason`, **`R, t` = `T_{i+1<-i}`** (`pipeline.hpp:79`),
    `pnp_mask` (final, after re-selection), `ransac_mask`, `pnp_residual_px`, reprojection statistics, linear
    pose `R_linear/t_linear`, refinement statistics, `validity`, and **`pose_accepted = pnp_ok && (gate disabled
    || validity accepted)`** (`pipeline.hpp:102`).
  - `ApplyImprovedDefaults` (`pipeline.hpp:110`), `ApplyStage2Frontend` (`pipeline.hpp:114`),
    `ProcessPair(i, j, fi, fj, frame_i, frame_j, K, params)` (`pipeline.hpp:116-119`).
- **Inputs → outputs:** two frames' features and depth, intrinsics → `PairResult` (pose in metres, residuals in
  pixels).
- **Dependencies:** `<string> <vector> <Eigen/Core>`, `frontend.hpp`, `ransac.hpp`, `refine.hpp`, `validity.hpp`.
- **Used by:** `apps/pipeline_main.cpp`, `experiments/depth_edge/depth_edge_experiment.cpp`,
  `include/slam/evaluation/baseline.hpp`, `include/slam/evaluation/experiment.hpp`, `src/pipeline/pipeline.cpp`,
  `tests/pipeline/test_pipeline_0_1.cpp`, `test_multiscale_0_1.cpp`, `test_ranked_frontend.cpp`,
  `test_validity.cpp`; `ProcessPair` also via `src/evaluation/baseline.cpp`,
  `src/evaluation/experiment.cpp`, `apps/baseline_main.cpp`, `apps/experiments_main.cpp`.
- **Covered by tests:** see `pipeline.cpp`.
- **Before studying it, understand:** all of the files above.

### `src/pipeline/pipeline.cpp`
- **Category:** production source
- **Status:** used — `scratch_core`.
- **Purpose and why it exists:** Implements the named configurations and `ProcessPair`, the exact sequence of
  stages for one pair (see the flow section below).
- **Responsibility in the architecture:** Per-pair orchestration.
- **Key types and functions:**
  - `Median`, `InlierStats` (`pipeline.cpp:12-35`): statistics helpers (local).
  - `ApplyImprovedDefaults` (`pipeline.cpp:39-48`) — **T0**: `selection = {harris = true, total_features =
    1400}`, FAST threshold 10, `mutual_ratio = true` (ratio stays 0.87), `depth_edge_m = 0.012`, Huber δ = 2 px,
    `reselect_inliers = true`, PnP RANSAC 650 iterations / 3 px. It modifies only these fields of whatever
    `params` it is given (seed 12345, pyramid 8×1.2, refine on, essential 2000/2/54321, gate off stay).
  - `ApplyStage2Frontend` (`pipeline.cpp:50-54`) — **M0**: resets to `PipelineParams{}` and sets the FAST
    per-cell fallback `cell_size = 30`, `min_threshold = 7`.
  - `ProcessPair` (`pipeline.cpp:56-193`).
- **Inputs → outputs:** as in the header.
- **Dependencies:** `pipeline.hpp`, `geometry/fundamental.hpp` (header-only Eigen; `geometry_scratch::
  EstimateFundamental8Point`), `<algorithm> <cmath> <limits>`.
- **Used by:** `apps/pipeline_main.cpp:317-318` (production), `src/evaluation/baseline.cpp`,
  `src/evaluation/experiment.cpp` (`ApplyIntegrationConfig`, `experiment.cpp:96-145`),
  `experiments/depth_edge/depth_edge_experiment.cpp`, tests.
- **Covered by tests:** `test_pipeline_0_1` (with `PipelineParams{}`, NOT T0): RANSAC PnP solves Bunny 0→1 with
  > 50 inputs / ≥ 40 inliers, pose error < 3° and < 3 cm, inlier mean < 3 px, R in SO(3), mask count matches,
  essential mask size, refinement lowers the error, `refine = false` keeps the linear pose, determinism.
  `test_multiscale_0_1` (`PipelineParams{}`, 8 levels). `test_ranked_frontend` (`PipelineParams{}` with and
  without `ranked_matcher`). `test_validity` (with `ApplyStage2Frontend` + gate). `test_experiment` checks that
  `ApplyIntegrationConfig("T0")` equals `ApplyImprovedDefaults` field by field (`test_experiment.cpp:81-88`)
  and M0 (`89-93`). End-to-end T0 is only run by the ctest `milestone3_scratch_pipeline`
  (`CMakeLists.txt:98-99`), whose exit status is non-zero only if every pair failed
  (`apps/pipeline_main.cpp:664`); no built test checks T0 pose accuracy on a pair.
- **Before studying it, understand:** all the stage files above.

### `include/slam/pipeline/accumulate.hpp`
- **Category:** public header (header-only; not a `scratch_core` source)
- **Status:** used — `scratch_pipeline`, `scratch_experiments`, `test_validity`.
- **Purpose and why it exists:** Chains accepted relative poses into an absolute camera trajectory and defines
  what happens to a non-accepted pair (previous pose held, its pose values never read). It was factored out so
  that the production application and the experiment harness accumulate identically and so that the
  "rejected pair does not move the trajectory" rule can be unit-tested.
- **Responsibility in the architecture:** Trajectory composition (the only "back end"; no optimisation).
- **Key types and functions:**
  - Rule (`accumulate.hpp:3-6`): **`T_wc[0] = anchor`, `T_wc[k+1] = T_wc[k] · T_{k+1<-k}⁻¹`** for an accepted
    pair, otherwise `T_wc[k+1] = T_wc[k]`.
  - `Inverse(T)` (`accumulate.hpp:15-20`): `(Rᵀ, −Rᵀt)`. `Compose(A, B)` (`accumulate.hpp:21-26`):
    `(A.R·B.R, A.R·B.t + A.t)`, i.e. `(A∘B)x = A(Bx)`.
  - `struct RelativeStep { accepted, held_source, T }` (`accumulate.hpp:28-32`), T = **`T_{k+1<-k}`**
    (`accumulate.hpp:31`).
  - `struct AccumulatedTrajectory { Twc, source }` (`accumulate.hpp:34-37`), `Twc` = camera→world.
  - `AccumulateTrajectory(anchor, steps, estimated_source)` (`accumulate.hpp:39-55`): source label
    `"gt_anchor"` for frame 0 (`45`); accepted: `Compose(Twc[k], Inverse(T))` (`48`); held: copy and label
    with `held_source` (`50-51`).
- **Inputs → outputs:** anchor `T_wc` (metres) + relative steps → absolute `T_wc` list (metres).
- **Dependencies:** `<string> <vector>`, `common.hpp`.
- **Used by:** `apps/pipeline_main.cpp:328-335` (anchor = ground-truth `Twc_gt[0]`; held_source
  `"held_previous_pose_pair_rejected"` if `pnp_ok` else `"held_previous_pose_pair_failed"`, line 331),
  `apps/experiments_main.cpp:292`, `tests/pipeline/test_validity.cpp`. (`apps/milestones/sequence_main.cpp:29-35`
  and `src/evaluation/baseline.cpp:34-40` define their own `Inverse`/`Compose` with the same formulas.)
- **Covered by tests:** `test_validity` section 7: a rejected pair holds the previous pose; the next accepted
  pair continues from it; a rejected NaN pose is never read; source labels; "accepted step:
  T_wc[1]⁻¹ T_wc[0] = T_{1<-0}" (lines 224-249).
- **Before studying it, understand:** composition and inversion of rigid transforms, `T_wc` vs `T_{j<-i}`.

---

### How one frame pair flows through scratch_core

Driver (production application `apps/pipeline_main.cpp`): parameters start as `PipelineParams{}` and
`ApplyImprovedDefaults` is applied immediately (`pipeline_main.cpp:98-101`), so **T0 is the default**; flags are
then applied in command-line order (`pipeline_main.cpp:107-182`). Frames are loaded with `LoadRgbdFrame` (Bunny)
or `LoadRgbdFiles` (TUM) (`pipeline_main.cpp:240-243`); features of each frame are extracted once with
`ExtractFeatures` (`pipeline_main.cpp:307, 315`) and reused for the next pair; `ProcessPair(i, i+1, ...)` is
called per consecutive pair (`pipeline_main.cpp:317-318`).

Inside `ProcessPair` (`src/pipeline/pipeline.cpp:56-193`):

0. (Before the call, per frame) `ExtractFeatures` (`frontend.cpp:42-70`): grayscale → pyramid (8 × 1.2) → per
   level FAST → [T0: Harris ranking of all FAST candidates + 1400-feature geometric level budget,
   `pipeline.cpp:40`; `PipelineParams{}`/M0: strongest 500 by FAST score per level] → orientation → rotated
   BRIEF. FAST threshold: T0 10 (`pipeline.cpp:41`); `PipelineParams{}`/M0 20 (`frontend.hpp:47`). FAST per-cell
   fallback: **off** in T0 (`fast.hpp:23-24`), on (30 px / 7) with `--stage2-frontend`
   (`pipeline.cpp:52-53`) or `--fast-cell/--fast-min-threshold` (`pipeline_main.cpp:134-137`).
1. **Matching and correspondences** — `MatchFrames` (`pipeline.cpp:65-66` → `frontend.cpp:72-151`).
   - T0: `MatchMutualRatio` with ratio 0.87 replaces the distance-filtered matches (`frontend.cpp:81-83`,
     `pipeline.cpp:42`).
   - `--stage2-frontend` / `--no-mutual-ratio`: brute force + `max(2·d_min, 30)` distance filter
     (`frontend.cpp:78-80`).
   - Option `--ranked-matcher` and the `--match-*` flags (`pipeline_main.cpp:152-165`): `features::MatchRanked`
     (`frontend.cpp:84-101`), overrides `mutual_ratio`.
   - Depth at truncated pixel, back-projection `X_i` in camera i with depth/5000 (`frontend.cpp:115-142`).
   - T0: 12 mm 3×3 depth-edge rejection (`frontend.cpp:120-137`, `pipeline.cpp:43`); off in
     `PipelineParams{}`/M0.
2. **Two-view stage (diagnostic only)** — normalised coordinates of all filtered matches
   (`pipeline.cpp:77-81`); pixel fundamental matrix by `geometry_scratch::EstimateFundamental8Point`
   (`pipeline.cpp:82-88`, stored in `PairResult::F`, not used further by any built target);
   `RansacEssential(x1, x2, K.fx, essential_ransac)` with 2000 iterations / 2 px Sampson / seed 54321
   (`pipeline.cpp:89-99`, `pipeline.hpp:48-51`) → `R_essential`, unit `t_essential_dir`
   (`X_{i+1} = R X_i + t`, |t| = 1). Not used for the accepted pose. Runs in every configuration.
3. **PnP inputs** — every correspondence with `has_3d`: `X = X_i`, `uv = uv_j`, index in `pnp_index`
   (`pipeline.cpp:102-109`).
4. **RANSAC PnP** — `RansacPnP(X, uv, K, pnp_ransac)` (`pipeline.cpp:110` → `ransac.cpp:75-118`): 6-point DLT
   samples, seed 12345, then DLT refit on all inliers. T0: **650 iterations, 3 px** (`pipeline.cpp:46-47`);
   `--stage2-frontend`/`PipelineParams{}`: 300 / 8 px (`ransac.hpp:32-33`); the experimental T0-ransac is
   `--pnp-iterations 300 --pnp-threshold 8` (`pipeline_main.cpp:124-127`). On failure: gate records
   `kSolverFailed`, return (`pipeline.cpp:113-118`). Result stored as linear pose `R_linear, t_linear`
   (`pipeline.cpp:119-128`).
5. **LM refinement** — if `refine` (default true; `--no-refine` turns off, `pipeline_main.cpp:130-131`):
   `RefinePnP` on the RANSAC inlier mask starting from the linear pose (`pipeline.cpp:129-146` →
   `refine.cpp:42-130`); T0 Huber δ = 2 px (`pipeline.cpp:44`), `PipelineParams{}`/M0 plain least squares
   (δ = 0, `refine.hpp:26-27`). If it succeeds, residuals of all PnP inputs are recomputed under the refined
   pose (`pipeline.cpp:139-144`).
6. **Inlier re-selection (T0 only)** — if `reselect_inliers && refined` (`pipeline.cpp:147-169`; T0 sets it at
   `pipeline.cpp:45`; off with `--no-reselect` or `--stage2-frontend`): mask := residual < PnP threshold (3 px)
   under the refined pose (`149`); a second `RefinePnP` on the new mask (`150-156`); recompute residuals,
   mask and `pnp_inliers` (`157-164`); if fewer than 6 inliers remain, `pnp_ok = false` with reason
   "Insufficient refined support" (`165-168`).
7. **Statistics** — inlier and all-point reprojection statistics (`pipeline.cpp:170-176`).
8. **Pose-validity gate** — rows from the final mask/residuals (`pipeline.cpp:180-187`);
   `EvaluatePose(pnp_ok, ..., pnp_ransac.threshold_px, validity)` (`pipeline.cpp:188-190`) is **always
   evaluated**; `pose_accepted = pnp_ok && (!validity.enabled || validity.accepted())` (`pipeline.cpp:191`).
   The gate is **off** in T0 and in `--stage2-frontend` (`pipeline.hpp:47`; `ApplyStage2Frontend` resets to
   `PipelineParams{}`); `--pose-gate` enables it (`pipeline_main.cpp:168-169`).
9. **Result** — `PairResult::R, t` = refined (and re-selected) `T_{i+1<-i}` with `X_{i+1} = R X_i + t`
   (`pipeline.hpp:79`).

After all pairs, the application builds `RelativeStep`s from `pose_accepted`, `R`, `t`
(`pipeline_main.cpp:328-334`) and calls `AccumulateTrajectory(Twc_gt[0], steps)` (`pipeline_main.cpp:335`):
`T_wc[k+1] = T_wc[k] · T_{k+1<-k}⁻¹` for accepted pairs, previous pose held otherwise.

Summary of what is default (T0) vs option:

| Step | T0 (default) | Option |
|---|---|---|
| Feature selection | Harris ranking, 1400 total, FAST thr 10 (`pipeline.cpp:40-41`) | `--stage2-frontend`: FAST score, 500/level, thr 20 + cell fallback 30/7; `--no-harris`, `--feature-budget`, `--fast-threshold` |
| FAST per-cell fallback | off (`fast.hpp:23-24`) | `--fast-cell C --fast-min-threshold T`, or via `--stage2-frontend` |
| Matching | `MatchMutualRatio` 0.87 (`pipeline.cpp:42`, `frontend.hpp:37`) | `--no-mutual-ratio`/`--stage2-frontend`: distance filter max(2·d_min, 30); `--ranked-matcher`/`--match-*`: `MatchRanked` |
| Depth-edge rejection | 12 mm (`pipeline.cpp:43`) | `--depth-edge-m M` (0 = off); off in `--stage2-frontend` |
| Essential RANSAC | always, 2000 / 2 px / seed 54321 (diagnostic) | — |
| PnP RANSAC | 650 it / 3 px / seed 12345 (`pipeline.cpp:46-47`) | `--pnp-iterations`, `--pnp-threshold` (T0-ransac 300/8), `--seed`; 300/8 in `--stage2-frontend` |
| LM refinement | on, Huber 2 px (`pipeline.cpp:44`) | `--no-refine`; `--huber-delta PX` (0 = LS); LS in `--stage2-frontend` |
| Inlier re-selection | on (`pipeline.cpp:45`) | `--no-reselect`; off in `--stage2-frontend` |
| Pose-validity gate | evaluated, not enforced (`pipeline.hpp:47`) | `--pose-gate` (+ `--gate-min-*`) |

Note on flag order: `--stage2-frontend` and `--legacy` reset all parameters (`pipeline.cpp:51`,
`pipeline_main.cpp:112-113`), and flags are processed left to right, so any flag given before them (including
`--pose-gate`) is discarded.

## Applications, evaluation harness, tests and fixtures

This part covers the executables in `apps/`, the C++ evaluation harness (`include/slam/evaluation/`,
`src/evaluation/`), the pipeline tests (`tests/pipeline/`), the frozen fixtures (`tests/fixtures/`), the
depth-edge experiment and the root `CMakeLists.txt`. All paths are relative to the repository root.

Two facts are needed to read every entry below:

- **Two configurations share one parameter struct.** `PipelineParams{}` (`include/slam/pipeline/pipeline.hpp:34-53`)
  is the *legacy* configuration: FAST threshold 20, 500 strongest keypoints per level, original
  brute-force matcher with the `max(2 d_min, 30)` filter, RANSAC 300 iterations / 8 px / seed 12345,
  least-squares LM, no inlier re-selection, gate off. The production default **T0** is obtained by
  `ApplyImprovedDefaults` (`src/pipeline/pipeline.cpp:39-48`): Harris ranking with a 1400-feature total budget,
  FAST threshold 10, `mutual_ratio = true` (ratio 0.87, `include/slam/pipeline/frontend.hpp:37`),
  `depth_edge_m = 0.012`, Huber delta 2 px, inlier re-selection, RANSAC 650 iterations / 3 px. Main's earlier
  Stage-1/2 configuration (**M0**) is `ApplyStage2Frontend` (`src/pipeline/pipeline.cpp:50-54`):
  `PipelineParams{}` plus the FAST per-cell fallback (30 px cells, threshold 7).
- **Pose convention.** Every pair result `(R, t)` is `T_{j<-i}` (`X_j = R X_i + t`). The trajectory is
  `T_wc[0] = T_gt[0]` (ground-truth anchor), `T_wc[k+1] = T_wc[k] * T_{k+1<-k}^-1` for an accepted pair, else
  the previous pose is held (`include/slam/pipeline/accumulate.hpp:39-55`). Ground truth anchors frame 0 and
  is used for evaluation only.

The production pipeline is visual odometry by frame-to-frame PnP. Nothing in these files builds a map, closes
loops or runs bundle adjustment.

---

### `apps/pipeline_main.cpp`
- **Category:** production application
- **Status:** used — built as `scratch_pipeline` (`CMakeLists.txt:47-49`, links `scratch_core`); registered as
  CTest `milestone3_scratch_pipeline` (`CMakeLists.txt:98-99`), which runs it with the default (T0)
  configuration and `--out <build>/milestone3_pipeline`.
- **Purpose and why it exists:** The complete, self-contained pipeline: RGB-D PNG frames in, a per-frame
  trajectory out. It is the only program that turns the per-pair estimator (`ProcessPair`) into a trajectory
  and writes the files read by the figures, the dashboard and the Blender scene.
- **Responsibility in the architecture:** Top-level driver. It (1) builds `PipelineParams` (T0 by default,
  `apps/pipeline_main.cpp:98-101`), (2) loads frames (Bunny or a TUM frame list), (3) extracts features once per
  frame and calls `ProcessPair(i, i+1, ...)` for every consecutive pair (`:303-321`), (4) accumulates the
  trajectory (`:323-337`), (5) computes errors against ground truth *after* every decision (`:339-366`),
  (6) writes the outputs. It contains no estimation code of its own.
- **Key types and functions:** `main` only, plus local helpers `F` (fixed-point formatting, empty string for
  non-finite, `:63-69`), `DirErrDeg` (`:70-75`), `Stats` (mean/median/max ignoring non-finite, `:79-93`).

  **Command-line interface** (parsing at `apps/pipeline_main.cpp:107-186`; an unknown option exits with 2).
  Defaults are those of T0 unless stated otherwise.

  | Flag | Effect (path:line) | Default |
  |---|---|---|
  | `--legacy` | `params = PipelineParams{}` — resets to the legacy configuration (L0) (`:112-113`). Not in the header usage comment. | off |
  | `--repo DIR` | repository root (`:114-115`); otherwise `FindRepo` from the cwd, then from the executable directory (`:187-194`; `FindRepo` walks up to `data/synthetic_bunny/intrinsics.txt`, `include/slam/pipeline/common.hpp:73-81`) | auto |
  | `--frames N` | use at most N frames; only applied when N > 1 (`:116-117`, `:244-246`) | all (36 Bunny frames) |
  | `--fast-threshold T` | `frontend.fast.threshold` (`:118-119`) | 10 (T0) |
  | `--pyramid-levels L` | `frontend.pyramid.levels` (`:120-121`) | 8 (scale 1.2) |
  | `--out DIR` | output directory; relative paths are relative to the repo (`:122-123`, `:262-264`) | `results/runs/pipeline` |
  | `--pnp-iterations I` | `pnp_ransac.iterations` (`:124-125`) | 650 (T0) |
  | `--pnp-threshold PX` | `pnp_ransac.threshold_px` (`:126-127`); also the inlier threshold of re-selection and the gate's reference threshold | 3 (T0) |
  | `--seed S` | `pnp_ransac.seed` (`:128-129`) | 12345 (`include/slam/pipeline/ransac.hpp:34`) |
  | `--no-refine` | `refine = false`: keep the linear RANSAC pose (`:130-131`). Inlier re-selection only runs when the pose was refined (`src/pipeline/pipeline.cpp:147`), so this also disables it. | refine on |
  | `--stage2-frontend` | `ApplyStage2Frontend(params)` — replaces the **whole** parameter set by main's Stage-1/2 configuration (`:132-133`). Flags given *before* it are lost; flags after it modify it. | off |
  | `--fast-cell C` | `frontend.fast.cell_size` (per-cell FAST fallback; ≤ 0 = off) (`:134-135`) | 0 |
  | `--fast-min-threshold T2` | `frontend.fast.min_threshold` (fallback threshold) (`:136-137`) | 0 |
  | `--no-harris` | `frontend.selection.harris = false` (`:138-139`) | Harris on |
  | `--feature-budget N` | `frontend.selection.total_features`; 0 = legacy per-level cap (`fast.max_keypoints` = 500 per level) (`:140-141`) | 1400 |
  | `--no-mutual-ratio` | `frontend.mutual_ratio = false`: original brute force + distance filter (`:142-143`) | mutual-ratio on |
  | `--ratio R` | `frontend.ratio` used by `MatchMutualRatio` (`:144-145`) | 0.87 |
  | `--depth-edge-m M` | `frontend.depth_edge_m`; > 0 rejects a 3D point whose 3×3 frame-i depth window has an invalid sample or a span ≥ M metres (`:146-147`, rule at `src/pipeline/frontend.cpp:120-137`) | 0.012 |
  | `--huber-delta PX` | `refine_params.huber_delta_px`; 0 = least squares (`:148-149`) | 2 |
  | `--no-reselect` | `reselect_inliers = false` (`:150-151`) | re-selection on |
  | `--ranked-matcher` | `frontend.ranked_matcher = true` (`features::MatchRanked` replaces the matcher, including mutual-ratio; `src/pipeline/frontend.cpp:77-100`) (`:152-153`) | off |
  | `--match-mutual` | ranked + `matcher.mutual` (`:154-155`) | off |
  | `--match-mutual-location-aware` | ranked + mutual + `matcher.mutual_location_aware` (`:156-157`) | off |
  | `--match-ratio R` | ranked + `matcher.ratio` (0 = off) (`:158-159`) | 0 |
  | `--match-max-level-diff N` | ranked + `matcher.max_level_diff` (−1 = off) (`:160-161`) | −1 |
  | `--match-rank` | ranked + `matcher.rank` (kept matches in quality order) (`:162-163`) | off |
  | `--match-same-location-radius PX` | ranked + `matcher.same_location_radius_px` (`:164-165`) | 3 px |
  | `--export-match-details` | with the ranked matcher, also write `matches/pair_<i>_<j>.csv` (`:166-167`, `:502-515`) | off |
  | `--pose-gate` | `validity.enabled = true` (`:168-169`): only gate-accepted poses enter the trajectory and `pose_validity.csv` is written | off (`pipeline.hpp:46-47`) |
  | `--gate-min-distinct-locations N` | `validity.min_distinct_locations` (`:170-171`) | 20 (`validity.hpp:38`) |
  | `--gate-min-coverage-cells N` | `validity.min_coverage_cells` (`:172-173`) | 7 (`validity.hpp:42`) |
  | `--gate-min-inlier-ratio R` | `validity.min_inlier_ratio` (`:174-175`) | 0.0 (recorded only) |
  | `--tum-frames CSV` | TUM frame list (`:176-177`) | — |
  | `--tum-sequence DIR` | TUM sequence directory; `rgb`/`depth` columns are relative to it (`:178-179`) | — |
  | `--intrinsics fx,fy,cx,cy` | intrinsics for TUM mode (`:180-181`, parsed `:210-213`) | — |

  The three TUM flags must be given together (`:206-209`); relative paths are resolved against the repo
  (`:214-215`). The frame list must have a `frame` column equal to 0, 1, 2, … and columns `rgb`, `depth`,
  `gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw` (`:222-229`); it is the `frames.csv` written by
  `tum_trajectory_eval` (`reference/opencv_slambook2/tests/tum/tum_trajectory_eval.cpp:540`). Depth scale is
  `FrontendParams::depth_scale = 5000` (`include/slam/pipeline/frontend.hpp:41`) in both modes.

  **How T0 defaults are set:** `PipelineParams params; ApplyImprovedDefaults(params);` before parsing
  (`:98-101`). Flags are applied in command-line order on top of that. `--legacy` and `--stage2-frontend`
  overwrite everything set earlier.

  **Output files** (directory `out`, default `<repo>/results/runs/pipeline`, `:262-264`; that
  directory is git-ignored, `.gitignore:53`):

  - `keypoints/frame_<k:06d>.csv` (`:274-298`), one row per detected keypoint:
    `index,x,y,score,angle_deg,level,scale,level_x,level_y,descriptor_hex` — x, y in original-image pixels
    (3 decimals), FAST score, orientation in degrees, pyramid level, x scale of the level, integer level
    coordinates, 256-bit descriptor as 64 hex digits (word 0 first, each word most-significant digit first).
  - `pairs.csv` (`:370-408`), one row per pair:
    `pair,frame_i,frame_j,keypoints_i,keypoints_j,raw_matches,filtered_matches,d_min,correspondences_3d2d,
    pnp_ok,pnp_inliers,pnp_best_sample_inliers,reproj_inlier_mean_px,reproj_inlier_median_px,
    reproj_inlier_max_px,reproj_all_median_px,rot_err_deg,trans_err_m,tx,ty,tz,r00..r22,est_rotation_deg,
    essential_ok,essential_inliers,essential_in_front,essential_rot_err_deg,essential_t_dir_err_deg,
    pnp_vs_essential_rot_deg,refined,refine_iterations,linear_rot_err_deg,linear_trans_err_m,
    linear_reproj_inlier_mean_px,linear_reproj_inlier_median_px,linear_reproj_inlier_max_px,
    refine_rms_initial_px,refine_rms_final_px`.
    Notes from the code: the column named `pnp_ok` holds `pose_accepted` (`:389`), which equals `pnp_ok`
    when the gate is off; `rot_err_deg`/`trans_err_m` and the `linear_*_err` columns are filled only for
    accepted pairs (`:351-356`); `tx..r22` are the final `T_{j<-i}`; `correspondences_3d2d` is the number of
    PnP inputs; `raw_matches`/`filtered_matches` with mutual-ratio on are the brute-force count and the
    mutual-ratio count (`src/pipeline/frontend.cpp:77-83`).
  - `correspondences/pair_<i>_<j>.csv` (`:409-438`), one row per filtered match:
    `match,query,train,hamming,level_i,level_j,u_i,v_i,u_j,v_j,depth_raw_i,depth_raw_j,X_i,Y_i,Z_i,
    essential_inlier,pnp_used,pnp_inlier,pnp_residual_px` plus, with the ranked matcher only,
    `second_distance,ratio,mutual,rank`. `X_i..Z_i` are empty when the point has no valid 3D (no depth or
    depth-edge rejection); `pnp_inlier` is the final mask (after re-selection).
  - `trajectory.csv` (`:442-489`): five `#` comment lines (frontend settings, description, `hud_note`,
    accumulation rule), then
    `frame,tx,ty,tz,r00..r22,qx,qy,qz,qw,rotation_error_deg,translation_error_m,pnp_inliers,pose_source`.
    `(t, R)` is `T_wc`; errors are absolute against ground truth; `pnp_inliers` is that of pair `k-1`
    (empty for frame 0); `pose_source` ∈ {`gt_anchor`, `scratch_pipeline`, `held_previous_pose_pair_failed`,
    `held_previous_pose_pair_rejected`}.
  - `trajectory.txt` (`:470-488`): TUM format `timestamp tx ty tz qx qy qz qw` with the frame index as the
    timestamp.
  - `summary.txt` (`:541-656`): the printed summary (counts, front-end means, per-pair error statistics,
    linear-vs-refined comparison, essential-matrix diagnostic, final-frame error, pair 0→1 detail).
  - `matcher.csv` (ranked matcher only, `:491-501`):
    `pair,frame_i,frame_j,queries,trains,invalid_query,invalid_train,raw,removed_distance,removed_mutual,
    removed_ratio,removed_level,kept,d_min,d_max`.
  - `matches/pair_<i>_<j>.csv` (ranked + `--export-match-details`, `:502-515`):
    `query,train,distance,second_distance,second_train,ratio,mutual,mutual_location,level_query,level_train,
    fate,rank`.
  - `pose_validity.csv` (`--pose-gate` only, `:518-539`):
    `pair,frame_i,frame_j,status,solver_ok,accepted,pnp_inputs,inlier_rows,distinct_src,distinct_dst,
    distinct_loc_src,distinct_loc_dst,inlier_ratio,cells_src,cells_dst,reproj_median_px,reproj_p90_px,
    orthogonality_error,det_error,reason,returned_rot_err_deg,returned_trans_err_m` (status is
    `accepted`/`rejected`/`solver_failed`; the `returned_*` errors are for every pose the solver returned).
  - TUM mode only: `frames.csv` (copy of the input list) and `command.txt` (the command line) (`:267-273`).

  **Trajectory accumulation:** each pair becomes a `RelativeStep{accepted = pose_accepted, T = (R, t)}`
  (`:328-334`) and `AccumulateTrajectory(Twc_gt[0], steps)` (`:335`) applies
  `T_wc[k+1] = Compose(T_wc[k], Inverse(T_{k+1<-k}))` (`include/slam/pipeline/accumulate.hpp:48`), i.e. right
  multiplication by the inverse relative pose; an unaccepted pair copies the previous pose (`:50-51`).
  `pose_accepted = pnp_ok && (!validity.enabled || validity.accepted())` (`src/pipeline/pipeline.cpp:191`);
  `pnp_ok` can be cleared after RANSAC by inlier re-selection when fewer than 6 inliers remain
  (`src/pipeline/pipeline.cpp:165-168`).

  **Exit status:** 0; 1 if a frame cannot be loaded or every pair failed (`:664`); 2 for argument / setup
  errors. Code-level caveats: a flag missing its value passes an empty string to `std::stoi` (uncaught
  exception); the summary takes `min_element` of the solved-pair inlier counts without an emptiness check
  (`:602`).
- **Inputs → outputs:** Bunny: `data/synthetic_bunny/{000000.png, 000000_depth.png, …, intrinsics.txt
  (w h fx fy cx cy), groundtruth.txt (TUM T_wc)}` (`:200-204`). TUM: the frame list + sequence directory +
  intrinsics. Outputs as listed above.
- **Dependencies:** `scratch_core` (`pipeline.hpp`, `common.hpp`, `accumulate.hpp`), which pulls in
  `feature_core` and Eigen.
- **Used by:** `CMakeLists.txt:98` (CTest); `tools/pipeline_analysis/run_baseline.sh:112,171` (byte-identity
  checks with `--stage2-frontend [--pose-gate]` and TUM re-runs); `visualization/rebuild_blender.sh:10`
  (`./build-scratch/scratch_pipeline --repo … --out results/runs/pipeline`); the analysis tools read its default output
  `results/runs/pipeline/` (`tools/pipeline_analysis/check_trajectory.py:94`, `trajectory_figures.py`,
  `descriptor_figure.py:57`, `make_figures.py`, `tools/figures/validate_frontend_audit.py`). The Blender scene and
  `render_outputs.py` do **not** read a run: they read the frozen T0 trajectory `tests/fixtures/bunny_t0_scene/`
; the dashboard recognises TUM runs by
  `command.txt` naming `scratch_pipeline` (`tools/evaluation/dashboard_app/methods.py:54`);
  `tools/pipeline_analysis/validity_evidence.py` and `test_correspondence_regions.py` read its exports.
- **Covered by tests:** only as a smoke test (`milestone3_scratch_pipeline` passes unless every pair fails).
  The stages it calls are covered by `tests/pipeline/*`. Byte identity of its output under
  `--stage2-frontend` is checked by `tools/pipeline_analysis/run_baseline.sh`, not by CTest.
- **Before studying it, understand:** `PipelineParams` and the three presets (`pipeline.hpp`,
  `pipeline.cpp:39-54`), `ProcessPair` and `PairResult` (`pipeline.hpp:55-119`), `accumulate.hpp`, the
  `T_{j<-i}` convention.

### `apps/baseline_main.cpp`
- **Category:** evaluation harness
- **Status:** used — built as `scratch_baseline` (`CMakeLists.txt:67-69`); registered as CTest
  `scratch_baseline_lock` (`CMakeLists.txt:110-112`, `--out <build>/scratch_baseline_report`,
  `SKIP_RETURN_CODE 77`).
- **Purpose and why it exists:** Stage-1/2 regression harness: runs the unchanged pipeline stages on fixed
  frame pairs, measures every stage, scores against ground truth, and compares with the locked values in
  `tests/fixtures/`. Also computes raw ATE of existing runs and runs multi-seed sweeps.
- **Responsibility in the architecture:** Command-line front end of `scratch::baseline`
  (`src/evaluation/baseline.cpp`). Not part of the per-frame pipeline.
- **Key types and functions:** `main` only. **Default configuration** is main's Stage-2 configuration:
  `ApplyStage2Frontend(params); params.validity.enabled = true;` (`apps/baseline_main.cpp:50-52`), because the
  locks were taken with it. **CLI** (`:53-103`; unknown option → exit 2):

  | Flag | Effect | Default |
  |---|---|---|
  | `--repo DIR` | repository root (`:56`, `:104-109`) | `FindRepo` |
  | `--fixtures CSV` | fixture list (`:57`, `:132`) | `tests/fixtures/baseline_pairs.csv` |
  | `--expected CSV` | Stage-1 lock; `none` = no comparison (`:58`, `:133`, `:224`) | `tests/fixtures/baseline_expected.csv` |
  | `--expected-gate CSV` | Stage-2 gate lock; `none` = no comparison (`:62`, `:134`, `:231`) | `tests/fixtures/baseline_expected_gate.csv` |
  | `--out DIR` | report directory (`:59`, `:135`) | `results/runs/scratch_baseline/stage1` |
  | `--only NAME` | one fixture (`:60`) | all |
  | `--write-expected CSV` | write a new Stage-1 lock; refuses to overwrite (`:61`, `:359-367`) | — |
  | `--write-expected-gate CSV` | write a new gate lock; refuses to overwrite (`:63`, `:368-376`) | — |
  | `--ate RUN_DIR` (repeatable) | raw ATE RMSE of an existing run, then exit (`:64`, `:113-130`) | — |
  | `--ate-expected M` | check the **last** `--ate` run against M, tolerance 1e-6 m (`:65`, `:124-127`) | — |
  | `--seeds LIST` | multi-seed mode, e.g. `12345,1-19` (`:66-71`) | — |
  | `--seed-count N` | multi-seed mode with `{12345, 1, …, N-1}` (`:72-73`) | — |
  | `--seed-out DIR` | multi-seed output (`:74`, `:146`) | `results/runs/scratch_baseline/stage3_seeds` |
  | `--config ID` | replace the whole configuration by an `IntegrationConfigIds` preset (`:75-81`); **not in the usage comment** | — |
  | `--ranked-matcher`, `--match-mutual`, `--match-mutual-location-aware`, `--match-ratio R`, `--match-max-level-diff N`, `--match-rank`, `--match-same-location-radius PX` | as in `scratch_pipeline` (`:82-98`) | off |
  | `--matcher-config ID` | `ApplyMatcherConfig(ID)` on the front end (`:86-92`) | — |

  Flags are applied in order, so `--config` after a matcher flag discards it.
  **Exit status:** 0 all fixtures match; 1 mismatch/error (or PnP inputs changed between seeds); 77 some
  fixture's images are missing but every present one matched (`:216-220`, `:377-379`); 2 argument errors.
- **Inputs → outputs:** Lock mode writes `<out>/baseline_pairs.json` (tool, paths, params, definitions, one
  entry per fixture with measurement, both verdicts, export check) and `<out>/baseline_pairs.csv` (same columns
  as `baseline_expected.csv`) (`:334-358`). Seed mode writes `seed_runs.csv`
  (`fixture,seed,inputs_identical,pnp_inputs,solver_ok,gate_status,accepted,inlier_rows,distinct_loc_dst,
  rot_err_deg,trans_err_m,seconds,reason`), `seed_summary.csv`
  (`fixture,seeds,solver_ok,accepted,rejected,solver_failed,inputs_changed,returned_over_2deg` then
  `{rot_returned,trans_returned,rot_accepted,trans_accepted,seconds}_{n,median,p95,max,std}`) and
  `seed_summary.json` (`:147-215`). `--ate` prints only.
- **Dependencies:** `scratch_baseline_lib`, `scratch_experiment_lib` (`baseline.hpp`, `experiment.hpp`).
- **Used by:** `CMakeLists.txt:110` (CTest); `tools/pipeline_analysis/run_baseline.sh:130,142`;
  `docs/DATA_AND_ARCHIVE.md`.
- **Covered by tests:** its library by `test_baseline`; the program itself by `scratch_baseline_lock`.
- **Before studying it, understand:** `include/slam/evaluation/baseline.hpp`, the fixture CSV format, the
  difference between solver success, the diagnostic `pose_valid` and the Stage-2 gate.

### `apps/experiments_main.cpp`
- **Category:** evaluation harness
- **Status:** used (offline) — built as `scratch_experiments` (`CMakeLists.txt:71-73`, links Threads); no CTest.
- **Purpose and why it exists:** Runs the Stage-3C matcher experiments and the integration configurations
  (M0…T0CG) over fixture pairs or a whole TUM sequence, for several RANSAC seeds, and writes per-run gate
  diagnostics, RANSAC traces and (sequence mode) trajectory metrics. It is how configurations were compared.
- **Responsibility in the architecture:** Experiment driver around `ProcessPair`; every decision is the
  production one, the extra quantities come from `scratch::experiment`.
- **Key types and functions:** `Frame` (RGB-D + ground-truth `T_wc`), `PairJob`, `MutualComparison`
  (strict vs location-aware mutual over the matches that passed the distance filter, with ground-truth
  correctness, `:56-91`), `RunRow`, `MatchRow`.
  **CLI** (`:173-189`; unknown option → exit 2):

  | Flag | Effect | Default |
  |---|---|---|
  | `--out DIR` | output directory; **required**, must not exist (`:177`, `:191-196`) | — |
  | `--configs IDS` | comma-separated `IntegrationConfigIds` (`:178`, `:199-205`) | `M0,T0` |
  | `--fixtures CSV` | fixture mode (pairs from a `baseline_pairs.csv`-style file) (`:179`) | — |
  | `--sequence-frames CSV` | sequence mode: a `frames.csv` (columns `rgb, depth, gt_*`) (`:180`) | — |
  | `--tum-sequence DIR` | sequence directory (`:181`); not validated, an empty value resolves to the repo root | — |
  | `--intrinsics fx,fy,cx,cy` | sequence mode intrinsics (`:182`, `:228-229`) | — |
  | `--threads T` | configurations run in parallel batches of T (`:183`, `:315-320`) | number of configurations |
  | `--seed-count N` / `--seeds LIST` | RANSAC seeds (`:184-187`) | `DefaultSeedSweep(5)` = 12345,1,2,3,4 (`:171`) |
  | `--repo DIR` | repository root (`:176`) | `FindRepo` |

  Exactly one of `--fixtures` / `--sequence-frames` is required (`:191`).
- **Inputs → outputs:** For each configuration, features are extracted once per frame (`:260-261`); for each
  pair and seed, `ProcessPair` runs with only `pnp_ransac.seed` changed (`:266-271`), then `TraceRansacPnP`,
  `TraceMatches`, `DiagnoseGate`, `EvaluateWithTruth`. Writes (`:322-338`):
  `runs.csv` (header `kRunHeader`, `:93-101`: config/label/pair/frames/seed/seconds, PnP inputs, solver,
  gate status/accepted/reason, all gate quantities, 12 `c_*` check flags, input structure, relative-support
  metrics, 3D covariance eigenvalues/condition/planarity/depths, `sample_inliers`, `refit_inliers`,
  `trace_ok`, and `eval_*` ground-truth columns); `matchsets.csv` (`kMatchHeader`, `:102-106`, first seed
  only); `ate.csv` in sequence mode
  (`config,seed,frames,accepted,rejected,solver_failed,eval_ate_raw_rmse_m,eval_final_pos_err_m,
  eval_final_rot_err_deg,eval_rpe_trans_rmse_m,eval_rpe_rot_rmse_deg`; trajectory via `AccumulateTrajectory`,
  raw ATE without alignment, RPE with Δ = 1 including held poses, `:291-313`); `command.txt`. Exit 1 if any
  RANSAC trace does not reproduce production (`:341`).
- **Dependencies:** `scratch_experiment_lib`, `scratch_baseline_lib` (fixtures, seeds, `RelativeGroundTruth`),
  `accumulate.hpp`, `std::thread`.
- **Used by:** `tools/pipeline_analysis/stage3c_analysis.py`, `tools/pipeline_analysis/integration_analysis.py`
  (read its outputs); recorded invocations in `results/archive/integration_2026-11/raw/*/command.txt` (their paths
  predate the current layout: `pnp_from_scratch/build/<exe>` is now `build/<exe>`, and
  `pnp_from_scratch/tests/fixtures/stage3_pairs.csv` is now `tests/fixtures/stage3_pairs.csv`).
- **Covered by tests:** its library by `test_experiment`; the program itself is not tested.
- **Before studying it, understand:** `include/slam/evaluation/experiment.hpp` (configurations, trace,
  diagnostics), `validity.hpp`, `accumulate.hpp`.

### `apps/feature_audit.cpp`
- **Category:** production application (diagnostic tool built from the production feature code)
- **Status:** used — built as `feature_audit` (`CMakeLists.txt:125-126`); registered as CTest
  `feature_audit_pair_0_1` (`CMakeLists.txt:127`, `feature_audit --repo <source dir>`).
- **Purpose and why it exists:** A controlled audit of the T0 front end on one Bunny pair: it isolates what
  Harris ranking and the 1400-feature quota change, and re-derives every mutual-ratio matching decision
  independently, failing if it disagrees with `features::MatchMutualRatio`.
- **Responsibility in the architecture:** Diagnostic; reuses `ExtractFeatures`, `BuildPyramid`,
  `HarrisResponse`, `LevelQuotas`, `HammingDistance`, `MatchMutualRatio`. Not used by estimation.
- **Key types and functions:** `stableId` (frame, level, integer level pixel; `:17`), `exportFrame`
  (extracts four variants with FAST threshold 10 and no per-level cap, `max_keypoints = 0`:
  `{harris false, 0}`, `{true, 0}`, `{false, 1400}`, `{true, 1400}`, `:20-40`), `decisions` (nearest and
  second-nearest Hamming distance, ratio `d1 < 0.87 d2`, `:42`). **CLI** (`:45`): `--repo DIR` (default the
  current directory, made absolute), `--i N` (default 0), `--j N` (default 1); every flag requires a value;
  `i, j ≥ 0` and `i ≠ j` (`:46`).
- **Inputs → outputs:** Reads `data/synthetic_bunny` frames i and j. Writes to
  `<repo>/results/reports/figures/frontend/pair_<i>_<j>/` (`:47`): `frame_<k>_{00_fast_all,01_harris_all,
  02_fast_quotas,03_harris_quotas}.csv` (columns `stable_id,index,level,x,y,level_x,level_y,fast_score,
  harris_response,fast_order,harris_order,kept_fast_quota,kept_harris_quota,descriptor_hex`), `frame_<k>_levels.csv`
  (`level,quota,fast_all,harris_all,fast_quota,harris_quota`), `matching_decisions.csv`
  (`query,train,query_id,train_id,u_i,v_i,u_j,v_j,d1,d2,ratio_forward,reverse_d1,reverse_d2,ratio_reverse,
  mutual_best,accepted`), `config.txt`. Exit 1 on any divergence or error.
- **Dependencies:** `scratch_core` (`frontend.hpp`), `feature_core` headers.
- **Used by:** `CMakeLists.txt:127`; `tools/figures/visualize_frontend.sh:9-10`; `visualization/rebuild_blender.sh:14`.
- **Covered by tests:** it is itself the check (`AUDIT PASS` / "Audit diverged from actual matcher", `:51`).
  Note the CTest writes into the source tree (`results/reports/figures/frontend/pair_0_1/`, which is not
  git-ignored).
- **Before studying it, understand:** `features/multiscale.hpp` (`SelectionParams`, `LevelQuotas`,
  `HarrisResponse`), `features::MatchMutualRatio`.

### `apps/milestones/main.cpp`
- **Category:** test (self-checking milestone program)
- **Status:** used — built as `pnp_from_scratch` from `main.cpp + src/pipeline/pnp.cpp + projection.cpp`
  (`CMakeLists.txt:35-38`, Eigen only); registered as CTest `milestone1_pnp_from_scratch` (`CMakeLists.txt:95`).
- **Purpose and why it exists:** Milestone 1: the from-scratch linear PnP (DLT) on the frozen reference
  correspondences of Bunny pair 0→1, compared with ground truth and with the reference (OpenCV) pipeline's
  own PnP result.
- **Responsibility in the architecture:** Historical validation of `solvePnPDLT` and `projectPoint` in
  isolation; not part of the pipeline.
- **Key types and functions:** `Check` (counted PASS/FAIL), `SelfTests` (projection with hand-computed values;
  DLT on noise-free projections recovers ground truth, `:48-70`), `Run`, `Validate` (orthonormality,
  det, cheirality ≥ 95 %, finite reprojection, rotation < 5°, translation < 40 % of the baseline,
  `:96-114`). **CLI** (`:120-124`): `--repo DIR` (default `FindRepo`), `--corr FILE` (default
  `tests/fixtures/reference_baseline/correspondences/pair_0_1_correspondences.csv`, `:131-132`; `pair_metrics.csv` is
  looked up next to it or one level up, `:133-134`). Unknown flags are ignored.
- **Inputs → outputs:** Reads rows with `pnp_used = 1` (`X_i,Y_i,Z_i,u_j,v_j,pnp_inlier`) and
  `pair_metrics.csv` (`pnp_rel_q*`, `pnp_rel_t*`, `pnp_inliers`). Writes
  `results/runs/milestones/frame_0_1_reprojection.csv` and `frame_0_1_pose.txt` (`:255-280`). Exit 0 iff
  every counted check passes; the "all correspondences" run is informational and not counted (`:251`).
- **Dependencies:** `common.hpp`, `pnp.hpp`, `projection.hpp`, Eigen.
- **Used by:** `CMakeLists.txt:95`; `docs/components/pipeline.md`.
- **Covered by tests:** self-checking.
- **Before studying it, understand:** `include/slam/pipeline/pnp.hpp` (DLT and its diagnostics), the
  relative ground truth `T_{1<-0} = T_wc[1]^-1 T_wc[0]` (`:166-170`).

### `apps/milestones/sequence_main.cpp`
- **Category:** test (milestone program)
- **Status:** used — built as `pnp_full_sequence` (`CMakeLists.txt:41-44`); registered as CTest
  `milestone2_pnp_full_sequence` (`CMakeLists.txt:96`).
- **Purpose and why it exists:** Milestone 2: the same linear PnP on all raw 3D→2D correspondences of every
  Bunny pair (no RANSAC), accumulated into a trajectory — to show what linear PnP without outlier rejection
  does.
- **Responsibility in the architecture:** Historical baseline; not part of the pipeline.
- **Key types and functions:** `PairRow`, local `Inverse`/`Compose`, `Stats`. **CLI** (`:80-84`): `--repo DIR`,
  `--corr-dir DIR` (default `tests/fixtures/reference_baseline/correspondences`, `:91`). Each 3D point is re-checked
  against `(u_i, v_i, depth_raw_i / 5000)` and the program aborts on disagreement > 2e-5 m (`:127-141`).
  Trajectory: `T_wc[i+1] = Compose(T_wc[i], Inverse(T))` for a successful pair, else held (`:177-187`).
- **Inputs → outputs:** Reads `pair_<i>_<i+1>_correspondences.csv` (`depth_raw_i, X_i..Z_i, u_i, v_i, u_j,
  v_j`). Writes to `results/runs/milestones/`: `scratch_pnp_pair_results.csv`, `scratch_pnp_trajectory.csv`
  (`frame,tx..tz,r00..r22,qx..qw,rotation_error_deg,translation_error_m,mean/median/max_reprojection_px,
  num_correspondences,num_positive_depth,success,cheirality_ok,pose_source`), `scratch_pnp_trajectory.txt`,
  `scratch_pnp_full_sequence_summary.txt` (`:195-301`).
- **Dependencies:** `common.hpp`, `pnp.hpp`, `projection.hpp`, Eigen.
- **Used by:** `CMakeLists.txt:96`; its trajectory is read by `visualization/check_blender_consistency.py:42`,
  `visualization/scripts/build_scene.py` (optional 4th trajectory) and
  `tools/pipeline_analysis/check_trajectory.py:94`.
- **Covered by tests:** registered as a test, but the program has no pass/fail checks: it returns 0 whenever
  the inputs exist and are consistent (`:302`), even if pairs fail. (CMake's comment at `CMakeLists.txt:94`
  calls it self-checking.) Its header comment refers to `src/pnp.cpp`; the file is `src/pipeline/pnp.cpp`.
- **Before studying it, understand:** `apps/milestones/main.cpp`, `pnp.hpp`, `accumulate.hpp` (same rule).

---

### `include/slam/evaluation/baseline.hpp`
- **Category:** evaluation harness
- **Status:** used — header of `scratch_baseline_lib` (include dir `include/slam/evaluation`,
  `CMakeLists.txt:59-62`).
- **Purpose and why it exists:** Declares the Stage-1 measurement / regression API: fixtures, ground-truth
  isolation, per-stage counts, lock files, export cross-check, multi-seed sweeps, raw ATE.
- **Responsibility in the architecture:** Evaluation layer above `scratch_core`; ground truth is held in a
  separate list (`FixtureGroundTruth`) that `MeasureFixture` never receives (`:15-18`).
- **Key types and functions:** `Fixture`, `FixtureGroundTruth`, `LoadFixtures` (CSV format documented at
  `:56-58`), counting helpers (`CountDistinct`, `CountDistinctLocations`, `CountCrossLevelNearDuplicates`,
  `CountSharedDestinationMatches`), `PairMeasurement` (`:74-107`), constants `kDuplicateRadiusPx = 2.0`,
  `kMinDistinctInliers = 6` (`:109-110`), `Measure`, `MeasureFixture`, `RelativeGroundTruth`
  (`T_wc[j]^-1 T_wc[i]`, `:121-122`), `ScoreMeasurement`, `NumericFields`, `GateFields` (gate_status 0/1/2 =
  accepted/rejected/solver_failed, `:130-132`), `FieldTolerance`, `Expected`/`LoadExpected`/`ExpectedCsv`/
  `GateExpectedCsv`, `Compare`/`CompareGate`, `ExportCheck`/`CheckAgainstExport`, `ToJson`,
  `DefaultSeedSweep`, `ParseSeedList`, `SeedRun`, `SweepStat`, `Summarize`, `SeedSummary`, `SweepFixture`,
  `SummarizeSweep`, `AteResult`, `RawAte`.
- **Inputs → outputs:** declarations only.
- **Dependencies:** `common.hpp`, `pipeline.hpp`.
- **Used by:** `apps/baseline_main.cpp`, `apps/experiments_main.cpp`, `tests/pipeline/test_baseline.cpp`.
- **Covered by tests:** `test_baseline`, `scratch_baseline_lock`.
- **Before studying it, understand:** `PairResult` (`pipeline.hpp`), `PoseValidity` (`validity.hpp`).
  Comment vs code: lines 6-8 say the harness uses "default PipelineParams, i.e. exactly what scratch_pipeline
  does"; on this branch `MeasureFixture` takes the parameters from the caller, `scratch_baseline` passes the
  Stage-2 configuration, and `scratch_pipeline`'s default is T0 — neither is `PipelineParams{}`.

### `src/evaluation/baseline.cpp`
- **Category:** evaluation harness
- **Status:** used — `scratch_baseline_lib` (`CMakeLists.txt:59`).
- **Purpose and why it exists:** Implements `baseline.hpp`.
- **Responsibility in the architecture:** Measures a `PairResult` without ground truth, scores it afterwards,
  and implements the lock comparison.
- **Key types and functions:**
  - `LoadFixtures` (`:71-117`): rejects empty paths, non-positive focal length, duplicate names, non-finite
    ground truth, empty files.
  - `CountDistinctLocations` (`:125-137`): single-linkage union-find, distance ≤ radius merges.
  - `CountCrossLevelNearDuplicates` (`:139-150`): keypoints with a keypoint of another level within radius.
  - `Measure` (`:161-214`): fills all counts; diagnostic `pose_valid` = solver ok, finite proper rotation
    (`|det−1|`, `‖RᵀR−I‖ ≤ 1e-6`), and ≥ 6 distinct inlier keypoints in each frame (`:201-212`). This is
    separate from, and older than, the Stage-2 gate.
  - `MeasureFixture` (`:216-227`): `LoadRgbdFiles` + `ExtractFeatures` + `ProcessPair` + `Measure`.
  - `ScoreMeasurement` (`:231-237`): rotation error = angle of `R_gtᵀ R`, translation error `‖t_gt − t‖`.
  - `NumericFields` / `GateFields` (`:240-273`): the exact locked field lists.
  - `FieldTolerance` (`:282-293`): `_deg` 1e-6, `_m` and `t_x/y/z` 1e-8, `_px` 1e-6, `_ratio` 1e-12,
    `rXY` 1e-9, everything else exact.
  - `CompareFields` (`:336-354`): a field missing on either side counts as a difference.
  - `CheckAgainstExport` (`:363-446`): compares counts exactly and printed floats within their printed
    precision against `<export_dir>/pairs.csv`, `correspondences/pair_<i>_<j>.csv` (`train`, `query`,
    `pnp_inlier`) and `keypoints/frame_<k>.csv` (`x, y, level`).
  - `DefaultSeedSweep`, `ParseSeedList` (ranges, rejects duplicates/reversed/empty, `:468-509`),
    `Summarize` (median and p95 by `Percentile`, population std, `:511-524`), `SweepFixture` (features once,
    PnP inputs checked identical across seeds, `:526-556`), `SummarizeSweep`.
  - `RawAte` (`:579-609`): unaligned RMSE of `trajectory.csv` positions vs `<run>/frames.csv` (TUM) or
    `data/synthetic_bunny/groundtruth.txt`; skips `#` comment lines.
- **Inputs → outputs:** fixture CSV, lock CSVs, images; JSON/CSV strings to the caller.
- **Dependencies:** `baseline.hpp` (→ `scratch_core`).
- **Used by:** `scratch_baseline`, `scratch_experiments`, `test_baseline`.
- **Covered by tests:** `test_baseline` (counting, fixtures, determinism, GT isolation, lock comparator,
  seed sweep); `scratch_baseline_lock` (the locked values).
- **Before studying it, understand:** `baseline.hpp`, `Percentile` and `CountLocationGroups` in `validity.hpp`.

### `include/slam/evaluation/experiment.hpp`
- **Category:** evaluation harness
- **Status:** used — header of `scratch_experiment_lib` (`CMakeLists.txt:63-66`).
- **Purpose and why it exists:** Stage-3C experiment support and the integration configurations; explicitly
  not used by `scratch_pipeline`'s estimate (`:3`).
- **Responsibility in the architecture:** Names every configuration compared on this branch; provides a
  step-by-step re-run of RANSAC PnP and offline gate diagnostics computed without ground truth;
  `EvaluateWithTruth` is the only ground-truth function (`:11`).
- **Key types and functions:** `MatcherConfigIds`, `ApplyMatcherConfig`, `DescribeMatcherConfig`,
  `IntegrationConfigIds`, `ApplyIntegrationConfig`, `RansacTrace`, `TraceRansacPnP`, `TraceMatches`,
  `GateDiagnostics`, `DiagnoseGate`, `CovarianceEigenvalues`, `CountLocationPairGroups`, `TruthEval`,
  `EvaluateWithTruth`.
- **Inputs → outputs:** declarations only.
- **Dependencies:** `common.hpp`, `pipeline.hpp`.
- **Used by:** `apps/experiments_main.cpp`, `apps/baseline_main.cpp` (`--config`, `--matcher-config`),
  `tests/pipeline/test_experiment.cpp`.
- **Covered by tests:** `test_experiment`.
- **Before studying it, understand:** `features::MatcherParams` (`include/slam/features/matcher.hpp:63-71`),
  `RansacPnP`, `validity.hpp`.

### `src/evaluation/experiment.cpp`
- **Category:** evaluation harness
- **Status:** used — `scratch_experiment_lib` (`CMakeLists.txt:63`).
- **Purpose and why it exists:** Implements the configurations, the RANSAC trace and the diagnostics. Features
  must be extracted **per configuration** (the integration configurations differ in their front-end settings); a
  first version shared one extraction across configurations, which was valid when only the matcher varied but not
  here, and was caught by the check that `scratch_experiments` reproduces `scratch_pipeline` at seed 12345.
- **Responsibility in the architecture:** Defines exactly what each configuration id means.
- **Key types and functions:**

  **Matcher presets** — `ApplyMatcherConfig` (`src/evaluation/experiment.cpp:45-71`) starts from
  `features::MatcherParams{}` (all off, radius 3 px, floor 30) and sets `ranked_matcher = (id != "M0")`.
  The ranked matcher always applies the distance floor `max(2 d_min, 30)` (`src/pipeline/frontend.cpp:89`).
  "Location-aware ratio" means the second-best neighbour ignores trains within `same_location_radius_px`
  (3 px) of the best (`matcher.hpp:68`).

  | Id | Meaning (code) |
  |---|---|
  | M0 | original matcher: `MatchBruteForce` + distance filter (ranked matcher off) |
  | M1 | strict mutual |
  | M2 | ratio 0.8 (location-aware, radius 3 px) |
  | M2n | ratio 0.8 with radius 0 (naive ratio; comparison only) |
  | M3 | strict mutual + ratio 0.8 |
  | M4 | M3 + `max_level_diff = 1` |
  | M5 | M3 + `rank = true` (quality order) |
  | M1L | mutual with `mutual_location_aware` |
  | M3L | location-aware mutual + ratio 0.8 |

  **Integration configurations** — `ApplyIntegrationConfig` (`:96-151`) always starts from
  `PipelineParams{}` (`:97`):

  | Id | Exact parameter set |
  |---|---|
  | M0…M3L (any id starting with `M`) | `ApplyStage2Frontend` (legacy + FAST cell fallback 30/7) + `validity.enabled = true` + the matcher preset (`:98-102`). So **M0** = main's Stage-2 configuration; **M3** = the same with ranked strict mutual + ratio 0.8 |
  | L0 | `PipelineParams{}` unchanged — the teammate's `--legacy` (gate off) (`:103`) |
  | T0 | `ApplyImprovedDefaults` (production default), gate off (`:108`, `:114`) |
  | T0G | T0 + `validity.enabled = true` (`:115-116`) |
  | T0R | T0 + ranked matcher, `MatcherParams{}` with mutual + ratio 0.8 (replaces mutual-ratio 0.87) (`:117-118`) |
  | T0M | T0 + ranked matcher, mutual only, ratio 0 (`:119-120`) |
  | T0Q | T0 + ranked matcher, ratio 0.8 only (location-aware) (`:121-122`) |
  | T0GR | T0R + gate on (`:123-125`) |
  | T0E | T0 with `depth_edge_m = 0` (depth-edge rejection off) (`:126-127`) |
  | T0H | T0 with `huber_delta_px = 0` (least squares) (`:128-129`) |
  | T0-harris | T0 with `selection.harris = false` (`:130-131`) |
  | T0-budget | T0 with `selection.total_features = 0` → legacy cap of 500 per level (`:132-133`) |
  | T0-fast20 | T0 with FAST threshold 20 (`:134-135`) |
  | T0-mr | T0 with `mutual_ratio = false` → brute force + distance filter (`:136-137`) |
  | T0-reselect | T0 with `reselect_inliers = false` (`:138-139`) |
  | T0-ransac | T0 with RANSAC 300 iterations / 8 px (`:140-141`) — the same as `scratch_pipeline --pnp-iterations 300 --pnp-threshold 8` |
  | T0C | T0 with RANSAC 300 / 8 px **and** `total_features = 0` (500 per level), gate off (`:142-145`) |
  | T0CG | T0C + gate on (`:142-145`) |

  The ids are listed in `IntegrationConfigIds` (`:89-94`); any other id fails.

  **RANSAC trace** — `TraceRansacPnP` (`:154-195`) re-runs the algorithm of `RansacPnP` (6-point DLT
  samples from `std::mt19937(seed)`, best = most inliers then smallest residual sum, then a DLT refit on the
  best sample's inliers). `TraceMatches` (`:197-207`) requires bitwise equality with
  `PairResult::R_linear/t_linear`, `ransac_mask`, the RANSAC inlier count and `pnp_best_sample_inliers`
  (comparison with `ransac_mask`, not the final mask, because inlier re-selection changes the latter).

  **Gate diagnostics** — `DiagnoseGate` (`:248-298`): each Stage-2 check as a separate flag (`c_*`),
  input-structure counts, relative-support ratios (`ratio_pairs`, `ratio_loc_dst`, `ratio_cells_dst`),
  `rows_per_location`, `largest_cell_share` (destination cells), covariance eigenvalues, condition
  `sqrt(e3/e1)`, planarity `e1/e3`, inlier depth min/median/max. `CountLocationPairGroups` (`:224-246`):
  single-linkage groups where both ends are within the radius. `EvaluateWithTruth` (`:301-321`): errors of
  the final, sample and refit poses and the share of inputs/inliers that reproject under the true motion
  within the RANSAC threshold.
- **Inputs → outputs:** `PairResult` and parameters in, structures out.
- **Dependencies:** `experiment.hpp`, `pnp.hpp`, `projection.hpp`, `ransac.hpp`, Eigen eigen-solver.
- **Used by:** `scratch_experiments`, `scratch_baseline`, `test_experiment`.
- **Covered by tests:** `test_experiment` (presets, T0/M0/L0 contents, leave-one-out single changes,
  trace = production for every matcher preset and 3 seeds, diagnostics, conditioning, coherent wrong
  consensus).
- **Before studying it, understand:** `src/pipeline/ransac.cpp` (`RansacPnP`), `src/pipeline/pipeline.cpp`
  (where `ransac_mask` and the final mask diverge), `validity.hpp`.

---

### `tests/pipeline/check.hpp`
- **Category:** test
- **Status:** used — header included by every `tests/pipeline` test except `test_improvements`
  (include dir `tests/pipeline`, `CMakeLists.txt:90,103,107`).
- **Purpose and why it exists:** Minimal self-checking test helpers without a test framework.
- **Responsibility in the architecture:** `CHECK(cond, fmt, ...)` prints `[PASS]`/`[FAIL]` and counts failures;
  `Finish(name)` prints the verdict and returns 0/1; `Repo()` finds the repository root from the cwd and
  exits with 2 if none (`:13-41`).
- **Key types and functions:** `scratch_test::Failures`, `CHECK`, `Finish`, `Repo`.
- **Inputs → outputs:** none.
- **Dependencies:** `common.hpp` (`FindRepo`). Uses `std::exit` without including `<cstdlib>` directly.
- **Used by:** the tests below.
- **Covered by tests:** n/a.
- **Before studying it, understand:** nothing beyond `FindRepo`.

### `tests/pipeline/test_png.cpp`
- **Category:** test
- **Status:** used — target and CTest `test_png` (`CMakeLists.txt:87-93`, run from the repo root).
- **Purpose and why it exists:** Verifies the from-scratch PNG decoder (no image library is used).
- **Responsibility in the architecture:** Covers `png.hpp`: CRC-32 and Adler-32 check values, stored /
  fixed-Huffman / back-reference zlib streams (reference streams from Python's zlib), corruption rejection,
  real Bunny RGB and depth fingerprints (FNV-1a; references from Pillow), that the dataset uses
  dynamic-Huffman blocks, and that depth samples equal the frozen baseline export at every matched pixel.
- **Key types and functions:** `Hex`, `Fingerprint`.
- **Inputs → outputs:** `data/synthetic_bunny/00000{0,1}*.png`,
  `tests/fixtures/reference_baseline/correspondences/pair_0_1_correspondences.csv`.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** PNG/zlib/DEFLATE structure, `include/slam/pipeline/png.hpp`.

### `tests/pipeline/test_frontend.cpp`
- **Category:** test
- **Status:** used — CTest `test_frontend`.
- **Purpose and why it exists:** Front-end checks on Bunny 0→1 in the Stage-1/2 front end
  (`FrontendParams{}` + cell fallback 30/7, set explicitly at `:30-33`).
- **Responsibility in the architecture:** Covers `ExtractFeatures` / `MatchFrames` with the original matcher:
  one raw match per keypoint, filtered = correspondences, back-projection with `depth/5000`, > 60 % of
  3D→2D correspondences within 3 px under the true motion, index bookkeeping of raw/filtered/correspondence,
  that the cell fallback increases object occupancy and keeps ≥ 95 % of the threshold-20 corners,
  determinism.
- **Key types and functions:** `main` only.
- **Inputs → outputs:** Bunny frames 0, 1, `intrinsics.txt`, `groundtruth.txt` (measurement only).
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a. It does **not** exercise the T0 front end (Harris, budget, mutual-ratio,
  depth-edge).
- **Before studying it, understand:** `frontend.hpp`, `features/fast.hpp` (cell fallback).

### `tests/pipeline/test_essential.cpp`
- **Category:** test
- **Status:** used — CTest `test_essential`.
- **Purpose and why it exists:** Verifies the essential-matrix module on synthetic scenes with known motion.
- **Responsibility in the architecture:** Covers `essential.hpp`: 8-point estimate equals `[t]×R` up to
  scale/sign, singular values `(s, s, 0)`, constraint enforcement, the four decompositions, cheirality
  selection, triangulated depths, 8 points accepted / 7 rejected, 1 px noise accuracy, Sampson distance
  behaviour across vs along epipolar lines.
- **Key types and functions:** `MakeScene`, `EssentialDistance`.
- **Inputs → outputs:** none (synthetic).
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** epipolar geometry, `essential.hpp`.

### `tests/pipeline/test_ransac.cpp`
- **Category:** test
- **Status:** used — CTest `test_ransac`.
- **Purpose and why it exists:** Verifies RANSAC around linear PnP and around the 8-point essential matrix.
- **Responsibility in the architecture:** Covers `ransac.hpp`: sampler, recovery with 30 % outliers using
  the **default `RansacParams` (300 it, 8 px, seed 12345)** (`:71`), accuracy vs an oracle on the true inliers,
  inlier-mask quality, determinism, failure reporting (0 iterations, 5 points), RANSAC essential on a wide
  scene, and a regression of the refit guard on real pair 0→1 (with `FrontendParams{}` and the pipeline's
  essential settings 2000 it / 2 px / seed 54321).
- **Key types and functions:** `Make` (Bunny-like synthetic data with outliers).
- **Inputs → outputs:** synthetic data; Bunny frames 0, 1.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `pnp.hpp`, `ransac.hpp`.

### `tests/pipeline/test_pipeline_0_1.cpp`
- **Category:** test
- **Status:** used — CTest `test_pipeline_0_1`.
- **Purpose and why it exists:** End-to-end check of `ProcessPair` on Bunny 0→1.
- **Responsibility in the architecture:** Uses `PipelineParams{}` (legacy, `:30`): pose within 3° / 3 cm,
  inlier reprojection < 3 px, proper rotation, mask/count consistency, RANSAC better than plain DLT,
  essential stage ran on all matches, refinement lowers the reprojection error, `refine = false` keeps the
  linear pose and the same mask, determinism.
- **Key types and functions:** `main` only.
- **Inputs → outputs:** Bunny frames 0, 1 and ground truth.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a. Not run with T0.
- **Before studying it, understand:** `pipeline.hpp`.

### `tests/pipeline/test_multiscale_0_1.cpp`
- **Category:** test
- **Status:** used — CTest `test_multiscale_0_1`.
- **Purpose and why it exists:** Verifies the 8-level pyramid front end and the level→original coordinate
  mapping.
- **Responsibility in the architecture:** With `PipelineParams{}`: keypoints on every level, total 300–500,
  level 0 equal to single scale, `x = (level_x + 0.5)·scale − 0.5` per axis, depth read at the truncated
  original pixel, a control showing that using level coordinates directly would be wrong, and the whole pair
  with multiscale features.
- **Key types and functions:** `main` only.
- **Inputs → outputs:** Bunny frames 0, 1 and ground truth.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `features/pyramid.hpp`, `features/multiscale.hpp`.

### `tests/pipeline/test_refine.cpp`
- **Category:** test
- **Status:** used — CTest `test_refine`.
- **Purpose and why it exists:** Verifies the Levenberg–Marquardt PnP refinement.
- **Responsibility in the architecture:** Covers `refine.hpp`: `ExpSO3` vs axis-angle, one undamped step
  (Jacobian correctness), convergence from 5° / 3 cm on exact data, improvement over linear PnP with 1 px
  noise (40 trials), masked-out points ignored, failure cases, determinism. All with the default
  `RefineParams` (least squares, `huber_delta_px = 0`); the Huber loss is tested only in `test_improvements`.
- **Key types and functions:** `Make`, `Rms`.
- **Inputs → outputs:** synthetic.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `refine.hpp`, SO(3) exponential map.

### `tests/pipeline/test_validity.cpp`
- **Category:** test
- **Status:** used — CTest `test_validity`.
- **Purpose and why it exists:** Verifies the Stage-2 pose-validity gate and accumulation with rejected pairs.
- **Responsibility in the architecture:** Covers `validity.hpp` (`EvaluatePose`, `CountOccupiedCells`,
  `Percentile`) with default `ValidityParams` and an 8 px threshold: non-finite / non-SO(3) poses, solver
  failure status, duplicate rows and cross-level duplicates, distinct src/dst, the 20/19-row boundary,
  coverage cells, median/p90 bounds, infinite/NaN residuals, inlier ratio, a valid RANSAC pose, held poses
  in `AccumulateTrajectory` (NaN pose of a rejected pair never read; `pose_source` values), determinism and
  row-order independence; and the gate inside `ProcessPair` on Bunny 0→1 with the Stage-2 configuration,
  locking 217 inlier rows, 151 distinct, 84 locations (`:296-297`), and that the gate changes no estimate.
- **Key types and functions:** `Spread`, `MakeScene`, `Has`.
- **Inputs → outputs:** synthetic; Bunny frames 0, 1.
- **Dependencies:** `scratch_core`, `check.hpp`, `accumulate.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a. (Header comment cites `src/validity.hpp` / `src/accumulate.hpp`; the files are
  under `include/slam/pipeline/`.)
- **Before studying it, understand:** `validity.hpp`, `accumulate.hpp`.

### `tests/pipeline/test_ranked_frontend.cpp`
- **Category:** test
- **Status:** used — CTest `test_ranked_frontend`.
- **Purpose and why it exists:** Verifies that the optional `MatchRanked` path inside the front end is
  consistent.
- **Responsibility in the architecture:** With `PipelineParams{}`: all descriptors valid; ranked matcher with
  all options off reproduces the original matcher's counts, correspondences, pose, mask, gate decision and
  essential stage exactly; counters add up; with mutual + ratio 0.8 + rank the kept matches are mutual,
  one-to-one, below ratio 0.8 and in rank order; determinism.
- **Key types and functions:** `SameCorrespondences`.
- **Inputs → outputs:** Bunny frames 0, 1.
- **Dependencies:** `scratch_core`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `features/matcher.hpp` (`MatchRanked`, `MatchCounters`).

### `tests/pipeline/test_improvements.cpp`
- **Category:** test
- **Status:** used — target `test_improvements`, CTest **`scratch_improvements`** (`CMakeLists.txt:120-122`;
  no `WORKING_DIRECTORY`, does not need the repository).
- **Purpose and why it exists:** Unit checks of the teammate's T0 components.
- **Responsibility in the architecture:** `LevelQuotas(1400, 8, 1.2)` sums to 1400 and decreases with level;
  Harris response corner > edge; `MatchMutualRatio` on exact and ambiguous descriptors;
  `geometry_scratch::EstimateFundamental8Point` (recovery, rank 2, epipolar residual, degenerate rejection);
  Huber refinement (delta 2 px) is closer to the truth than least squares with 10 contaminated points.
  Uses exceptions instead of `check.hpp`.
- **Key types and functions:** `check(bool, msg)`.
- **Inputs → outputs:** synthetic.
- **Dependencies:** `scratch_core` (`features/matcher.hpp`, `features/multiscale.hpp`,
  `geometry/fundamental.hpp`, `refine.hpp`).
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `multiscale.hpp`, `include/slam/geometry/fundamental.hpp`.

### `tests/pipeline/test_baseline.cpp`
- **Category:** test
- **Status:** used — target and CTest `test_baseline` (`CMakeLists.txt:100-103`, links `scratch_baseline_lib`).
- **Purpose and why it exists:** Tests the measurement harness itself.
- **Responsibility in the architecture:** Fixture loading (three fixtures in order, intrinsics, Bunny ground
  truth equal to `groundtruth.txt`, rejection of duplicates/missing columns/empty/missing files), counting
  helpers, cross-level duplicate detection, the diagnostic `pose_valid`, bit-identical repeat runs, ground-truth
  isolation (scoring with a wrong ground truth changes no estimated field), pose conventions
  (`RelativeGroundTruth = T_wc[j]^-1 T_wc[i]`, accumulation, inverted convention is worse, quaternion order
  `(qx, qy, qz, qw)`), the lock comparator and tolerances, seed parsing, `Summarize`, and the seed sweep.
  Runs the pipeline only on the Bunny fixture, with the Stage-2 configuration (`:146-148`).
- **Key types and functions:** `TempFile` (writes into `fs::temp_directory_path()`), `H`, `FakePair`.
- **Inputs → outputs:** `tests/fixtures/baseline_pairs.csv`, Bunny frames.
- **Dependencies:** `scratch_baseline_lib`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `baseline.hpp`.

### `tests/pipeline/test_experiment.cpp`
- **Category:** test
- **Status:** used — target and CTest `test_experiment` (`CMakeLists.txt:104-107`).
- **Purpose and why it exists:** Tests the Stage-3C / integration support library.
- **Responsibility in the architecture:** The nine matcher presets; that T0 equals `ApplyImprovedDefaults`
  with gate off and no cell fallback, M0 equals main's Stage-2 configuration, L0 equals `PipelineParams{}`
  (`:80-96`); that each of T0-harris, T0-budget, T0-fast20, T0-mr, T0-reselect, T0E, T0H changes exactly one
  T0 component (`:97-110`); all ids apply; on Bunny 0→1 every matcher preset is accepted, deterministic, and
  `TraceRansacPnP` reproduces production for seeds 12345, 1, 7; `DiagnoseGate` quantities on constructed
  data; covariance conditioning; a spatially coherent wrong consensus that the unchanged gate accepts but
  relative support separates.
- **Key types and functions:** `Build`, `Backproject`.
- **Inputs → outputs:** synthetic; Bunny frames 0, 1.
- **Dependencies:** `scratch_experiment_lib`, `check.hpp`.
- **Used by:** CTest.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `experiment.hpp`, `validity.hpp`.

---

### `tests/fixtures/baseline_pairs.csv`
- **Category:** fixture
- **Status:** used — default `--fixtures` of `scratch_baseline` (`apps/baseline_main.cpp:132`); read by
  `test_baseline` (`tests/pipeline/test_baseline.cpp:54`).
- **Purpose and why it exists:** The three locked frame pairs of the Stage-1/2 regression: `bunny_0_1`,
  `tum_xyz_stride4_46_47` (a pair where the solver returns a 22° wrong pose), `tum_xyz_stride4_73_74`.
- **Responsibility in the architecture:** Defines what is measured; ground truth sits in the same rows but is
  separated by `LoadFixtures`.
- **Key types and functions:** columns (`include/slam/evaluation/baseline.hpp:56-58`):
  `name,dataset,frame_i,frame_j,rgb_i,depth_i,rgb_j,depth_j,fx,fy,cx,cy,gt_i_x,gt_i_y,gt_i_z,gt_i_qx,gt_i_qy,
  gt_i_qz,gt_i_qw,gt_j_x,…,gt_j_qw,export_dir,export_pair`. Paths are repository-relative. Bunny
  `export_dir = tests/fixtures/stage1_bunny_export`, pair 0; TUM `export_dir =
  tests/fixtures/stage1_tum_exports/rgbd_dataset_freiburg1_xyz/n100_stride4` (tracked frozen export, manifest
  `tests/fixtures/stage1_tum_export_sha256.txt`; a byte-identical copy of the archived run, `tests/fixtures/README.md`).
- **Inputs → outputs:** read only.
- **Dependencies:** `data/synthetic_bunny/`; the TUM rows name `data/rgbd_dataset_freiburg1_xyz/…`, read from
  `$SLAM_TUM_ROOT/rgbd_dataset_freiburg1_xyz/…` when `SLAM_TUM_ROOT` is set (`LoadFixtures`); missing images → exit 77.
- **Used by:** `scratch_baseline`, `test_baseline`, `tools/pipeline_analysis/run_baseline.sh` (via
  `scratch_baseline`).
- **Covered by tests:** `test_baseline`, `scratch_baseline_lock`.
- **Before studying it, understand:** `LoadFixtures`.

### `tests/fixtures/baseline_expected.csv`
- **Category:** fixture
- **Status:** used — default `--expected` of `scratch_baseline` (`apps/baseline_main.cpp:133`).
- **Purpose and why it exists:** Stage-1 lock: every `NumericFields` value of the three fixture pairs in the
  Stage-1/2 front end (M0 configuration), written with 17 significant digits.
- **Responsibility in the architecture:** Regression lock of the legacy/M0 configuration; any algorithmic
  change in that configuration shows up as a difference.
- **Key types and functions:** columns `name` + `frame_i,frame_j,keypoints_i,keypoints_j,cross_level_dup_i,
  cross_level_dup_j,raw_matches,filtered_matches,unique_src_filtered,unique_dst_filtered,shared_dst_filtered,
  pnp_inputs,unique_src_pnp,unique_dst_pnp,inlier_rows,distinct_inlier_i,distinct_inlier_j,
  distinct_inlier_loc_i,distinct_inlier_loc_j,shared_dst_inliers,solver_ok,pose_valid,refined,
  refine_iterations,reproj_inlier_mean_px,est_rotation_deg,est_translation_m,t_x,t_y,t_z,r00..r22,
  gt_motion_deg,gt_motion_m,rot_err_deg,trans_err_m` (`src/evaluation/baseline.cpp:240-263`). Compared with
  `FieldTolerance`.
- **Inputs → outputs:** read only (`--write-expected` creates a new file; never overwrites).
- **Dependencies:** —
- **Used by:** `scratch_baseline`; `tools/pipeline_analysis/run_baseline.sh` (step 3).
- **Covered by tests:** `scratch_baseline_lock`.
- **Before studying it, understand:** `NumericFields`, `FieldTolerance`.

### `tests/fixtures/baseline_expected_gate.csv`
- **Category:** fixture
- **Status:** used — default `--expected-gate` of `scratch_baseline` (`apps/baseline_main.cpp:134`).
- **Purpose and why it exists:** Stage-2 lock of the gate evaluation on the same three pairs (Bunny accepted,
  46→47 rejected, 73→74 accepted).
- **Responsibility in the architecture:** Locks the gate's decisions and quantities separately so that the
  Stage-1 field set stays unchanged.
- **Key types and functions:** columns `name,gate_status,pose_accepted,gate_pnp_inputs,gate_inlier_rows,
  gate_distinct_src,gate_distinct_dst,gate_distinct_loc_src,gate_distinct_loc_dst,gate_inlier_ratio,
  gate_cells_src,gate_cells_dst,gate_reproj_median_px,gate_reproj_p90_px` (`src/evaluation/baseline.cpp:265-273`);
  `gate_status` 0 accepted, 1 rejected, 2 solver_failed.
- **Inputs → outputs:** read only.
- **Dependencies:** —
- **Used by:** `scratch_baseline`; `run_baseline.sh`.
- **Covered by tests:** `scratch_baseline_lock`.
- **Before studying it, understand:** `GateFields`, `validity.hpp`.

### `tests/fixtures/baseline_trajectories.csv`
- **Category:** fixture
- **Status:** used — only by `tools/pipeline_analysis/run_baseline.sh` step 4 (not by CTest).
- **Purpose and why it exists:** Raw-ATE lock of frozen Stage-1 runs.
- **Responsibility in the architecture:** For each row, `run_baseline.sh` calls
  `scratch_baseline --ate <run_dir> --ate-expected <ate>` (tolerance 1e-6 m) if the directory exists.
- **Key types and functions:** columns `name,run_dir,frames,ate_raw_rmse_m`. Rows: TUM fr1/xyz runs under
  `tests/fixtures/stage1_ate_runs/scratch/rgbd_dataset_freiburg1_xyz/...` (n20, n100, n100_stride2, n100_stride4, n300,
  full) and `bunny_36` with `run_dir = tests/fixtures/stage1_bunny_export` (frozen), 0.143328109 m.
- **Inputs → outputs:** read only.
- **Dependencies:** the referenced run directories: `tests/fixtures/stage1_ate_runs/` (24 files, manifest
  `stage1_ate_runs_sha256.txt`) and `tests/fixtures/stage1_bunny_export/`, all tracked frozen copies.
- **Used by:** `run_baseline.sh`.
- **Covered by tests:** no CTest.
- **Before studying it, understand:** `RawAte`. Note: `bunny_36` locks the Stage-1 configuration's
  trajectory; it reads the frozen copy in `tests/fixtures/stage1_bunny_export` (not `scratch_pipeline`'s mutable
  default output `results/runs/pipeline`).

### `tests/fixtures/baseline_trajectories_gate.csv`
- **Category:** fixture
- **Status:** used — only by `run_baseline.sh` (same loop).
- **Purpose and why it exists:** Raw-ATE lock of the Stage-2 (gate on) TUM runs, read from
  `tests/fixtures/stage1_ate_runs/scratch_gate/rgbd_dataset_freiburg1_xyz/...`.
- **Responsibility in the architecture:** Same as above, for the gate configuration.
- **Key types and functions:** columns `name,run_dir,frames,ate_raw_rmse_m`; six TUM rows (no Bunny row).
  Only `n100_stride4` differs from the Stage-1 lock (0.099262525 m vs 0.514672706 m).
- **Inputs → outputs:** read only.
- **Dependencies:** `tests/fixtures/stage1_ate_runs/scratch_gate/` (tracked frozen copies).
- **Used by:** `run_baseline.sh`.
- **Covered by tests:** no CTest (the files themselves: `fixture_checksums`).
- **Before studying it, understand:** `RawAte`.

### `tests/fixtures/stage1_ate_runs/` and `tests/fixtures/stage1_ate_runs_sha256.txt`
- **Category:** fixture (frozen data)
- **Status:** used — the `run_dir` of the 12 TUM rows of `baseline_trajectories{,_gate}.csv`.
- **Purpose and why it exists:** `trajectory.csv` and `frames.csv` (the only files `RawAte` reads) of the 12
  frozen Stage-1/Stage-2 TUM runs, `{scratch,scratch_gate}/rgbd_dataset_freiburg1_xyz/{n20,n100,n100_stride2,
  n100_stride4,n300,full}/`; 24 files, 1 166 355 bytes. Copied on 2026-11-06 from the external archive
  (`2026-11-04_main/output/…`), byte-identical to it (checked against the archive's `MANIFEST.sha256`), so the ATE
  locks run without the 2.8 GB archive.
- **Inputs → outputs:** read only; no program writes here by default (files are read-only on disk).
- **Covered by tests:** `fixture_checksums` (`sha256sum -c stage1_ate_runs_sha256.txt`).

### `tests/fixtures/bunny_t0_scene/` and `tests/fixtures/bunny_t0_scene_sha256.txt`
- **Category:** fixture (frozen data)
- **Status:** used — `trajectory_data.SCRATCH_SOURCE` (the Blender scene's scratch-PnP camera, its autoload
  fingerprint, `check_blender_consistency.py`, `test_blender_data.py`) and `tools/figures/render_outputs.py`.
- **Purpose and why it exists:** `trajectory.csv` and `pairs.csv` of the **T0** `scratch_pipeline` Bunny run (36 poses,
  defaults: Harris ranking, budget 1400, FAST 10, RANSAC 650 it / 3 px, depth edge 12 mm, Huber 2 px). The scene was
  built from exactly this trajectory: with it, `trajectory_data._hash` reproduces the scene's stored fingerprint
  `42aef737…`, which no other trajectory did. Copied (no re-run) from the milestone-3 CTest run of 2026-11-04;
  `pairs.csv` comes from the same run directory (the whole source run is kept in the external backup). The
  scene therefore never reads a mutable run.
- **Inputs → outputs:** read only; read-only on disk; `fixture_checksums` detects any change.
- **Covered by tests:** `fixture_checksums`, `blender_data_bridge`, `check_blender_consistency.py`.

### `tests/fixtures/reference_baseline/` and `tests/fixtures/reference_baseline_sha256.txt`
- **Category:** fixture (frozen data)
- **Status:** used — default input of `pnp_from_scratch` (`apps/milestones/main.cpp:132`) and `pnp_full_sequence`
  (`apps/milestones/sequence_main.cpp:91`), i.e. the CTests `milestone1_pnp_from_scratch` and
  `milestone2_pnp_full_sequence`; read by `test_png` (`tests/pipeline/test_png.cpp:93`), `make_figures.py`,
  `descriptor_figure.py` and `trajectory_figures.py`.
- **Purpose and why it exists:** the OpenCV reference pipeline's Bunny export at the `baseline-pre-migration` tag
  (41 files): `correspondences/pair_<i>_<j>_correspondences.csv` (35 pairs), `pair_metrics.csv`, `metrics.json`,
  `slam_trajectory.csv`, `reference_features/pair_0_1_{keypoints,raw_matches}.csv` (ORB keypoints 366 / 367 and raw
  matches of pair 0→1) and `reference_orb_distances_0_1.csv` (all-pairs ORB Hamming histogram, written once by
  `tools/pipeline_analysis/export_reference_orb_distances.py`). It lets the milestone programs and figure tools
  compare against the reference without running it.
- **Inputs → outputs:** read only; read-only on disk.
- **Covered by tests:** `fixture_checksums` (`sha256sum -c reference_baseline_sha256.txt`).

### `tests/fixtures/README.md`
- **Category:** documentation
- **Purpose:** one row per lock file and frozen data directory: what it is, its manifest, where it came from and what
  reads it.

### `tests/fixtures/stage3_pairs.csv`
- **Category:** fixture
- **Status:** used offline — input of `scratch_experiments --fixtures` (recorded in
  `results/archive/integration_2026-11/raw/fixtures_20seeds*/command.txt`, under its pre-reorganisation path); not read by
  any CTest.
- **Purpose and why it exists:** The Stage-3C fixture set: the three `baseline_pairs.csv` pairs plus
  `tum_xyz_stride4_81_82`.
- **Responsibility in the architecture:** Pair list for the matcher/integration experiments.
- **Key types and functions:** same columns as `baseline_pairs.csv`. Its Bunny `export_dir` is
  `tests/fixtures/stage1_bunny_export`; `scratch_experiments` does not use `export_dir`.
- **Inputs → outputs:** read only.
- **Dependencies:** Bunny and TUM fr1/xyz data.
- **Used by:** `scratch_experiments` (by command line).
- **Covered by tests:** no.
- **Before studying it, understand:** `LoadFixtures`, `apps/experiments_main.cpp`.

### `tests/fixtures/baseline_bunny_sha256.txt`
- **Category:** fixture
- **Status:** used — only by `tools/pipeline_analysis/run_baseline.sh:121-122`.
- **Purpose and why it exists:** SHA-256 manifest (75 lines, `sha256sum` format, paths `./…`) of a full
  Bunny `scratch_pipeline` output in the Stage-1 configuration: `pairs.csv`, `summary.txt`,
  `trajectory.csv`, `trajectory.txt`, 35 `correspondences/pair_<i>_<j>.csv`, 36 `keypoints/frame_<k>.csv`.
  The files in `tests/fixtures/stage1_bunny_export/` match all 75 hashes.
- **Responsibility in the architecture:** Byte-identity lock. Because this branch's `scratch_pipeline` writes
  a different format (extra `# frontend:` line in `trajectory.csv`, fewer correspondence columns),
  `run_baseline.sh` checks only the subset `trajectory.txt` + `keypoints/` of a fresh
  `scratch_pipeline --stage2-frontend` run against it (`run_baseline.sh:105-108,121-122`).
- **Key types and functions:** `<sha256>  ./<path>` per line.
- **Inputs → outputs:** read only.
- **Dependencies:** —
- **Used by:** `run_baseline.sh`; `docs/DATA_AND_ARCHIVE.md`.
- **Covered by tests:** no CTest.
- **Before studying it, understand:** the output layout of `scratch_pipeline`.

### `tests/fixtures/baseline_bunny_gate_sha256.txt`
- **Category:** fixture
- **Status:** used — only by `run_baseline.sh:123-124`.
- **Purpose and why it exists:** The same manifest for main's Stage-2 configuration (gate on): 76 lines —
  the 75 paths above plus `pose_validity.csv`; `summary.txt` and `trajectory.csv` hashes differ from the
  Stage-1 manifest, all others are identical.
- **Responsibility in the architecture:** `run_baseline.sh` checks the subset `trajectory.txt`,
  `pose_validity.csv`, `keypoints/` of `scratch_pipeline --stage2-frontend --pose-gate`.
- **Key types and functions:** as above.
- **Inputs → outputs:** read only. (No exported directory in the repository corresponds to this manifest.)
- **Dependencies:** —
- **Used by:** `run_baseline.sh`.
- **Covered by tests:** no CTest.
- **Before studying it, understand:** `--pose-gate` output files.

### `tests/fixtures/stage1_bunny_export/`
- **Category:** fixture
- **Status:** used — `export_dir` of the `bunny_0_1` fixture (`tests/fixtures/baseline_pairs.csv`), read by
  `CheckAgainstExport` in `scratch_baseline` (and thus `scratch_baseline_lock`).
- **Purpose and why it exists:** The frozen complete Bunny output of main's Stage-1 `scratch_pipeline`
  (8-level pyramid, FAST 20, original matcher, RANSAC 300 it / 8 px, LM refinement; 35/35 pairs solved;
  final position error 0.2527 m per its `summary.txt`). It lets the harness prove it reproduces the numbers
  the pipeline reported, independent of any locally generated output.
- **Responsibility in the architecture:** Reference export; byte-identical to `baseline_bunny_sha256.txt`.
- **Key types and functions:** contents: `pairs.csv` (main's 47-column format, same leading columns as this
  branch's), `trajectory.csv` (four `#` lines, no `# frontend:` line), `trajectory.txt`, `summary.txt`,
  `correspondences/pair_<i>_<j>.csv` ×35 (main's format: this branch's 19 columns plus `depth_edge`,
  `epi_a,epi_b,epi_c,epi_dist_px`, `epi_gt_*`, `epi_pnp_*`), `keypoints/frame_<k:06d>.csv` ×36 (same format
  as this branch).
- **Inputs → outputs:** read only. `CheckAgainstExport` reads `pairs.csv` (row `pair = 0`),
  `correspondences/pair_0_1.csv` (`query`, `train`, `pnp_inlier`) and `keypoints/frame_000000.csv`,
  `frame_000001.csv` (`x`, `y`, `level`).
- **Dependencies:** —
- **Used by:** `scratch_baseline`; `docs/DATA_AND_ARCHIVE.md`.
- **Covered by tests:** `scratch_baseline_lock` (export check counts as a mismatch if not reproduced,
  `apps/baseline_main.cpp:329`).
- **Before studying it, understand:** `scratch_pipeline` output columns; `CheckAgainstExport`.

---

### `experiments/depth_edge/depth_edge_experiment.cpp`
- **Category:** experiment
- **Status:** built as `depth_edge_experiment` (`CMakeLists.txt:53-55`); no CTest; run by hand.
- **Purpose and why it exists:** Read-only experiment: runs RANSAC PnP + LM twice per Bunny pair — on all
  3D→2D correspondences and on those whose depth sample is not at a depth edge — to measure the effect of
  depth-edge correspondences. Precedes the production `depth_edge_m` option.
- **Responsibility in the architecture:** Outside the pipeline; reimplements only the geometric stage
  (`Geometric`, `:102-127`: `RansacPnP` then `RefinePnP` on the RANSAC inliers; no inlier re-selection, no
  essential stage, no gate).
- **Key types and functions:** `EdgeStats`, `Classify` (`:58-89`): circular window of radius
  `ceil(radius_factor · level_scale)`; edge if any window pixel lacks depth or two 4-adjacent valid pixels
  differ by more than `τ = (Z/fx)·tan(incidence_deg)`. **CLI** (`:152-157`): `--incidence-deg A` (default 80),
  `--radius-factor F` (default 3); repository root only from the cwd (`:158`); unknown flags ignored.
- **Inputs → outputs:** Bunny frames, `intrinsics.txt`, `groundtruth.txt`. Writes
  `results/runs/depth_edge/`: `pairs.csv` (counts, inliers, errors and reprojection for both
  sets), `trajectory_all.csv` / `trajectory_interior.csv` (`frame,tx..tz,r00..r22,translation_error_m,
  rotation_error_deg`; accumulation `T[i+1] = T[i]·rel^-1`, failed pair held, `:228-235`),
  `correspondences/pair_<i>_<j>.csv` (window statistics, edge flag, inlier flags of both runs).
- **Dependencies:** `scratch_core`.
- **Used by:** `tools/pipeline_analysis/depth_edge_audit.py` (reads its output); `docs/components/pipeline.md`.
- **Covered by tests:** no.
- **Before studying it, understand:** `frontend.hpp` (`MatchFrames`, `Correspondence`), `ransac.hpp`,
  `refine.hpp`. Comment vs code: line 172 says `const PipelineParams params; // the production defaults` and
  line 7 says the "all" set equals the production pipeline; on this branch `PipelineParams{}` is the legacy
  configuration (L0), not T0, so the experiment does not run the production front end (no Harris/budget,
  original matcher, no 12 mm depth-edge rejection, 300 it / 8 px, least squares).

---

### `CMakeLists.txt`
- **Category:** build
- **Status:** used — the single root build (`cmake -S . -B build`).
- **Purpose and why it exists:** Builds the production library and applications, the evaluation harness,
  the tests, and optionally the OpenCV reference/experiments.
- **Responsibility in the architecture:** C++17, Release by default (`:4-8`); every executable is placed in
  the build directory root (`:11`).
- **Key types and functions:**
  - Option `SLAM_WITH_OPENCV` (default OFF, `:16`): ON adds `experiments/rgbd_icp`,
    `reference/opencv_slambook2` and `cmake/orb_benchmark.cmake` (`:125-129`).
  - `find_package(Eigen3 REQUIRED)` (`:19`); `include(cmake/feature_core.cmake)` → library `feature_core` (`:23`).
  - Library `scratch_core` (`:27-32`): `src/pipeline/{png,frontend,projection,pnp,essential,ransac,refine,
    validity,pipeline}.cpp`, public include dir `include/slam/pipeline`, links Eigen and `feature_core`.
  - Executables: `pnp_from_scratch` (`apps/milestones/main.cpp` + pnp/projection sources, Eigen only,
    `:35-38`), `pnp_full_sequence` (`:41-44`), `scratch_pipeline` (`:47-49`), `depth_edge_experiment`
    (`:53-55`), `scratch_baseline` (`:67-69`), `scratch_experiments` (+ Threads, `:70-73`),
    `test_improvements` (`:120-121`), `feature_audit` (`:125-126`), the six feature unit tests and the
    pipeline tests below.
  - Libraries `scratch_baseline_lib` (`src/evaluation/baseline.cpp`) and `scratch_experiment_lib`
    (`src/evaluation/experiment.cpp`), public include dir `include/slam/evaluation` (`:59-66`).
  - **CTests** (`enable_testing()`, `:78`), 24 in total:
    - `features_fast_test`, `features_orientation_test`, `features_brief_test`, `features_matcher_test`,
      `features_pyramid_test`, `features_matcher_ranked_test` (`tests/unit/`, link `feature_core`, `:81-86`);
    - `test_png`, `test_frontend`, `test_essential`, `test_ransac`, `test_pipeline_0_1`,
      `test_multiscale_0_1`, `test_refine`, `test_validity`, `test_ranked_frontend` (link `scratch_core`,
      run in the repo root, `:87-93`);
    - `milestone1_pnp_from_scratch`, `milestone2_pnp_full_sequence`, `milestone3_scratch_pipeline`
      (`:95-99`);
    - `test_baseline`, `test_experiment` (`:100-107`);
    - `scratch_baseline_lock` (skip code 77, `:108-112`);
    - `fixture_checksums` (`sha256sum -c` of the five fixture-directory manifests: Bunny export, TUM export, ATE-lock
      runs, the Blender scene's T0 trajectory, the reference-pipeline baseline; `:113-118`);
    - `scratch_improvements` (`:122`);
    - `feature_audit_pair_0_1` (`:127`).
- **Inputs → outputs:** source tree → `build/<target>`. Note side effects of `ctest`: milestones 1 and 2
  write into `results/runs/milestones/` (git-ignored), `feature_audit_pair_0_1` writes into
  `results/reports/figures/frontend/pair_0_1/` (not git-ignored); milestone 3 and the baseline lock write
  into the build tree.
- **Dependencies:** CMake ≥ 3.16, Eigen3, Threads; OpenCV/g2o/Sophus only with `SLAM_WITH_OPENCV=ON`.
- **Used by:** `tools/pipeline_analysis/run_baseline.sh:103`, `tools/figures/visualize_frontend.sh`,
  `visualization/rebuild_blender.sh` (these two use a `build-scratch/` directory).
- **Covered by tests:** n/a.
- **Before studying it, understand:** `cmake/feature_core.cmake`.

## Reference implementation (OpenCV/g2o, not production)

Everything in this section lives under `reference/`. None of it is part of the production pipeline
(`scratch_pipeline`, library `scratch_core`), which uses only Eigen and the C++ standard library. The reference
code is the earlier, slambook2-derived ("Visual SLAM: From Theory to Practice", 2nd ed., chapters 6 and 7)
OpenCV + g2o + Sophus implementation. It is kept for comparison, for the synthetic-Bunny demos, and for the TUM
evaluator `tum_trajectory_eval`. Files mark copied book code with `// Book code starts` / `// Book code ends`.

**How it is built.** The root `CMakeLists.txt` only adds this directory when it is configured with
`-DSLAM_WITH_OPENCV=ON` (option `SLAM_WITH_OPENCV`, default OFF, `CMakeLists.txt:16`;
`add_subdirectory(reference/opencv_slambook2)`, `CMakeLists.txt:130-132`). All executables are written directly
into `build/` (`CMAKE_RUNTIME_OUTPUT_DIRECTORY`, `CMakeLists.txt:11`). Most demos use paths such as
`../data/...` and `../results/runs/...`, so they assume the working directory is `build/`. `results/runs/` is
ignored by git.

**None of the reference executables is registered with CTest.** There is no `add_test` in
`reference/opencv_slambook2/CMakeLists.txt` or `reference/icp_main_variant/CMakeLists.txt`. The self-checking
ones are run by `tools/evaluation/run_reference_suite.sh` (from `build/`), which judges them by exit status.

**What the reference code does NOT have.** There is no mapping and no loop closure. There is no
keyframe or multi-frame bundle adjustment. Every "bundleAdjustment" here is a single-pose (pose-only)
optimisation over one frame pair. The trajectory programs chain pairwise poses. They anchor frame 0 to ground
truth and hold the previous pose when a pair fails.

Libraries defined in `reference/opencv_slambook2/CMakeLists.txt`:

| target | sources | links |
|---|---|---|
| `camera` | `src/camera/camera.cpp` | OpenCV (`:17-19`) |
| `features` | `src/features/features.cpp` | OpenCV PUBLIC, `feature_core` PRIVATE (`:25-27`) |
| `geometry` | `src/geometry/geometry.cpp` | OpenCV PUBLIC; `camera`, Eigen PRIVATE; private include `include/slam` for `geometry/fundamental.hpp` (`:32-36`) |
| `optimization` | header-only `include/optimization/g2o_types.hpp` | INTERFACE g2o::core, Sophus (`:39-41`) |
| `tracking` | `src/tracking/{pnp,icp,essential_matrix,fast,brief,hamming_matching}.cpp` | OpenCV, Eigen, Sophus PUBLIC; `optimization`, g2o core/stuff/solver_dense, `camera`, **`rgbd_icp`** PRIVATE; `-msse4` (`:47-60`) |
| `render` | `src/render/*.cpp` | OpenCV, Eigen, Sophus, `camera` PUBLIC (`:67-74`) |
| `dataset` | `src/dataset/tum.cpp` | Eigen only (`:136-138`) |
| `tracking_main_icp` | the same `tracking` sources except `src/icp.cpp` from `reference/icp_main_variant/` | no `rgbd_icp` (`reference/icp_main_variant/CMakeLists.txt:13-25`) |

`rgbd_icp` is the teammate's dense RGB-D ICP library in `experiments/rgbd_icp/` (`experiments/rgbd_icp/CMakeLists.txt:6-8`).

---

### The two ICP variants (read this before any ICP file)

The repository has two incompatible implementations of the same two book functions,
`pose_estimation_3d3d(pts1, pts2, R, t)` and `bundleAdjustment(pts1, pts2, R, t)`:

| | **Teammate variant** (`reference/opencv_slambook2/src/tracking/icp.cpp` + `include/tracking/icp.hpp`), library `tracking` | **Main variant** (`reference/icp_main_variant/src/icp.cpp` + `include/tracking/icp.hpp`), library `tracking_main_icp` |
|---|---|---|
| `pose_estimation_3d3d` | Wrapper (`src/tracking/icp.cpp:8-24`). It copies `pts1` into `target` and `pts2` into `source`, then calls `rgbd_icp::rigidSVD(source, target)`. | The book's closed-form SVD (`src/icp.cpp:19-63`). |
| SVD formula | `rigidSVD` (`experiments/rgbd_icp/src/rgbd_icp.cpp:44-63`): centroids a (source), b (target); H = Σ(sᵢ−a)(tᵢ−b)ᵀ; `D(2,2) = det(V Uᵀ) < 0 ? -1 : 1`; R = V D Uᵀ; t = b − R a. | W = Σ q1ᵢ q2ᵢᵀ (centred pts1 times centred pts2ᵀ, `:37-40`); R = U Vᵀ (`:51`); t = p̄1 − R p̄2 (`:55`). |
| Reflection handling | Correct Kabsch/Umeyama correction: only the last singular direction is flipped (`rgbd_icp.cpp:59-61`). | `if (R_.determinant() < 0) R_ = -R_;` (`src/icp.cpp:52-54`). For a 3×3 matrix det(−R) = −det(R), so the result is a proper rotation. In general it is **not** the least-squares rotation; that would be U diag(1,1,−1) Vᵀ, which `FitRigid` uses. The case only arises for degenerate or very noisy point sets. |
| Input checks | Throws `std::invalid_argument` if the sizes differ, there are fewer than 3 pairs, or a point is non-finite (`rgbd_icp.cpp:45-50`). | None. N = 0 divides by zero (`:28-29`). Prints W, U, V to stdout (`:41,48-49`). |
| `bundleAdjustment` | **Does not optimise.** It calls `pose_estimation_3d3d(pts1, pts2, R, t)` again and returns that result (`src/tracking/icp.cpp:29-33`). The comment says the fixed-correspondence problem "already has an exact SVD solution" (`:26-28`). | The book's g2o pose-only refinement (`src/icp.cpp:93-138`). Levenberg–Marquardt, `BlockSolverX` + dense linear solver (`:97-100`). One `VertexPose` (left-multiplicative SE(3) update, `include/optimization/g2o_types.hpp:18-22`), **initialised to identity** (`:107`); the incoming R, t are ignored. One `EdgeProjectXYZRGBDPoseOnly` per pair: error = pts1ᵢ − T·pts2ᵢ, Jacobian [−I, (T p)^] (`:66-91`). Identity information, no robust kernel, `optimize(10)` (`:122`), verbose. |
| Extra API | None. The header points to `tracking/rgbd_icp.hpp` / `icp_sequence` for real RGB-D ICP (`include/tracking/icp.hpp:19`). | `FitRigid` (`src/icp.cpp:144-166`): least-squares fit over given indices. Returns false for fewer than 3 points (`:146`) or a collinear set, i.e. `s(1) <= 1e-12·max(1, s(0))` (`:160`). D = diag(1,1,−1) when det(U Vᵀ) < 0 (`:161-162`); R = U D Vᵀ, t = c1 − R c2 (`:163-164`). `EstimateRigidRansac` (`:168-213`, defaults `include/tracking/icp.hpp:37-38`: **threshold 0.01 m, 300 iterations, seed 12345**): `std::mt19937(seed)` (`:185`), 3 indices by partial Fisher–Yates (`:192`), minimal `FitRigid`. Inlier test ‖pts1 − (R pts2 + t)‖ < threshold (strict, `:182`). Best hypothesis = more inliers; ties go to the smaller residual sum (`:195`). A least-squares refit on the inliers is kept only if it loses no inlier (`:203-206`). `ok` requires at least 3 inliers (`:211`). Fixed iteration count, no adaptive stopping. |
| Transform convention | **pts1 ≈ R·pts2 + t**, i.e. x_i = R x_j + t with pts1 from frame i, pts2 from frame j: T_{i←j}. | **pts1 ≈ R·pts2 + t** for all four functions (header comment `include/tracking/icp.hpp:28`; edge error `src/icp.cpp:74`; `t_ = p1 − R p2` `:55`; `t = c1 − R c2` `:164`): T_{i←j}. |

Both variants therefore return the *native* ICP direction T_{i←j}. This is the opposite of PnP and ground truth
(x_j = R x_i + t, T_{j←i}). Every caller inverts the result once: R' = Rᵀ, t' = −Rᵀt
(e.g. `tests/synthetic/slam_trajectory_test.cpp:235-236`, `tests/tum/tum_trajectory_eval.cpp:224`).

**Practical consequence of the teammate variant.** Any program linked against `tracking` that reports an
"SVD vs g2o" difference now compares two identical SVD results. This applies to `icp_test`,
`icp_synthetic_test`, `slam_trajectory_test`, `trajectory_validation_test` and `trajectory_diagnostics_test`.
The difference is therefore zero, and the "after g2o" statistics equal the "before" ones. The
comments in those files still describe a g2o refinement.

**Why they are never linked together.** Both `.cpp` files define `pose_estimation_3d3d` and `bundleAdjustment`
with identical signatures (global namespace, `const std::vector<cv::Point3f>&, const std::vector<cv::Point3f>&,
cv::Mat&, cv::Mat&`). Linking both would give duplicate-symbol errors (or an ODR violation). Both
also provide a header with the same include name `tracking/icp.hpp`. `reference/icp_main_variant/CMakeLists.txt`
keeps them apart as follows:
1. It builds a separate library `tracking_main_icp`. This library recompiles the reference `pnp.cpp`,
   `essential_matrix.cpp`, `fast.cpp`, `brief.cpp`, `hamming_matching.cpp` together with main's `src/icp.cpp`
   (`:13-19`), instead of linking `tracking`. It does not link `rgbd_icp` (`:23-25`).
2. Every main-variant executable links `camera features geometry tracking_main_icp` and never `tracking`
   (`add_main_icp_exe`, `:28-32`). `camera`, `features` and `geometry` contain no ICP code.
3. `target_include_directories(... BEFORE ...)` puts main's `include/` ahead of the reference `include/`
   (`:21-22`, `:30`). As a result, `#include "tracking/icp.hpp"` resolves to main's header (which declares
   `RigidRansacResult`, `EstimateRigidRansac`, `FitRigid`) in those targets.
4. The whole subdirectory is optional: `BUILD_MAIN_ICP_VARIANT` (default ON,
   `reference/opencv_slambook2/CMakeLists.txt:147-150`).

Which executable uses which variant: **teammate** (`tracking`): `icp_test`, `icp_synthetic_test`,
`slam_trajectory_test`, `trajectory_validation_test`, `trajectory_diagnostics_test` (and every other
`add_test_exe` target, which links `tracking` but does not call ICP). **Main** (`tracking_main_icp`): `icp_rigid_test`,
`tum_trajectory_eval`, `slam_trajectory_test_main_icp`, `trajectory_validation_test_main_icp`.

---

### Warning: `slam_trajectory_test` writes into its dataset directory

The teammate version `reference/opencv_slambook2/tests/synthetic/slam_trajectory_test.cpp` has **no
output-directory option**. Its first argument is the dataset directory, which defaults to `../data/synthetic_bunny`
(`:356`). After processing, it unconditionally writes three files there:
- `<dataset>/pnp_trajectory.txt` (`:494-497`)
- `<dataset>/icp_trajectory.txt` (`:498-501`)
- `<dataset>/slam_trajectory.csv` (`:505-522`)

`--export-dir` only *adds* `pair_metrics.csv` and `pair_<i>_<j>_*.csv` files in another directory (`:524-533`). It
does not redirect the three trajectory files. `--export-dir` does not create its directory either: no
`create_directories` call; `ofstream` fails silently. All three files are tracked in git
(`git ls-files data/synthetic_bunny`). Running the program on `data/synthetic_bunny` therefore overwrites committed
data. Also, this build does not reproduce the committed trajectories: on 2026-11-06, 35 of the 36 rows of
`pnp_trajectory.txt`, `icp_trajectory.txt` and `slam_trajectory.csv` differed (book ORB/ICP before the feature migration;
this build's ICP is the SVD-only teammate variant). **Always run it on a copy**, as the tooling does:
`cp -r data/synthetic_bunny $tmp/ds` (`tools/evaluation/run_reference_suite.sh`, `tools/pipeline_analysis/export_correspondences.sh`,
`tools/evaluation/run_diagnostics.py`). Those scripts then compare the copy's three files byte for byte with the
committed ones. `COMMANDS.md:60-62` and `README.md:283` show the direct command on `../data/synthetic_bunny`. The
main variant `slam_trajectory_test_main_icp` accepts `--out-dir DIR`, but without it it falls back to the dataset
directory as well (`reference/icp_main_variant/tests/slam_trajectory_test.cpp:510`). `trajectory_validation_test`
and `trajectory_diagnostics_test` *read* `pnp_trajectory.txt`/`icp_trajectory.txt` from the dataset directory.
Run them on the same copy, after `slam_trajectory_test`.

---

## `reference/opencv_slambook2/`

### `reference/opencv_slambook2/CMakeLists.txt`
- **Category:** build
- **Status:** Added by the root `CMakeLists.txt:132` only with `-DSLAM_WITH_OPENCV=ON`. Defines the libraries in the table above and 22 executables: `render_bunny_test`, `synthetic_pose_eval_test`, 11 `add_test_exe` programs, 7 unit programs (`orb_from_scratch_test`, `gauss_newton_curve_fit_test`, `g2o_curve_fit_test`, `essential_matrix_test`, `fast_test`, `brief_test`, `vo_pipeline_test`), `tum_dataset_test`, `tum_orb_correspondences`. The main-ICP subdirectory adds 4 more.
- **Purpose and why it exists:** The former root build of the OpenCV reference code, moved here in the reorganisation with "targets, sources and flags ... unchanged" (`:1-5`).
- **Key types and functions:** `add_test_exe(name folder)` builds `tests/<folder>/<name>.cpp` and links `camera features geometry tracking` (`:87-90`). `-msse4` for `tracking` and `orb_from_scratch_test` (`_mm_popcnt_u32`, `:56`, `:111`). `BUILD_MAIN_ICP_VARIANT` (`:147-150`). Sets `OpenGL_GL_PREFERENCE GLVND` to silence a g2o CMake warning (`:6`).
- **Inputs → outputs:** CMake configuration → targets in `build/`.
- **Dependencies:** `find_package(OpenCV, Eigen3, g2o CONFIG, Sophus)` (`:8-11`). It also uses `feature_core` (root `cmake/feature_core.cmake`) and `rgbd_icp` (`experiments/rgbd_icp/`, which the root adds first, `CMakeLists.txt:131`).
- **Used by:** root `CMakeLists.txt`.
- **Covered by tests:** n/a (no `add_test` here).
- **Before studying it, understand:** CMake PUBLIC/PRIVATE/INTERFACE linkage; how include directories propagate.

### `reference/opencv_slambook2/include/camera/camera.hpp`
- **Category:** reference implementation
- **Status:** Library `camera`. Included by almost every reference test.
- **Purpose and why it exists:** Shared pinhole helpers. `pixel2cam` appears identically in three slambook2 ch7 files and was extracted once (`:10-14`).
- **Key types and functions:** `PinholeCamera {fx, fy, cx, cy}` with `K()` (`:5-8`); `pixel2cam` (book, `:13`); `cam2pixel` (not book, used by the rasterizer, `:16-18`).
- **Inputs → outputs:** see camera.cpp.
- **Dependencies:** OpenCV core.
- **Used by:** `geometry.cpp`, `essential_matrix.cpp`, `rasterizer.hpp`, all integration/synthetic tests, `tum_trajectory_eval`.
- **Covered by tests:** indirectly (e.g. `essential_matrix_test`, `vo_pipeline_test`).
- **Before studying it, understand:** the pinhole model, normalised image coordinates.

### `reference/opencv_slambook2/src/camera/camera.cpp`
- **Category:** reference implementation
- **Status:** Library `camera`.
- **Purpose and why it exists:** Implements the two projections.
- **Key types and functions:** `pixel2cam`: ((u−cx)/fx, (v−cy)/fy) (`:4-9`). `cam2pixel`: (fx X/Z + cx, fy Y/Z + cy) (`:12-17`). No distortion, no check for Z ≤ 0.
- **Inputs → outputs:** pixel + K → normalised coordinates; camera point + K → pixel.
- **Dependencies:** OpenCV core.
- **Used by:** as above.
- **Covered by tests:** indirectly.
- **Before studying it, understand:** intrinsic matrix K.

### `reference/opencv_slambook2/include/features/features.hpp`
- **Category:** reference implementation
- **Status:** Library `features`. Used by the integration/synthetic tests and `tum_trajectory_eval`.
- **Purpose and why it exists:** The book's `find_feature_matches()` signature (slambook2 ch7 `orb_cv.cpp`, `pose_estimation_*.cpp`, `triangulation.cpp`) with the book's filter, distance ≤ max(2·d_min, 30) (`:7-12`).
- **Key types and functions:** `find_feature_matches(img_1, img_2, kp1, kp2, matches, all_matches = nullptr)` (`:15-20`). `all_matches` receives every nearest-neighbour match before the filter.
- **Inputs → outputs:** two 8-bit images → OpenCV keypoints and `DMatch`es.
- **Dependencies:** OpenCV core/features2d (types only).
- **Used by:** see features.cpp.
- **Covered by tests:** no dedicated test; exercised by `feature_matching_test` (demo, no verdict).
- **Before studying it, understand:** `cv::KeyPoint`, `cv::DMatch` (queryIdx → image 1, trainIdx → image 2).

### `reference/opencv_slambook2/src/features/features.cpp`
- **Category:** reference implementation
- **Status:** Library `features` (links `feature_core` PRIVATE, `CMakeLists.txt:27`).
- **Purpose and why it exists:** **An OpenCV-facing wrapper around the production `feature_core`.** No OpenCV detector or descriptor is used. Images are converted into `features::GrayImage`, the project-owned modules run, and the results are converted back into `cv::KeyPoint`/`cv::DMatch` so the reference callers stay unchanged. The comment at `:19-22` describes this library boundary.
- **Key types and functions:**
  - `ToGray` rejects non-8-bit images (`:27-30`).
  - `Extract`: `features::DetectFast` with default `features::FastParams()` (threshold 30, FAST-9, border 16, strongest 500: `include/slam/features/fast.hpp:14-18`), then `AssignOrientations` (intensity centroid), then `ComputeBrief` on `GaussianSmooth(gray)` (`:32-37`).
  - `find_feature_matches` (`:60-78`): `MatchBruteForce`, then `FilterMatchesByDistance(raw, 30, &d_min, &d_max)` (`:68-70`). It prints max/min distance.
  - Keypoints are converted with size 31 (= 2·`kBriefPatchRadius`+1), angle in degrees in [0, 360), response = FAST score, octave 0 (`:39-49`).
  - Single scale only. This is not the production default front end (Harris ranking, 1400 features, FAST 10, mutual + ratio matching), which lives in `scratch_core`.
- **Inputs → outputs:** two `cv::Mat` (gray or colour, 8-bit) → keypoints, filtered matches, optional raw matches.
- **Dependencies:** OpenCV core/features2d, `feature_core` (`features/fast.hpp`, `orientation.hpp`, `brief.hpp`, `matcher.hpp`).
- **Used by:** `feature_matching_test`, `two_view_pose_test`, `triangulation_test`, `pnp_test`, `icp_test`, `pnp_synthetic_test`, `pnp_multiframe_test`, `icp_synthetic_test`, `slam_trajectory_test`(+`_main_icp`), `trajectory_validation_test`(+`_main_icp`), `trajectory_diagnostics_test`, `tum_trajectory_eval`.
- **Covered by tests:** the underlying modules by `tests/unit/features_*_test` (production); the wrapper itself has none.
- **Before studying it, understand:** the production feature modules in `include/slam/features/` (FAST, orientation, rotated BRIEF, Hamming matching).

### `reference/opencv_slambook2/include/geometry/geometry.hpp`
- **Category:** reference implementation
- **Status:** Library `geometry`.
- **Purpose and why it exists:** The book's 2D–2D functions from `pose_estimation_2d2d.cpp` / `triangulation.cpp` (`:6-21`).
- **Key types and functions:** `pose_estimation_2d2d(kp1, kp2, matches, R, t)`; `triangulation(kp1, kp2, matches, R, t, points)`.
- **Inputs → outputs:** see geometry.cpp.
- **Dependencies:** OpenCV core.
- **Used by:** `two_view_pose_test`, `triangulation_test`.
- **Covered by tests:** none (demos only).
- **Before studying it, understand:** epipolar geometry, E = [t]ₓR.

### `reference/opencv_slambook2/src/geometry/geometry.cpp`
- **Category:** reference implementation
- **Status:** Library `geometry`.
- **Purpose and why it exists:** Book code. The book's `cv::findFundamentalMat` call was replaced by the project's Eigen 8-point estimator.
- **Key types and functions:**
  - `pose_estimation_2d2d` (`:14-58`). K is **hard-coded** to the TUM Freiburg2 values (520.9, 521.0, 325.1, 249.7) (`:18`). These happen to equal the synthetic Bunny intrinsics.
  - F comes from `geometry_scratch::EstimateFundamental8Point` (production `include/slam/geometry/fundamental.hpp`). It throws `std::runtime_error` if degenerate (`:34-36`) and is **only printed**.
  - E comes from OpenCV `findEssentialMat(points1, points2, 521, (325.1, 249.7))` with OpenCV's default robust method (`:46-47`). H from `findHomography(..., RANSAC, 3)` is only printed (`:51`).
  - `recoverPose` (`:54-55`) gives R, t with x₂ = R x₁ + t and unit-norm t.
  - `triangulation` (`:60-87`): P1 = [I|0], P2 = [R|t] (float), normalised coordinates via `pixel2cam`, `cv::triangulatePoints`, dehomogenise.
- **Inputs → outputs:** keypoints + matches → R, t (scale-free); triangulated points in frame-1 coordinates, in units of ‖t‖ = 1.
- **Dependencies:** OpenCV calib3d (incl. legacy `calib3d_c.h`), Eigen, `camera`, production `geometry/fundamental.hpp`.
- **Used by:** `two_view_pose_test`, `triangulation_test`.
- **Covered by tests:** none.
- **Before studying it, understand:** 8-point algorithm, cheirality in `recoverPose`, linear triangulation.

### `reference/opencv_slambook2/include/optimization/g2o_types.hpp`
- **Category:** reference implementation
- **Status:** Header-only INTERFACE target `optimization`, linked PRIVATE by `tracking` and `tracking_main_icp`.
- **Purpose and why it exists:** The book's `VertexPose` is defined identically in `pose_estimation_3d2d.cpp` and `pose_estimation_3d3d.cpp` and is extracted once here (`:6-8`).
- **Key types and functions:** `VertexPose : g2o::BaseVertex<6, Sophus::SE3d>` (`:9-27`). `setToOriginImpl` sets identity. `oplusImpl` applies the left update T ← exp(δ)·T with δ = (ρ, φ) (Sophus order: translation first) (`:18-22`). `read`/`write` are stubs.
- **Inputs → outputs:** n/a.
- **Dependencies:** g2o core, Sophus.
- **Used by:** `src/tracking/pnp.cpp`, `reference/icp_main_variant/src/icp.cpp`. The teammate `icp.cpp` no longer uses it.
- **Covered by tests:** indirectly via `pnp_test` (demo).
- **Before studying it, understand:** Lie-group perturbation (left vs right), g2o vertices/edges.

### `reference/opencv_slambook2/include/tracking/pnp.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking` (and `tracking_main_icp`).
- **Purpose and why it exists:** Book `pose_estimation_3d2d.cpp` pose-only BA declarations (`:8-26`).
- **Key types and functions:** `VecVector2d`, `VecVector3d` (aligned allocators); `bundleAdjustmentGaussNewton`, `bundleAdjustmentG2O` (3D points in frame 1, pixels in frame 2, K, in/out `Sophus::SE3d pose`).
- **Inputs → outputs:** see pnp.cpp.
- **Dependencies:** OpenCV core, Eigen, Sophus.
- **Used by:** `pnp_test`. It is also included, without calls, by `pnp_synthetic_test`, `pnp_multiframe_test`, `slam_trajectory_test` and `trajectory_validation_test`.
- **Covered by tests:** none self-checking.
- **Before studying it, understand:** reprojection error, PnP.

### `reference/opencv_slambook2/src/tracking/pnp.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking` and `tracking_main_icp`.
- **Purpose and why it exists:** Book code: two pose-only reprojection-error minimisers.
- **Key types and functions:**
  - `bundleAdjustmentGaussNewton` (`:18-86`): 10 iterations (`:24`). Analytic 2×6 Jacobian (`:45-57`), H = ΣJᵀJ, b = −ΣJᵀe, `ldlt` solve, left update `exp(dx)·pose` (`:76`). Stops on NaN, on cost increase, or when ‖dx‖ < 1e-6.
  - `EdgeProjection` (`:89-127`): error = measurement − π(K·T·P).
  - `bundleAdjustmentG2O` (`:129-176`): `BlockSolver<6,3>`, dense linear solver, **Gauss–Newton**, vertex initialised to identity (`:145`), identity information, `optimize(10)`, verbose.
  - Neither function uses the input pose of `bundleAdjustmentG2O`, and both print to stdout. The pose maps frame-1 points into frame 2: x₂ = R x₁ + t.
  - Note: the PnP *solver* used by the trajectory demos is OpenCV `solvePnP`/`solvePnPRansac`, called in the test files, not anything in this file.
- **Inputs → outputs:** 3D points (frame 1), 2D pixels (frame 2), K → `Sophus::SE3d` T₂₁.
- **Dependencies:** g2o core/solver_dense, Sophus, OpenCV core, `optimization`.
- **Used by:** `pnp_test`.
- **Covered by tests:** none self-checking.
- **Before studying it, understand:** Gauss–Newton on SE(3), the 2×6 projection Jacobian (slambook2 §7.7).

### `reference/opencv_slambook2/include/tracking/icp.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking` (teammate variant). **Shadowed** in `tracking_main_icp` targets by main's header of the same name.
- **Purpose and why it exists:** Compatibility declarations for paired 3D points (target = pts1, source = pts2, `:6`).
- **Key types and functions:** `pose_estimation_3d3d(pts1, pts2, R, t)` (`:8-12`); `bundleAdjustment(points_3d, source_points_3d, R, t)` (`:14-18`). There is no RANSAC API. `:19` points to `tracking/rgbd_icp.hpp` / `icp_sequence` for full RGB-D ICP.
- **Inputs → outputs:** pts1 ≈ R pts2 + t (T_{i←j}).
- **Dependencies:** OpenCV core.
- **Used by:** `icp_test`, `icp_synthetic_test`, `slam_trajectory_test`, `trajectory_validation_test`, `trajectory_diagnostics_test`.
- **Covered by tests:** none self-checking. The underlying `rigidSVD` is tested by `rgbd_icp_test` (`experiments/rgbd_icp`).
- **Before studying it, understand:** the ICP comparison above.

### `reference/opencv_slambook2/src/tracking/icp.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking` only (teammate variant).
- **Purpose and why it exists:** The teammate replaced the book SVD and the g2o refinement with a thin wrapper around `rgbd_icp::rigidSVD`.
- **Key types and functions:** `pose_estimation_3d3d` (`:8-24`) and `bundleAdjustment` (`:29-33`, identical result, no optimisation). Details, constants and conventions are in the comparison table above.
- **Inputs → outputs:** paired `cv::Point3f` → `cv::Mat` R (3×3 double), t (3×1 double), pts1 ≈ R pts2 + t.
- **Dependencies:** `rgbd_icp` (`tracking/rgbd_icp.hpp`), OpenCV core. It does **not** use g2o, although `tracking` still links g2o for `pnp.cpp`.
- **Used by:** as icp.hpp.
- **Covered by tests:** `rgbd_icp_test` covers `rigidSVD`. Nothing tests this wrapper directly.
- **Before studying it, understand:** Kabsch/Umeyama SVD alignment; `experiments/rgbd_icp/src/rgbd_icp.cpp:44-63`.

### `reference/opencv_slambook2/include/tracking/essential_matrix.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking`.
- **Purpose and why it exists:** A from-scratch E estimation and pose recovery, written without OpenCV's essential-matrix calls (not book code) (`:7-13`).
- **Key types and functions:** `EstimateEssentialMatrix(pts1_cam, pts2_cam)`, which needs at least 8 points in normalised camera coordinates. `RecoverPoseFromEssential(E, pts1, pts2, R, t)`. `EstimateRelativePoseEssentialMatrix(kp1, kp2, matches, K, R, t)`. Convention x₂ = R x₁ + t, E = [t]ₓR, unit-norm t (`:8-13`).
- **Inputs → outputs:** see .cpp.
- **Dependencies:** OpenCV core/features2d types.
- **Used by:** `essential_matrix_test`, `vo_pipeline_test`.
- **Covered by tests:** `essential_matrix_test`, `vo_pipeline_test` (see their notes on exit codes).
- **Before studying it, understand:** 8-point algorithm, the four (R, t) decompositions of E, cheirality.

### `reference/opencv_slambook2/src/tracking/essential_matrix.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking` and `tracking_main_icp`.
- **Purpose and why it exists:** Implementation of the above.
- **Key types and functions:**
  - `EstimateEssentialMatrix` (`:32-52`): each row is x₂x₁, x₂y₁, x₂, y₂x₁, y₂y₁, y₂, x₁, y₁, 1 (`:38`). e is the last right singular vector of A (`:41-42`). It then projects onto the essential manifold, singular values (s, s, 0) with s = mean of the first two (`:46-49`).
  - **No Hartley normalisation and no RANSAC.** "Normalized" in the header only means normalised camera coordinates. There is no size check: fewer than 8 points is only rejected in the wrapper (`:108`).
  - `RecoverPoseFromEssential` (`:54-103`): makes det U, det V positive (`:59-60`), W = [[0,−1,0],[1,0,0],[0,0,1]], candidates {UWVᵀ, UWᵀVᵀ} × {±u₃} (`:67-73`). For each candidate it solves d₂x₂ − R d₁x₁ = t by least squares for the two depths (`:85-88`) and counts points with both depths > 0. The candidate with the most positive points wins; it returns false if none has any (`:97`).
- **Inputs → outputs:** normalised correspondences → E (`cv::Mat` 3×3); R, t with ‖t‖ = 1.
- **Dependencies:** Eigen (Dense, SVD), OpenCV core, `camera`.
- **Used by:** as header.
- **Covered by tests:** `essential_matrix_test`, `vo_pipeline_test`.
- **Before studying it, understand:** SVD-based E decomposition (Hartley & Zisserman §9.6).

### `reference/opencv_slambook2/include/tracking/fast.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking`.
- **Purpose and why it exists:** The first from-scratch FAST detector, which uses OpenCV types. It is superseded by `feature_core`'s `features::DetectFast`. The two are distinct: this one is in the global namespace and takes a `cv::Mat`.
- **Key types and functions:** `FastParams {threshold = 30, contiguous = 9}` (`:10-13`); `DetectFast(gray, params)` (`:15`). The header says single scale, **no non-maximum suppression**, no pyramid (`:7-9`).
- **Inputs → outputs:** 8-bit gray `cv::Mat` → `cv::KeyPoint`s.
- **Dependencies:** OpenCV core/features2d.
- **Used by:** `fast_test`, `vo_pipeline_test`.
- **Covered by tests:** `fast_test`.
- **Before studying it, understand:** the FAST segment test.

### `reference/opencv_slambook2/src/tracking/fast.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking`, `tracking_main_icp`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** 16-pixel radius-3 circle offsets (`:11-12`). `IsCorner` labels each circle pixel +1/−1/0 against centre ± threshold and looks for a run of ≥ `contiguous` equal non-zero labels, with wrap-around (`:14-34`). `DetectFast` scans with a 3-pixel border, keypoint size 7, no score (`:38-48`).
- **Inputs → outputs:** as header.
- **Dependencies:** OpenCV core.
- **Used by:** `fast_test`, `vo_pipeline_test`.
- **Covered by tests:** `fast_test`.
- **Before studying it, understand:** FAST-9.

### `reference/opencv_slambook2/include/tracking/brief.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking`.
- **Purpose and why it exists:** An early from-scratch, **unrotated** BRIEF (not ORB's steered BRIEF), superseded by `feature_core`.
- **Key types and functions:** `BriefDescriptor = std::vector<uint32_t>` (256 bits) (`:13`); `kBriefPatchHalfSize = 13` (`:15`); `ComputeBrief(gray, keypoints)` (`:20-21`). Border keypoints get an empty descriptor (`:17-19`).
- **Inputs → outputs:** gray image + keypoints → descriptors.
- **Dependencies:** OpenCV core/features2d.
- **Used by:** `brief_test`, `vo_pipeline_test`.
- **Covered by tests:** `brief_test`.
- **Before studying it, understand:** BRIEF binary tests.

### `reference/opencv_slambook2/src/tracking/brief.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking`, `tracking_main_icp`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** The pattern is 256 point pairs, uniform in [−13, 13]², from `mt19937(12345)`, generated once (`:12-24`). Bit i is set if I(p₁) < I(p₂) (`:46`). There is no smoothing and no orientation. The border test is `x < 13 || x >= cols−13` and the same for y (`:35-36`).
- **Inputs → outputs:** as header.
- **Dependencies:** OpenCV core, `<random>`.
- **Used by / Covered by tests:** `brief_test`, `vo_pipeline_test`.
- **Before studying it, understand:** BRIEF.

### `reference/opencv_slambook2/include/tracking/hamming_matching.hpp`
- **Category:** reference implementation
- **Status:** Library `tracking`.
- **Purpose and why it exists:** Hamming distance and a nearest-neighbour matcher for the early BRIEF.
- **Key types and functions:** `HammingDistance(a, b)` (`:13`); `MatchHamming(desc1, desc2, max_distance = 60)`, with no ratio test or cross-check (`:15-21`). The header says matches are kept "below max_distance", but the code keeps distance **≤** `max_distance` (`hamming_matching.cpp:30`).
- **Inputs → outputs:** descriptor lists → `cv::DMatch`.
- **Dependencies:** OpenCV types.
- **Used by:** `vo_pipeline_test`.
- **Covered by tests:** indirectly by `vo_pipeline_test`.
- **Before studying it, understand:** XOR/popcount.

### `reference/opencv_slambook2/src/tracking/hamming_matching.cpp`
- **Category:** reference implementation
- **Status:** Compiled into `tracking`, `tracking_main_icp` (`-msse4`).
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** `_mm_popcnt_u32(a[i]^b[i])` summed (`:8-14`). The brute-force loop skips empty descriptors and keeps the first strictly smaller distance (`:16-35`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header; SSE4.2.
- **Before studying it, understand:** brute-force matching.

### `reference/opencv_slambook2/include/render/mesh.hpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Triangle-mesh container for the synthetic Bunny renderer (not book code).
- **Key types and functions:** `Mesh {vertices, faces, face_normals}` (`:6-10`); `ComputeFaceNormals` (`:13`); `Aabb` with `Center()` and `BoundingSphereRadius() = ½‖max−min‖` (`:15-19`); `ComputeAabb` (`:21`).
- **Inputs → outputs:** see mesh.cpp.
- **Dependencies:** Eigen.
- **Used by:** `ply_mesh_io`, `rasterizer`, `render_bunny_test`.
- **Covered by tests:** none (exercised by `render_bunny_test`).
- **Before studying it, understand:** indexed triangle meshes.

### `reference/opencv_slambook2/src/render/mesh.cpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** normal = normalize((v1−v0)×(v2−v0)), or zero if the norm is ≤ 1e-12 (`:7-20`); component-wise min/max for the box (`:22-31`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header.
- **Before studying it, understand:** cross products, winding order.

### `reference/opencv_slambook2/include/render/ply_mesh_io.hpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Loads the Stanford Bunny (`data/meshes/bunny/bun_zipper.ply`) and writes coloured PLY scenes.
- **Key types and functions:** `Rgb8` (`:10-12`); `ReadPLYMesh(path, mesh, error)`, which supports ASCII and binary little-endian, skips extra properties and fan-triangulates polygons (`:14-21`); `WritePLYMesh(path, vertices, faces, colors)`, ASCII (`:23-28`).
- **Inputs → outputs:** PLY file ↔ `Mesh`.
- **Dependencies:** Eigen.
- **Used by:** `render_bunny_test`, `visualization.hpp`.
- **Covered by tests:** none.
- **Before studying it, understand:** the PLY format.

### `reference/opencv_slambook2/src/render/ply_mesh_io.cpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** Header parser for `format`/`element`/`property [list]`/`end_header` (`:102-148`). Binary scalars are read by type width (`:24-82`). Vertex x/y/z are located by name (`:160-184`). The face element's first property must be a list (`:186-189`), and polygons become (0, k, k+1) fans (`:206`). Other elements are skipped (`:84-98`, `:209`). Big-endian files are rejected (`:150-154`). Normals are computed after loading (`:213`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header.
- **Before studying it, understand:** binary file parsing.

### `reference/opencv_slambook2/include/render/rasterizer.hpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** CPU software rasteriser that produced the synthetic RGB-D Bunny dataset.
- **Key types and functions:** `RenderTarget {rgb (CV_8UC3, BGR), depth_m (CV_32F, background +inf)}` (`:9-12`). `Light {direction_world (−0.3,−1,−0.5), ambient 0.4, diffuse 0.6}` (`:14-21`). `RenderOptions {background 30, albedo 180, near 0.01 m, backface_cull true}` (`:23-28`). `RenderMesh(mesh, cam, w, h, R_wc, t_wc, light, opts)` with **x_cam = R·x_world + t**: despite the `_wc` names, the arguments are world→camera (`:30-34`). `DepthMetersToTUM16U(depth, scale = 5000)` (`:36-39`).
- **Inputs → outputs:** mesh + camera + extrinsics → colour and metric depth images.
- **Dependencies:** OpenCV core, Eigen, `camera`.
- **Used by:** `render_bunny_test`.
- **Covered by tests:** none.
- **Before studying it, understand:** z-buffering, barycentric interpolation. Note that the comments at `:31` and `:37-38` cite `ARCHITECTURE.md` and `tests/pnp_test.cpp`/`tests/icp_test.cpp`. These paths predate the current layout: no `ARCHITECTURE.md` exists any more (the architecture is in `docs/architecture/`), and the tests are now under `reference/opencv_slambook2/tests/integration/`.

### `reference/opencv_slambook2/src/render/rasterizer.cpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:**
  - A triangle is skipped if any vertex has z ≤ near; there is no clipping (`:40`).
  - Back faces are culled when n·centroid ≥ 0 (`:46`).
  - Bounding box, screen-space barycentric weights at pixel centres (x+0.5, y+0.5) (`:64-81`). Perspective-correct depth by interpolating 1/z (`:85-86`). Z-test (`:89`).
  - Flat Lambert shading per face: ambient + diffuse·max(0, −n·l), clamped to 1 (`:69-73`).
  - `DepthMetersToTUM16U`: round(z·5000), clamped to 65535; 0 for invalid (`:100-111`).
  - The pixel-centre convention (x+0.5) differs from OpenCV's integer pixel centres. `trajectory_diagnostics_test` measures this with a half-pixel principal-point variant (`trajectory_diagnostics_test.cpp:413-422`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header.
- **Before studying it, understand:** rasterisation pipeline.

### `reference/opencv_slambook2/include/render/trajectory.hpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Orbit camera path for the synthetic dataset.
- **Key types and functions:** `OrbitTrajectoryParams {center, radius 0.5, height 0.1, world_up (0,1,0), num_keyframes 60, start 0°, end 360°}` (`:7-15`). `GenerateOrbitTrajectory` returns **T_world_camera** (camera → world) (`:17-21`).
- **Inputs → outputs:** params → `std::vector<Sophus::SE3d>` T_wc.
- **Dependencies:** Eigen, Sophus.
- **Used by:** `render_bunny_test`.
- **Covered by tests:** none.
- **Before studying it, understand:** look-at camera construction.

### `reference/opencv_slambook2/src/render/trajectory.cpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** An in-plane basis is built from `up` and a reference axis (`:21-24`). Angles are sampled over [start, end) (`:27`). Centre C = centre + height·up + radius·(cos θ a + sin θ b) (`:31-32`). R_cw has columns (right, −cam_up, forward), so camera +Y points down (CV convention) (`:42-45`). The pose is stored as `SE3d(q, C)` (`:47-48`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header.
- **Before studying it, understand:** camera-to-world vs world-to-camera.

### `reference/opencv_slambook2/include/render/visualization.hpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Adds trajectory/frustum geometry into a PLY mesh. PLY edges are not reliably shown by viewers, so lines are drawn as thin quads (`:11-13`).
- **Key types and functions:** `AppendLineAsQuad`, `AppendFrustumWireframe(T_wc, cam, w, h, vis_near, ...)`, `AppendTrajectoryPolyline` (colour gradient) (`:14-31`).
- **Inputs → outputs:** poses → extra vertices/faces/colours.
- **Dependencies:** Eigen, Sophus, `camera`, `ply_mesh_io`.
- **Used by:** `render_bunny_test`.
- **Covered by tests:** none.
- **Before studying it, understand:** frustum corners from K⁻¹.

### `reference/opencv_slambook2/src/render/visualization.cpp`
- **Category:** reference implementation
- **Status:** Library `render`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:** A quad of width `thickness` perpendicular to the segment (`:7-29`). Frustum corners are the image corners back-projected to depth `vis_near` and mapped by T_wc (`:37-47`); 8 segments (`:49-56`). Polyline colours are linearly interpolated (`:59-73`).
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header.
- **Before studying it, understand:** —

### `reference/opencv_slambook2/include/dataset/tum.hpp`
- **Category:** reference implementation
- **Status:** Library `dataset` (Eigen + std only), "ported verbatim from main (P4)" (`CMakeLists.txt:138-140`).
- **Purpose and why it exists:** TUM RGB-D adapter: file lists, associate.py-style association, ground-truth interpolation, official intrinsics, depth conversion.
- **Key types and functions:**
  - The header comment gives the conventions (`:3-23`): `groundtruth.txt` is T_wc with X_world = R X_cam + t, and depth is raw/5000.
  - Types: `Stamped`, `GtSample`, `Pose` (T_wc), `Association`, `CameraIntrinsics` (default 640×480, `depth_factor` 5000, distortion stored but "not applied", `:72-78`), `Frame`, `SelectionReport`.
  - Functions: `ReadFileList`, `ReadGroundtruth`, `Associate`, `QuaternionToR` (qx, qy, qz, qw order), `InterpolateGroundtruth`, inline `DepthMetres` (0 → 0), `ValidDepth` (`:68-70`), `OfficialIntrinsics`, `SelectFrames`.
- **Inputs → outputs:** sequence directory → selected frames with absolute paths and interpolated T_wc.
- **Dependencies:** Eigen Core/Geometry.
- **Used by:** `tum_trajectory_eval`, `tum_dataset_test`.
- **Covered by tests:** `tum_dataset_test` (self-checking).
- **Before studying it, understand:** TUM file formats; quaternion slerp.

### `reference/opencv_slambook2/src/dataset/tum.cpp`
- **Category:** reference implementation
- **Status:** Library `dataset`.
- **Purpose and why it exists:** Implementation.
- **Key types and functions:**
  - `OpenLines` skips blank lines and `#` lines (`:14-27`). The lists are stable-sorted by time (`:44`, `:63`), and quaternions are normalised (`:60`).
  - `Associate` (`:67-90`) forms every pair with |dt| ≤ max_dt (O(N·M)). It sorts by (dt, t_rgb, t_depth), assigns greedily with each image used at most once, and finally sorts by RGB time.
  - `InterpolateGroundtruth` (`:96-118`) fails outside the span or when the bracketing samples are more than `max_gap` apart. It returns an exact sample unchanged, otherwise interpolates translation linearly and rotation by slerp (Eigen takes the shortest arc).
  - `OfficialIntrinsics` (`:120-141`) matches a substring of the sequence name: freiburg1 (517.3, 516.5, 318.6, 255.3), freiburg2 (520.9, 521.0, 325.1, 249.7), freiburg3 (535.4, 539.2, 320.1, 247.6, no distortion) (`:127-129`).
  - `SelectFrames` (`:143-186`) counts and records missing images and missing ground truth. `stride` applies to usable frames (`:176`); `max_frames` stops early.
- **Inputs → outputs / Dependencies / Used by / Covered by tests:** as header; `<filesystem>`.
- **Before studying it, understand:** TUM `associate.py`.

### `reference/opencv_slambook2/tests/report_utils.hpp`
- **Category:** reference test/demo (shared helper)
- **Status:** Header included as `"../report_utils.hpp"` by the tests in `tests/{unit,integration,synthetic}/` and, through an extra include directory, by main's two trajectory tests (`reference/icp_main_variant/CMakeLists.txt:42-44`).
- **Purpose and why it exists:** Reporting helpers shared across tests; "Contains no SLAM/book algorithm logic" (`:3-6`).
- **Key types and functions:** `Stats`/`ComputeStats` (min, max, mean, median, RMS) (`:18-40`); `PrintStats`; `RotationAngleDeg(R1, R2) = acos((tr(R1ᵀR2)−1)/2)` (`:49-54`); `TranslationDiff` (L2) (`:56-58`); `EigenRotToCv`, `EigenVecToCv`; `Rgb`; `WritePLY` (points + colours, ASCII) (`:76-90`).
- **Inputs → outputs:** —
- **Dependencies:** Eigen, OpenCV core.
- **Used by:** most reference tests.
- **Covered by tests:** none.
- **Before studying it, understand:** rotation angle from trace.

### `reference/opencv_slambook2/tests/unit/brief_test.cpp`
- **Category:** reference test/demo (self-checking; exit 0/1)
- **Status:** `brief_test` (links `tracking`). Run by `tools/evaluation/run_reference_suite.sh:47`.
- **Purpose and why it exists:** Tests the early `ComputeBrief` on a synthetic ramp image.
- **Key types and functions:** Checks determinism, a 256-bit size, a discriminative Hamming distance in (20, 236), and an empty descriptor at the border (1, 1) (`:28-44`). The size is printed but not asserted (`:32`).
- **Inputs → outputs:** none → PASS/FAIL, exit code (`:49`).
- **Dependencies:** OpenCV core, `tracking`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** brief.cpp.

### `reference/opencv_slambook2/tests/unit/essential_matrix_test.cpp`
- **Category:** reference test/demo
- **Status:** `essential_matrix_test` (links `camera tracking`). Run by `run_reference_suite.sh:48` as "self-check".
- **Purpose and why it exists:** Exact synthetic check of `EstimateEssentialMatrix` + `RecoverPoseFromEssential`: 60 points, depths 3–8, R about 0.2 rad, t = (0.3, −0.05, 0.1), `mt19937(42)` (`:27-40`).
- **Key types and functions:** Rotation error and translation **direction** error, each with a 1° threshold (`:71-72`).
- **Inputs → outputs:** none → printed verdicts. **It returns 0 even when a check prints FAIL** (`:74`). Only a cheirality failure exits 1 (`:49-52`). The exit status used by `run_reference_suite.sh` is therefore not a real verdict.
- **Dependencies:** Eigen, OpenCV core, `camera`, `tracking`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** essential_matrix.cpp.

### `reference/opencv_slambook2/tests/unit/fast_test.cpp`
- **Category:** reference test/demo (self-checking)
- **Status:** `fast_test` (links `tracking`). `run_reference_suite.sh:46`.
- **Purpose and why it exists:** The early `DetectFast` must fire within 2 px of the 4 corners of a 20×20 white square and not at the flat interior (40, 40) (`:16-55`).
- **Key types and functions:** threshold 30, contiguous 9.
- **Inputs → outputs:** none → exit 0/1 (`:57`).
- **Dependencies:** OpenCV core, `tracking`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** fast.cpp.

### `reference/opencv_slambook2/tests/unit/g2o_curve_fit_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `g2o_curve_fit_test` (self-contained; OpenCV, Eigen, g2o). `run_reference_suite.sh:58` ("demo").
- **Purpose and why it exists:** slambook2 ch6 `g2oCurveFitting.cpp`: fit y = exp(ax² + bx + c) with g2o.
- **Key types and functions:** `CurveFittingVertex` (3-D), `CurveFittingEdge` (1-D error, analytic Jacobian) (`:22-72`). True (1, 2, 1), initial (2, −1, 5), N = 100, σ = 1, `cv::RNG` (`:75-87`). Gauss–Newton. Not book code: 10 single `optimize(1)` calls record χ² per iteration (`:120-132`).
- **Inputs → outputs:** none → writes `../results/runs/optimization/g2o_cost.csv` (`:143-149`); always exit 0.
- **Dependencies:** g2o core/stuff/solver_dense, Eigen, OpenCV core.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** g2o vertex/edge API.

### `reference/opencv_slambook2/tests/unit/gauss_newton_curve_fit_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `gauss_newton_curve_fit_test` (self-contained). `run_reference_suite.sh:57`.
- **Purpose and why it exists:** slambook2 ch6 `gaussNewton.cpp`, the same curve fitted with a hand-written Gauss–Newton.
- **Key types and functions:** H += JJᵀ/σ², b += −eJ/σ², `ldlt` solve, at most 100 iterations; stops on NaN or cost increase (`:30-77`).
- **Inputs → outputs:** none → `../results/runs/optimization/gauss_newton_cost.csv` (`:87-93`); exit 0.
- **Dependencies:** Eigen, OpenCV (RNG; `opencv.hpp`).
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** Gauss–Newton normal equations.

### `reference/opencv_slambook2/tests/unit/orb_from_scratch_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `orb_from_scratch_test` (self-contained, `-msse4`). `run_reference_suite.sh:59`.
- **Purpose and why it exists:** slambook2 ch7 `orb_self.cpp` ("Created by xiang", `:2`): OpenCV FAST at threshold 40, then a hand-written ORB descriptor and brute-force matching.
- **Key types and functions:**
  - `ORB_pattern[256*4]` (`:88-345`).
  - `ComputeORB` (`:348-399`): intensity-centroid angle over a 16×16 patch (`half_patch_size` 8), rotated pattern, border `half_boundary` 16. The book comment at `:26` says 8 pixels, but the code uses 16.
  - `BfMatch`: d_max 40, keeps distance < 40 (`:402-423`).
- **Inputs → outputs:** reads `../data/tum_sample/{1,2}.png` (`:14-15`) → `../results/runs/orb/matches.png` (`:77-79`). Image loading is guarded only by `assert` (`:44`), which is compiled out in the default Release build.
- **Known bug (book code, not fixed here):** the rotated BRIEF pattern reaches a radius of 18.38 px, but `ComputeORB`
  rejects only keypoints within 16 px of the border, so keypoints near the bottom or right edge read up to 3 px past
  the image buffer. It segfaults intermittently (3 of 30 runs, depending on the memory layout); a border of 19 px
  would fix it.
- **Dependencies:** OpenCV (core, features2d, imgcodecs, highgui via `opencv.hpp`).
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** ORB (oriented FAST + steered BRIEF).

### `reference/opencv_slambook2/tests/unit/tum_dataset_test.cpp`
- **Category:** reference test/demo (self-checking)
- **Status:** `tum_dataset_test` (links `dataset`). It is not in `run_reference_suite.sh` and not in CTest, so it is run manually.
- **Purpose and why it exists:** Checks the TUM adapter on hand-made files in `temp_directory_path()/tum_dataset_test`, which it removes afterwards (`:35-36`, `:166`).
- **Key types and functions:** list parsing and errors (`:39-48`); association within 0.02 s, greedy closest, shuffled-input determinism (`:51-69`); quaternion and T_wc convention, T_{j←i} = T_wc[j]⁻¹T_wc[i] (`:72-97`); interpolation (midpoint, exact, out of span, gap, q vs −q shortest arc) (`:100-120`); depth /5000 (`:123-126`); intrinsics freiburg1/3, unknown sensor (`:129-138`); `SelectFrames` with a missing image and missing ground truth, stride, max_frames (`:141-165`).
- **Inputs → outputs:** none → exit 0/1 (`:168`).
- **Dependencies:** `dataset`, `<filesystem>`.
- **Used by:** —
- **Covered by tests:** n/a.
- **Before studying it, understand:** tum.hpp.

### `reference/opencv_slambook2/tests/unit/vo_pipeline_test.cpp`
- **Category:** reference test/demo
- **Status:** `vo_pipeline_test` (links `camera tracking`). `run_reference_suite.sh:49` ("self-check").
- **Purpose and why it exists:** End-to-end early from-scratch two-view VO: FAST → BRIEF → `MatchHamming(60)` → 8-point E → pose. It runs on two synthetic images of 5×4 square markers of varying size at depth 5 m. The truth is R = 0.05 rad, t = (0.3, −0.05, 0.15) (`:58-111`). The header explains why translation direction is expected to be rough: there is no RANSAC (`:9-28`).
- **Key types and functions:** `DrawMarker`; rotation and translation-direction errors are printed.
- **Inputs → outputs:** none → printed errors. It exits 1 only if there are fewer than 8 matches or a cheirality failure (`:128-144`). **No accuracy threshold is checked**; otherwise it returns 0 (`:160`).
- **Dependencies:** Eigen, OpenCV core, `camera`, `tracking`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** essential_matrix_test.cpp.

### `reference/opencv_slambook2/tests/integration/feature_matching_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `feature_matching_test` (add_test_exe). `run_reference_suite.sh:60` with `../data/tum_sample/{1,2}.png`.
- **Purpose and why it exists:** Adapted from slambook2 `orb_cv.cpp` `main()`, routed through `find_feature_matches` (`:1-2`). The comment says `features::find_feature_matches()`, but the function is in the global namespace.
- **Key types and functions:** statistics on Hamming distances and the retained percentage (`:35-39`); draws keypoints, all matches and good matches (`:45-52`).
- **Inputs → outputs:** `img1 img2` → `../results/runs/feature_matching/{keypoints,all_matches,good_matches}.png` (`:55-62`); exit 0.
- **Dependencies:** OpenCV highgui/imgcodecs/features2d, `features`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** features.cpp.

### `reference/opencv_slambook2/tests/integration/two_view_pose_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `two_view_pose_test`. `run_reference_suite.sh:61`.
- **Purpose and why it exists:** Adapted from slambook2 `pose_estimation_2d2d.cpp` `main()`: estimates the pose, checks E = t^R, and evaluates the epipolar residual y₂ᵀ[t]ₓR y₁ per match.
- **Key types and functions:** `kEpipolarResidualThreshold = 1e-2` (`:22`, used only to report a percentage). K is hard-coded to TUM fr2 (`:50`). It draws epipolar lines from F = K⁻ᵀ[t]ₓRK⁻¹ (`:74-98`).
- **Inputs → outputs:** `img1 img2` → `../results/runs/two_view_pose/epipolar_lines_img{1,2}.png`; exit 0.
- **Dependencies:** OpenCV calib3d/imgproc/highgui, `features`, `geometry`, `camera`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** geometry.cpp; epipolar constraint.

### `reference/opencv_slambook2/tests/integration/triangulation_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `triangulation_test`. `run_reference_suite.sh:62`.
- **Purpose and why it exists:** Adapted from slambook2 `triangulation.cpp` `main()`.
- **Key types and functions:** `get_color(depth)` (book, `:21-26`); per-match depth in both cameras (`:54-65`); cheirality percentage; camera centres (0 and −Rᵀt) (`:88-97`). Depths are in units of ‖t‖ = 1, not metres (`:107-109`).
- **Inputs → outputs:** `img1 img2` → `../results/runs/triangulation/{img1,img2}_reprojection.png`, `points_and_cameras.ply`; exit 0.
- **Dependencies:** OpenCV, `features`, `geometry`, `camera`, report_utils `WritePLY`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** triangulation, scale ambiguity.

### `reference/opencv_slambook2/tests/integration/pnp_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `pnp_test`. `run_reference_suite.sh:63` with the TUM sample images and depths.
- **Purpose and why it exists:** Adapted from slambook2 `pose_estimation_3d2d.cpp` `main()`. It back-projects frame-1 keypoints with depth/5000 and K = TUM fr2 (`:36-48`), then runs OpenCV `solvePnP` (iterative) (`:54-56`), `bundleAdjustmentGaussNewton` (`:74`) and `bundleAdjustmentG2O` (`:82`).
- **Key types and functions:** Both BA functions start from identity (default `Sophus::SE3d`), not from the `solvePnP` result. The file reports pairwise rotation/translation differences and reprojection statistics for the three solutions (`:89-131`).
- **Inputs → outputs:** `img1 img2 depth1 depth2` (depth2 is unused) → stdout; exit 0.
- **Dependencies:** OpenCV calib3d, `features`, `tracking` (pnp.cpp), Sophus.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** pnp.cpp.

### `reference/opencv_slambook2/tests/integration/icp_test.cpp`
- **Category:** reference test/demo (book demo)
- **Status:** `icp_test` (links `tracking` = **teammate ICP**). `run_reference_suite.sh:64`.
- **Purpose and why it exists:** Adapted from slambook2 `pose_estimation_3d3d.cpp` `main()`. It builds 3D–3D pairs from both depth maps (/5000, K = TUM fr2) (`:36-50`), then calls `pose_estimation_3d3d` (`:54`) followed by `bundleAdjustment` (`:69`).
- **Key types and functions:** It prints R, t and their inverse, checks p1 ≈ R p2 + t on 5 samples, and reports the residual "before (SVD) / after (g2o)" (`:83-111`). **The comments (`:1-3`, `:61-63`) and the printed labels say g2o, but with the teammate `bundleAdjustment` both results are the same SVD** (`src/tracking/icp.cpp:29-33`). The "diff SVD-vs-g2o" is therefore zero.
- **Inputs → outputs:** `img1 img2 depth1 depth2` → stdout; exit 0.
- **Dependencies:** OpenCV, `features`, `tracking`.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** the ICP comparison.

### `reference/opencv_slambook2/tests/synthetic/render_bunny_test.cpp`
- **Category:** reference test/demo (dataset generator)
- **Status:** `render_bunny_test` (links `render`). `run_reference_suite.sh:65` renders into a temporary directory (`--num-frames 36 --radius-scale 4 --height-scale 1`).
- **Purpose and why it exists:** The generator of the synthetic RGB-D Bunny dataset (`data/synthetic_bunny`, see its README).
- **Key types and functions:**
  - Defaults: 36 frames, 640×480, radius-scale 3, height-scale 0.5 (×bounding radius) (`:57-58`). The orbit is centred on the mesh bounding box (`:77-86`). The camera is fixed at (520.9, 521.0, 325.1, 249.7) (`:88`).
  - Each frame is rendered with T_cw = T_wc⁻¹ (`:106-111`) and saved as 16-bit depth at /5000.
  - `groundtruth.txt` stores T_wc in TUM format with timestamp = frame index (`:121-124`). `intrinsics.txt` holds "w h fx fy cx cy" (`:95-96`).
  - Note: `--width/--height` change the image size but not cx, cy.
- **Inputs → outputs:** `<mesh.ply> <output_dir> [options]` → `<output_dir>/{000000.png, 000000_depth.png, ..., groundtruth.txt, intrinsics.txt}` and `../results/runs/synthetic_bunny/scene_visualization.ply` (`:145-148`).
- **Dependencies:** `render`, OpenCV imgcodecs, Sophus.
- **Used by:** run_reference_suite.sh; documented in `data/synthetic_bunny/README.md`, `COMMANDS.md`.
- **Covered by tests:** none; `run_reference_suite.sh` renders into a temporary directory and compares the result byte-wise with `data/synthetic_bunny` (`dataset_regeneration.txt`).
- **Before studying it, understand:** the render module.

### `reference/opencv_slambook2/tests/synthetic/synthetic_pose_eval_test.cpp`
- **Category:** reference test/demo
- **Status:** `synthetic_pose_eval_test` (OpenCV, Eigen, Sophus only, `CMakeLists.txt:79-81`). `run_reference_suite.sh:69`.
- **Purpose and why it exists:** Prints the ground-truth relative pose between two frames in both conventions: T_ji = T_wc[j]⁻¹T_wc[i] (PnP, x_j = R x_i + t) and T_ij = T_ji⁻¹ (ICP native) (`:66-74`).
- **Key types and functions:** `ReadGroundtruth` (skips `#`), `PrintPose` (angle, ‖t‖).
- **Inputs → outputs:** `<groundtruth.txt> <i> <j>` → stdout; exit 1 on bad indices.
- **Dependencies:** OpenCV core, Eigen, Sophus.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** relative pose from absolute poses.

### `reference/opencv_slambook2/tests/synthetic/pnp_synthetic_test.cpp`
- **Category:** reference test/demo
- **Status:** `pnp_synthetic_test`. `run_reference_suite.sh:66` (`../data/synthetic_bunny 0 1`).
- **Purpose and why it exists:** Runs one Bunny pair through `find_feature_matches`, back-projection with frame-i depth (/5000), and OpenCV `solvePnP` (`:96-134`). The result is compared with the ground truth T_{j←i}. A diagnostic `solvePnPRansac(100, 8 px, 0.99)` follows (`:150-151`).
- **Key types and functions:** The header claims it tests "the EXISTING, unmodified PnP implementation (src/tracking/pnp.cpp...)" (`:1-3`). **In fact it calls only OpenCV `solvePnP`/`solvePnPRansac`; nothing in `pnp.cpp` is called.** `ReadGroundtruth` here does not skip `#` lines (`:51-52`). This is harmless for the committed `groundtruth.txt`, which has no comments.
- **Inputs → outputs:** `[dataset] [i] [j]` → stdout; exit 1 if fewer than 4 correspondences.
- **Dependencies:** OpenCV calib3d, `features`, Sophus.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** PnP convention x_j = R x_i + t.

### `reference/opencv_slambook2/tests/synthetic/pnp_multiframe_test.cpp`
- **Category:** reference test/demo
- **Status:** `pnp_multiframe_test`. `run_reference_suite.sh:68` (`0 10`).
- **Purpose and why it exists:** The same as `pnp_synthetic_test` over consecutive pairs from `start` to `end` (default 0 to 10) (`:132-135`). It tabulates errors of plain `solvePnP` and the RANSAC diagnostic, and counts pairs with rotation > 5° or translation > 5 cm (`:169-180`).
- **Key types and functions:** `RunPair` (`:70-127`). Like `pnp_synthetic_test`, the header's reference to "src/tracking/pnp.cpp's plain solvePnP call" is inaccurate: the call is OpenCV's.
- **Inputs → outputs:** `[dataset] [start] [end]` → stdout; exit 0.
- **Dependencies / Used by / Covered by tests / Before studying it:** as pnp_synthetic_test.

### `reference/opencv_slambook2/tests/synthetic/icp_synthetic_test.cpp`
- **Category:** reference test/demo
- **Status:** `icp_synthetic_test` (**teammate ICP**). `run_reference_suite.sh:67`.
- **Purpose and why it exists:** One Bunny pair through the fixed-correspondence ICP compatibility API (`:1-15`). The 3D–3D pairs use both depth maps (`:109-122`). It calls `pose_estimation_3d3d`, then `bundleAdjustment` (`:144`, `:149`), inverts once to the common convention (`:157`), and compares with ground truth.
- **Key types and functions:** `InvertPose` (`:74-77`). The header gives the convention derivation: native x_i = R x_j + t (`:5-15`). `ReadGroundtruth` does not skip `#`. The comments refer to `tests/icp_test.cpp`, which is now `tests/integration/icp_test.cpp`.
- **Inputs → outputs:** `[dataset] [i] [j]` → stdout; exit 1 if fewer than 3 pairs.
- **Dependencies:** OpenCV, `features`, `tracking`, Sophus.
- **Used by:** run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** the ICP comparison.

### `reference/opencv_slambook2/tests/synthetic/slam_trajectory_test.cpp`
- **Category:** reference test/demo (the reference trajectory pipeline)
- **Status:** `slam_trajectory_test` (links `tracking` = **teammate ICP**). It is run (always on a dataset copy) by `tools/evaluation/run_reference_suite.sh:70`, `tools/pipeline_analysis/export_correspondences.sh:26` and `tools/evaluation/run_diagnostics.py:160`. Its output files are read by `visualization/scripts/slam_coords.py` (and the Blender scene) and by `trajectory_validation_test`/`trajectory_diagnostics_test`.
- **Purpose and why it exists:** Builds full global PnP and ICP trajectories over the synthetic Bunny sequence. **Read the write-location warning above.**
- **Key types and functions:**
  - `RunPnpPair` (`:123-182`): `find_feature_matches` (with raw matches), the frame-i depth at the truncated pixel (/5000), then **`solvePnPRansac(..., false, 100, 8.0f, 0.99, inliers)`** (`:167`) gives T_{j←i}. It needs at least 4 correspondences.
  - `RunIcpPair` (`:191-244`): pairs with depth in both frames, then `pose_estimation_3d3d` + `bundleAdjustment` on **all** correspondences, with no outlier rejection (`:231-232`), then a single inversion to T_{j←i} (`:235-236`).
  - Accumulation `T_wc[i+1] = T_wc[i] · T_rel⁻¹`, frame 0 = GT frame 0; a failed pair holds the previous pose (`:405-424`). The header gives the derivation (`:21-30`).
  - Export-only `PairDetail`, `WritePairMetrics`, `WritePairDetail` (`:99-113`, `:269-351`); options `--export-dir`, `--export-pair K|all` (`:357-370`).
  - Header discrepancies: it says ICP's `bundleAdjustment` is "completely unmodified" book code (`:5-6`), but in this build it is the teammate's SVD wrapper. It cites "src/tracking/pnp.cpp's solvePnP call pattern" (`:3`), but `pnp.cpp` contains no PnP solver call.
- **Inputs → outputs:** `[dataset_dir=../data/synthetic_bunny] [--export-dir D] [--export-pair K|all]` → stdout tables. It **always writes** `<dataset>/{pnp_trajectory.txt, icp_trajectory.txt, slam_trajectory.csv}` (`:494-522`) and optionally `D/pair_metrics.csv`, `D/pair_<k>_<k+1>_{keypoints,raw_matches,correspondences}.csv`.
- **Dependencies:** OpenCV calib3d/imgcodecs, `features`, `tracking`, Sophus.
- **Used by:** see Status. `visualization/scripts/slam_coords.py:10` documents its pose convention (T_wc).
- **Covered by tests:** `trajectory_validation_test` checks its accumulation formula and output files; `trajectory_diagnostics_test` checks that a re-run reproduces them.
- **Before studying it, understand:** relative vs absolute poses; composition order; the ICP comparison.

### `reference/opencv_slambook2/tests/synthetic/trajectory_validation_test.cpp`
- **Category:** reference test/demo
- **Status:** `trajectory_validation_test` (**teammate ICP**). `run_reference_suite.sh:50` calls it "self-check" with `../data/synthetic_bunny` (read-only use).
- **Purpose and why it exists:** Independent checks of the accumulation maths and of the files written by `slam_trajectory_test`. It contains read-only duplicates of `RunPnpPair`/`RunIcpPair` (`:1-9`, `:145-206`).
- **Key types and functions:**
  - `AccumulateStep(T, T_rel) = T · T_rel⁻¹` (`:130-132`). Its comment cites "slam_trajectory_test.cpp lines 259/265"; the corresponding lines are now `:413/:419`.
  - Section 6: zero motion, tolerance 1e-9.
  - Section 7: perfect motion through ground truth, tolerances 1e-6 m / 1e-4°.
  - Section 8: identity anchor.
  - Sections 4/5: one and two real PnP steps.
  - Section 10: one ICP step.
  - Sections 9/10: file checks for `pnp_trajectory.txt`/`icp_trajectory.txt` (timestamps = index, frame 0 = GT, unit quaternions, a spot check of pair 0→1 against a fresh run with tolerance 1e-6 m) (`:217-407`).
- **Inputs → outputs:** `[dataset_dir]` → stdout PASS/FAIL lines. It **always returns 0** (`:410`), so the "self-check" exit status is not a verdict. It reads `<dataset>/pnp_trajectory.txt`, `icp_trajectory.txt`; it writes nothing.
- **Dependencies:** OpenCV, `features`, `tracking`, Sophus.
- **Used by:** run_reference_suite.sh; `COMMANDS.md:41,65`.
- **Covered by tests:** n/a.
- **Before studying it, understand:** slam_trajectory_test.cpp.

### `reference/opencv_slambook2/tests/synthetic/trajectory_diagnostics_test.cpp`
- **Category:** reference test/demo (diagnostic)
- **Status:** `trajectory_diagnostics_test` (**teammate ICP**). It is run by `tools/evaluation/run_diagnostics.py:166-168` (on the dataset copy, after `slam_trajectory_test`) and by `run_reference_suite.sh:71`.
- **Purpose and why it exists:** Repeats the reference per-pair calls and measures intermediate quantities against ground truth (`:1-20`).
- **Key types and functions:**
  - GT replay of the accumulation plus two wrong-formula controls (`:234-259`).
  - Per-correspondence values: GT reprojection error, depth consistency, `SilhouetteDistance` (Chebyshev distance to zero depth, cap 10) and `DepthJump` (`:96-121`, `:358-397`).
  - "Perfect correspondence" re-solves (`:399-411`).
  - A **half-pixel principal-point variant**, K with cx−0.5, cy−0.5 (`:227`, `:413-422`).
  - Subset re-solves: interior (silhouette distance ≥ 4) and GT-consistent (< 1 px or < 5 mm) (`:424-439`).
  - Re-accumulation compared with the trajectory files (`:500-518`).
  - `RunIcp` records "T_svd_native" and "T_g2o_native" (`:157-172`). With the teammate variant these are identical, so all `icp_g2o_*` columns equal `icp_svd_*` and `svd_vs_g2o_*` = 0.
- **Inputs → outputs:** `<dataset_dir> <out_dir>` → `<out_dir>/{gt_replay.csv, gt_replay_controls.csv, correspondence_diagnostics.csv, pair_solver_checks.csv, reaccumulation_check.csv}`. It reads `<dataset>/pnp_trajectory.txt`, `icp_trajectory.txt`. The output directory is not created by the program.
- **Dependencies:** OpenCV calib3d/imgcodecs, `features`, `tracking`, Sophus.
- **Used by:** `tools/evaluation/run_diagnostics.py` (results under `results/reports/diagnostics/`), run_reference_suite.sh.
- **Covered by tests:** n/a.
- **Before studying it, understand:** slam_trajectory_test.cpp; reprojection error.

### `reference/opencv_slambook2/tests/tum/tum_orb_correspondences.cpp`
- **Category:** reference test/demo (diagnostic)
- **Status:** `tum_orb_correspondences` (OpenCV + Eigen only, `CMakeLists.txt:147-148`). The dashboard (`tools/evaluation/dashboard_app/methods.py:348-379`, `data.py`) reads its output; the programme itself is run manually.
- **Purpose and why it exists:** Genuine OpenCV ORB correspondences on the pairs of an existing `tum_trajectory_eval` run, used to compare against the project features. It is separate from every trajectory (`:1-22`).
- **Key types and functions:**
  - `ReadFrames` reads the evaluator's `frames.csv`; the frame column must be 0, 1, 2, ... (`:54-93`). `ParsePairs` (`:95-116`).
  - Detector `cv::ORB::create(nfeatures = 500)` with OpenCV defaults (`:139`, `:186`); `BFMatcher(NORM_HAMMING)`; the same filter ≤ max(2·d_min, 30) (`:206-210`).
  - Truncated-pixel depth / `depth_factor` (5000) (`:217-222`), then `solvePnPRansac(100, 8 px, 0.99)` (`:228`).
  - Default intrinsics are freiburg1 (`:140`), with no undistortion.
  - It refuses an output directory inside the sequence's parent directory, i.e. `data/` (`:161-166`).
- **Inputs → outputs:** `SEQUENCE_DIR --frames-csv CSV --out DIR [--pairs] [--features] [--fx..] [--depth-factor]` → `DIR/{frames.csv, command.txt, summary.csv, pair_<i>_<j>/{matches.csv, keypoints_i.csv, keypoints_j.csv, pair.json}}`.
- **Dependencies:** OpenCV core/features2d/calib3d/imgcodecs, Eigen.
- **Used by:** `tools/evaluation/dashboard_app`.
- **Covered by tests:** `tools/evaluation/dashboard_app/test_methods.py` covers the dashboard's reading of the output, not the programme.
- **Before studying it, understand:** tum_trajectory_eval's `frames.csv`.

### `reference/opencv_slambook2/tests/tum/tum_trajectory_eval.cpp`
- **Category:** reference test/demo (the OpenCV TUM evaluator)
- **Status:** Built as **`tum_trajectory_eval` by `reference/icp_main_variant/CMakeLists.txt:36-37`** (main ICP + `dataset`). It is not built against the teammate `tracking`, and could not be, because it uses `RigidRansacResult`/`EstimateRigidRansac`. It is used by `tools/evaluation/run_tum_stress.py:68,166`, `tools/evaluation/tum_metrics.py`, `synthetic_runs.py`, `diagnostics/*`, the dashboard, and referenced in `apps/pipeline_main.cpp` and `include/slam/evaluation/baseline.hpp`.
- **Purpose and why it exists:** The reference pipeline (as in main's `slam_trajectory_test`) on a TUM RGB-D sequence, through the dataset adapter (`:1-13`).
- **Key types and functions:**
  - `RunPair` (`:108-231`). `find_feature_matches` is silenced. PnP uses the frame-i depth, then `solvePnPRansac(100, 8 px, 0.99)` (`:152`).
  - ICP needs depth in both frames. `EstimateRigidRansac(p1, p2)` uses the defaults 1 cm / 300 / 12345 (`:212`). Then `bundleAdjustment` runs **on the inliers** (`:221`): main's g2o LM, which starts from identity, so the RANSAC pose is used only to select inliers. The result is inverted once (`:224`).
  - `AcceptPnp` rejects a returned PnP pose if its inliers contain fewer than `--pnp-min-distinct-inliers` distinct keypoints in either frame. The default is **6**; 0 turns the check off (`:240-248`, `:455`).
  - Convention control: the PnP inliers are projected with the GT pose and with the deliberately wrong T_cw reading (`:194-206`, `:518-522`).
  - Accumulation as in the synthetic test; a failed or rejected pair holds the pose (`:531-536`).
  - Opt-in `--export-pairs` diagnostics (`WriteDiagnostics`, `:311-440`) include export-only `solvePnP` refits on the inliers (`:157-182`).
  - Defaults: 20 frames, stride 1, max-dt 0.02 s, max-gt-gap 0.05 s, out `../results/runs/tum/<sequence>` (`:451-475`). It refuses to write into the data directory (`:477-482`). Intrinsics come from `OfficialIntrinsics` (exit 2 if unknown); distortion is not corrected.
- **Inputs → outputs:** `SEQUENCE_DIR [options]` → `<out>/{frames.csv, pairs.csv, trajectory.csv, summary.txt}` (+ `diagnostics/pair_<i>_<i+1>/`).
- **Dependencies:** OpenCV calib3d/highgui, Sophus, `dataset`, `features`, `camera`, `tracking_main_icp`.
- **Used by:** `tools/evaluation/` (stress runs, TUM metrics, dashboard, diagnostics).
- **Covered by tests:** no C++ test. `tools/evaluation/test_pose_conventions.py` and `diagnostics/test_diagnostics.py` test the Python readers of its output.
- **Before studying it, understand:** tum.hpp; main ICP variant; slam_trajectory_test.cpp.

---

## `reference/icp_main_variant/`

### `reference/icp_main_variant/CMakeLists.txt`
- **Category:** build
- **Status:** Added by `reference/opencv_slambook2/CMakeLists.txt:147-150` when `BUILD_MAIN_ICP_VARIANT=ON` (the default), so it is only built with `-DSLAM_WITH_OPENCV=ON`.
- **Purpose and why it exists:** Preserves main's ICP verbatim ("implementation worktree, branch main, uncommitted state of 2026-11-04", `:1-4`) beside the teammate's, without ever linking the two (`:6-8`).
- **Key types and functions:** library `tracking_main_icp` (`:13-25`); `add_main_icp_exe` (`:28-32`). Targets: `icp_rigid_test` (`:34`), `tum_trajectory_eval` (`:36-37`, + `dataset`), `slam_trajectory_test_main_icp`, `trajectory_validation_test_main_icp` (`:40-41`). The last two get the reference `tests/synthetic` include directory so that their `"../report_utils.hpp"` resolves to `reference/opencv_slambook2/tests/report_utils.hpp` (`:42-44`). The separation mechanism is explained in the ICP comparison above.
- **Inputs → outputs:** —
- **Dependencies:** variables and targets inherited from the parent (`camera`, `features`, `geometry`, `optimization`, `dataset`, OpenCV, g2o, Sophus).
- **Used by:** parent CMakeLists.
- **Covered by tests:** n/a.
- **Before studying it, understand:** CMake `BEFORE` include ordering.

### `reference/icp_main_variant/include/tracking/icp.hpp`
- **Category:** reference implementation
- **Status:** Public header of `tracking_main_icp`, placed BEFORE the reference include path.
- **Purpose and why it exists:** Main's ICP API: book declarations plus the robust project code.
- **Key types and functions:** `pose_estimation_3d3d` (`:10-14`). `bundleAdjustment(points_3d, points_2d, R, t)` (`:16-20`): the second parameter is named `points_2d` (book naming), although it takes 3D points. `RigidRansacResult {ok, R, t, inliers, num_inliers}` (`:30-36`). `EstimateRigidRansac(pts1, pts2, threshold_m = 0.01, iterations = 300, seed = 12345)` (`:37-38`). `FitRigid(pts1, pts2, idx, R, t)` (`:43-44`). Convention pts1 = R·pts2 + t (`:28`, `:40`).
- **Inputs → outputs:** see icp.cpp.
- **Dependencies:** OpenCV core, Eigen.
- **Used by:** `icp_rigid_test`, `tum_trajectory_eval`, `slam_trajectory_test_main_icp`, `trajectory_validation_test_main_icp`.
- **Covered by tests:** `icp_rigid_test`.
- **Before studying it, understand:** the ICP comparison above.

### `reference/icp_main_variant/src/icp.cpp`
- **Category:** reference implementation
- **Status:** Compiled only into `tracking_main_icp`.
- **Purpose and why it exists:** Book SVD + g2o (`:18-139`, book) and robust 3D–3D alignment (`:141-213`, project code). The project code was added because feature correspondences contain mismatches (header `:23-29`).
- **Key types and functions:** all listed with line numbers in the comparison table: `R = -R` (`:52-54`), `EdgeProjectXYZRGBDPoseOnly` (`:66-91`), `bundleAdjustment` LM, identity start, `optimize(10)` (`:93-138`), `FitRigid` with D = diag(1,1,−1) (`:144-166`), `EstimateRigidRansac` (`:168-213`).
- **Inputs → outputs:** paired `cv::Point3f` → R, t (cv::Mat or Eigen), with pts1 ≈ R pts2 + t.
- **Dependencies:** g2o core/solver_dense (Levenberg), Sophus, Eigen SVD, OpenCV core, `optimization`.
- **Used by:** as header.
- **Covered by tests:** `icp_rigid_test` covers `EstimateRigidRansac`, `FitRigid` and the book SVD on clean data. **`bundleAdjustment` is not tested.**
- **Before studying it, understand:** Kabsch/Umeyama; RANSAC; g2o pose-only edges.

### `reference/icp_main_variant/tests/icp_rigid_test.cpp`
- **Category:** reference test/demo (self-checking; exit 0/1)
- **Status:** `icp_rigid_test` (`reference/icp_main_variant/CMakeLists.txt:34`). Not in CTest and not in `run_reference_suite.sh`.
- **Purpose and why it exists:** Self-check of main's robust ICP and of its relative-pose convention (`:1-5`). The comment refers to `src/tracking/icp.cpp`; the file is now `reference/icp_main_variant/src/icp.cpp`.
- **Key types and functions:**
  - `Make` generates Bunny-like points 0.45–0.58 m away, moved by x_{i+1} = R x_i + t, with optional mismatches and noise (`:35-50`). `ToCommon` inverts once (`:53-56`).
  - Checks: identity, translation only, 10° rotation, the convention (native = T_{i←i+1}, and a second inversion would be detectable) (`:67-100`).
  - Agreement with the book SVD on clean data (`:103-113`).
  - 35 % mismatches with 1 mm noise: rotation < 0.5°, translation < 5 mm, no false inliers, > 90 % true inliers kept (`:116-124`). Without inlier selection the error is much worse (`:125-133`).
  - Determinism (`:134-135`); fewer than 3 points gives not ok (`:138-140`).
- **Inputs → outputs:** none → exit 0/1 (`:142`).
- **Dependencies:** `tracking_main_icp`, Eigen.
- **Used by:** —
- **Covered by tests:** n/a.
- **Before studying it, understand:** main's icp.cpp.

### `reference/icp_main_variant/tests/slam_trajectory_test.cpp`
- **Category:** reference test/demo
- **Status:** `slam_trajectory_test_main_icp`. No script calls it (grep finds the name only in its CMakeLists).
- **Purpose and why it exists:** Main's version of the reference trajectory pipeline. It differs from the teammate file only in the ICP step and the output option (verified by `diff`).
- **Key types and functions:**
  - `RunIcpPair` first runs `EstimateRigidRansac(pts1, pts2)` (1 cm, `:237`) and returns failure if not ok. It then runs `bundleAdjustment(in1, in2, ...)` **on the inliers only** (`:244`); there is no separate `pose_estimation_3d3d` call. The inlier count is reported (`ICPInl` column).
  - `--out-dir DIR` redirects the three trajectory files (`:373-381`). It creates `DIR` (`:511`) but **defaults to the dataset directory** (`:510`).
  - Everything else (PnP RANSAC 100/8 px/0.99, accumulation, export flags) is identical to the teammate file.
- **Inputs → outputs:** `[dataset] [--out-dir D] [--export-dir E] [--export-pair K|all]` → `<D or dataset>/{pnp_trajectory.txt, icp_trajectory.txt, slam_trajectory.csv}` (+ export CSVs).
- **Dependencies:** `features`, `camera`, `geometry`, `tracking_main_icp`, OpenCV, Sophus; `report_utils.hpp` via include path.
- **Used by:** —
- **Covered by tests:** `trajectory_validation_test_main_icp` (same caveats as the teammate validator).
- **Before studying it, understand:** the teammate slam_trajectory_test entry; main's icp.cpp.

### `reference/icp_main_variant/tests/trajectory_validation_test.cpp`
- **Category:** reference test/demo
- **Status:** `trajectory_validation_test_main_icp`. No script calls it.
- **Purpose and why it exists:** Main's version of the trajectory validator. The only difference is that `RunIcpPairOnce` uses `EstimateRigidRansac` and then `bundleAdjustment` on the inliers (`:198-204`). If RANSAC fails, the inlier lists are empty and the g2o call has no edges; the pose stays at identity and is not reported as a failure.
- **Key types and functions:** as the teammate version (sections 4–10; always exit 0).
- **Inputs → outputs:** `[dataset_dir]` → stdout. It reads `<dataset>/{pnp,icp}_trajectory.txt`, so it has no `--out-dir`. A run of `slam_trajectory_test_main_icp --out-dir D` therefore cannot be validated unless the files are in the dataset directory passed here.
- **Dependencies:** as `slam_trajectory_test_main_icp`.
- **Used by:** —
- **Covered by tests:** n/a.
- **Before studying it, understand:** the teammate trajectory_validation_test entry.

## Experiments, tooling, visualization and configuration

None of the files in this section belong to the production per-frame pipeline (`scratch_pipeline`). They are
experiments (C++ that is built only with `-DSLAM_WITH_OPENCV=ON` or through a separate CMake project), evaluation
and figure tooling (Python/shell, which reads files written by C++ programs), the Blender visualization, and the
build/environment configuration. Python and Blender scripts never estimate a pose. All geometry comes from the C++
outputs they read.

Two terms used throughout:
- *T_wc* is a camera-to-world pose: `X_world = R X_cam + t`. Its translation is the camera centre.
- *T_i<-j* maps camera-j coordinates into camera i: `X_i = R X_j + t`.

---

### Experiments: `experiments/rgbd_icp/` (the teammate's dense RGB-D ICP)

This experiment is provisional. The CMake file says so (`experiments/rgbd_icp/CMakeLists.txt:1`: "provisional ICP;
no final ICP decision yet"). It is the only place in the repository with **loop verification and pose-graph
optimisation** (`findLoops`, `optimizeGraph`). The production pipeline has neither.

#### Registration overview (applies to the next four entries)

`registerFramesWithFeatures` (`src/rgbd_icp.cpp:532-566`) registers a *source* frame b (frame i+1) to a *target*
frame a (frame i). Every returned `Pose` maps source points into the target frame, so it is T_i<-(i+1):
`target ≈ R·source + t`. `rigidSVD` returns `{R, b - R a}`, with a the source centroid and b the target centroid
(`src/rgbd_icp.cpp:44-65`). The stages are:

1. **Features**
   - `extractFeatures` (`:512-526`) runs feature_core: BT.601 gray, `BuildPyramid`, FAST with
     `Config::fast_threshold`, then `ExtractMultiscale` with `SelectionParams{harris=true, total_features=Config::features}`.
   - `matchedFeaturePoints` (`:288-308`) matches with `features::MatchMutualRatio(…, Config::ratio)`.
   - Each matched pixel is lifted to 3D only if `stableDepth` holds in **both** frames (`:237-251`). The test needs:
     - a border of 2 px;
     - all nine pixels of the 3×3 window with depth > 0;
     - a window depth range (max − min) below `depth_edge` (12 mm).
   - Back-projection is `((u-cx)z/fx, (v-cy)z/fy, z)` with `z = raw / depth_scale` (`:234-236`).
2. **Feature-pair SVD ("baseline")**: closed-form Kabsch/Umeyama without scale on all lifted pairs
   (`rigidSVD`, `:543`). A reflection is corrected with `D = diag(1,1,sign det(VUᵀ))` (`:60-63`). It needs at
   least 3 pairs; otherwise the status stays `INSUFFICIENT_FEATURES`.
3. **RANSAC initialisation ("robust")** (`ransac`, `:309-355`):
   - Draws 3 distinct indices with `std::mt19937(seed)` and skips collinear samples (cross-product norm ≤ 1e-7, `:67-69`).
   - Fits with SVD and counts inliers with residual `< ransac_threshold` (9 mm).
   - Keeps the sample with the most inliers, ties broken by the smaller median residual.
   - Fails if there are fewer than `min_inliers` (12) inliers.
   - Then refits and rescores up to 4 times until the inlier set stops changing, and refits once more (`:346-353`).
   - Status `RANSAC_FAILED` on failure.
4. **Dense clouds** (`cloud`, `:252-286`):
   - Sampling: every `stride`-th pixel (3) with stable depth.
   - Target (frame i) normals: the cross product of the central differences of back-projected neighbours (`:260-267`).
   - If a cloud has more than `max_points` (2800) points, a seeded shuffle (`Config::seed`) keeps `max_points` of them.
   - Status `INSUFFICIENT_CLOUD` if either cloud has fewer than `min_pairs` (70) points.
5. **Point-to-point Gauss–Newton** (`pointICP`, `:356-420`). Starts from the RANSAC pose and runs up to
   `point_iterations` (18) iterations. Each iteration:
   - **Nearest-neighbour refresh** at the current estimate, with an exact balanced 3D KD-tree (`KDTree`, `:99-144`).
     Pairs farther than `max_distance` (24 mm) are dropped. The rest are sorted by distance, the closest
     `max(3, trim·N)` are kept (trim 0.83), and each target point is used at most once (`nearestPairs`, `:149-171`).
     The pairs are then held fixed for the linearisation and the line search (comment `:363-364`).
   - Jacobian `J = [-[x]×  I]` (rotation first, then translation) at the transformed point (`pointJacobian`, `:25-29`).
   - Normal equations `H = ΣJᵀJ`, `b = -ΣJᵀr`, with a fixed damping `1e-5·tr(H)/6 + 1e-9` (`:381`). This is not an
     adaptive LM schedule. Solved with LDLT.
   - The step is scaled so that ‖rotation‖ ≤ 0.1 rad and ‖translation‖ ≤ 0.03 m (`:389-392`).
   - Backtracking with α ∈ {1, ½, ¼, ⅛, 1/16}. A candidate is rejected if `safeMotion` fails: its motion relative to
     the **initial** pose exceeds `max_angle_deg` (13°) or `max_translation` (0.07 m) (`:172-175`). A candidate is
     accepted if the fixed-pair squared cost does not increase (`:395-409`).
   - Stop reasons: `CONVERGED` (applied step < 1e-7 in both parts), `NO_DESCENT`, `INSUFFICIENT_SUPPORT`,
     `NUMERICAL_FAILURE` or `ITERATION_LIMIT`.
   - Final acceptance (`finish`, `:176-184`): at least one iteration, at least `min_pairs` pairs, and NN RMSE < `max_distance`.
6. **Point-to-plane Gauss–Newton with robust weighting** (`planeICP`, `:421-492`). Same structure, starting from the
   point-to-point result, for up to `plane_iterations` (16) iterations.
   - Residual `r = n_tᵀ(T·s − t)` using the **target** normal.
   - Huber IRLS weight `w = 1` if |r| ≤ `huber` (4 mm), else `huber/|r|` (`:441-442`).
   - The line search uses the Huber cost (`huber()`, `:185-188`).
   - Both GN stages use the same left update, `leftIncrement`: `(R,t) ← (Exp(w)R, Exp(w)t + v)` (`:30-32`). The header
     notes that this is an SO(3)×R³ retraction, not a full SE(3) exponential (`include/tracking/rgbd_icp.hpp:86-88`).
7. **Acceptance**: `accepted = point_info.accepted && plane_info.accepted`. The status is then `OK` or `ICP_REJECTED`
   (`:563-564`).
8. **Loop verification** (`findLoops`, `:567-595`). This runs once, after odometry. It uses no ground truth.
   - Candidates: frame pairs with j ≥ i + 25 whose *estimated* relative pose is under 0.12 m and 32°. The 8 closest
     are kept.
   - Each candidate is re-registered with `registerFrames`. It becomes a loop edge (measurement = point-to-plane pose)
     only if all of these hold:
     - the registration is accepted;
     - inliers ≥ max(35, `min_inliers`);
     - inlier ratio ≥ 0.65;
     - it agrees with the odometry prediction to within 8° and 0.045 m (`:587-592`).

     The agreement test means a loop can only correct small drift.
9. **Pose graph** (`optimizeGraph`, `:625-743`, default 25 iterations at `include/tracking/rgbd_icp.hpp:115-116`).
   - Edge residual `[log(R_e)/σ_R ; t_e/σ_t]` with `E = Z⁻¹·(T_i⁻¹T_j)` (`:598-604`).
     - Loop edges: σ_R = 0.015 rad, σ_t = 3 mm.
     - Odometry edges: σ_R = 0.025 rad, σ_t = 5 mm.
   - Cost: component-wise soft-L1, `4(√(1+r²/4) − 1)` (`:606-615`).
   - Solver: IRLS weights, central-difference Jacobians (h = 1e-6) of a left perturbation (`:616-623, 676-686`),
     and Levenberg–Marquardt with λ starting at 1e-5, ×10 on rejection and ÷3 on acceptance (`:664-732`).
   - Gauge: one pose is fixed per connected component, the lowest-index pose (union-find, `:640-659`).

All `Config` defaults are in `include/tracking/rgbd_icp.hpp:23-33`:
- `depth_scale` 5000 (`:24`); `ratio` 0.87, `depth_edge` 0.012 m and `ransac_threshold` 0.009 m (`:24-25`).
- `features` 1400, `fast_threshold` 10, `ransac_iterations` 650 and `min_inliers` 12 (`:26-27`).
- `stride` 3, `max_points` 2800, `min_pairs` 70, `point_iterations` 18 and `plane_iterations` 16 (`:28-29`).
- `max_distance` 0.024 m, `trim` 0.83, `huber` 0.004 m, `max_angle_deg` 13 and `max_translation` 0.07 m (`:30-31`).
- `seed` 42 (`:32`).

The frontend values (1400 features, FAST 10, ratio 0.87, 12 mm depth edge, 650 iterations) are the same numbers as
the production T0 frontend. The RANSAC here, however, is 3D–3D with a metric threshold, not the PnP RANSAC in pixels.

### `experiments/rgbd_icp/CMakeLists.txt`
- **Category:** build/config (experiment)
- **Status:** used. Added by the root `CMakeLists.txt:130-131` only when `SLAM_WITH_OPENCV=ON` (`pixi run build`).
- **Purpose and why it exists:**
  - Defines the library `rgbd_icp` (`src/rgbd_icp.cpp`; public include dir `include` + OpenCV), the executables
    `icp_sequence` and `rgbd_icp_test`, and the CTest `rgbd_icp_geometry`.
  - Links OpenCV, `Eigen3::Eigen` and `feature_core` (`:6-13`).
  - The header comment (`:1-3`) says the target definitions are unchanged from the former root CMakeLists.txt.
- **Key functions / CLI:** none.
- **Inputs → outputs:** none (binaries in `build/`, because the root sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY`, `CMakeLists.txt:11`).
- **Dependencies:** OpenCV (found here), Eigen, and `feature_core` from `cmake/feature_core.cmake` via the root.
- **Used by:**
  - `cmake/orb_benchmark.cmake`, because `orb_benchmark` links `rgbd_icp`.
  - `reference/opencv_slambook2/src/tracking/icp.cpp`, which includes `tracking/rgbd_icp.hpp` and calls `rgbd_icp::rigidSVD`.
- **Covered by tests:** defines `rgbd_icp_geometry`.
- **Before studying it, understand:** the root `SLAM_WITH_OPENCV` option, and that a second, standalone build of the
  same sources exists (`icp_cpp/CMakeLists.txt`).

### `experiments/rgbd_icp/include/tracking/rgbd_icp.hpp`
- **Category:** experiment
- **Status:** used (by `rgbd_icp.cpp`, `icp_sequence.cpp`, `rgbd_icp_test.cpp`, `orb_benchmark.cpp`, and the
  reference `src/tracking/icp.cpp`).
- **Purpose and why it exists:** the public API of the dense RGB-D registration experiment.
- **Key functions / CLI:**
  - `Pose {R, t}` with `apply`, `inverse` and `operator*` (`:12-18`).
  - `Camera`, `Config` (all defaults, `:23-33`), `ImageFeatures`, `Frame {rgb, depth}`.
  - `IterationInfo`, `SolverInfo` (per-iteration RMSE/objective before and after, step sizes, an FNV-1a hash of the
    assignment), `PairClouds`, `PairResult` (baseline/robust/point/plane poses and status), `Edge`, `GraphResult`.
  - Free functions: `skew`, `expSO3`, `logSO3`, `leftIncrement`, `pointJacobian`, `angleDeg`, `rigidSVD`, `ransac`,
    `pointICP`, `planeICP`, `readCamera`, `readFrame`, `matchedFeaturePoints`, `extractFeatures`,
    `registerFramesWithFeatures`, `registerFrames`, `findLoops`, `optimizeGraph`.
- **Inputs → outputs:** n/a.
- **Dependencies:** Eigen, OpenCV core (`cv::Mat` in `Frame`), and `features/types.hpp` (`BriefDescriptor`).
- **Used by:** see Status.
- **Covered by tests:** `rgbd_icp_test.cpp`.
- **Before studying it, understand:** the conventions in the overview above.
  - The second frame argument is the *source*; poses are T_target<-source.
  - `leftIncrement` is a left SO(3)×R³ update.

### `experiments/rgbd_icp/src/rgbd_icp.cpp`
- **Category:** experiment
- **Status:** used. Compiled into the library `rgbd_icp`, both in the root build (OpenCV on) and in `build-icp/`.
- **Purpose and why it exists:** the teammate's corrected 3D–3D registration pipeline: feature pairs → RANSAC/SVD
  → point-to-point GN → Huber point-to-plane GN, plus optional loop verification and pose graph. It was kept as an
  experiment to compare against PnP. It is not called by `scratch_pipeline`.
- **Key functions / CLI:** see the overview (stages 1–9) with line numbers. Other functions:
  - `readCamera` reads `intrinsics.txt` as `width height fx fy cx cy` (`:493-500`).
  - `readFrame` reads `%06d.png` (colour) and `%06d_depth.png`, and requires 16-bit depth of the intrinsics size (`:501-511`).
- **Inputs → outputs:** RGB-D frames + intrinsics → `PairResult`; poses + edges → `GraphResult`. It writes no files.
- **Dependencies:**
  - feature_core: `GrayFromInterleaved`, `BuildPyramid`, `ExtractMultiscale` and `MatchMutualRatio`.
  - Eigen: JacobiSVD, LDLT, AngleAxis.
  - OpenCV `imgcodecs` (image reading only).
- **Used by:** `icp_sequence`, `orb_benchmark`, `rgbd_icp_test`, and the reference `pose_estimation_3d3d` (through `rigidSVD`).
- **Covered by tests:** `rgbd_icp_test`. The test checks:
  - SVD, reflection handling, RANSAC, the Jacobians (by finite differences), GN descent, the motion gate,
    partial overlap and the pose graph.
  - It does **not** cover `findLoops`, `cloud` or `stableDepth` on real images, apart from one blank-frame rejection.
- **Before studying it, understand:**
  - Kabsch/Umeyama alignment, and Gauss–Newton on SO(3)×R³ with a left perturbation.
  - Point-to-plane ICP, Huber IRLS and KD-tree nearest neighbours.
  - Soft-L1 pose-graph LM and gauge fixing.
  - Note: the GN damping is a fixed small constant (comment `:380`), not LM.

### `experiments/rgbd_icp/apps/icp_sequence.cpp`
- **Category:** experiment
- **Status:** used.
  - Root build: `build/icp_sequence`.
  - Standalone build: `pixi run icp` (= `./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp`, `pixi.toml:15`).
  - Also run by `visualization/rebuild_blender.sh:23` and `tools/figures/run_gn_orb.sh:14`.
- **Purpose and why it exists:** runs the RGB-D registration over a whole sequence and writes the trajectories and
  metrics that the Blender Bunny scene and `render_outputs.py` read.
- **Key functions / CLI:**
  - `icp_sequence [DATASET=data/synthetic_bunny] [OUT=results/runs/rgbd_icp]` (`:66-67`; the tracked evidence
    `results/benchmarks/rgbd_icp/` is written only when passed explicitly: `pixi run icp`, `rebuild_blender.sh`,
    `run_gn_orb.sh`); all other parameters are `Config`
    defaults.
  - The frame count is the number of consecutive `%06d.png` files, not the ground-truth length (`:78-86`).
  - For each pair i→i+1 it calls `registerFrames(frame i, frame i+1, …, seed = 42 + i)`.
  - Each of the four trajectories (`baseline`, `robust`, `point`, `plane`) is accumulated as `T_wc[i+1] = T_wc[i]·T_i<-(i+1)`
    (`:132-137`).
  - **Failed pair:** the method's last *accepted* relative pose is reused (constant-velocity fallback; identity before
    the first acceptance), and the failure is counted.
  - After odometry: `findLoops` runs on the `plane` trajectory. If any loop is accepted, `optimizeGraph` runs, and its
    result is kept as `pose_graph` only if it converged and did not increase the cost (`:149-167`).
  - Ground truth is read only after all estimation (comment `:168-169`).
  - Exit code: 0 if all pairs were accepted, 2 if some failed, 1 on error.
- **Inputs → outputs:**
  - Inputs:
    - `DATASET/intrinsics.txt`, `%06d.png`, `%06d_depth.png`.
    - Optionally `groundtruth.txt` (TUM rows `stamp tx ty tz qx qy qz qw`), used for evaluation only. Its row count
      must equal the frame count (`:174-175`).
  - Files written to `OUT/`:
    - `solver_metadata.txt`.
    - `pair_metrics.csv`: per pair, status, matches, valid_3d, inliers, inlier_ratio, iteration counts, pair counts,
      RMSE, converged flags and stop reasons (`:104-107`).
    - `iteration_history.csv`: one row per accepted GN update (`:100-103`).
    - `audit_pair_15_16/{clouds.csv, stage_poses.csv}`: only for pair 15→16, and only if accepted (`:112-113, 11-43`).
    - `loop_edges.csv` (`i,j`).
    - `<method>_trajectory.txt` for `baseline`, `robust`, `point`, `plane` and, if kept, `pose_graph`. Rows are
      `frame tx ty tz qx qy qz qw` of T_wc. With ground truth, they are anchored at GT frame 0 (`world = gt[0]·pose`,
      `:189`). There is no fitted alignment.
    - `summary_metrics.csv` (only when ground truth exists):
      - ATE RMSE (raw positions);
      - mean RPE (Δ=1) rotation and translation;
      - final position and rotation error;
      - accepted and failed pair counts (`:177-211`).
    - `selected_trajectory.txt`: `pose_graph` if kept, else `plane` (`:215-221`).
    - `run_complete.txt`.
  - `run_complete.txt`, `selected_trajectory.txt` and `pose_graph_trajectory.txt` are deleted at start (`:75-77`), so a
    failed run cannot look fresh.
  - The tracked `results/benchmarks/rgbd_icp/` also contains files that `icp_sequence` does **not** write:
    - `blender_trajectory.csv` and `blender_source.json`, written by `visualization/scripts/trajectory_data.py`.
    - `blender_consistency.csv`, written by `visualization/check_blender_consistency.py`.
  - The current tracked run is 36 frames, 0 failed pairs, one loop edge `0,35`, selection `pose_graph`
    (`results/benchmarks/rgbd_icp/run_complete.txt`, `loop_edges.csv`, `selected_trajectory.txt`).
- **Dependencies:** `rgbd_icp`; Eigen (quaternions); `<filesystem>`.
- **Used by:**
  - `visualization/scripts/trajectory_data.py`, `tools/figures/render_outputs.py` and `tools/figures/validate_outputs.py`
    (read `results/benchmarks/rgbd_icp/`).
  - `visualization/rebuild_blender.sh` and `tools/figures/run_gn_orb.sh` (run it).
- **Covered by tests:** no direct test of the program. `tools/figures/validate_outputs.py` checks its
  `iteration_history.csv` and `pair_metrics.csv` for consistency.
- **Before studying it, understand:** `rgbd_icp.hpp` conventions.
  - The trajectory text files are T_wc in TUM order (qx qy qz qw).
  - The metrics use only a GT-frame-0 anchor, no alignment.

### `experiments/rgbd_icp/tests/rgbd_icp_test.cpp`
- **Category:** test
- **Status:** used. It is the CTest `rgbd_icp_geometry`, registered in two places:
  - the root build with OpenCV (`experiments/rgbd_icp/CMakeLists.txt:13`);
  - `build-icp` (`icp_cpp/CMakeLists.txt:24`).
- **Purpose and why it exists:** self-checking geometry test of the `rgbd_icp` library on synthetic data. It uses
  100 Gaussian points and a known pose `Exp(.2,-.1,.4)`, `t = (.3,-.2,.5)`.
- **Key functions / CLI:** `main()` throws on the first failed `require`. Exit 0 means PASS. It checks:
  - **SVD:** recovers the known transform; inverse composition; reflection handling keeps det R = +1; empty input throws.
  - **RANSAC:** with 30 of 100 targets corrupted, finds exactly the 70 inliers and the pose to 1e-9 m.
  - **Point GN:**
    - identity at the true pose;
    - every logged update has a non-increasing fixed-pair objective;
    - the analytic Jacobian matches central finite differences of `leftIncrement`, for both the point and the plane
      projection (`:64-76`);
    - recovers a small non-zero motion.
  - **Plane GN:** recovers the same small motion with a non-increasing Huber objective.
  - **Other ICP behaviour:** partial overlap with 15 far outliers; the motion gate rejects (`NO_DESCENT`); 0
    iterations is not accepted; no overlap is rejected; a blank frame gives `INSUFFICIENT_FEATURES`.
  - **Pose graph:** reduces the cost to below 1e-6 with frame 0 fixed; a disconnected graph is handled.
- **Inputs → outputs:** none → stdout PASS/FAIL.
- **Dependencies:** `rgbd_icp`. In `build-icp` it is also compiled together with `reference/opencv_slambook2/src/tracking/icp.cpp`
  (`icp_cpp/CMakeLists.txt:20-22`), although the test calls nothing from it.
- **Used by:** CTest; `tools/figures/run_gn_orb.sh:12` and `visualization/rebuild_blender.sh:20` (`ctest --test-dir build-icp`).
- **Covered by tests:** n/a (it is the test).
- **Before studying it, understand:** finite-difference Jacobian checks, and the left-increment convention.

### `experiments/rgbd_icp/icp_cpp/CMakeLists.txt`
- **Category:** build/config (experiment)
- **Status:** used by `pixi run icp-configure` / `icp-build` (`pixi.toml:13-14`), `visualization/rebuild_blender.sh:18-19`
  and `tools/figures/run_gn_orb.sh:10-11`. The binaries land in `build-icp/`.
- **Purpose and why it exists:** a standalone CMake project (`corrected_rgbd_icp`) that builds the ICP experiment
  without the root project. It needs only OpenCV core+imgcodecs, Eigen and feature_core.
- **Key functions / CLI:** what it builds:
  - **OpenCV:** `find_package(OpenCV QUIET COMPONENTS core imgcodecs)`, with a manual `find_path`/`find_library`
    fallback (`:5-12`).
  - **Shared definitions:** includes `../../../cmake/feature_core.cmake` (`:14`) and `../../../cmake/orb_benchmark.cmake` (`:33`).
  - **Library and programs:** builds `rgbd_icp` with `-Wall -Wextra -Wpedantic`, plus `icp_sequence` and `rgbd_icp_test`.
  - **CTests:** `rgbd_icp_geometry` and, when Python 3 is found, `blender_data_bridge`
    (= `python visualization/tests/test_blender_data.py`, `:27-31`). `orb_benchmark_smoke` comes from the included file.
- **Inputs → outputs:** none.
- **Dependencies:** OpenCV core/imgcodecs (+ features2d through orb_benchmark.cmake), Eigen3, C++17.
- **Used by:** pixi `icp*` tasks, `rebuild_blender.sh`, `run_gn_orb.sh`.
- **Covered by tests:** registers the three CTests above.
- **Before studying it, understand:**
  - The paths are relative to this directory (`../src`, `../../../cmake`).
  - This build does not define `SLAM_WITH_OPENCV`. It is independent of `build/`.

---

### `experiments/orb_opencv/orb_benchmark.cpp`
- **Category:** benchmark
- **Status:** used. Built by `cmake/orb_benchmark.cmake`, i.e. in the root build with OpenCV and in `build-icp/`. Run by:
  - `visualization/rebuild_blender.sh:30` and `tools/figures/run_gn_orb.sh:19`, with
    `data/synthetic_bunny results/reports/figures/orb_benchmark --repeats 3`;
  - the CTest `orb_benchmark_smoke` (`--frames 2 --repeats 1`).
- **Purpose and why it exists:** a controlled frontend comparison of the project's ORB-style features
  (`rgbd_icp::extractFeatures`, feature_core) against OpenCV ORB.
  - Both feed the **same** project matcher (`MatchMutualRatio`, 0.87), depth gates, 3D RANSAC and the two GN ICP
    stages (comment `:1-2`).
  - OpenCV ORB is created with the project budget: `ORB::create(1400, 1.2, 8, 16, 0, 2, HARRIS_SCORE, 31, 10)` (`:39`).
  - Its descriptors are converted bit by bit into `BriefDescriptor` (`:49-52`).
- **Key functions / CLI:** `orb_benchmark DATASET OUTPUT [--repeats N] [--frames N]`. Defaults:
  `data/synthetic_bunny`, `results/reports/figures/orb_benchmark`, 3 repeats, all frames (`:92-100`).
  - **Setup:** `cv::setNumThreads(1)`, `setRNGSeed(42)`.
  - **Timing:** extraction and matching are timed as the median of N runs after one warm-up; the method order
    alternates per frame (`:119-137`). Registration is timed once per pair.
  - **Trajectory:** the plane pose of an accepted pair, otherwise the constant-velocity fallback (`:154-156`).
  - **Ground truth:** read only after all estimation (`:160`).
  - **Feature precision against GT:** the fraction of valid 3D pairs whose GT-transformed source lies within
    `ransac_threshold` of its target (`:172-182`).
- **Inputs → outputs:** dataset frames + `intrinsics.txt` (+ optional `groundtruth.txt`) → `OUTPUT/`:
  - `frames.csv` (keypoints and extraction time per method/frame);
  - `pairs.csv` (status, matches, valid_3d, inliers, GT-correct count/precision, timings, rotation/translation error);
  - `summary.csv` (per method: mean counts, GT precision, median extraction ms, raw ATE, mean accepted errors);
  - `<method>_trajectory.txt` (T_wc relative to frame 0, **not** anchored at GT);
  - `keypoints/frame_%06d_<method>.csv` (`index,x,y,level,angle_rad`);
  - `protocol.txt` (a written protocol of the fixed settings, `:211-223`);
  - `run_complete.txt`.
- **Dependencies:** `rgbd_icp`, feature_core, OpenCV `features2d` (`cmake/orb_benchmark.cmake:4`), Eigen.
- **Used by:** `tools/figures/render_orb_benchmark.py` and `tools/figures/render_orb_keypoints.py`.
- **Covered by tests:** `orb_benchmark_smoke` (it runs; nothing is asserted beyond exit 0).
- **Before studying it, understand:** `rgbd_icp` registration.
  - Per the protocol text, this is a frontend comparison, "not identical implementations or an ORB-SLAM benchmark"
    (`:218`).
  - There is no loop closure, and ATE includes constant-velocity fallbacks (`:222`).

---

### `experiments/orb_extractor_benchmark/` (overview)
- **Category:** benchmark (standalone, historical)
- **Status:** not part of any CMake build of the repository, and no pixi task runs it. It is built and run by hand
  inside its own directory (README §1). Its own `.gitignore` excludes `bin/`, `build/`, `results/` and `datasets/`, so
  the `bin/orb_extract` named in the README is absent until built.
- **Purpose and why it exists:** a self-contained snapshot of the project's ORB-style extractor for dataset-level
  comparison with OpenCV ORB.
  - It vendors `include/features/*.hpp` and `src/features/*.cpp` as they were at commit `fb23599`. I checked this:
    `git show fb23599:` equals the snapshot for `fast`, `matcher` and `multiscale`.
  - Today the snapshot's `matcher.*` and `multiscale.*` differ from the current `include/slam/features/` and
    `src/features/`; `brief`, `fast`, `orientation`, `pyramid` and `types` are identical. **It does not track
    production.**
  - Its README calls the code "byte-for-byte copies" from `slam-main.zip` (`README.md:3`).
  - Matching in the benchmark is OpenCV mutual-NN Hamming for **both** extractors (`README.md:5`, caveat 3). The
    project matcher is not used.
- **Key functions / CLI (per README):**
  - Setup:
    - `python3 -m venv .venv && pip install -r requirements.txt`;
    - `cmake -S . -B build && cmake --build build && cp build/orb_extract bin/orb_extract`.
  - `python benchmark.py {synthetic|oxford|hpatches|tum|kitti|folder} --root … --output …`.
  - Metrics:
    - Homography datasets (synthetic, Oxford, HPatches): repeatability and match precision.
    - TUM, KITTI and plain folders: only an F-RANSAC inlier fraction.
  - `python run_all.py …` runs several datasets and builds one HTML report.
  - `python extract_image.py IMG --output DIR [--extractor scratch|opencv|both]`.
  - `./bin/orb_extract --input image.pgm --output features.csv --levels 8 --features-per-level 500 --fast-threshold 30`.
  - Tests: `python -m unittest discover -s tests -v`.
- **Inputs → outputs:** images or datasets → `results/<name>/{pairs.csv, summary.csv, settings.json, plots/,
  visualizations/, dashboards/}`. A small sample run is tracked in `sample_results/`.
- **Dependencies:**
  - C++17 only for `orb_extract` (no OpenCV/Eigen).
  - Python: numpy, opencv-python, scipy, matplotlib, Pillow (`requirements.txt`). This is a venv, not pixi.
- **Used by:** nothing else in the repository.
- **Covered by tests:** its own `tests/` (Python unittest), not CTest.
- **Before studying it, understand:** that it is frozen history. To study the current feature code, read
  `include/slam/features/` and `src/features/` instead.

Files (one line each):
- `README.md`: setup, per-dataset commands, output definitions and comparison caveats.
- `CMakeLists.txt`: builds a static `feature_core` from the vendored sources and the `orb_extract` executable (C++17, Release by default).
- `.gitignore`: ignores datasets, results, build/bin, Python caches, editors.
- `requirements.txt`: numpy, opencv-python, scipy, matplotlib, Pillow.
- `LICENSE`: MIT, Mehrish Khan.
- `THIRD_PARTY_NOTICES.md`: slambook2 (MIT) and Stanford Bunny (research/non-commercial) notices.
- `include/features/{brief,fast,matcher,multiscale,orientation,pyramid,types}.hpp`: vendored feature headers (fb23599).
- `src/features/{brief,fast,matcher,multiscale,orientation,pyramid,types}.cpp`: vendored feature sources (fb23599).
- `src/orb_extract.cpp`: CLI adapter; binary PGM in → keypoint/descriptor CSV out; options `--levels --scale-factor --fast-threshold --features-per-level --cell-size --min-threshold`.
- `benchmark.py`: dataset evaluator; runs `orb_extract` through subprocess and OpenCV ORB, matches both with OpenCV, writes CSV/plots/dashboards.
- `extract_image.py`: keypoints, descriptors and a preview image for one image (scratch, OpenCV or both).
- `run_all.py`: runs every installed dataset and assembles one combined report (`build_report.assemble`).
- `build_report.py`: builds an offline HTML index from existing result folders (it does not invent metrics).
- `visualization.py`: spatially thinned correspondence drawings and HTML pair dashboards.
- `tests/test_metrics.py`: unit tests of homography correctness/repeatability logic and synthetic cases.
- `tests/test_visualization.py`: unit tests of the drawing and dashboard functions.
- `tests/test_master_report.py`: unit test of `build_report.assemble`.
- `sample_data/bunny.png`: a Stanford Bunny render for smoke tests.
- `sample_results/…`: tracked output of a 2-pair synthetic run (`pairs.csv`, `summary.csv`, `settings.json`, `plots/`, `dashboards/`, `visualizations/`, `single_image/`). `settings.json` records an absolute `/mnt/data/…` binary path from the machine that produced it.

---

### Tooling: `tools/pipeline_analysis/`

These read the outputs of `build/scratch_pipeline` (default `results/runs/pipeline/`),
`build/scratch_experiments`, `build/depth_edge_experiment` and the frozen `tests/fixtures/reference_baseline/`. Most locate the
repository root as `Path(__file__).parents[2]`. None is registered with CTest.

### `tools/pipeline_analysis/check_trajectory.py`
- **Category:** tooling
- **Status:** used manually: `python3 tools/pipeline_analysis/check_trajectory.py [--csv OUT]`. Exit 1 if an identity check fails.
- **Purpose:** a NumPy check of the scratch trajectory transform chain. Per frame it compares GT, reference PnP/ICP,
  the milestone-2 linear PnP and the scratch pipeline (camera centre, position/rotation error, look-at error to the
  least-squares bunny centre). It verifies these identities:
  - frame 0 = GT;
  - quaternion = rotation matrix;
  - `T_wc[i+1]⁻¹T_wc[i]` = exported `T_{i+1<-i}`;
  - camera centre = `-Rᵀt`.
- **Inputs → outputs:**
  - Inputs:
    - `data/synthetic_bunny/{groundtruth.txt, slam_trajectory.csv}`;
    - `results/runs/milestones/scratch_pnp_trajectory.csv` (from `build/pnp_full_sequence`, optional);
    - `results/runs/pipeline/{trajectory.csv, pairs.csv}` (from `build/scratch_pipeline`, required).
  - Outputs: a printed table, plus an optional CSV.
- **C++ dependency / test:** `scratch_pipeline`, optionally `pnp_full_sequence`. It is a self-checking script, not CTest.

### `tools/pipeline_analysis/correspondence_regions.py`
- **Category:** tooling (library module)
- **Status:** imported by the figure scripts and by `test_correspondence_regions.py`.
- **Purpose:** visual-only neighbourhood colouring of matches. It runs k-means (K = 12, 20 iterations,
  farthest-point seeding) on the frame-i endpoints, with hue from the angle around the centroid. A match is flagged
  `neighbourhood_changed` if its label is not among the K_VOTE = 5 nearest inlier labels in frame i+1. The
  inlier/outlier class always comes from the C++ `pnp_inlier`/`pnp_used` columns.
- **Inputs → outputs:** arrays from `correspondences/pair_<i>_<j>.csv` → labels and colours.
- **C++ dependency / test:** `scratch_pipeline` export. Not a test.

### `tools/pipeline_analysis/depth_edge_audit.py`
- **Category:** tooling
- **Status:** manual: `python3 tools/pipeline_analysis/depth_edge_audit.py [--out DIR]`.
- **Purpose:** a read-only audit of correspondences on depth edges:
  - neighbourhood depth statistics per class;
  - a GT "fixed 3D point" test;
  - four figures (correspondences by class, edge map, statistics, per-pair error all vs interior).
- **Inputs → outputs:**
  - Inputs:
    - `results/runs/depth_edge/{correspondences/pair_*.csv, pairs.csv, trajectory_all.csv, trajectory_interior.csv}`
      (from `build/depth_edge_experiment`, `experiments/depth_edge/depth_edge_experiment.cpp:21`);
    - the dataset images and ground truth.
  - Output: PNG/PDF in `results/runs/figures/depth_edge/`.
- **C++ dependency / test:** `depth_edge_experiment`. Not a test.

### `tools/pipeline_analysis/descriptor_figure.py`
- **Category:** tooling
- **Status:** manual.
- **Purpose:** a Hamming-distance histogram for pair 0→1, explaining why the same filter keeps more scratch than
  reference matches. A "true match" reprojects within 8 px under the GT motion.
- **Inputs → outputs:**
  - Inputs:
    - `results/runs/pipeline/keypoints/frame_00000{0,1}.csv`;
    - `tests/fixtures/reference_baseline/reference_orb_distances_0_1.csv`;
    - `tests/fixtures/reference_baseline/reference_features/`;
    - GT and the depth PNG.
  - Output: `descriptor_distances_0_1.png|pdf` in `--out` (default `results/runs/figures/`).
- **C++ dependency / test:** `scratch_pipeline`. Not a test.

### `tools/pipeline_analysis/integration_analysis.py`
- **Category:** tooling
- **Status:** manual: `integration_analysis.py OUT_DIR fixtures=DIR[,…] [stride4=…] [bunny=…] [full=…]`.
- **Purpose:** offline tables for the integration experiments (branch integration/teammate). Its thresholds are
  BAD 2°/0.05 m and GROSS 5°. GT enters only through the `eval_` columns.
- **Inputs → outputs:** `runs.csv`, `matchsets.csv` and `ate.csv` from `build/scratch_experiments`
  (`apps/experiments_main.cpp:328-334`) → `OUT_DIR/{fixture_pairs.csv, sequences.csv, attribution.csv}` and
  markdown tables on stdout.
- **C++ dependency / test:** `scratch_experiments`. Not a test.

### `tools/pipeline_analysis/make_figures.py`
- **Category:** tooling (also a shared module: `ROOT`, `PIPE`, `BASE`, `OUT`, `rows()`; imported as `mf` by the other scripts)
- **Status:** manual: `--pipeline DIR --out DIR`.
- **Purpose:** figures comparing the scratch pipeline with the frozen reference (baseline ORB + RANSAC PnP):
  - correspondence images, keypoints and pyramid levels;
  - `pnp_comparison.png/.md` over whichever of `pipeline`, `pipeline_linear` and `pipeline_single` exist.
- **Inputs → outputs:**
  - Inputs:
    - `results/runs/pipeline*/` (keypoints, correspondences, pairs, trajectory);
    - `tests/fixtures/reference_baseline/` (frozen);
    - the Bunny PNGs.
  - Output: PNG/PDF in `results/runs/figures/`.
- **C++ dependency / test:** `scratch_pipeline` (the docstring gives the `--no-refine` and `--pyramid-levels 1` runs, `:30-32`). Not a test.

### `tools/pipeline_analysis/png_reader.py`
- **Category:** tooling (library)
- **Status:** imported by `make_figures.py`, `depth_edge_audit.py` and `descriptor_figure.py`.
- **Purpose:** a pure-Python + NumPy PNG decoder:
  - CRC check, zlib, the 5 scanline filters;
  - 8/16-bit grey, grey+alpha, RGB, RGBA;
  - no palette, no interlacing.
- **Inputs → outputs:** PNG path → ndarray.
- **C++ dependency / test:** none; it mirrors the C++ decoder. Not a test. The docstring (`:2`) names `src/png.cpp`;
  the file is now `src/pipeline/png.cpp`.

### `tools/pipeline_analysis/run_baseline.sh`
- **Category:** tooling (regression harness)
- **Status:** manual: `tools/pipeline_analysis/run_baseline.sh --build DIR [--rerun-stride4] [--rerun-full] [--archive DIR]`.
- **Purpose:** the Stage-1/Stage-2 regression harness. Steps:
  0. **build-directory guard** (`:56-75`, before anything is configured): `--build DIR` or `SLAM_BUILD_DIR` is
     required (no default); an existing DIR must have been configured from this repository (`CMakeCache.txt`) and
     must already register `scratch_baseline_lock` (read from `CTestTestfile.cmake` as text; nothing is run in it);
     otherwise exit 2. Why: on 2026-11-05 a stale pre-reorganisation `build/` ran its old tests and rewrote 13 of
     their by-products, and a later `ctest -N` check rewrote that tree's `Testing/Temporary/LastTest.log`; so there is
     no default and the guard only reads files as text;
  1. configures and builds `--build DIR`;
  2. Bunny byte identity against **subsets** of `tests/fixtures/baseline_bunny{,_gate}_sha256.txt`: only
     `trajectory.txt`, `keypoints/` and `pose_validity.csv`, because the teammate output format changed (`:105-124`);
  3. `ctest --test-dir DIR`;
  4. `scratch_baseline` against the fixture locks;
  5. raw ATE of the frozen runs listed in `tests/fixtures/baseline_trajectories{,_gate}.csv` (tracked in
     `tests/fixtures/stage1_ate_runs/` and `stage1_bunny_export/`);
  6. optionally re-runs the TUM `n100_stride4` / `full` frame lists and requires byte identity with the complete
     archived run `$SLAM_ARCHIVE_DIR/2026-11-04_main/output/scratch/rgbd_dataset_freiburg1_xyz/<run>/` (`--archive DIR`
     or `SLAM_ARCHIVE_DIR`; verified against the archive's `MANIFEST.sha256` first; exit 2 if unavailable).
- **Inputs → outputs:** `tests/fixtures/*`, the archive and the TUM sequence (re-runs only; `$SLAM_TUM_ROOT` or
  `data/rgbd_dataset_freiburg1_xyz`) → console
  verdict. The report goes to `results/runs/scratch_baseline/stage1/` (via `scratch_baseline`). Temporary directories
  are removed; the archive is only read. When a re-run differs, `describe_run_diff.py` explains the difference
  (FORMAT / NUMERICAL / TEXT, per column with max |diff|) without changing the verdict, which stays byte identity.
- **C++ dependency / test:** `scratch_pipeline`, `scratch_baseline`, the CTest suite. It is a test harness (shell), not CTest.

### `tools/pipeline_analysis/describe_run_diff.py`
- **Category:** tooling (reporting only)
- **Status:** called by `run_baseline.sh` `compare_run` only after a byte comparison found a difference; also manual:
  `describe_run_diff.py REFERENCE_DIR NEW_DIR NAME...`.
- **Purpose:** explains a difference between a re-run and its reference without deciding anything. FORMAT: comment
  lines added/removed, columns present in only one file, row counts, files present on one side only. NUMERICAL: per
  common column, the number of differing cells, the largest absolute difference and the cell where it occurs. TEXT:
  non-numeric cells and line differences of non-CSV files. Nothing is ignored.
- **Inputs → outputs:** two run directories → console text. Standard library only.
- **Covered by tests:** no automated test; checked manually by comparing a run with itself (no difference
  reported) and with the archived stride-4 run (the known differences, `docs/DATA_AND_ARCHIVE.md`).

### `tools/pipeline_analysis/stage3c_analysis.py`
- **Category:** tooling
- **Status:** manual: `stage3c_analysis.py FIXTURE_DIR SEQUENCE_DIR OUT_DIR`.
- **Purpose:** analysis of the Stage-3C configurations M0…M5, M1L, M3L. It evaluates candidate gate metrics
  (computed by C++ without GT) against GT labels (bad: > 2° or > 0.05 m).
- **Inputs → outputs:** `runs.csv`, `matchsets.csv` and `ate.csv` from `build/scratch_experiments` →
  `OUT_DIR/{config_fixtures, config_sequence, gate_73_74, candidate_gate, metric_correlation, mutual_comparison}.csv`.
- **C++ dependency / test:** `scratch_experiments`. Not a test.

### `tools/pipeline_analysis/test_correspondence_regions.py`
- **Category:** test (Python, standalone)
- **Status:** manual: `python3 tools/pipeline_analysis/test_correspondence_regions.py`. Exit 1 on failure. Not in CTest.
- **Purpose:** checks the neighbourhood colouring and the exported match indexing (query/train vs keypoint files).
- **Inputs → outputs:** `results/runs/pipeline/{correspondences/pair_0_1.csv, keypoints/frame_00000{0,1}.csv, pairs.csv}` → PASS/FAIL lines.
- **C++ dependency / test:** `scratch_pipeline`. Yes, it is a test.

### `tools/pipeline_analysis/trajectory_figures.py`
- **Category:** tooling
- **Status:** manual: `--out DIR`.
- **Purpose:** trajectory figures: 3D comparison, position/rotation error per frame, per-pair RPE (scratch
  linear/refined vs reference PnP), and a final comparison.
- **Inputs → outputs:**
  - Inputs:
    - `data/synthetic_bunny/slam_trajectory.csv`;
    - `tests/fixtures/reference_baseline/metrics.json`;
    - `results/runs/pipeline/{trajectory.csv, pairs.csv}`;
    - `data/meshes/bunny/bun_zipper.ply`.
  - Output: PNG/PDF in `results/runs/figures/`.
- **C++ dependency / test:** `scratch_pipeline`. Not a test.

### `tools/pipeline_analysis/validity_evidence.py`
- **Category:** tooling
- **Status:** manual: `validity_evidence.py RUN_DIR [RUN_DIR …] [--csv OUT]`.
- **Purpose:** evidence for the Stage-2 pose-validity thresholds. It recomputes, from each pair's correspondences:
  - inlier counts and distinct inlier locations (2 px single linkage);
  - inlier ratio;
  - coverage on an 80 px grid (it must match `ValidityParams::coverage_cell_px`);
  - reprojection statistics.

  It then joins the GT error.
- **Inputs → outputs:** `RUN_DIR/{pairs.csv, correspondences/pair_*.csv}` → a printed table, plus an optional CSV.
- **C++ dependency / test:** `scratch_pipeline`. Not a test. The docstring (`:4`) cites `src/validity.hpp`; it is
  now `include/slam/pipeline/validity.hpp`.

---

### Tooling: `tools/figures/`

### `tools/figures/generate_results.py`
- **Category:** tooling
- **Status:** `pixi run -e results results` (`pixi.toml:32`), optionally with `--skip-cpp`.
  **Broken after the reorganisation; see the warning below.**
- **Purpose:** all figures, CSVs and summary tables of the **reference** (OpenCV) Bunny experiment, with the
  numbers cross-checked. It runs `build/slam_trajectory_test` on a temporary copy of `data/synthetic_bunny/` with
  `--export-dir results/runs/reference_results/data/cpp_export --export-pair 0`, and requires that run to reproduce the committed
  trajectory files byte for byte (`:174-193`).
- **Inputs → outputs:**
  - Inputs:
    - the dataset, the mesh, `slam_trajectory.csv`, `pnp/icp_trajectory.txt`;
    - the C++ export;
    - two C++ source files read by `source_constants()`, used for labels.
  - Outputs (under `results/runs/reference_results/`, ignored): `data/*.csv`, `tables/summary.{md,json}`, `figures/fig01…fig15.{png,pdf}`.
- **C++ dependency / test:** `slam_trajectory_test` (reference, needs `SLAM_WITH_OPENCV=ON`). Not a test, but it
  aborts on any cross-check mismatch.
- **Warning:** `source_constants()` reads the two C++ files from `reference/opencv_slambook2/` (paths fixed
  2026-11-06), but its regex `distance <= max(N * min_dist, X)` (`:163`) matches nothing in the repository (the
  filter is no longer inline in `features.cpp`, on main either), so the script stops there. Pre-existing; not fixed.

### `tools/figures/generate_correspondence_figure.py`
- **Category:** tooling
- **Status:** `pixi run -e results correspondences [--pair N] [--skip-cpp]` (`pixi.toml:34`).
- **Purpose:** publication figures of the actual sparse ORB correspondences of one Bunny pair (reference pipeline),
  cross-checked against the dataset images.
- **Inputs → outputs:** the export of `build/slam_trajectory_test --export-dir/--export-pair` on a temporary dataset
  copy (`results/runs/reference_results/data/correspondence_export/`) → `results/runs/reference_results/figures/bunny_correspondences_annotated_<i>_<j>.*` and
  `bunny_pnp_icp_correspondences*.{png,pdf,svg}`.
- **C++ dependency / test:** `slam_trajectory_test`. Not a test.

### `tools/figures/generate_correspondence_map.py`
- **Category:** tooling
- **Status:** `pixi run -e results correspondence-map [--pair N] [--skip-cpp]` (`pixi.toml:35`).
- **Purpose:** MASt3R-style correspondence maps. The colour is the correspondence identity (hue/saturation from the
  frame-i position); the marker size is the depth.
- **Inputs → outputs:** the same export, loaded through `generate_correspondence_figure.load()` →
  `results/runs/reference_results/figures/bunny_correspondences_<i>_<j>{,_inliers,_outliers}.{png,pdf,svg}`.
- **C++ dependency / test:** `slam_trajectory_test`. Not a test.

### `tools/figures/render_outputs.py`
- **Category:** tooling (also a module: `frontend()`, `manifest()`)
- **Status:** run by `visualization/rebuild_blender.sh:6,33` and `visualize_frontend.sh:5`. Flags: `--frontend-only`, `--check-dependencies`.
- **Purpose:** PNG plots of measurements exported by C++. It draws:
  - the frontend stages for each `results/reports/figures/frontend/pair_*`;
  - the ICP plots (`icp()`, `:164-257`): trajectories of every ICP stage plus the scratch PnP, errors, iteration
    history, and PnP refinement before/after;
  - a pipeline flowchart.

  It also writes `image_manifest.json` with SHA-256 hashes.
- **Inputs → outputs:**
  - Inputs:
    - `results/reports/figures/frontend/pair_*/…` (from `build-scratch/feature_audit`);
    - `results/benchmarks/rgbd_icp/{run_complete, selected_trajectory, *_trajectory, iteration_history, pair_metrics}`;
    - `tests/fixtures/bunny_t0_scene/{trajectory, pairs}.csv` (frozen T0 run, the same trajectory as the Blender scene);
    - `data/synthetic_bunny/groundtruth.txt`.
  - Outputs: `results/reports/figures/{frontend/…, icp/…, 00_pipeline_flowchart.png, image_manifest.json}`.
- **C++ dependency / test:** `feature_audit`, `icp_sequence`, `scratch_pipeline`. Not a test.

### `tools/figures/render_frontend_audit.py`
- **Category:** tooling
- **Status:** run by `rebuild_blender.sh:15` and `visualize_frontend.sh:11`: `--i N --j N` (default 15/16).
- **Purpose:** a thin wrapper that calls `render_outputs.frontend(i, j)` and `manifest()` for one pair.
- **Inputs → outputs:** `results/reports/figures/frontend/pair_<i>_<j>/` → PNGs in the same tree.
- **C++ dependency / test:** `feature_audit`. Not a test.

### `tools/figures/validate_frontend_audit.py`
- **Category:** test (Python assertion script)
- **Status:** run by `rebuild_blender.sh:16` and `visualize_frontend.sh:12`. Not CTest.
- **Purpose:** asserts the frontend audit CSVs are consistent with the estimator:
  - Harris ranking does not change membership or descriptors;
  - responses decrease within each level;
  - level quotas sum to 1400.

  It compares the keypoints and correspondences with `scratch_pipeline`'s export.
- **Inputs → outputs:** `results/reports/figures/frontend/pair_<i>_<j>/frame_*_{00_fast_all,…,levels}.csv` and
  `results/runs/pipeline/{keypoints, correspondences}` → assertion failure or silence.
- **C++ dependency / test:** `feature_audit`, `scratch_pipeline`. Yes, it is a check.

### `tools/figures/validate_outputs.py`
- **Category:** test (Python assertion script)
- **Status:** run by `rebuild_blender.sh:34`. Not CTest.
- **Purpose:** checks the PNG manifest:
  - the hashes match and the images open;
  - every image is at least 600×400.

  It also checks `results/benchmarks/rgbd_icp/iteration_history.csv`:
  - at least 70 pairs per row, all values finite, a non-increasing objective;
  - the counts per pair/stage equal `pair_metrics.csv` iterations.
- **Inputs → outputs:** `results/reports/figures/image_manifest.json`, `results/benchmarks/rgbd_icp/` → assertions.
- **C++ dependency / test:** `icp_sequence`. Yes, it is a check.

### `tools/figures/render_orb_benchmark.py`
- **Category:** tooling
- **Status:** run by `rebuild_blender.sh:31` and `run_gn_orb.sh:20`. Flags: `--results DIR --dataset DIR`.
- **Purpose:** plots of the ORB benchmark summary and frames (it requires exactly two methods and `run_complete.txt`).
- **Inputs → outputs:** `results/reports/figures/orb_benchmark/{summary.csv, frames.csv, …}` → PNGs in the same directory.
- **C++ dependency / test:** `orb_benchmark`. Not a test.

### `tools/figures/render_orb_keypoints.py`
- **Category:** tooling
- **Status:** run by `rebuild_blender.sh:32` and `run_gn_orb.sh:21`. Flags: `--frames 0 15 --tolerance 2 --results DIR`.
- **Purpose:** draws the exported keypoint positions of both methods and their spatial overlap within a pixel tolerance.
- **Inputs → outputs:** `results/reports/figures/orb_benchmark/keypoints/frame_%06d_{project_orb_style,opencv_orb}.csv`
  → `…/keypoint_visuals/`.
- **C++ dependency / test:** `orb_benchmark`. Not a test.

### `tools/figures/render_trajectory_overview.py`
- **Category:** visualization
- **Status:** manual: `blender -b results/blender/bunny_slam_demo.blend --python tools/figures/render_trajectory_overview.py -- OUT.png`.
- **Purpose:** a read-only overview render of the saved Bunny scene from a temporary camera 50° above the orbit. It
  does not save the scene.
- **Inputs → outputs:** the `.blend` → one PNG.
- **C++ dependency / test:** none directly (scene built from `icp_sequence`/`scratch_pipeline` outputs). Not a test.

### `tools/figures/run_gn_orb.sh`
- **Category:** tooling (driver)
- **Status:** manual: `bash tools/figures/run_gn_orb.sh [--blender]`. With `--blender` it simply execs `visualization/rebuild_blender.sh`.
- **Purpose:** reproduces the GN ICP and ORB benchmark:
  1. configures and builds `build-icp` with `BUILD_ORB_BENCHMARK=ON`;
  2. runs `ctest`;
  3. runs `icp_sequence` → `results/benchmarks/rgbd_icp`; exit 2 is tolerated with a warning;
  4. runs `orb_benchmark` → `results/reports/figures/orb_benchmark`;
  5. runs the two ORB renderers.
- **Inputs → outputs:** dataset → `results/benchmarks/rgbd_icp/`, `results/reports/figures/orb_benchmark/`.
- **C++ dependency / test:** `icp_sequence`, `orb_benchmark`, the CTests of `build-icp`. Runs tests, but is not a test itself.

### `tools/figures/visualize_frontend.sh`
- **Category:** tooling (driver)
- **Status:** manual: `bash tools/figures/visualize_frontend.sh [i=15] [j=16]`.
- **Purpose:** builds only `feature_audit` in `build-scratch` (production CMake, no OpenCV), runs it for one pair,
  then renders and validates the frontend audit.
- **Inputs → outputs:** dataset → `results/reports/figures/frontend/pair_<i>_<j>/`.
- **C++ dependency / test:** `feature_audit`. Runs `validate_frontend_audit.py`.

---

### Tooling: reference-pipeline runners (`tools/evaluation/run_diagnostics.py`, `run_reference_suite.sh`, `tools/pipeline_analysis/export_correspondences.sh`)

### `tools/evaluation/run_diagnostics.py`
- **Category:** tooling
- **Status:** `pixi run -e results diagnostics [--skip-cpp]` (`pixi.toml`).
- **Purpose:** trajectory-error diagnostics of the **reference** Bunny pipeline. It runs:
  - `build/slam_trajectory_test <copy> --export-dir results/reports/diagnostics/cpp/export --export-pair all`, with a
    byte-identity check;
  - `build/trajectory_diagnostics_test <copy> results/reports/diagnostics/cpp`.

  It stops if the GT replay of the accumulation fails, then writes tables and figures. Its three "export run
  reproduces … byte for byte" checks currently fail, for the reason given under `export_correspondences.sh`.
- **Inputs → outputs:**
  - Inputs: the C++ exports `gt_replay.csv`, `gt_replay_controls.csv`, `reaccumulation_check.csv`,
    `pair_solver_checks.csv`, `correspondence_diagnostics.csv` and `pair_metrics.csv`, plus the dataset.
  - Outputs: `results/reports/diagnostics/*.csv` and `results/reports/diagnostics/figures/*.png` (both ignored),
    `summary_generated.md` (ignored) and the tracked `worst_pairs.md`.
- **C++ dependency / test:** `slam_trajectory_test`, `trajectory_diagnostics_test` (reference). Not a test.

### `tools/evaluation/run_reference_suite.sh`
- **Category:** tooling (test runner)
- **Status:** manual: `tools/evaluation/run_reference_suite.sh OUT_DIR` (after `pixi run build`).
- **Purpose:** the reference library registers no CTests; this runs every reference test/demo program in `build/`
  with documented arguments, plus the production `features_*_test`, and records:
  - exit status and time in `OUT_DIR/tests/summary.tsv`;
  - path-normalised logs `OUT_DIR/tests/<program>.log`;
  - a byte-wise comparison of a freshly rendered Bunny dataset with `data/synthetic_bunny`
    (`OUT_DIR/tests/dataset_regeneration.txt`).

  Programs that write files do so into a temporary dataset copy or `results/runs/`; tracked data is never modified.
- **Inputs → outputs:** `build/*`, `data/tum_sample`, `data/synthetic_bunny`, the bunny mesh →
  `OUT_DIR/tests/{*.log, summary.tsv, dataset_regeneration.txt}`.
- **C++ dependency / test:** needs the full build with `SLAM_WITH_OPENCV=ON`. A test runner, not CTest.

### `tools/pipeline_analysis/export_correspondences.sh`
- **Category:** tooling
- **Status:** manual (after `pixi run build`).
- **Purpose:** exports the raw 3D→2D correspondences of all pairs from the reference pipeline. It runs
  `slam_trajectory_test --export-dir … --export-pair all` on a temporary copy and aborts if the trajectory files change.
- **Inputs → outputs:** `build/slam_trajectory_test`, `data/synthetic_bunny` →
  `results/runs/reference_results/data/raw_correspondences/pair_<i>_<i+1>_correspondences.csv` + a log.
- **C++ dependency / test:** `slam_trajectory_test`. Not a test. The milestone programs default to the frozen export
  in `tests/fixtures/reference_baseline/correspondences/` instead; `--corr FILE` / `--corr-dir DIR` select a fresh
  export.
- **Known issue:** it currently aborts ("export run changed pnp_trajectory.txt"): the current reference build does not
  reproduce the committed `data/synthetic_bunny/{pnp,icp}_trajectory.txt` and `slam_trajectory.csv` (35 of 36 rows of
  each differ, checked 2026-11-06; they were produced by the book ORB/ICP before the feature migration).

### `tools/pipeline_analysis/export_reference_orb_distances.py`
- **Category:** tooling (provenance of frozen data)
- **Status:** manual, default (OpenCV) pixi environment: `pixi run python tools/pipeline_analysis/export_reference_orb_distances.py`.
- **Purpose:** recomputes the Hamming-distance histogram of all frame-0 × frame-1 ORB descriptor pairs with
  `ORB::create()` defaults and asserts that the best-match distances equal the frozen
  `tests/fixtures/reference_baseline/reference_features/pair_0_1_raw_matches.csv`.
- **Inputs → outputs:** `data/synthetic_bunny/00000{0,1}.png` → `results/runs/reference_orb_distances_0_1.csv` (never the
  frozen copy `tests/fixtures/reference_baseline/reference_orb_distances_0_1.csv`, which `descriptor_figure.py` reads).
- **C++ dependency / test:** none (Python + OpenCV). Not a test.

---

### Tooling: `tools/evaluation/` (TUM evaluation, dashboard, diagnostics; ported from main)

These read output directories of the **reference** TUM evaluator `build/tum_trajectory_eval` (source
`reference/opencv_slambook2/tests/tum/tum_trajectory_eval.cpp`; built only with `SLAM_WITH_OPENCV=ON` and
`BUILD_MAIN_ICP_VARIANT`). A run directory holds:
- `frames.csv`, `pairs.csv`, `trajectory.csv` and `summary.txt`;
- optionally `diagnostics/pair_<i>_<j>/` (from `--export-pairs`).

The dashboard also reads `scratch_pipeline --tum-frames` runs, which it recognises by a `command.txt` that names
`scratch_pipeline`. Convention (`tum_metrics.py:10-12`): trajectory rows are T_wc, and frame 0 of PnP/ICP is anchored
to GT. Several docstrings cite `tests/tum/…` and `include/dataset/tum.hpp` relative to `reference/opencv_slambook2/`.
No Python test here is registered with CTest.

### `tools/evaluation/tum_metrics.py`
- **Category:** tooling (library + metrics definitions)
- **Status:** imported by the dashboard, plotting, stress runner, diagnostics and the Blender export.
- **Purpose:** loads, validates and computes metrics for one evaluator run:
  - raw and aligned ATE (one rigid SE(3) fit without scale);
  - final error and RPE over δ frames (Sturm 2012);
  - pair success rate.

  Failed pairs hold the previous pose and are flagged `held`. Windows containing them are excluded from RPE.
- **Inputs → outputs:** `frames.csv`, `pairs.csv`, `trajectory.csv` → Python dicts.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test (tested by `test_tum_metrics.py`).

### `tools/evaluation/plot_tum_results.py`
- **Category:** tooling
- **Status:** manual, or via `run_tum_stress.py --figures`: `--input RUN [--output DIR] [--dataset DIR] [--max-dt] [--max-gt-gap]`.
- **Purpose:** headless figures, `metrics.json` and `report.md` for one evaluator run. Exit 2 on unsupported input
  ("nothing fabricated").
- **Inputs → outputs:** the run CSVs (+ optional dataset) → `<input>/figures/`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/run_tum_stress.py`
- **Category:** tooling (driver)
- **Status:** manual: `--dataset DIR --output results/runs/stress/<session> [--binary build/tum_trajectory_eval] [--force] [--figures] [--cases …]`.
- **Purpose:** reproducible evaluator runs with these cases:
  - `full`, `n20`, `n100`, `n300`;
  - `n100_stride2`, `n100_stride4`;
  - `repeat_a`/`repeat_b`, compared file by file.

  Each case gets sanity checks and a status: ok, sanity_failed, infrastructure_error or skipped.
- **Inputs → outputs:** TUM sequence → `<output>/<case>/{command.txt, config.json, stdout/stderr.log, status.json, CSVs}`
  plus `stress_summary.{json,csv}`.
- **C++ dependency / test:** runs `tum_trajectory_eval`. A checking driver, not a unit test.

### `tools/evaluation/synthetic_runs.py`
- **Category:** test helper
- **Status:** imported by `test_tum_metrics.py`, `test_plot_tum_results.py`, `diagnostics/test_diagnostics.py` and `dashboard_app/test_methods.py`.
- **Purpose:** writes synthetic evaluator outputs (the three CSVs) with known poses and failures.
- **Inputs → outputs:** pose lists → a temporary run directory.
- **C++ dependency / test:** mimics the `tum_trajectory_eval` schema. A helper, not a test.

### `tools/evaluation/test_tum_metrics.py`
- **Category:** test (Python, standalone)
- **Status:** manual: `python3 tools/evaluation/test_tum_metrics.py`. Exit 1 on failure.
- **Purpose:** checks `tum_metrics` against synthetic trajectories with known errors.
- **Inputs → outputs:** none (temporary data) → PASS/FAIL.
- **C++ dependency / test:** none. Yes, it is a test.

### `tools/evaluation/test_plot_tum_results.py`
- **Category:** test (Python, standalone)
- **Status:** manual. Exit 1 on failure.
- **Purpose:** runs `plot_tum_results.py` as a subprocess on synthetic runs and checks its outputs and exit codes.
- **Inputs → outputs:** none → PASS/FAIL.
- **C++ dependency / test:** none. Yes, it is a test.

### `tools/evaluation/test_pose_conventions.py`
- **Category:** test (Python, standalone)
- **Status:** manual: `[--dataset DIR]` (default `data/rgbd_dataset_freiburg1_xyz`).
- **Purpose:** end-to-end pose-convention tests with non-identity poses. It checks:
  - T_wc;
  - `T_j<-i = T_wc[j]⁻¹T_wc[i]`;
  - accumulation `T_wc[i+1] = T_wc[i]·T_j<-i⁻¹`;
  - the viewer `[R row-major, t]`.

  It uses synthetic data, and the real evaluator when available.
- **Inputs → outputs:** synthetic data / optional evaluator run → PASS/FAIL.
- **C++ dependency / test:** optionally `tum_trajectory_eval`. Yes, it is a test.

### `tools/evaluation/dashboard.py`
- **Category:** tooling (local web app)
- **Status:** manual: `PYTHONNOUSERSITE=1 pixi run -e results python tools/evaluation/dashboard.py [--host 127.0.0.1] [--port 8765] [--open] [--check]`.
- **Purpose:** a standard-library `ThreadingHTTPServer`. It:
  - serves `dashboard_app/static/` and the files under `results/runs/` (URL prefix `/output/`);
  - provides JSON endpoints `/api/{runs, run, timeline, pair, registration, scene, compare, groups, sources,
    source_compare, source_scene, corr_index, corr, pair_compare}` (`:69-98`).

  `--check` loads every run and exits.
- **Inputs → outputs:** every run under `results/runs/` (read on each request, cached by mtime) → HTTP responses. A run's
  dataset is read from `<repo>/data/<sequence>` or, if absent, `$SLAM_TUM_ROOT/<sequence>` (`data.py` `dataset_dir`).
- **C++ dependency / test:** `tum_trajectory_eval`, `scratch_pipeline --tum-frames`, `tum_orb_correspondences`. Not a test.

### `tools/evaluation/dashboard_app/data.py`
- **Category:** tooling (dashboard data layer)
- **Status:** imported by `dashboard.py`.
- **Purpose:** discovers evaluator runs under `ROOT/output` and computes metrics with `tum_metrics.compute_metrics`.
  It also provides the timeline (`failure_timeline`), configuration, status, match exports, images and colourised
  depth. It may substitute the export of an equivalent run (byte-identical CSVs), and says so in its response.
- **Inputs → outputs:** run directories + dataset images → dicts for the API.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/dashboard_app/methods.py`
- **Category:** tooling (dashboard comparison layer)
- **Status:** imported by `dashboard.py`.
- **Purpose:** side-by-side method sources, compared only when their `frames.csv` selections are identical:
  - `ref_pnp` / `ref_icp` from reference evaluator runs;
  - `scratch_pnp` from `scratch_pipeline --tum-frames` runs;
  - OpenCV ORB correspondences from `tum_orb_correspondences` (`:348-396`).
- **Inputs → outputs:** `results/runs/<run>/` → dicts.
- **C++ dependency / test:** `tum_trajectory_eval`, `scratch_pipeline`, `tum_orb_correspondences`. Not a test.

### `tools/evaluation/dashboard_app/test_methods.py`
- **Category:** test (Python, standalone)
- **Status:** manual. Exit 1 on failure.
- **Purpose:** tests `methods.py` on a temporary synthetic run tree, and on the real `results/runs/` if a scratch TUM
  run exists (otherwise SKIPPED).
- **Inputs → outputs:** synthetic data / `results/runs/` → PASS/FAIL.
- **C++ dependency / test:** optionally `scratch_pipeline` output. Yes, it is a test.

### `tools/evaluation/dashboard_app/static/` (`index.html`, `app.js`, `methods.js`, `view3d.js`, `style.css`)
- **Category:** tooling (frontend)
- **Status:** served by `dashboard.py`.
- **Purpose:** browser UI:
  - `app.js`: runs, metrics, timeline, pairs; missing values are shown as "n/a";
  - `methods.js`: comparison of reference vs scratch;
  - `view3d.js`: a canvas-2D z-buffered 3D viewer. In scene mode it uses the TUM world (z up) and T_wc. In
    registration mode it uses camera i's frame, with T_i<-j applied to the source.

  Nothing in these files computes a metric.
- **Inputs → outputs:** `/api/*` JSON → page.
- **C++ dependency / test:** indirect. Not a test.

### `tools/evaluation/diagnostics/common.py`
- **Category:** tooling (library)
- **Status:** imported by all diagnostic scripts, the dashboard and `export_tum_scene.py`.
- **Purpose:** loading, validation (`validate_pair` recomputes residuals and reports disagreements) and drawing
  helpers. It also defines the conventions `T_j_from_i`, T_wc and ICP-native `T_i_from_j`.
- **Inputs → outputs:** run + `diagnostics/pair_<i>_<j>/` export → arrays and figures.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/failure_timeline.py`
- **Category:** tooling
- **Status:** manual: `--run DIR [--output DIR] [--mark N]`. Also used as a module by the dashboard.
- **Purpose:** a per-pair timeline. It shows:
  - solver status;
  - GT-free warnings: PnP inliers < 20 or ratio < 0.25, ICP inliers < 10, PnP/ICP disagreement > 5° or > 0.05 m;
  - GT error flags.
- **Inputs → outputs:** `pairs.csv`, `trajectory.csv` → `<run>/figures/failure_timeline/{timeline.png/pdf/csv/json, index.html}`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/diagnostics/pair_view.py`
- **Category:** tooling
- **Status:** manual, and called by `pair_page.py`: `--run --pair --dataset [--max-lines 150]`.
- **Purpose:** draws the correspondences of one pair on the RGB images, coloured by PnP class (inlier, outlier, no depth).
- **Inputs → outputs:** export + dataset → `<run>/../pairs/pair_<i>_<j>/{matches.png, matches_drawn.json}`.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/pair_report.py`
- **Category:** tooling
- **Status:** manual, and called by `pair_page.py`: `--run --pair --dataset [--compare N --context a-b]`.
- **Purpose:** measurements and a written markdown report of one pair, optionally against a reference pair. It
  separates observed facts from hypotheses. Exit 1 if the export is inconsistent.
- **Inputs → outputs:** export → `pair_summary.json`, `pair_<i>_<j>_report.md`.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/pair_page.py`
- **Category:** tooling
- **Status:** manual, and called by `make_diagnostics.py`.
- **Purpose:** an HTML page for one pair: matches, depth, PnP residuals under the returned pose vs GT, ICP
  registration, camera orientations, and the report.
- **Inputs → outputs:** export + dataset → `<run>/../pairs/pair_<i>_<j>/index.html` + PNGs.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/icp_registration.py`
- **Category:** tooling
- **Status:** manual / `make_diagnostics.py`: `--run --pair --dataset [--step 4]`.
- **Purpose:** before/after ICP views of the **reference** ICP (EstimateRigidRansac) under three transforms: identity,
  ICP's T_i<-j and GT T_i<-j. It draws the actual correspondences and a depth-consistency map.
- **Inputs → outputs:** export + dataset → `<run>/../icp/pair_<i>_<j>/{registration.png, registration.json}`.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/feature_tracks.py`
- **Category:** tooling
- **Status:** manual / `make_diagnostics.py`: `--run --frames a-b --dataset [--max-tracks 120]`.
- **Purpose:** chains exported per-pair matches into tracks (only where keypoints are identical across the
  exports) and draws them.
- **Inputs → outputs:** consecutive pair exports → `<run>/../tracking/frames_<a>_<b>/{tracking.png, tracks.csv, tracking.json}`.
- **C++ dependency / test:** `tum_trajectory_eval --export-pairs`. Not a test.

### `tools/evaluation/diagnostics/scene_3d.py`
- **Category:** tooling
- **Status:** manual / `make_diagnostics.py`.
- **Purpose:** a static render of the RGB-D scene placed with **GT** poses, with the GT/PnP/ICP trajectories and frustums.
- **Inputs → outputs:** run + dataset → `<run>/../scene/{scene.png, scene.json}`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/diagnostics/trajectory_divergence.py`
- **Category:** tooling
- **Status:** manual / `make_diagnostics.py`.
- **Purpose:** GT→estimate connectors coloured by error, frustums at selected frames, and RGB strips. It uses raw
  estimates only.
- **Inputs → outputs:** run + dataset → `<run>/../divergence/{divergence.png, .csv, .json}`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/diagnostics/render3d.py`
- **Category:** tooling (library)
- **Status:** imported by `scene_3d.py`, `trajectory_divergence.py` and `icp_registration.py`.
- **Purpose:** a minimal software renderer: a pinhole `View` with OpenCV axes, z-buffered point splats, and lines drawn on top.
- **Inputs → outputs:** arrays → image.
- **C++ dependency / test:** none. Not a test.

### `tools/evaluation/diagnostics/build_viewer.py`
- **Category:** tooling
- **Status:** manual / `make_diagnostics.py`: `--run --dataset [--pair] [--scene-every 10] [--frames]`.
- **Purpose:** builds the offline interactive viewer for one run:
  - frames as JPEG;
  - raw and aligned poses;
  - a sampled depth scene;
  - exported pairs.
- **Inputs → outputs:** run + dataset → `<run>/../viewer/{index.html, data.js, frames/, viewer.json}`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test.

### `tools/evaluation/diagnostics/viewer_html.py`
- **Category:** tooling (library)
- **Status:** imported by `build_viewer.py`.
- **Purpose:** writes the viewer page (`index.html` + `data.js`) with a correspondences view and a 3D view; the URL
  fragment keeps the state.
- **Inputs → outputs:** dict → HTML/JS.
- **C++ dependency / test:** none. Not a test.

### `tools/evaluation/diagnostics/make_diagnostics.py`
- **Category:** tooling (driver)
- **Status:** manual: `--dataset --frames N --stride S --export-pairs a-b --failure N [--reference N] [--windows …] [--baseline DIR] [--binary build/tum_trajectory_eval] [--force]`.
- **Purpose:** one command for the full visual diagnostics of one configuration:
  - runs the evaluator with `--export-pairs`;
  - optionally checks byte identity against a baseline run;
  - then runs every view above and writes `manifest.json`.
- **Inputs → outputs:** TUM sequence → `results/runs/diagnostics/<sequence>/n<frames>_stride<stride>/{evaluator, pairs, icp,
  tracking, scene, divergence, viewer}/`, plus `results/runs/diagnostics/index.html`.
- **C++ dependency / test:** runs `tum_trajectory_eval`. Not a test.

### `tools/evaluation/diagnostics/test_diagnostics.py`
- **Category:** test (Python, standalone)
- **Status:** manual: `[--dataset DIR]`. Synthetic checks always run. The evaluator checks run only if
  `build/tum_trajectory_eval` and the dataset exist; otherwise they are SKIPPED (`:287, 314`).
- **Purpose:** tests the export and diagnostic modules. With the evaluator present, it checks that `--export-pairs`
  on/off gives identical standard outputs.
- **Inputs → outputs:** synthetic / evaluator → PASS/FAIL.
- **C++ dependency / test:** optionally `tum_trajectory_eval`. Yes, it is a test.

---

### Visualization: `visualization/`

**Coordinate conversion SLAM camera → Blender** (`visualization/scripts/slam_coords.py`):

- **Input:** trajectory poses are T_wc. The translation is the camera centre in the C++ world frame (Y up, metres).
  The quaternion maps CV camera axes (+X right, +Y down, +Z forward) to world axes (`:7-15`).
- **Per-camera fix:** `R_FIX = diag(1, −1, −1)`, a 180° rotation about the camera's own X axis (`:27`).
  `trajectory_to_blender_pose(pos, quat_cv)` returns `pos` unchanged and `quat = R_wc · R_FIX` (`:51-68`). This turns
  the CV camera (looking along +Z, Y down) into a Blender camera (looking along −Z, Y up). It does not invert,
  transpose or re-compose the pose.
- **Bunny scene only:** every object is parented under a "WorldRoot" empty rotated +90° about X,
  `WORLD_ROOT_ROTATION` (`:37`), so the Y-up C++ world displays Z-up in Blender: `(x, y, z) → (x, −z, y)`.
  `build_scene.py` adds a recentring translation (`build_scene.py:44-50`).
- **TUM scenes:** use the TUM world directly (z up), with no WorldRoot. Cameras still get `R_wc @ R_FIX`
  (`build_tum_scene.py:11-20`).

### `visualization/README.md`
- **Category:** visualization (documentation)
- **Status:** current, with some stale passages (below).
- **Purpose and why it exists:** explains the Bunny scene:
  - GT (green), legacy PnP (orange), corrected ICP (magenta), Scratch PnP (cyan);
  - that `results/benchmarks/rgbd_icp/selected_trajectory.txt` selects pose-graph or point-to-plane;
  - controls, and the TUM scene workflow and conventions.
- **Key functions / CLI:**
  - `bash visualization/rebuild_blender.sh` and `blender -y results/blender/bunny_slam_demo.blend`;
  - the TUM commands (`export_tum_scene.py` → `build_tum_scene.py` → `compose_reconstructions.py` → `test_tum_reconstruction.py`).
- **Inputs → outputs:** n/a.
- **Dependencies:** n/a.
- **Used by:** readers; linked from the root `README.md:54`.
- **Covered by tests:** n/a.
- **Before studying it, understand:** it also records the rebuilt scene's equivalence with the original
  Blender-4.0 scene (one non-functional difference: the saved viewport orientation, 5.9e-6°), the Blender 6.0
  deprecations and why background renders show no HUD.

### `visualization/rebuild_blender.sh`
- **Category:** tooling (end-to-end driver)
- **Status:** `pixi run blender-rebuild` (`pixi.toml:16`) or `bash visualization/rebuild_blender.sh`.
- **Purpose and why it exists:** the one command that rebuilds every input of the Bunny scene from C++ and then the scene:
  1. Production CMake in `build-scratch` (no OpenCV) + `ctest`, then `scratch_pipeline --repo` (`:7-10`).
  2. `feature_audit` + render/validate for pairs 0/1 and 15/16 (`:12-17`).
  3. `build-icp` (with `BUILD_ORB_BENCHMARK=ON`) + `ctest`, then `icp_sequence → results/benchmarks/rgbd_icp` (exit 2 tolerated) (`:18-29`).
  4. `orb_benchmark` and the ORB renderers; `render_outputs.py`; `validate_outputs.py` (`:30-34`).
  5. `blender --background --factory-startup --python visualization/scripts/build_scene.py` (`:44`).
  6. `check_blender_consistency.py` → `results/benchmarks/rgbd_icp/blender_consistency.csv` (`:45-46`).
  The fresh T0 run (`:13`, `--out results/runs/pipeline`) is only compared with the frozen scene trajectory
  (`:15-19`, informational); the scene never reads it.
- **Key functions / CLI:** no arguments.
- **Inputs → outputs:** dataset → `results/runs/pipeline/`, `results/reports/figures/`, `results/benchmarks/rgbd_icp/`,
  `results/blender/bunny_slam_demo.blend`.
- **Dependencies:** a C++17 compiler, CMake, OpenCV, Eigen, system `python3` with matplotlib/numpy/Pillow, and
  `blender` on PATH. The pixi default environment has no Python package listed.
- **Used by:** `tools/figures/run_gn_orb.sh --blender`.
- **Covered by tests:** runs both CTest suites and the Python/Blender checks.
- **Before studying it, understand:** the three build trees (`build/`, `build-scratch/`, `build-icp/`). The last line
  (`:39`) prints where the images are: `xdg-open results/reports/figures`.

### `visualization/export_scene_state.py` and `visualization/compare_scene_state.py`
- **Category:** visualization (verification)
- **Status:** manual: `blender -b --factory-startup SCENE.blend --python visualization/export_scene_state.py -- OUT.json`,
  then `python3 visualization/compare_scene_state.py OLD.json NEW.json [--tol 1e-9] [--expect PROP] [--expect-text NAME]`.
- **Purpose:** compare two builds of a scene (e.g. before/after a rebuild). The exporter (read-only, never saves)
  writes metadata, per-object scale/dimensions, mesh vertex fingerprints, curve control points, camera intrinsics,
  every object's world matrix and visibility at every frame, the scene properties and text blocks, and a second pass
  with the scene's own runtime registered (`slam_hud_panel`: sidebar settings, every frame with the frame handler
  active). The comparator checks each category separately (camera position / orientation / viewing direction per
  frame, all matrices, Bunny, trajectories, intrinsics, visibility, properties, texts, runtime) and reports intended
  changes (`--expect`) with both values; exit 0 only if every category passes.
- **Covered by tests:** no automated test; checked manually by comparing a scene with itself (all categories pass)
  and with a copy whose camera was moved by 1 µm (the camera-position category fails).

### `visualization/check_blender_consistency.py`
- **Category:** test (Blender, read-only)
- **Status:** run by `rebuild_blender.sh:36-37`:
  `blender -b -y results/blender/bunny_slam_demo.blend --python visualization/check_blender_consistency.py -- OUT.csv`.
- **Purpose and why it exists:** verifies the saved scene against the data. It first rejects a stale scene by
  comparing `scene["slam_corrected_source"]["source_sha256"]` with `trajectory_data.load_corrected` (`:33-36`). Then,
  per frame and method, it checks:
  - camera position and rotation;
  - view direction;
  - curve points;
  - HUD values;
  - the animation range;
  - the intrinsic projection.

  The thresholds are asserts at `:159-181` (e.g. position < 1e-5 m, rotation < 0.05°, projection < 0.001 px).
  The repository is `trajectory_data.find_repo_root` from the `.blend` (`:34`; `$SLAM_REPO_ROOT` overrides). The
  scratch camera is compared with the frozen fixture `SCRATCH_SOURCE`; a scene recording another scratch source,
  a missing fixture or a missing camera is an error (`:44-51`), never a silently skipped camera.
- **Key functions / CLI:** the output CSV path follows `--`. The default is `results/reports/diagnostics/blender_consistency.csv`.
- **Inputs → outputs:** the `.blend`, `load_corrected(ROOT)` rows (`results/benchmarks/rgbd_icp/`, `data/synthetic_bunny/`) and
  the frozen scratch trajectory `tests/fixtures/bunny_t0_scene/trajectory.csv` → CSV + `BLENDER_*` console lines.
- **Dependencies:** `bpy`, `mathutils`, `slam_animation`, `slam_coords.R_FIX`, `trajectory_data`.
- **Used by:** `rebuild_blender.sh`.
- **Covered by tests:** it is a check.
- **Before studying it, understand:** `slam_coords.py`. The docstring (`:8-10`) says it compares against
  `data/synthetic_bunny/slam_trajectory.csv`, but the code compares against `load_corrected` (the corrected ICP from
  `results/benchmarks/rgbd_icp/` and the legacy PnP from `data/synthetic_bunny/pnp_trajectory.txt`). The code wins.

### `visualization/scripts/slam_coords.py`
- **Category:** visualization (library)
- **Status:** imported by `build_scene.py`, `slam_animation.py`, `slam_hud_panel.py`, `build_tum_scene.py` and
  `check_blender_consistency.py`.
- **Purpose and why it exists:** the single place for the pose conversion, so the scripts cannot drift apart
  (`:1-5`). See the conversion summary above.
- **Key functions / CLI:** `R_FIX` (`:27`), `WORLD_ROOT_ROTATION` (`:37`), `pose_from_row(row, prefix)` (`:40-48`),
  `trajectory_to_blender_pose(pos, quat_cv)` (`:51-68`).
- **Inputs → outputs:** pose → Blender pose.
- **Dependencies:** `mathutils` (Blender only).
- **Used by:** see Status.
- **Covered by tests:** indirectly, by `check_blender_consistency.py`.
- **Before studying it, understand:** the OpenCV and Blender camera axis conventions.

### `visualization/scripts/trajectory_data.py`
- **Category:** visualization (data bridge, no Blender dependency)
- **Status:** called by `build_scene.main()` (`load_corrected(REPO_ROOT, write=True)`, `build_scene.py:911`), by
  `check_blender_consistency.py`, and by the embedded autoload text. Also runnable standalone.
- **Purpose and why it exists:** builds the Blender table from the C++ T_wc files.
  - **Sources:**
    - GT from `data/synthetic_bunny/groundtruth.txt`;
    - **legacy PnP** from `data/synthetic_bunny/pnp_trajectory.txt`;
    - **corrected ICP** from `ICP_RESULTS/<selected>_trajectory.txt` (`ICP_RESULTS = results/benchmarks/rgbd_icp`,
      `:15`), where the selection must be `plane` or `pose_graph` (`:72-80`).
  - **Paths:** `ICP_RESULTS` and `SCRATCH_SOURCE` (`:15-16`) are the scene's only input paths; `build_scene.py`,
    `check_blender_consistency.py` and `test_blender_data.py` import them. `find_repo_root(start)` (`:21-38`):
    `$SLAM_REPO_ROOT` if set, else the nearest directory at or above `start` holding `ROOT_MARKERS` (`:18`), else
    `RuntimeError`; the scene's autoload text applies the same rule.
  - **Validation:**
    - equal frame counts and stamps;
    - frame 0 within 1e-5 m / 0.01° of GT;
    - `pair_metrics.csv` covers every pair.
  - **Per-frame values:** recomputes the per-frame errors; `icp_valid` comes from the pair status.
  - **Source fingerprint:** a SHA-256 over the base names and bytes (not the paths) of all source files, including
    the scratch trajectory (`_hash`, `:62-67`; `:109-111`).
- **Key functions / CLI:** `read_tum`, `errors`, `load_corrected(root, write=False)`. `python3 trajectory_data.py`
  writes the files.
- **Inputs → outputs:** above → with `write=True`, `results/benchmarks/rgbd_icp/{blender_trajectory.csv, blender_source.json}`.
- **Dependencies:** standard library only.
- **Used by:** `build_scene.py`, `check_blender_consistency.py`, `test_blender_data.py`.
- **Covered by tests:** `visualization/tests/test_blender_data.py`.
- **Before studying it, understand:**
  - It raises if `results/benchmarks/rgbd_icp/run_complete.txt` or the frozen `tests/fixtures/bunny_t0_scene/trajectory.csv` is missing.
  - The "ICP" in the Bunny scene is the corrected `rgbd_icp` trajectory, not `data/synthetic_bunny/icp_trajectory.txt`.

### `visualization/scripts/build_scene.py`
- **Category:** visualization
- **Status:** run by `rebuild_blender.sh:44`, or `blender [--background] --factory-startup --python visualization/scripts/build_scene.py`
  (Blender 5.x; the saved scene needs Blender ≥ 5.0).
- **Purpose and why it exists:** builds and saves `results/blender/bunny_slam_demo.blend` (`SAVE_PATH`, `:94`):
  - the bunny mesh;
  - four keyframed real Camera objects (GT, PnP, ICP, ScratchPnP) with trajectory curves;
  - WorldRoot, a ground plane, lights and render settings;
  - an embedded autoload text block (`:826-857`) that locates the repository at load time (`$SLAM_REPO_ROOT`, else
    the nearest directory at or above the `.blend` holding the repository markers; no fixed relative path),
    re-registers the HUD and checks the source fingerprint.

  It labels ICP "ICP + loop closure" when the pose graph was selected, and PnP "PnP (legacy)" (`:923-924`). It
  verifies the displayed values against `load_corrected` before clearing the scene (`verify_trajectory_conventions`, `:308-319`).
- **Key functions / CLI:** constants at the top (`SHOW_*`, `CAMERA_SAMPLE_STEP`, `PRESENTATION_MODE`, `METHOD_STYLES` at `:150`); `main()` at `:915`.
- **Inputs → outputs:**
  - Inputs:
    - `data/meshes/bunny/bun_zipper.ply`;
    - `data/synthetic_bunny/intrinsics.txt`;
    - `load_corrected` rows;
    - the frozen `tests/fixtures/bunny_t0_scene/trajectory.csv` (`SCRATCH_TRAJECTORY_CSV_PATH`, `:93`; required when
      `SHOW_SCRATCH_PNP`, `:925-926`).
  - Outputs: the `.blend` and `results/benchmarks/rgbd_icp/blender_trajectory.csv`/`blender_source.json` (via `write=True`).
- **Dependencies:** `bpy`, `mathutils`, `slam_animation`, `slam_hud_panel`, `slam_coords`, `trajectory_data`.
- **Used by:** `rebuild_blender.sh`. `build_tum_scene.py` imports its `METHOD_STYLES`.
- **Covered by tests:** `check_blender_consistency.py`.
- **Before studying it, understand:** `slam_coords.py`. The docstring (`:9-17`) says the scene reads
  `results/benchmarks/rgbd_icp/blender_trajectory.csv`; the constant `TRAJECTORY_CSV_PATH` (`:87`) is defined but not read. The
  rows come from `load_corrected`, which *writes* that CSV.

### `visualization/scripts/slam_animation.py`
- **Category:** visualization
- **Status:** imported by `build_scene.py`, `slam_hud_panel.py`, `check_blender_consistency.py`.
- **Purpose and why it exists:** per-frame update. Blender frame f shows pose `clamp(f, 0, N−1)`. It also handles
  progressive trails, drift vectors, visibility, and the HUD error lookup `error_at()`. Methods with fewer poses are
  hidden after their last pose.
- **Key functions / CLI:** `update_animation`, `world_position`, `error_at`, role constants (`:32-36`).
- **Inputs → outputs:** `scene["slam_trajectory"]` and the object tags → scene state.
- **Dependencies:** `bpy`, `slam_coords`.
- **Used by:** see Status.
- **Covered by tests:** `check_blender_consistency.py` (HUD values).
- **Before studying it, understand:** Blender handlers and ID properties.

### `visualization/scripts/slam_hud_panel.py`
- **Category:** visualization
- **Status:** registered by `build_scene.py`, by the embedded autoload text, or with `--python …/slam_hud_panel.py`.
- **Purpose and why it exists:**
  - the sidebar "Trajectories" tab (toggles, intrinsics, per-frame centres and errors);
  - camera helper operators;
  - a POST_PIXEL HUD overlay;
  - the `frame_change_post` handler.

  It never recomputes a pose.
- **Key functions / CLI:** `SLAMVizSettings`, `SLAM_PT_trajectories`, `get_gt_camera()`/`set_active_camera()`, `register()`.
- **Inputs → outputs:** scene data → UI.
- **Dependencies:** `bpy`, `slam_animation`, `slam_coords`.
- **Used by:** `build_scene.py`.
- **Covered by tests:** no automated test of the UI.
- **Before studying it, understand:** the Blender UI/handler API.

### `visualization/scripts/export_tum_scene.py`
- **Category:** visualization (exporter; runs outside Blender)
- **Status:** manual: `PYTHONNOUSERSITE=1 pixi run -e results python visualization/scripts/export_tum_scene.py --run RUN [--dataset DIR] [--out DIR] [--force] [--include-frames a,b]`.
- **Purpose:** exports one `tum_trajectory_eval` run for the TUM Blender scene:
  - `trajectory.json`;
  - three PLY clouds per density (placed with GT, PnP and ICP poses), with manifests;
  - `comparison_<density>.json`;
  - depth previews and match images.

  Density presets (`:48`): preview = every 10th frame on an 8 px grid; render = every 3rd frame on a 4 px grid.
- **Inputs → outputs:** evaluator run + dataset → `results/runs/blender/<sequence>/<run>/`.
- **C++ dependency / test:** `tum_trajectory_eval`. Not a test. Imports `tools/evaluation/{tum_metrics, diagnostics/common}`.

### `visualization/scripts/tum_reconstruction.py`
- **Category:** visualization (library, NumPy)
- **Status:** imported by `export_tum_scene.py` and `test_tum_reconstruction.py`.
- **Purpose:** three reconstructions that share frames and pixels and differ only in the placing pose:
  `X_world = R_wc X_cam + t_wc`, with `z = raw / depth_factor`. Estimated modes drop frames with held poses, decided
  from the evaluator's pair status only, without GT.
- **Inputs → outputs:** depth/RGB arrays + poses → point arrays.
- **C++ dependency / test:** `tum_trajectory_eval` data. Not a test.

### `visualization/scripts/build_tum_scene.py`
- **Category:** visualization
- **Status:** manual: `blender --background --python visualization/scripts/build_tum_scene.py -- --export DIR [--density preview|render] [--frame N] [--save …] [--render …] [--view …] [--mode …] [--render-all PREFIX] [--samples 16] [--check]`.
- **Purpose:** builds the TUM scene: point clouds, GT/PnP/ICP curves, animated cameras, frustums, connectors, an
  optional aligned view and an overview camera. Blender frame = trajectory frame + 1. `--check` verifies the scene
  numerically against the export. Scenes go to `visualization/scenes/tum/` (ignored).
- **Inputs → outputs:** `export_tum_scene.py` export → `.blend` / renders.
- **C++ dependency / test:** indirect (`tum_trajectory_eval`). `--check` is a built-in check. The docstring (`:4`)
  says the Bunny scene is at `scenes/bunny_slam_demo.blend`; it is `results/blender/`.

### `visualization/scripts/build_tum_scene_io.py`
- **Category:** visualization (library)
- **Status:** imported by `build_tum_scene.py` and the tests.
- **Purpose:** binary little-endian PLY reader (xyz float32 + rgb uint8).
- **Inputs → outputs:** PLY → arrays.
- **C++ dependency / test:** none. Not a test.

### `visualization/scripts/tum_scene_panel.py`
- **Category:** visualization
- **Status:** embedded into the TUM `.blend` as a text block that registers on load.
- **Purpose:** the "TUM SLAM" sidebar. It provides frame selection, a reconstruction selector, and the RGB, depth
  and match images, timestamps, errors and pose status. Everything comes from `scene["tum_data"]`.
- **Inputs → outputs:** scene data → UI.
- **C++ dependency / test:** none directly. Not a test.

### `visualization/scripts/compose_reconstructions.py`
- **Category:** visualization
- **Status:** manual: `--export DIR --prefix PREFIX --density preview|render [--out PNG]`.
- **Purpose:** a side-by-side image of the three renders made by `build_tum_scene.py --render-all`, captioned from
  `comparison_<density>.json`. The images are not modified.
- **Inputs → outputs:** `<prefix>_{gt,pnp,icp}.png` + JSON → one PNG.
- **C++ dependency / test:** none. Not a test.

### `visualization/scripts/test_tum_reconstruction.py`
- **Category:** test (Python, standalone)
- **Status:** manual: `python3 visualization/scripts/test_tum_reconstruction.py [--export DIR …]`. Exit 1 on failure.
- **Purpose:** synthetic checks of the reconstruction. With `--export`, it recomputes every exported cloud from the
  dataset and `trajectory.csv` and compares it with the PLY.
- **Inputs → outputs:** synthetic data / export → PASS/FAIL.
- **C++ dependency / test:** optionally `tum_trajectory_eval` output. Yes, it is a test.

### `visualization/tests/test_blender_data.py`
- **Category:** test (Python unittest)
- **Status:** CTest `blender_data_bridge`, **only in the `build-icp` project** (`experiments/rgbd_icp/icp_cpp/CMakeLists.txt:27-31`);
  not in the root build. Can also be run directly.
- **Purpose:** tests `trajectory_data.load_corrected` on a temporary copy of the real files:
  - 36 rows, `pose_graph` selected, ATE < 0.02 m;
  - a missing completion marker raises;
  - a `plane` selection works;
  - the fingerprint changes when a source changes;
  - a missing pose raises;
  - a rejected pair shows as `icp_valid = 0`.
- **Inputs → outputs:** copies of `data/synthetic_bunny/*`, `results/benchmarks/rgbd_icp/*`, the mesh and the frozen
  `tests/fixtures/bunny_t0_scene/trajectory.csv` (paths from `trajectory_data.ICP_RESULTS`/`SCRATCH_SOURCE`) → unittest result.
- **C++ dependency / test:** reads only tracked or frozen files (no `scratch_pipeline` run is needed). Yes, it is a test (CTest `blender_data_bridge` in `build-icp`).

---

### Configuration and build

### `pixi.toml`
- **Category:** build/config
- **Status:** used (`pixi run …`).
- **Purpose and why it exists:** the conda-forge environments and task shortcuts (workspace `implementation`, linux-64).
- **Key functions / CLI:**
  - **Default environment.** Dependencies `opencv >=4,<5`, `cmake >=4.4.3,<5`, `cxx-compiler`, `eigen >=5.0.1,<6`,
    `g2o 2019.11.23`, `sophus 1.24.6+` (`:18-24`). Tasks:
    - `configure` = `cmake -B build -S . -DSLAM_WITH_OPENCV=ON` (`:11`).
    - `build` = `cmake --build build -j`, depends on `configure` (`:12`).
    - `icp-configure` = `cmake -S experiments/rgbd_icp/icp_cpp -B build-icp -DCMAKE_BUILD_TYPE=Release` (`:13`).
    - `icp-build` = `cmake --build build-icp -j2`, depends on `icp-configure` (`:14`).
    - `icp` = `./build-icp/icp_sequence data/synthetic_bunny results/benchmarks/rgbd_icp`, depends on `icp-build` (`:15`).
    - `blender-rebuild` = `bash visualization/rebuild_blender.sh` (`:16`).
  - **Environment `results`.** Feature `results` only, `no-default-feature = true` (`:38-39`): `python 3.12.*`,
    `numpy >=2`, `matplotlib-base >=3.9` (`:29`). Tasks, all with `PYTHONNOUSERSITE=1`:
    - `results` → `tools/figures/generate_results.py`;
    - `diagnostics` → `tools/evaluation/run_diagnostics.py`;
    - `correspondences` → `tools/figures/generate_correspondence_figure.py`;
    - `correspondence-map` → `tools/figures/generate_correspondence_map.py` (`:31-35`).
  - The comment `:9` gives the production-only build (`cmake -S . -B build && cmake --build build -j`), which has no
    pixi task.
- **Inputs → outputs:** n/a.
- **Dependencies:** pixi, conda-forge; the lock is in `pixi.lock`.
- **Used by:** everything that says `pixi run`.
- **Covered by tests:** n/a.
- **Before studying it, understand:**
  - The default environment always installs OpenCV/g2o/Sophus, even though production needs only Eigen.
  - No pixi task runs `ctest` or the production-only build.
  - The `results` tasks need `build/` from `pixi run build` (the reference executables).

### `.gitignore`
- **Category:** build/config
- **Status:** used.
- **Purpose and why it exists:** the ignored categories, in file order:
  - **pixi:** `.pixi/*`, except `config.toml`.
  - **CMake/C++ build trees and artifacts:** `build/`, `build-icp/`, `build-scratch/`, `cmake-build-*`, caches,
    object/library files.
  - **Python caches.**
  - **Blender:** generated scenes `visualization/scenes/*.blend` and `visualization/scenes/tum/`, Blender's
    `.blend1-3` backups, crash files, renders in `results/render/`.
  - **Runtime output:** `results/runs/`, the default output of every program and script.
  - **Generated diagnostics:** `results/reports/diagnostics/{cpp, figures}/`, `*.csv`, `summary_generated.md`.
  - **TUM sequences:** `data/rgbd_dataset_*` (an external dataset, normally located through `SLAM_TUM_ROOT`).
  - **Logs** (`*.log`).
  - **Editors and OS files.**
- **Key functions / CLI:** n/a.
- **Inputs → outputs:** n/a.
- **Dependencies:** n/a.
- **Used by:** git.
- **Covered by tests:** n/a.
- **Before studying it, understand:** `results/benchmarks/rgbd_icp/` and `results/blender/*.blend` are **tracked**.

### `cmake/orb_benchmark.cmake`
- **Category:** build/config
- **Status:** included by the root `CMakeLists.txt:133` (inside `if(SLAM_WITH_OPENCV)`) and by
  `experiments/rgbd_icp/icp_cpp/CMakeLists.txt:33`.
- **Purpose and why it exists:** a single definition of the `orb_benchmark` executable, shared by both builds.
- **Key functions / CLI:**
  - `option(BUILD_ORB_BENCHMARK … ON)`;
  - `find_library(opencv_features2d REQUIRED)`;
  - `add_executable(orb_benchmark experiments/orb_opencv/orb_benchmark.cpp)`, linked to `rgbd_icp` + features2d,
    with `-Wall -Wextra -Wpedantic`;
  - CTest `orb_benchmark_smoke` = `orb_benchmark data/synthetic_bunny <binary dir>/orb-smoke --frames 2 --repeats 1` (`:1-10`).
- **Inputs → outputs:** n/a.
- **Dependencies:** the target `rgbd_icp` must already exist (defined in `experiments/rgbd_icp/`).
- **Used by:** both builds above.
- **Covered by tests:** defines `orb_benchmark_smoke`.
- **Before studying it, understand:** it locates the repository root as `${CMAKE_CURRENT_LIST_DIR}/..`, so it works
  from either including project.
