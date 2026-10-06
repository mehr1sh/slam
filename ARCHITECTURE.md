# Architecture

The project is a from-scratch C++ visual SLAM system under construction. Its
code is organized by SLAM function, not by book chapter, so that each stage of
the target pipeline has one place to live.

## Target pipeline and where each stage lives

```
frame input ─▶ features ─▶ matching ─▶ geometry (2D-2D, PnP, triangulation) ─▶ tracking
                                                                                 │
        loop closing ◀── pose graph / bundle adjustment ◀── local mapping ◀── landmarks
```

| Stage | Current home | State |
|---|---|---|
| Image I/O | `pnp_from_scratch/src/png.*` | from scratch |
| Features, matching | `src/features/` (`feature_core`, `cmake/feature_core.cmake`) | from scratch, shared by both builds |
| Camera model | `pnp_from_scratch/src/projection.*` (scratch), `src/camera/` (reference) | both |
| Two-view geometry | `pnp_from_scratch/src/essential.*` (scratch), `src/geometry/` (reference, OpenCV) | both |
| PnP + RANSAC + refinement | `pnp_from_scratch/src/{pnp,ransac,refine}.*` | from scratch |
| ICP / 3D-3D alignment | `src/tracking/icp.cpp` | reference |
| Triangulation | `src/geometry/` | reference (OpenCV) only |
| Frame-to-frame tracking | `pnp_from_scratch/src/pipeline.*`, `pipeline_main.cpp` | from scratch; no keyframes |
| Landmarks / map, local mapping, backend optimization, loop closing, system | — | not started |

`pnp_from_scratch/` is a separate CMake project today so that its no-OpenCV
property can be verified by construction (it links only Eigen and
`feature_core`). As the from-scratch stages mature they are meant to move into
function-named modules (`features/` is the first to have done so), and the
mapping, backend and loop-closing layers will be added next to them, once
there is multi-frame state (`Frame`, `Map`, `KeyFrame`) for them to hold.

Everything under `src/render/`, `data/`, `visualization/`, `scripts/`,
`pnp_from_scratch/experiments/` and `pnp_from_scratch/tools/` is evaluation
infrastructure: the synthetic test sequence (a rendered test object with exact
ground truth), Blender visualization, plots and diagnostics. None of it is
part of the SLAM computation.

## The reference (book-derived) modules

The reference modules below follow the book's structure.
The book's companion code, slambook2 (<https://github.com/gaoxiang12/slambook2>,
MIT; `ch6/` = nonlinear optimization, `ch7/` = visual odometry), is the algorithm
reference -- it is not vendored here, see `THIRD_PARTY_NOTICES.md`; `src/` is the actual
implementation, built by lifting each algorithm essentially verbatim into
the module it functionally belongs to. Code taken directly from the book is
marked with `// Book code starts` / `// Book code ends`.

```
include/<module>/*.hpp   public interface of each module
src/<module>/*.cpp       implementation
  camera/        PinholeCamera, pixel2cam()
  features/      find_feature_matches() -- identical across 5 book files, extracted once
                 (features.cpp: the book interface, now implemented with feature_core
                 since migration checkpoint 01; the rest of features/ is feature_core)
  geometry/      pose_estimation_2d2d() (F/E/H + recoverPose), triangulation()
  optimization/  VertexPose -- identical g2o SE3 vertex duplicated in the book's
                 pose_estimation_3d2d.cpp and pose_estimation_3d3d.cpp, extracted once
                 (header-only: no .cpp, so it only exists under include/)
  tracking/      PnP (pnp.cpp) and ICP (icp.cpp) pose estimation + BA refinement;
                 fast/brief/hamming_matching/essential_matrix.cpp are earlier
                 from-scratch prototypes on OpenCV image types, superseded by
                 feature_core and pnp_from_scratch/ (kept with their unit tests)
tests/           one executable per component, adapted from the book's own main()s
```

Headers live under `include/<module>/`, implementation under `src/<module>/`,
the conventional C++ split -- each library target declares `include/` as a
`PUBLIC` include directory (`target_include_directories`), so consumers get
the right path without a project-wide `include_directories()`. Dependencies
between modules are declared with `PUBLIC`/`PRIVATE` on `target_link_libraries`
based on whether the dependency's types appear in the module's own public
header: e.g. `geometry` links `camera` `PRIVATE` (used inside `geometry.cpp`
for `pixel2cam()`, but `geometry.hpp` itself never mentions `PinholeCamera`),
while `tracking` links `Eigen3::Eigen`/`Sophus::Sophus` `PUBLIC` (their types
appear directly in `pnp.hpp`/`icp.hpp`) but g2o `PRIVATE` (only used inside
`pnp.cpp`/`icp.cpp` for the BA solver setup, never in the public headers).

Only `find_feature_matches` and `VertexPose` were extracted into shared
code -- both are byte-for-byte duplicated across multiple book files, so
sharing them is deduplication, not invented abstraction. Everything else
(the EdgeProjection/EdgeProjectXYZRGBDPoseOnly g2o edges, the solver setup
in each BA function) differs between PnP and ICP and stays as the book has
it, one copy each in `tracking/pnp.cpp` / `tracking/icp.cpp`.

Not created yet: `mapping/`, `backend/`, `loop_closing/`, `system/` (see the
target pipeline above).

## What's book-verbatim vs. changed

Book code is used as-is; only two kinds of minimal, environment-forced
changes were made (both `// Book code starts/ends` blocks keep the
original syntax otherwise):

1. One added legacy-constants `#include` per file needing
   `CV_LOAD_IMAGE_COLOR`/`CV_FM_8POINT`/`CV_LOAD_IMAGE_UNCHANGED` -- OpenCV
   split these into a compat header after the book was written against
   OpenCV 2.4/3. No symbol renamed.
2. `pixel2cam()`/`find_feature_matches`/`VertexPose`, each defined
   identically in multiple book files, are defined once and declared in a
   header so every caller links against the same definition -- required
   simply because C++ forbids multiple definitions across translation
   units in one binary; this is integration, not rewriting.

`tests/unit/orb_from_scratch_test.cpp`, `gauss_newton_curve_fit_test.cpp`,
`g2o_curve_fit_test.cpp` are unmodified copies (only two hardcoded file
paths fixed in the first).

## Coordinate convention

- `(R, t)` transforms points from frame/cloud 1 into frame/cloud 2:
  `x2 = R*x1 + t`.
- Triangulated points are in camera 1's frame (`P1 = K[I|0]`).
- `pixel2cam(p, K)`: pixel -> normalized camera coordinates, `K^-1 * p`.
- 2D-2D `t` (from `recoverPose`) is unit-norm -- monocular scale ambiguity,
  never metres. PnP/ICP recover real (depth-scaled) `t` because they use
  the TUM depth maps.

## Dependencies (pixi.toml)

The from-scratch track needs only a C++17 compiler and Eigen. The reference
modules need OpenCV, Eigen, g2o, Sophus -- all via conda-forge, pinned to versions where
the book's OpenCV-2.4/3-era code compiles with the minimal changes above
(OpenCV 4.x, not 5.x).
