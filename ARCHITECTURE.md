# Architecture

The final system is organized by SLAM function, not by book chapter.
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
  geometry/      pose_estimation_2d2d() (F/E/H + recoverPose), triangulation()
  optimization/  VertexPose -- identical g2o SE3 vertex duplicated in the book's
                 pose_estimation_3d2d.cpp and pose_estimation_3d3d.cpp, extracted once
                 (header-only: no .cpp, so it only exists under include/)
  tracking/      PnP (pnp.cpp) and ICP (icp.cpp) pose estimation + BA refinement
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

Not created yet: `mapping/`, `backend/`, `loop_closing/`, `system/` --
there's no multi-frame state (`Frame`, `Map`, `KeyFrame`) yet for any of
these to hold. They get designed when we actually build that layer.

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

OpenCV, Eigen, g2o, Sophus -- all via conda-forge, pinned to versions where
the book's OpenCV-2.4/3-era code compiles with the minimal changes above
(OpenCV 4.x, not 5.x).
