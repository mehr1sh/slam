# Third-party notices

This repository's own code is MIT-licensed (see `LICENSE`). It also contains
the third-party material listed below, each under its own terms.

## 1. slambook2 — code (MIT)

Source: <https://github.com/gaoxiang12/slambook2>, the companion code of
*14 Lectures on Visual SLAM: From Theory to Practice* (2nd ed.) by Xiang Gao
and Tao Zhang et al.

Several algorithm implementations were taken from the book's `ch6/` and `ch7/`
examples, with minimal changes (header/linking fixes, file-output paths).
Inside the source files these blocks are marked `// Book code starts` …
`// Book code ends`.

| File(s) | Derived from (slambook2) | What |
|---|---|---|
| `src/camera/camera.cpp`, `include/camera/camera.hpp` | `ch7/pose_estimation_*.cpp` | `pixel2cam()` |
| `src/features/features.cpp`, `include/features/features.hpp` | `ch7/pose_estimation_*.cpp`, `ch7/triangulation.cpp` | `find_feature_matches()` (ORB + brute-force Hamming + distance filter) |
| `src/geometry/geometry.cpp`, `include/geometry/geometry.hpp` | `ch7/pose_estimation_2d2d.cpp`, `ch7/triangulation.cpp` | 2D-2D pose estimation, triangulation |
| `include/optimization/g2o_types.hpp` | `ch7/pose_estimation_3d2d.cpp`, `ch7/pose_estimation_3d3d.cpp` | g2o `VertexPose` |
| `src/tracking/pnp.cpp`, `include/tracking/pnp.hpp` | `ch7/pose_estimation_3d2d.cpp` | Gauss-Newton and g2o PnP bundle adjustment |
| `src/tracking/icp.cpp`, `include/tracking/icp.hpp` | `ch7/pose_estimation_3d3d.cpp` | SVD 3D-3D alignment, g2o refinement |
| `tests/integration/{feature_matching,two_view_pose,triangulation,pnp,icp}_test.cpp` | the matching `ch7/*.cpp` `main()` functions | adapted example programs |
| `tests/unit/orb_from_scratch_test.cpp` | `ch7/orb_self.cpp` | near-verbatim copy |
| `tests/unit/gauss_newton_curve_fit_test.cpp` | `ch6/gaussNewton.cpp` | near-verbatim copy |
| `tests/unit/g2o_curve_fit_test.cpp` | `ch6/g2oCurveFitting.cpp` | near-verbatim copy |

License of slambook2 (reproduced as its terms require):

```
MIT License

Copyright (c) 2018 Xiang Gao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Stanford Bunny — `data/meshes/bunny/bun_zipper.ply`

Source: The Stanford 3D Scanning Repository, Stanford University Computer
Graphics Laboratory, <https://graphics.stanford.edu/data/3Dscanrep/>.

The repository permits research use and free redistribution with
acknowledgment of the source. The models and images made from them must not
be used for commercial purposes without Stanford's permission. The synthetic
RGB-D images in `data/synthetic_bunny/` are renderings of this model and
carry the same restriction. **The MIT license of this repository does not
apply to these files.**

## 3. TUM RGB-D benchmark frames — `data/tum_sample/`

`1.png`, `2.png`, `1_depth.png` and `2_depth.png` are byte-identical to the
copies shipped in slambook2's `ch7/`. They are frames of the TUM RGB-D
benchmark (<https://cvg.cit.tum.de/data/datasets/rgbd-dataset>), which is
licensed under Creative Commons Attribution 4.0 (CC BY 4.0). Please cite:

J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers,
"A Benchmark for the Evaluation of RGB-D SLAM Systems",
Proc. of the International Conference on Intelligent Robot Systems (IROS), 2012.
