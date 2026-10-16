#pragma once

// RANSAC from scratch, around the scratch linear PnP and the scratch 8-point
// essential matrix.
//
//   repeat `iterations` times:
//     draw a minimal random sample (deterministic generator, fixed seed)
//     fit a model to the sample                    (DLT PnP: 6 points; 8-point E: 8)
//     residual of EVERY correspondence under it    (reprojection px; Sampson px)
//     inliers = residual < threshold
//     score   = number of inliers; ties -> smaller sum of inlier residuals
//   best model -> refit on all its inliers -> final residuals and inlier mask
//
// Randomness: std::mt19937 (its output sequence is fixed by the C++
// standard) with indices drawn by a partial Fisher-Yates shuffle using
// `rng() % remaining` (std::uniform_int_distribution is avoided because its
// algorithm is implementation-defined), so runs are reproducible everywhere.

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "essential.hpp"
#include "projection.hpp"

namespace scratch {

struct RansacParams {
  int iterations = 300;
  double threshold_px = 8.0;  // inlier threshold in pixels
  uint32_t seed = 12345;
};

// k distinct indices out of [0, n), partial Fisher-Yates.
std::vector<int> SampleIndices(std::mt19937 &rng, int n, int k);

struct RansacPnPResult {
  bool ok = false;
  std::string reason;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // X_{i+1} = R X_i + t  (T_{i+1<-i})
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  std::vector<bool> inlier_mask;    // final: residual under the REFIT pose < threshold
  std::vector<double> residual_px;  // final reprojection error of every correspondence (inf if behind the camera)
  int num_inliers = 0;              // final
  int best_sample_inliers = 0;      // inliers of the best minimal-sample model, before the refit
  int models_tried = 0;             // minimal samples that gave a valid DLT model
  bool refit_used = false;          // false: refit failed, best sample model kept
};

// X: 3D points in camera i; uv: matched pixels in frame i+1.
RansacPnPResult RansacPnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                          const Intrinsics &K, const RansacParams &params = RansacParams());

struct RansacEssentialResult {
  bool ok = false;
  std::string reason;
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  RecoveredPose pose;               // from the refit E, on the inliers
  std::vector<bool> inlier_mask;    // Sampson distance under the refit E < threshold
  int num_inliers = 0;
  int best_sample_inliers = 0;
  int models_tried = 0;
};

// x1, x2: normalized coordinates; the threshold is converted to normalized
// units with `focal_px` (Sampson distance in pixels ~ f * normalized distance).
RansacEssentialResult RansacEssential(const Points2 &x1, const Points2 &x2, double focal_px,
                                      const RansacParams &params = RansacParams());

}  // namespace scratch
