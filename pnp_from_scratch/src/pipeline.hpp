#pragma once

// The complete scratch pipeline for one frame pair i -> i+1:
//
//   RGB-D frames
//     -> front end (grayscale, FAST, orientation, rotated BRIEF, Hamming, filter)
//     -> 2D-2D matches  -> RANSAC 8-point essential matrix -> decomposition
//                          -> cheirality -> (R, t/|t|)       [two-view estimate]
//     -> 3D->2D correspondences (frame-i depth)
//                       -> RANSAC around the scratch linear PnP -> refit on inliers
//                          -> (R, t) = T_{i+1<-i}           [metric pose]
//
// Both estimates use the same convention X_{i+1} = R X_i + t. The metric PnP
// pose is the one accumulated into the trajectory; the essential-matrix
// estimate is scale-free and is reported next to it for comparison.

#include <string>
#include <vector>

#include <Eigen/Core>

#include "frontend.hpp"
#include "ransac.hpp"

namespace scratch {

struct PipelineParams {
  FrontendParams frontend;  // FAST threshold 20, strongest 500, match floor 30
  RansacParams pnp_ransac;  // 300 iterations, 8 px, seed 12345
  RansacParams essential_ransac;
  PipelineParams() {
    essential_ransac.iterations = 2000;  // the minimal 8-point sample is noise-sensitive: more samples
    essential_ransac.threshold_px = 2.0; // Sampson distance
    essential_ransac.seed = 54321;
  }
};

struct PairResult {
  int i = 0, j = 0;
  int keypoints_i = 0, keypoints_j = 0, raw_matches = 0, filtered_matches = 0;
  int d_min = 0, d_max = 0;
  std::vector<Correspondence> correspondences;  // all filtered matches

  // essential-matrix stage (all filtered matches, normalized coordinates)
  bool essential_ok = false;
  std::string essential_reason;
  int essential_inliers = 0, essential_in_front = 0;
  Eigen::Matrix3d R_essential = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_essential_dir = Eigen::Vector3d::Zero();  // unit vector
  std::vector<bool> essential_mask;                            // per correspondence

  // PnP stage (correspondences with frame-i depth)
  std::vector<int> pnp_index;  // correspondence index of each PnP input
  bool pnp_ok = false;
  std::string pnp_reason;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();  // T_{i+1<-i}
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  int pnp_inliers = 0, pnp_best_sample_inliers = 0;
  std::vector<bool> pnp_mask;          // per PnP input
  std::vector<double> pnp_residual_px; // per PnP input, under the final pose
  double reproj_inlier_mean = 0, reproj_inlier_median = 0, reproj_inlier_max = 0, reproj_all_median = 0;
};

PairResult ProcessPair(int i, int j, const FrameFeatures &fi, const FrameFeatures &fj, const RgbdFrame &frame_i,
                       const RgbdFrame &frame_j, const Intrinsics &K, const PipelineParams &params);

}  // namespace scratch
