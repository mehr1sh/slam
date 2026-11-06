#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <cstdint>
#include <vector>

// Book code starts
// pose_estimation_3d3d.cpp
void pose_estimation_3d3d(
  const std::vector<cv::Point3f> &pts1,
  const std::vector<cv::Point3f> &pts2,
  cv::Mat &R, cv::Mat &t
);

void bundleAdjustment(
  const std::vector<cv::Point3f> &points_3d,
  const std::vector<cv::Point3f> &points_2d,
  cv::Mat &R, cv::Mat &t
);
// Book code ends

// Robust 3D-3D alignment (project code). Feature correspondences contain
// mismatches, and a plain least-squares fit (pose_estimation_3d3d) uses every
// one of them. This runs RANSAC over minimal 3-point rigid fits, counts the
// correspondences with |pts1 - (R pts2 + t)| < threshold_m as inliers, and
// refits by least squares on the inliers (kept only if it does not lose
// inliers). Same convention as pose_estimation_3d3d: pts1 = R * pts2 + t.
// Deterministic: std::mt19937(seed), indices by partial Fisher-Yates.
struct RigidRansacResult {
  bool ok = false;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  std::vector<bool> inliers;
  int num_inliers = 0;
};
RigidRansacResult EstimateRigidRansac(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2,
                                      double threshold_m = 0.01, int iterations = 300, uint32_t seed = 12345);

// Least-squares rigid fit pts1[k] = R pts2[k] + t over the given indices (SVD of
// the cross-covariance; a reflection is corrected by flipping the last singular
// vector). Returns false for fewer than 3 points or a degenerate (collinear) set.
bool FitRigid(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2,
              const std::vector<int> &idx, Eigen::Matrix3d &R, Eigen::Vector3d &t);
