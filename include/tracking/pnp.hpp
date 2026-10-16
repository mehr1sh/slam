#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

// Book code starts
// pose_estimation_3d2d.cpp
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const cv::Mat &K,
  Sophus::SE3d &pose
);

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const cv::Mat &K,
  Sophus::SE3d &pose
);
// Book code ends
