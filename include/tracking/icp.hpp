#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

// Compatibility API for paired 3D points; target=pts1, source=pts2.
// pose_estimation_3d3d.cpp
void pose_estimation_3d3d(
  const std::vector<cv::Point3f> &pts1,
  const std::vector<cv::Point3f> &pts2,
  cv::Mat &R, cv::Mat &t
);

void bundleAdjustment(
  const std::vector<cv::Point3f> &points_3d,
  const std::vector<cv::Point3f> &source_points_3d,
  cv::Mat &R, cv::Mat &t
);
// For full RGB-D ICP use tracking/rgbd_icp.hpp and the icp_sequence executable.
