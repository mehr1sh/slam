#pragma once

#include <opencv2/core/core.hpp>
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
