#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

// Book code starts
// pose_estimation_2d2d.cpp / triangulation.cpp
void pose_estimation_2d2d(
  const std::vector<cv::KeyPoint> &keypoints_1,
  const std::vector<cv::KeyPoint> &keypoints_2,
  const std::vector<cv::DMatch> &matches,
  cv::Mat &R, cv::Mat &t);

// triangulation.cpp
void triangulation(
  const std::vector<cv::KeyPoint> &keypoint_1,
  const std::vector<cv::KeyPoint> &keypoint_2,
  const std::vector<cv::DMatch> &matches,
  const cv::Mat &R, const cv::Mat &t,
  std::vector<cv::Point3d> &points);
// Book code ends