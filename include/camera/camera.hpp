#pragma once

#include <opencv2/core/core.hpp>

struct PinholeCamera {
  double fx, fy, cx, cy;
  cv::Mat K() const { return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1); }
};

// Book code starts
// pixel2cam(): appears identically in pose_estimation_2d2d.cpp,
// pose_estimation_3d2d.cpp, pose_estimation_3d3d.cpp.
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);
// Book code ends

// Not book code: inverse of pixel2cam -- projects a camera-space 3D point to
// a pixel. Used by the render module's rasterizer.
cv::Point2d cam2pixel(const cv::Point3d &P, const cv::Mat &K);
