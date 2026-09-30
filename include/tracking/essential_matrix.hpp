#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

// From-scratch normalized 8-point Essential Matrix estimation and pose
// recovery (no OpenCV pose/essential-matrix calls). Same convention as
// pose_estimation_2d2d (geometry.cpp) and PnP: x2 = R*x1 + t, i.e. R,t
// transform frame-1 camera-space points into frame-2 camera-space points,
// equivalently E = [t]_x R (verified against the existing t^R check in
// tests/integration/two_view_pose_test.cpp before implementing).
// Translation is unit-norm: the Essential matrix fixes t only up to scale.

// pts1_cam/pts2_cam must already be normalized camera coordinates
// (pixel2cam() applied), not raw pixels. Requires at least 8 correspondences.
cv::Mat EstimateEssentialMatrix(
  const std::vector<cv::Point2d> &pts1_cam,
  const std::vector<cv::Point2d> &pts2_cam);

// Returns false if no candidate has positive depth (in both cameras) for
// any correspondence.
bool RecoverPoseFromEssential(
  const cv::Mat &E,
  const std::vector<cv::Point2d> &pts1_cam,
  const std::vector<cv::Point2d> &pts2_cam,
  cv::Mat &R, cv::Mat &t);

// End-to-end convenience wrapper mirroring pose_estimation_2d2d's own
// interface (geometry.hpp): pixel keypoints/matches + K in, R/t out.
bool EstimateRelativePoseEssentialMatrix(
  const std::vector<cv::KeyPoint> &keypoints_1,
  const std::vector<cv::KeyPoint> &keypoints_2,
  const std::vector<cv::DMatch> &matches,
  const cv::Mat &K,
  cv::Mat &R, cv::Mat &t);
