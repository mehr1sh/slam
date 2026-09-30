#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

// Book code starts
// find_feature_matches(): identical body in orb_cv.cpp's main(), and
// duplicated as a standalone function in pose_estimation_2d2d.cpp,
// triangulation.cpp, pose_estimation_3d2d.cpp, pose_estimation_3d3d.cpp.
//
// Modification: added the optional `all_matches` out-param (default
// nullptr, so every existing call site is unaffected) so feature_matching_test
// can visualize the pre-filter matches the same way orb_cv.cpp's main()
// does, without duplicating the detect/compute/match/filter logic below.
void find_feature_matches(
  const cv::Mat &img_1, const cv::Mat &img_2,
  std::vector<cv::KeyPoint> &keypoints_1,
  std::vector<cv::KeyPoint> &keypoints_2,
  std::vector<cv::DMatch> &matches,
  std::vector<cv::DMatch> *all_matches = nullptr);
// Book code ends
