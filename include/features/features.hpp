#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

// find_feature_matches(): detects, describes and matches features in two
// images and applies the distance filter distance <= max(2 * d_min, 30).
// Signature and filter rule from the book (slambook2 ch7, where it appears in
// orb_cv.cpp and the pose_estimation_*.cpp / triangulation.cpp examples);
// the implementation is the project-owned feature pipeline (features/fast,
// orientation, brief, matcher) -- see src/features/features.cpp.
//
// `all_matches` (optional): every nearest-neighbour match before the filter.
void find_feature_matches(
  const cv::Mat &img_1, const cv::Mat &img_2,
  std::vector<cv::KeyPoint> &keypoints_1,
  std::vector<cv::KeyPoint> &keypoints_2,
  std::vector<cv::DMatch> &matches,
  std::vector<cv::DMatch> *all_matches = nullptr);
