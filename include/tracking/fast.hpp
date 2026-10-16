#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

// From-scratch FAST corner detector (no OpenCV FAST/feature-detector
// calls): the classic 16-pixel Bresenham-circle segment test (Rosten &
// Drummond). Single scale, no non-max suppression, no scale pyramid.
struct FastParams {
  int threshold = 30;   // intensity difference vs. the center pixel
  int contiguous = 9;   // required contiguous arc length, out of 16
};

std::vector<cv::KeyPoint> DetectFast(const cv::Mat &gray, const FastParams &params = FastParams());
