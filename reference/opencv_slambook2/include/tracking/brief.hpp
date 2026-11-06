#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cstdint>
#include <vector>

// From-scratch BRIEF binary descriptor (no OpenCV descriptor-extractor
// calls). 256-bit descriptor (8 x uint32_t) -- the same encoding this
// repo's tests/unit/orb_from_scratch_test.cpp already uses for its own
// from-scratch ORB descriptor. Fixed, deterministic sampling pattern
// (uniform offsets within a patch, fixed RNG seed), generated once.
using BriefDescriptor = std::vector<uint32_t>;

constexpr int kBriefPatchHalfSize = 13;

// A keypoint too close to the image border to safely sample its patch
// gets an empty descriptor (size 0) -- same convention as this repo's
// existing ComputeORB()'s bad_points/empty-descriptor handling.
std::vector<BriefDescriptor> ComputeBrief(const cv::Mat &gray,
                                          const std::vector<cv::KeyPoint> &keypoints);
