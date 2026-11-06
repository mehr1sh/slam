#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cstdint>
#include <vector>

// Small, reusable Hamming-distance utility: XOR + popcount over
// fixed-width binary descriptors (any std::vector<uint32_t>, e.g.
// BriefDescriptor from tracking/brief.hpp). Same popcount technique
// (_mm_popcnt_u32) this repo's tests/unit/orb_from_scratch_test.cpp
// already uses for its own from-scratch descriptor matching.
int HammingDistance(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b);

// Simple nearest-neighbor matcher: for each descriptor in desc1, the
// single closest (by Hamming distance) descriptor in desc2, kept only if
// below max_distance. No ratio test / cross-check yet.
std::vector<cv::DMatch> MatchHamming(
  const std::vector<std::vector<uint32_t>> &desc1,
  const std::vector<std::vector<uint32_t>> &desc2,
  int max_distance = 60);
