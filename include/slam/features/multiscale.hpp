#pragma once

// Multiscale feature extraction on an image pyramid (project-owned):
// on EVERY level, independently,
//   DetectFast (unchanged: threshold, 3x3 NMS, border, cap -- all in that
//   level's pixels) -> intensity-centroid orientation on that level image ->
//   Gaussian smoothing of that level image -> rotated BRIEF on it.
// So each descriptor is computed at the scale where its keypoint was found.
// Keypoints are concatenated level by level (level 0 first), each level in
// deterministic order. SelectionParams enables project-owned Harris ranking
// and a geometric total-feature budget across levels. The two-argument API
// retains the historical extraction for regression tests and ablations.

#include <vector>

#include "features/fast.hpp"
#include "features/pyramid.hpp"
#include "features/types.hpp"

namespace features {

struct MultiscaleKeypoint {
  Keypoint level_kp; // x, y, score and angle in the coordinates of its level
  int level = 0;
  float scale = 1.0f; // level-0 size / level size (x axis)
  float x = 0, y = 0; // position in the level-0 (original) image
};

struct MultiscaleFeatures {
  std::vector<MultiscaleKeypoint> keypoints;
  std::vector<BriefDescriptor>
      descriptors;                  // one per keypoint, computed on its level
  std::vector<int> per_level_count; // keypoints found on each level
  std::vector<bool> valid;          // ComputeBrief's flag per keypoint: every test inside the level image
                                    // (false: all-zero descriptor; MatchRanked excludes it)
};

struct SelectionParams {
  bool harris =
      false; // false preserves the historical extraction for ablations
  int total_features =
      0; // >0: geometric per-level allocation; 0: legacy per-level cap
};
double HarrisResponse(const GrayImage &image, int x, int y);
std::vector<int> LevelQuotas(int total, int levels, double scale_factor);
MultiscaleFeatures
ExtractMultiscale(const ImagePyramid &pyramid, const FastParams &fast,
                  const SelectionParams &selection = SelectionParams());

} // namespace features
