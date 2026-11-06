#pragma once

// Multiscale feature extraction on an image pyramid (project-owned):
// on EVERY level, independently,
//   DetectFast (unchanged: threshold, 3x3 NMS, border, cap -- all in that
//   level's pixels) -> intensity-centroid orientation on that level image ->
//   Gaussian smoothing of that level image -> rotated BRIEF on it.
// So each descriptor is computed at the scale where its keypoint was found.
// Keypoints are concatenated level by level (level 0 first), each level in
// DetectFast's own deterministic order. No cross-level suppression, no
// Harris re-ranking and no per-level quotas (deliberately not added yet).

#include <vector>

#include "features/fast.hpp"
#include "features/pyramid.hpp"
#include "features/types.hpp"

namespace features {

struct MultiscaleKeypoint {
  Keypoint level_kp;  // x, y, score and angle in the coordinates of its level
  int level = 0;
  float scale = 1.0f; // level-0 size / level size (x axis)
  float x = 0, y = 0; // position in the level-0 (original) image
};

struct MultiscaleFeatures {
  std::vector<MultiscaleKeypoint> keypoints;
  std::vector<BriefDescriptor> descriptors;  // one per keypoint, computed on its level
  std::vector<int> per_level_count;          // keypoints found on each level
};

MultiscaleFeatures ExtractMultiscale(const ImagePyramid &pyramid, const FastParams &fast);

}  // namespace features
