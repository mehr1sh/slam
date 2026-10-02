#pragma once

// Scratch-pipeline front end: RGB-D frames -> 2D-2D matches and 3D->2D
// correspondences, using only the project-owned feature modules
// (features/fast, orientation, brief, matcher) and the scratch PNG decoder.
//
//   RGB -> grayscale -> image pyramid (optional; 1 level = single scale)
//       -> per level: FAST (score, 3x3 NMS, border, strongest N)
//       -> intensity-centroid orientation -> Gaussian-smoothed rotated BRIEF
//       -> brute-force Hamming nearest neighbour -> distance filter
//       -> depth lookup in frame i -> 3D point in camera i  (+ 2D point in frame i+1)

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "features/fast.hpp"
#include "features/pyramid.hpp"
#include "features/types.hpp"
#include "png.hpp"
#include "projection.hpp"

namespace scratch {

struct FrontendParams {
  features::FastParams fast;        // threshold set to 20 below; border 16; strongest 500 (per level)
  features::PyramidParams pyramid;  // levels set to 1 below (single scale); scale factor 1.2
  int match_floor = 30;             // keep matches with distance <= max(2 * d_min, match_floor)
  double depth_scale = 5000.0;      // depth PNG value = metres * depth_scale (TUM convention); 0 = no depth
  FrontendParams() {
    fast.threshold = 20;
    pyramid.levels = 1;  // single scale by default; 8 = the reference ORB's pyramid
  }
};

struct RgbdFrame {
  PngImage rgb;    // 8-bit, 3 or 4 channels
  PngImage depth;  // 16-bit, 1 channel
};

// Loads <dataset>/<index padded to 6 digits>.png and _depth.png; checks the formats.
bool LoadRgbdFrame(const std::string &dataset_dir, int index, RgbdFrame &frame, std::string *error = nullptr);

struct FrameFeatures {
  features::GrayImage gray;
  // keypoints[k].x/.y are ORIGINAL-image coordinates (mapped from the level);
  // score and angle are those computed on the keypoint's level.
  std::vector<features::Keypoint> keypoints;
  std::vector<features::BriefDescriptor> descriptors;  // computed on each keypoint's level
  std::vector<int> level;                   // pyramid level of each keypoint
  std::vector<float> scale;                 // original size / level size
  std::vector<float> level_x, level_y;      // position on its level
  std::vector<int> per_level_count;
};

FrameFeatures ExtractFeatures(const PngImage &rgb, const FrontendParams &params);

// One filtered match with everything the geometry stages need.
struct Correspondence {
  int query = -1, train = -1;  // keypoint index in frame i / frame i+1
  int hamming = 0;
  int level_i = 0, level_j = 0;  // pyramid levels of the two keypoints
  Eigen::Vector2d uv_i = Eigen::Vector2d::Zero(), uv_j = Eigen::Vector2d::Zero();  // pixels
  uint16_t depth_raw_i = 0, depth_raw_j = 0;  // raw depth at the (truncated) keypoint pixel
  bool has_3d = false;                        // depth_raw_i > 0
  Eigen::Vector3d X_i = Eigen::Vector3d::Zero();  // ((u-cx)/fx Z, (v-cy)/fy Z, Z) in camera i
};

struct PairMatches {
  std::vector<features::Match> raw;        // nearest neighbour of every frame-i descriptor
  std::vector<features::Match> filtered;   // after the distance filter
  int d_min = 0, d_max = 0;
  std::vector<Correspondence> correspondences;  // one per filtered match, in the same order
};

PairMatches MatchFrames(const FrameFeatures &fi, const FrameFeatures &fj, const PngImage &depth_i,
                        const PngImage &depth_j, const Intrinsics &K, const FrontendParams &params);

}  // namespace scratch
