#include "features/features.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "features/brief.hpp"
#include "features/fast.hpp"
#include "features/matcher.hpp"
#include "features/orientation.hpp"

// find_feature_matches(): the signature and the final distance filter come
// from the book (slambook2 ch7); the detection, description and matching in
// between are the project-owned feature_core modules:
//   grayscale -> FAST (threshold 30, score, 3x3 NMS, border 16, strongest 500)
//   -> intensity-centroid orientation -> Gaussian-smoothed rotated BRIEF
//   -> brute-force Hamming nearest neighbour -> distance <= max(2 d_min, 30).
//
// Transitional boundary (vision-library migration, docs/library_removal_audit.md):
// the images arrive and the keypoints/matches leave in the existing library
// types, so the 11 callers are unchanged in this checkpoint. Only buffer
// access and the final type conversion touch those types; the conversion is
// removed in the types-migration stage.

namespace {

features::GrayImage ToGray(const cv::Mat &img) {
  if (img.depth() != CV_8U) throw std::invalid_argument("find_feature_matches: 8-bit images expected");
  return features::GrayFromInterleaved(img.data, img.cols, img.rows, img.step, img.channels());
}

void Extract(const cv::Mat &img, std::vector<features::Keypoint> &kps, std::vector<features::BriefDescriptor> &desc) {
  const features::GrayImage gray = ToGray(img);
  kps = features::DetectFast(gray, features::FastParams());
  features::AssignOrientations(gray, kps);
  desc = features::ComputeBrief(features::GaussianSmooth(gray), kps);
}

std::vector<cv::KeyPoint> ToLibraryKeypoints(const std::vector<features::Keypoint> &kps) {
  std::vector<cv::KeyPoint> out;
  out.reserve(kps.size());
  for (const auto &k : kps) {
    float deg = float(k.angle * 180.0 / M_PI);
    if (deg < 0) deg += 360.0f;
    // size = descriptor patch diameter (31), response = FAST score, octave 0 (single scale)
    out.emplace_back(k.x, k.y, float(2 * features::kBriefPatchRadius + 1), deg, float(k.score), 0);
  }
  return out;
}

std::vector<cv::DMatch> ToLibraryMatches(const std::vector<features::Match> &m) {
  std::vector<cv::DMatch> out;
  out.reserve(m.size());
  for (const auto &x : m) out.emplace_back(x.query, x.train, float(x.distance));
  return out;
}

}  // namespace

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches,
                          std::vector<cv::DMatch> *all_matches) {
  std::vector<features::Keypoint> k1, k2;
  std::vector<features::BriefDescriptor> d1, d2;
  Extract(img_1, k1, d1);
  Extract(img_2, k2, d2);

  const std::vector<features::Match> raw = features::MatchBruteForce(d1, d2);
  int d_min = 0, d_max = 0;
  const std::vector<features::Match> kept = features::FilterMatchesByDistance(raw, 30, &d_min, &d_max);
  printf("-- Max dist : %f \n", double(d_max));
  printf("-- Min dist : %f \n", double(d_min));

  keypoints_1 = ToLibraryKeypoints(k1);
  keypoints_2 = ToLibraryKeypoints(k2);
  matches = ToLibraryMatches(kept);
  if (all_matches) *all_matches = ToLibraryMatches(raw);
}
