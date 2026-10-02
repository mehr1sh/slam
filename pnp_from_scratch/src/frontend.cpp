#include "frontend.hpp"

#include <cstdio>

#include "features/brief.hpp"
#include "features/matcher.hpp"
#include "features/orientation.hpp"

namespace scratch {

bool LoadRgbdFrame(const std::string &dataset_dir, int index, RgbdFrame &frame, std::string *error) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06d", index);
  const std::string base = dataset_dir + "/" + name;
  if (!LoadPng(base + ".png", frame.rgb, error) || !LoadPng(base + "_depth.png", frame.depth, error)) return false;
  if (frame.rgb.bit_depth != 8 || frame.rgb.channels < 3) {
    if (error) *error = base + ".png: expected 8-bit RGB(A)";
    return false;
  }
  if (frame.depth.bit_depth != 16 || frame.depth.channels != 1) {
    if (error) *error = base + "_depth.png: expected 16-bit single-channel depth";
    return false;
  }
  if (frame.rgb.width != frame.depth.width || frame.rgb.height != frame.depth.height) {
    if (error) *error = base + ": RGB and depth sizes differ";
    return false;
  }
  return true;
}

FrameFeatures ExtractFeatures(const PngImage &rgb, const FrontendParams &params) {
  const std::vector<uint8_t> bytes(rgb.samples.begin(), rgb.samples.end());  // 8-bit samples
  FrameFeatures f;
  f.gray = features::GrayFromRGB(bytes.data(), rgb.width, rgb.height, size_t(rgb.width) * rgb.channels, rgb.channels);
  f.keypoints = features::DetectFast(f.gray, params.fast);
  features::AssignOrientations(f.gray, f.keypoints);
  f.descriptors = features::ComputeBrief(features::GaussianSmooth(f.gray), f.keypoints);
  return f;
}

PairMatches MatchFrames(const FrameFeatures &fi, const FrameFeatures &fj, const PngImage &depth_i,
                        const PngImage &depth_j, const Intrinsics &K, const FrontendParams &params) {
  PairMatches pm;
  pm.raw = features::MatchBruteForce(fi.descriptors, fj.descriptors);
  pm.filtered = features::FilterMatchesByDistance(pm.raw, params.match_floor, &pm.d_min, &pm.d_max);
  for (const features::Match &m : pm.filtered) {
    const features::Keypoint &a = fi.keypoints[m.query], &b = fj.keypoints[m.train];
    Correspondence c;
    c.query = m.query;
    c.train = m.train;
    c.hamming = m.distance;
    c.uv_i = Eigen::Vector2d(a.x, a.y);
    c.uv_j = Eigen::Vector2d(b.x, b.y);
    // depth at the truncated pixel, as in the reference pipeline
    c.depth_raw_i = depth_i.samples[size_t(int(a.y)) * depth_i.width + int(a.x)];
    c.depth_raw_j = depth_j.samples[size_t(int(b.y)) * depth_j.width + int(b.x)];
    c.has_3d = c.depth_raw_i > 0;
    if (c.has_3d) {
      const double Z = c.depth_raw_i / params.depth_scale;
      c.X_i = Eigen::Vector3d((a.x - K.cx) / K.fx * Z, (a.y - K.cy) / K.fy * Z, Z);
    }
    pm.correspondences.push_back(c);
  }
  return pm;
}

}  // namespace scratch
