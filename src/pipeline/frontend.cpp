#include "frontend.hpp"

#include <cstdio>

#include "features/matcher.hpp"
#include "features/multiscale.hpp"

namespace scratch {

bool LoadRgbdFrame(const std::string &dataset_dir, int index, RgbdFrame &frame,
                   std::string *error) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06d", index);
  const std::string base = dataset_dir + "/" + name;
  return LoadRgbdFiles(base + ".png", base + "_depth.png", frame, error);
}

bool LoadRgbdFiles(const std::string &rgb_path, const std::string &depth_path,
                   RgbdFrame &frame, std::string *error) {
  if (!LoadPng(rgb_path, frame.rgb, error) ||
      !LoadPng(depth_path, frame.depth, error))
    return false;
  if (frame.rgb.bit_depth != 8 || frame.rgb.channels < 3) {
    if (error)
      *error = rgb_path + ": expected 8-bit RGB(A)";
    return false;
  }
  if (frame.depth.bit_depth != 16 || frame.depth.channels != 1) {
    if (error)
      *error = depth_path + ": expected 16-bit single-channel depth";
    return false;
  }
  if (frame.rgb.width != frame.depth.width ||
      frame.rgb.height != frame.depth.height) {
    if (error)
      *error = rgb_path + ": RGB and depth sizes differ";
    return false;
  }
  return true;
}

FrameFeatures ExtractFeatures(const PngImage &rgb,
                              const FrontendParams &params) {
  const std::vector<uint8_t> bytes(rgb.samples.begin(),
                                   rgb.samples.end()); // 8-bit samples
  FrameFeatures f;
  f.gray =
      features::GrayFromRGB(bytes.data(), rgb.width, rgb.height,
                            size_t(rgb.width) * rgb.channels, rgb.channels);
  // per pyramid level: FAST -> orientation -> rotated BRIEF on that level's
  // image
  const features::MultiscaleFeatures m = features::ExtractMultiscale(
      features::BuildPyramid(f.gray, params.pyramid), params.fast,
      params.selection);
  f.descriptors = m.descriptors;
  f.descriptor_valid = m.valid;
  f.per_level_count = m.per_level_count;
  for (const features::MultiscaleKeypoint &k : m.keypoints) {
    features::Keypoint o = k.level_kp;
    o.x = k.x; // original-image coordinates (identical to the level ones at
               // level 0)
    o.y = k.y;
    f.keypoints.push_back(o);
    f.level.push_back(k.level);
    f.scale.push_back(k.scale);
    f.level_x.push_back(k.level_kp.x);
    f.level_y.push_back(k.level_kp.y);
  }
  return f;
}

PairMatches MatchFrames(const FrameFeatures &fi, const FrameFeatures &fj,
                        const PngImage &depth_i, const PngImage &depth_j,
                        const Intrinsics &K, const FrontendParams &params) {
  PairMatches pm;
  std::vector<const features::RankedMatch *> info; // ranked path: the kept match behind each filtered entry
  if (!params.ranked_matcher) {
    pm.raw = features::MatchBruteForce(fi.descriptors, fj.descriptors);
    pm.filtered = features::FilterMatchesByDistance(pm.raw, params.match_floor,
                                                    &pm.d_min, &pm.d_max);
    if (params.mutual_ratio)
      pm.filtered = features::MatchMutualRatio(fi.descriptors, fj.descriptors,
                                               params.ratio);
  } else {
    std::vector<float> tx, ty, qx, qy;
    for (const features::Keypoint &k : fj.keypoints) tx.push_back(k.x), ty.push_back(k.y);
    for (const features::Keypoint &k : fi.keypoints) qx.push_back(k.x), qy.push_back(k.y);
    features::MatcherParams mp = params.matcher;
    mp.floor_distance = params.match_floor;
    features::MatchSet ms = features::MatchRanked(fi.descriptors, fj.descriptors, tx, ty, fi.level, fj.level, mp,
                                                  fi.descriptor_valid, fj.descriptor_valid, qx, qy);
    pm.ranked = true;
    pm.counters = ms.counters;
    pm.d_min = ms.counters.d_min, pm.d_max = ms.counters.d_max;
    for (const features::RankedMatch &m : ms.raw) pm.raw.push_back(m.match());
    for (const features::RankedMatch &m : ms.kept) pm.filtered.push_back(m.match());
    pm.ranked_raw = std::move(ms.raw);
    std::vector<int> raw_of_query(fi.descriptors.size(), -1);
    for (size_t k = 0; k < pm.ranked_raw.size(); ++k) raw_of_query[pm.ranked_raw[k].query] = int(k);
    for (const features::Match &m : pm.filtered) info.push_back(&pm.ranked_raw[raw_of_query[m.query]]);
  }
  for (size_t f = 0; f < pm.filtered.size(); ++f) {
    const features::Match &m = pm.filtered[f];
    const features::Keypoint &a = fi.keypoints[m.query],
                             &b = fj.keypoints[m.train];
    Correspondence c;
    c.query = m.query;
    c.train = m.train;
    c.hamming = m.distance;
    c.level_i = fi.level[m.query];
    c.level_j = fj.level[m.train];
    c.uv_i = Eigen::Vector2d(a.x, a.y);
    c.uv_j = Eigen::Vector2d(b.x, b.y);
    // depth at the truncated pixel, as in the reference pipeline
    c.depth_raw_i =
        depth_i.samples[size_t(int(a.y)) * depth_i.width + int(a.x)];
    c.depth_raw_j =
        depth_j.samples[size_t(int(b.y)) * depth_j.width + int(b.x)];
    c.has_3d = c.depth_raw_i > 0;
    if (c.has_3d && params.depth_edge_m > 0) {
      int x = int(a.x), y = int(a.y);
      double lo = 1e30, hi = 0;
      if (x < 1 || y < 1 || x >= depth_i.width - 1 || y >= depth_i.height - 1)
        c.has_3d = false;
      else
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            double v =
                depth_i.samples[size_t(y + dy) * depth_i.width + x + dx] /
                params.depth_scale;
            if (v <= 0)
              c.has_3d = false;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
          }
      c.has_3d = c.has_3d && hi - lo < params.depth_edge_m;
    }
    if (c.has_3d) {
      const double Z = c.depth_raw_i / params.depth_scale;
      c.X_i =
          Eigen::Vector3d((a.x - K.cx) / K.fx * Z, (a.y - K.cy) / K.fy * Z, Z);
    }
    if (!info.empty()) {
      const features::RankedMatch &r = *info[f];
      c.has_second = r.has_second, c.second_distance = r.second_distance, c.ratio = r.ratio();
      c.mutual = r.mutual, c.rank = r.rank;
    }
    pm.correspondences.push_back(c);
  }
  return pm;
}

} // namespace scratch
