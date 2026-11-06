#include "features/multiscale.hpp"

#include "features/brief.hpp"
#include "features/orientation.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <stdexcept>

namespace features {

double HarrisResponse(const GrayImage &img, int x, int y) {
  if (x < 4 || y < 4 || x >= img.width - 4 || y >= img.height - 4)
    return 0;
  double a = 0, b = 0, c = 0;
  for (int dy = -3; dy <= 3; ++dy)
    for (int dx = -3; dx <= 3; ++dx) {
      int u = x + dx, v = y + dy;
      double gx = (img.at(u + 1, v - 1) + 2. * img.at(u + 1, v) +
                   img.at(u + 1, v + 1) - img.at(u - 1, v - 1) -
                   2. * img.at(u - 1, v) - img.at(u - 1, v + 1)) /
                  1020.;
      double gy = (img.at(u - 1, v + 1) + 2. * img.at(u, v + 1) +
                   img.at(u + 1, v + 1) - img.at(u - 1, v - 1) -
                   2. * img.at(u, v - 1) - img.at(u + 1, v - 1)) /
                  1020.;
      a += gx * gx;
      b += gy * gy;
      c += gx * gy;
    }
  return a * b - c * c - .04 * (a + b) * (a + b);
}
std::vector<int> LevelQuotas(int total, int levels, double scale) {
  if (total < 0 || levels < 1 || scale < 1)
    throw std::invalid_argument("Invalid feature budget");
  std::vector<int> q(levels);
  std::vector<double> fractional(levels);
  double sum = 0;
  for (int l = 0; l < levels; ++l)
    sum += std::pow(scale, -l);
  int used = 0;
  for (int l = 0; l < levels; ++l) {
    double n = total * std::pow(scale, -l) / sum;
    q[l] = int(n);
    fractional[l] = n - q[l];
    used += q[l];
  }
  std::vector<int> ids(levels);
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(),
                   [&](int a, int b) { return fractional[a] > fractional[b]; });
  for (int i = 0; i < total - used; ++i)
    ++q[ids[i]];
  return q;
}
MultiscaleFeatures ExtractMultiscale(const ImagePyramid &pyramid,
                                     const FastParams &fast,
                                     const SelectionParams &selection) {
  MultiscaleFeatures out;
  std::vector<int> quotas;
  if (selection.total_features > 0) {
    double scale = pyramid.scale_x.size() > 1 ? pyramid.scale_x[1] : 1.2;
    quotas = LevelQuotas(selection.total_features, int(pyramid.levels.size()),
                         scale);
  }
  for (size_t l = 0; l < pyramid.levels.size(); ++l) {
    const GrayImage &img = pyramid.levels[l];
    FastParams candidates = fast;
    if (selection.harris || !quotas.empty())
      candidates.max_keypoints = 0;
    std::vector<Keypoint> kps = DetectFast(img, candidates);
    if (selection.harris) {
      // Compute each response once, rather than resampling the 7x7 patch on
      // every sort comparison. Stable sorting preserves equal-score order.
      std::map<std::pair<int, int>, double> responses;
      for (const auto &kp : kps)
        responses[{int(kp.x), int(kp.y)}] =
            HarrisResponse(img, int(kp.x), int(kp.y));
      std::stable_sort(kps.begin(), kps.end(),
                       [&](const Keypoint &a, const Keypoint &b) {
                         return responses.at({int(a.x), int(a.y)}) >
                                responses.at({int(b.x), int(b.y)});
                       });
    }
    int limit = quotas.empty() ? fast.max_keypoints : quotas[l];
    if ((!quotas.empty() || limit > 0) && int(kps.size()) > limit)
      kps.resize(limit);
    AssignOrientations(img, kps);
    std::vector<bool> valid;
    const std::vector<BriefDescriptor> desc =
        ComputeBrief(GaussianSmooth(img), kps, &valid);
    out.per_level_count.push_back(int(kps.size()));
    for (size_t k = 0; k < kps.size(); ++k) {
      MultiscaleKeypoint m;
      m.level_kp = kps[k];
      m.level = int(l);
      m.scale = float(pyramid.scale_x[l]);
      m.x = LevelToBaseX(pyramid, int(l), kps[k].x);
      m.y = LevelToBaseY(pyramid, int(l), kps[k].y);
      out.keypoints.push_back(m);
      out.descriptors.push_back(desc[k]);
      out.valid.push_back(valid[k]);
    }
  }
  return out;
}

} // namespace features
