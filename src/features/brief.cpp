#include "features/brief.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace features {

namespace {

// Standard-normal sample from two uniforms of a std::mt19937 (whose output
// sequence the C++ standard fixes), via Box-Muller. std::normal_distribution
// is not used: its algorithm is implementation-defined, so the pattern would
// differ between standard libraries.
double Gaussian(std::mt19937 &rng) {
  const double u1 = (double(rng()) + 0.5) / 4294967296.0;  // (0, 1)
  const double u2 = (double(rng()) + 0.5) / 4294967296.0;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

}  // namespace

const std::vector<BriefTest> &BriefPattern() {
  static const std::vector<BriefTest> pattern = [] {
    std::mt19937 rng(kBriefSeed);
    const double sigma = (2 * kBriefPatchRadius + 1) / 5.0;  // S = 31, sigma = S / 5
    const int r2 = kBriefPatchRadius * kBriefPatchRadius;
    auto point = [&](int &x, int &y) {
      do {  // rejection sampling: keep the point inside the disc
        x = int(std::lround(sigma * Gaussian(rng)));
        y = int(std::lround(sigma * Gaussian(rng)));
      } while (x * x + y * y > r2);
    };
    std::vector<BriefTest> p;
    p.reserve(kBriefTests);
    while (int(p.size()) < kBriefTests) {
      BriefTest t;
      point(t.x1, t.y1);
      point(t.x2, t.y2);
      if (t.x1 == t.x2 && t.y1 == t.y2) continue;  // a test against itself is always 0
      p.push_back(t);
    }
    return p;
  }();
  return pattern;
}

GrayImage GaussianSmooth(const GrayImage &img, double sigma, int radius) {
  std::vector<double> k(2 * radius + 1);
  double sum = 0;
  for (int i = -radius; i <= radius; ++i) sum += k[i + radius] = std::exp(-0.5 * i * i / (sigma * sigma));
  for (double &w : k) w /= sum;
  const int W = img.width, H = img.height;
  auto mirror = [](int i, int n) {  // reflect without repeating the edge pixel
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
  };
  std::vector<double> tmp(size_t(W) * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      double s = 0;
      for (int i = -radius; i <= radius; ++i) s += k[i + radius] * img.at(mirror(x + i, W), y);
      tmp[size_t(y) * W + x] = s;
    }
  GrayImage out(W, H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      double s = 0;
      for (int i = -radius; i <= radius; ++i) s += k[i + radius] * tmp[size_t(mirror(y + i, H)) * W + x];
      out.at(x, y) = uint8_t(std::clamp(std::lround(s), 0L, 255L));
    }
  return out;
}

bool ComputeBriefDescriptor(const GrayImage &smoothed, const Keypoint &kp, BriefDescriptor &out) {
  out = BriefDescriptor{};
  const double c = std::cos(kp.angle), s = std::sin(kp.angle);
  const int cx = int(kp.x), cy = int(kp.y);
  const auto &pattern = BriefPattern();
  for (int i = 0; i < kBriefTests; ++i) {
    const BriefTest &t = pattern[i];
    // rotate each test location by the keypoint angle (image axes: x right, y down)
    const int px = cx + int(std::lround(c * t.x1 - s * t.y1)), py = cy + int(std::lround(s * t.x1 + c * t.y1));
    const int qx = cx + int(std::lround(c * t.x2 - s * t.y2)), qy = cy + int(std::lround(s * t.x2 + c * t.y2));
    if (!smoothed.contains(px, py) || !smoothed.contains(qx, qy)) {
      out = BriefDescriptor{};
      return false;
    }
    if (smoothed.at(px, py) < smoothed.at(qx, qy)) out.set(i);
  }
  return true;
}

std::vector<BriefDescriptor> ComputeBrief(const GrayImage &smoothed, const std::vector<Keypoint> &keypoints,
                                          std::vector<bool> *valid) {
  std::vector<BriefDescriptor> d(keypoints.size());
  if (valid) valid->assign(keypoints.size(), false);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    const bool ok = ComputeBriefDescriptor(smoothed, keypoints[i], d[i]);
    if (valid) (*valid)[i] = ok;
  }
  return d;
}

int HammingDistance(const BriefDescriptor &a, const BriefDescriptor &b) {
  int n = 0;
  for (int w = 0; w < 4; ++w) {
    uint64_t v = a.bits[w] ^ b.bits[w];
    while (v) {  // count set bits (Kernighan)
      v &= v - 1;
      ++n;
    }
  }
  return n;
}

}  // namespace features
