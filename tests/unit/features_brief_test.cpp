// Self-checking test of features/brief (pattern, smoothing, rotated
// descriptor, Hamming distance).

#include <cmath>
#include <set>
#include <tuple>

#include "features/brief.hpp"
#include "features/fast.hpp"
#include "features/orientation.hpp"
#include "feature_test_utils.hpp"

using namespace features;

int main() {
  std::printf("=== features/brief ===\n");

  // --- pattern: fixed, deterministic, inside the disc, no degenerate tests ---
  const auto &pat = BriefPattern();
  bool inside = true, distinct = true;
  std::set<std::tuple<int, int, int, int>> uniq;
  uint64_t h = 1469598103934665603ull;  // FNV-1a over the pattern
  for (const auto &t : pat) {
    inside &= t.x1 * t.x1 + t.y1 * t.y1 <= 225 && t.x2 * t.x2 + t.y2 * t.y2 <= 225;
    distinct &= !(t.x1 == t.x2 && t.y1 == t.y2);
    uniq.insert({t.x1, t.y1, t.x2, t.y2});
    for (int v : {t.x1, t.y1, t.x2, t.y2}) h = (h ^ uint64_t(v + 64)) * 1099511628211ull;
  }
  FCHECK(pat.size() == 256, "256 binary tests");
  FCHECK(inside, "every test point within the radius-15 disc");
  FCHECK(distinct, "no test compares a pixel with itself");
  FCHECK(uniq.size() >= 250, "%d distinct tests", int(uniq.size()));
  FCHECK(&BriefPattern() == &pat, "pattern generated once and reused");
  std::printf("  pattern fingerprint (FNV-1a): %016llx\n", (unsigned long long)h);
  FCHECK(h == 0x24bb0ef5a0a96263ull, "pattern equals the fixed seeded pattern (fingerprint unchanged)");
  double spread = 0;
  for (const auto &t : pat) spread += t.x1 * t.x1 + t.y1 * t.y1;
  spread = std::sqrt(spread / pat.size() / 2);
  FCHECK(spread > 5.0 && spread < 7.0, "per-axis spread %.2f ~ sigma 6.2 (truncated Gaussian)", spread);

  // --- smoothing ---
  {
    GrayImage flat(30, 20, 77);
    FCHECK(GaussianSmooth(flat).pixels == flat.pixels, "smoothing a constant image changes nothing");
    GrayImage dot(31, 31, 0);
    dot.at(15, 15) = 255;
    const GrayImage s = GaussianSmooth(dot);
    bool sym = true;
    for (int y = 0; y < 31; ++y)
      for (int x = 0; x < 31; ++x) sym &= s.at(x, y) == s.at(30 - x, y) && s.at(x, y) == s.at(y, x);
    FCHECK(sym && s.at(15, 15) > s.at(16, 15) && s.at(19, 15) > 0 && s.at(20, 15) == 0,
           "impulse response: symmetric, peaked, 9x9 support");
  }

  // --- descriptors on a textured image ---
  const GrayImage img = ftest::Texture(200, 200, 3);
  const GrayImage sm = GaussianSmooth(img);
  std::vector<Keypoint> kps = DetectFast(img);
  AssignOrientations(img, kps);
  std::vector<bool> valid;
  const auto d = ComputeBrief(sm, kps, &valid);
  int nvalid = 0, ones = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    nvalid += valid[i];
    for (int b = 0; b < 256; ++b) ones += d[i].bit(b);
  }
  FCHECK(nvalid == int(kps.size()), "all %d keypoints (border 16) get a valid descriptor", int(kps.size()));
  const double frac = double(ones) / (256.0 * d.size());
  FCHECK(frac > 0.35 && frac < 0.65, "bits are balanced: %.3f ones", frac);
  {
    // On pure noise, descriptors of different keypoints are unrelated -> ~128 bits apart.
    const GrayImage noise = ftest::Noise(200, 200, 21);
    std::vector<Keypoint> nk = DetectFast(noise);
    AssignOrientations(noise, nk);
    const auto nd = ComputeBrief(GaussianSmooth(noise), nk);
    double mean_d = 0;
    int pairs = 0;
    for (size_t i = 0; i + 1 < nd.size(); i += 2, ++pairs) mean_d += HammingDistance(nd[i], nd[i + 1]);
    mean_d /= pairs;
    FCHECK(mean_d > 110 && mean_d < 146, "noise image, %d unrelated keypoint pairs: mean distance %.1f (~128)", pairs,
           mean_d);
  }
  FCHECK(HammingDistance(d[0], ComputeBrief(sm, {kps[0]})[0]) == 0, "same keypoint -> identical descriptor");
  {
    BriefDescriptor tmp;
    Keypoint edge;
    edge.x = 3;
    edge.y = 100;
    FCHECK(!ComputeBriefDescriptor(sm, edge, tmp), "keypoint 3 px from the edge -> invalid (pattern leaves the image)");
  }

  // --- rotation: exact 90-degree image rotation ---
  {
    const GrayImage rot = ftest::Rotate90(img), rsm = GaussianSmooth(rot);
    int n = 0, steered = 0, unsteered = 0;
    for (const Keypoint &k : kps) {
      Keypoint r;
      r.x = k.y;
      r.y = float(img.width - 1) - k.x;
      if (r.x < 16 || r.y < 16 || r.x >= rot.width - 16 || r.y >= rot.height - 16) continue;
      r.angle = IntensityCentroidAngle(rot, int(r.x), int(r.y));
      BriefDescriptor a, b;
      ComputeBriefDescriptor(sm, k, a);
      ComputeBriefDescriptor(rsm, r, b);
      steered += HammingDistance(a, b);
      Keypoint k0 = k, r0 = r;  // without steering (angle 0 in both images)
      k0.angle = r0.angle = 0;
      ComputeBriefDescriptor(sm, k0, a);
      ComputeBriefDescriptor(rsm, r0, b);
      unsteered += HammingDistance(a, b);
      ++n;
    }
    FCHECK(n > 50 && double(steered) / n < 8.0, "rotated BRIEF, %d keypoints: mean distance %.2f after a 90 deg rotation",
           n, double(steered) / n);
    FCHECK(double(unsteered) / n > 60.0, "unrotated BRIEF on the same points: mean distance %.1f (not invariant)",
           double(unsteered) / n);
  }

  // --- Hamming distance ---
  {
    BriefDescriptor a, b;
    for (int i : {0, 63, 64, 130, 255}) b.set(i);
    FCHECK(HammingDistance(a, b) == 5 && HammingDistance(b, b) == 0, "Hamming distance counts differing bits");
  }
  return ftest::Finish("features_brief_test");
}
