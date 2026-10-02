// Self-checking test of features/matcher (brute-force Hamming NN, the
// distance filter) and of the whole chain on a shifted image.

#include <algorithm>
#include <cmath>

#include "features/brief.hpp"
#include "features/fast.hpp"
#include "features/matcher.hpp"
#include "features/orientation.hpp"
#include "feature_test_utils.hpp"

using namespace features;

namespace {

BriefDescriptor WithBits(std::initializer_list<int> bits) {
  BriefDescriptor d;
  for (int b : bits) d.set(b);
  return d;
}

Match M(int q, int t, int dist) {
  Match m;
  m.query = q;
  m.train = t;
  m.distance = dist;
  return m;
}

}  // namespace

int main() {
  std::printf("=== features/matcher ===\n");

  // --- nearest neighbour ---
  const std::vector<BriefDescriptor> query = {WithBits({1, 2, 3}), WithBits({100, 200})};
  const std::vector<BriefDescriptor> train = {WithBits({1, 2, 3, 4, 5}), WithBits({100}), WithBits({1, 2, 3}),
                                              WithBits({200})};
  const auto m = MatchBruteForce(query, train);
  FCHECK(m.size() == 2, "one match per query descriptor");
  FCHECK(m[0].train == 2 && m[0].distance == 0, "exact copy found (train 2, distance 0)");
  FCHECK(m[1].train == 1 && m[1].distance == 1, "tie between train 1 and 3 (distance 1) -> lowest index 1");
  FCHECK(MatchBruteForce({}, train).empty() && MatchBruteForce(query, {}).empty(), "empty input -> no matches");

  // --- filter: keep distance <= max(2 * d_min, 30) ---
  {
    int dmin, dmax;
    auto k = FilterMatchesByDistance({M(0, 0, 10), M(1, 1, 25), M(2, 2, 31), M(3, 3, 40), M(4, 4, 19)}, 30, &dmin,
                                     &dmax);
    FCHECK(dmin == 10 && dmax == 40 && k.size() == 3 && k[0].query == 0 && k[1].query == 1 && k[2].query == 4,
           "d_min 10 -> limit max(20, 30) = 30: keeps 10, 25, 19");
    k = FilterMatchesByDistance({M(0, 0, 20), M(1, 1, 40), M(2, 2, 41)});
    FCHECK(k.size() == 2, "d_min 20 -> limit 40: keeps 20, 40; drops 41");
  }

  // --- whole chain on an image and its shifted copy ---
  {
    const GrayImage a = ftest::Texture(320, 240, 5);
    GrayImage b(320, 240);
    const int sx = 7, sy = -4;  // b(x, y) = a(x - sx, y - sy)
    for (int y = 0; y < 240; ++y)
      for (int x = 0; x < 320; ++x) b.at(x, y) = a.at(std::clamp(x - sx, 0, 319), std::clamp(y - sy, 0, 239));
    auto run = [](const GrayImage &img, std::vector<Keypoint> &k) {
      k = DetectFast(img);
      AssignOrientations(img, k);
      return ComputeBrief(GaussianSmooth(img), k);
    };
    std::vector<Keypoint> ka, kb;
    const auto da = run(a, ka), db = run(b, kb);
    const auto filtered = FilterMatchesByDistance(MatchBruteForce(da, db));
    int correct = 0;
    for (const Match &mm : filtered)
      correct += std::abs(kb[mm.train].x - ka[mm.query].x - sx) < 0.5f &&
                 std::abs(kb[mm.train].y - ka[mm.query].y - sy) < 0.5f;
    FCHECK(filtered.size() > 100, "shifted image: %d filtered matches", int(filtered.size()));
    FCHECK(correct >= 0.9 * filtered.size(), "%d / %d filtered matches have exactly the true shift (7, -4)", correct,
           int(filtered.size()));
  }
  return ftest::Finish("features_matcher_test");
}
