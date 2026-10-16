// Self-checking test of features/fast (segment test, score, non-maximum
// suppression, border, strongest-N, determinism). Exit status 0 = pass.

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "features/fast.hpp"
#include "feature_test_utils.hpp"

using namespace features;

namespace {

// 7x7 image, centre 100, the given circle positions set to `v`.
GrayImage Ring(const std::vector<int> &positions, int v) {
  GrayImage img(7, 7, 100);
  for (int i : positions) img.at(3 + kFastCircle[i][0], 3 + kFastCircle[i][1]) = uint8_t(v);
  return img;
}

std::vector<int> Arc(int start, int len) {
  std::vector<int> a;
  for (int k = 0; k < len; ++k) a.push_back((start + k) % 16);
  return a;
}

}  // namespace

int main() {
  std::printf("=== features/fast ===\n");

  // --- segment test ---
  FCHECK(IsFastCorner(Ring(Arc(0, 9), 200), 3, 3, 30, 9), "9 contiguous brighter pixels -> corner");
  FCHECK(IsFastCorner(Ring(Arc(0, 9), 10), 3, 3, 30, 9), "9 contiguous darker pixels -> corner");
  FCHECK(!IsFastCorner(Ring(Arc(0, 8), 200), 3, 3, 30, 9), "8 contiguous brighter pixels -> not a corner");
  FCHECK(IsFastCorner(Ring(Arc(12, 9), 200), 3, 3, 30, 9), "arc wrapping 15 -> 0 is contiguous");
  FCHECK(!IsFastCorner(Ring(Arc(0, 9), 130), 3, 3, 30, 9), "difference exactly at the threshold does not count");
  {
    auto mixed = Ring(Arc(0, 5), 200);
    for (int i : Arc(5, 5)) mixed.at(3 + kFastCircle[i][0], 3 + kFastCircle[i][1]) = 10;
    FCHECK(!IsFastCorner(mixed, 3, 3, 30, 9), "5 brighter + 5 darker is not a 9-arc");
  }

  // --- score ---
  FCHECK(FastScore(Ring(Arc(0, 9), 200), 3, 3, 30) == 9 * (200 - 100 - 30), "score = sum (I-c-t) over brighter = %d",
         FastScore(Ring(Arc(0, 9), 200), 3, 3, 30));
  FCHECK(FastScore(Ring(Arc(0, 12), 200), 3, 3, 30) > FastScore(Ring(Arc(0, 9), 200), 3, 3, 30),
         "a longer arc scores higher");

  // --- detection on a synthetic square: corners found, edges not ---
  {
    GrayImage sq(128, 128, 50);
    for (int y = 40; y < 88; ++y)
      for (int x = 40; x < 88; ++x) sq.at(x, y) = 200;
    FastParams p;
    p.max_keypoints = 0;
    auto kps = DetectFast(sq, p);
    const int corners[4][2] = {{40, 40}, {87, 40}, {40, 87}, {87, 87}};
    int found = 0;
    for (auto &c : corners)
      found += std::any_of(kps.begin(), kps.end(), [&](const Keypoint &k) {
        return std::abs(k.x - c[0]) <= 2 && std::abs(k.y - c[1]) <= 2;
      });
    bool all_near = std::all_of(kps.begin(), kps.end(), [&](const Keypoint &k) {
      for (auto &c : corners)
        if (std::abs(k.x - c[0]) <= 3 && std::abs(k.y - c[1]) <= 3) return true;
      return false;
    });
    FCHECK(found == 4, "all 4 square corners detected (%d keypoints total)", int(kps.size()));
    FCHECK(all_near, "no keypoint on straight edges or flat regions");
    GrayImage flat(64, 64, 128);
    FCHECK(DetectFast(flat).empty(), "flat image -> no keypoints");
  }

  // --- noise image (thousands of corners): NMS, border, cap, ordering, determinism ---
  {
    const GrayImage img = ftest::Noise(320, 240, 7);
    FastParams all;
    all.max_keypoints = 0;
    const auto every = DetectFast(img, all);
    const auto kps = DetectFast(img);  // defaults: threshold 30, border 16, max 500
    FCHECK(every.size() > 500, "noise image has %d corners after NMS (> 500, so the cap matters)",
           int(every.size()));
    FCHECK(kps.size() == 500, "strongest-500 cap: %d kept", int(kps.size()));
    bool border_ok = true, nms_ok = true, sorted_ok = true;
    for (size_t i = 0; i < kps.size(); ++i) {
      const auto &k = kps[i];
      border_ok &= k.x >= 16 && k.y >= 16 && k.x < img.width - 16 && k.y < img.height - 16;
      if (i > 0) {
        const auto &a = kps[i - 1];
        sorted_ok &= a.score > k.score || (a.score == k.score && (a.y < k.y || (a.y == k.y && a.x < k.x)));
      }
    }
    for (size_t i = 0; i < every.size(); ++i)
      for (size_t j = i + 1; j < every.size(); ++j)
        nms_ok &= std::max(std::abs(every[i].x - every[j].x), std::abs(every[i].y - every[j].y)) > 1;
    FCHECK(border_ok, "every keypoint >= 16 px from the image edge");
    FCHECK(nms_ok, "3x3 NMS: no two keypoints are 8-neighbours");
    FCHECK(sorted_ok, "order: score descending, then y, then x");
    FCHECK(std::equal(kps.begin(), kps.end(), every.begin(),
                      [](const Keypoint &a, const Keypoint &b) { return a.x == b.x && a.y == b.y; }),
           "the 500 kept are the first 500 of the full ordered list");
    const auto again = DetectFast(img);
    FCHECK(std::equal(kps.begin(), kps.end(), again.begin(),
                      [](const Keypoint &a, const Keypoint &b) {
                        return a.x == b.x && a.y == b.y && a.score == b.score;
                      }),
           "deterministic: two runs give identical keypoints");
    FastParams higher;
    higher.threshold = 60;
    higher.max_keypoints = 0;
    FCHECK(DetectFast(img, higher).size() < every.size(), "a higher threshold gives fewer corners (%d < %d)",
           int(DetectFast(img, higher).size()), int(every.size()));
  }

  // --- per-cell threshold fallback (spatial distribution) ---
  {
    // 32-px cells start at the border (16): x cells [16,48) [48,80) ..., y cells [16,48) [48,80).
    // strong square (contrast 100) and weak square A (contrast 12) share cell (0,0);
    // weak square B (contrast 12) is alone in cell (2,1).
    GrayImage img(160, 96, 100);
    auto square = [&](int x0, int y0, int v) {
      for (int y = y0; y < y0 + 8; ++y)
        for (int x = x0; x < x0 + 8; ++x) img.at(x, y) = uint8_t(v);
    };
    square(20, 20, 200);
    square(34, 34, 112);
    square(88, 56, 112);
    auto near = [](const std::vector<Keypoint> &k, int x0, int y0) {
      return std::any_of(k.begin(), k.end(), [&](const Keypoint &p) {
        return p.x >= x0 - 2 && p.x <= x0 + 9 && p.y >= y0 - 2 && p.y <= y0 + 9;
      });
    };
    FastParams off;
    off.threshold = 20;
    off.max_keypoints = 0;
    FastParams on = off;
    on.cell_size = 32;
    on.min_threshold = 7;
    const auto a = DetectFast(img, off), b = DetectFast(img, on);
    FCHECK(near(a, 20, 20) && !near(a, 34, 34) && !near(a, 88, 56), "threshold 20 alone: only the strong square");
    FCHECK(near(b, 20, 20) && near(b, 88, 56), "fallback: the weak square in the otherwise empty cell is found");
    FCHECK(!near(b, 34, 34), "fallback: the weak square sharing a cell with a corner is not added");
    int strong = 0;
    for (const auto &k : a) strong += near(b, int(k.x) - 1, int(k.y) - 1);
    FCHECK(strong == int(a.size()), "fallback keeps every corner found at the main threshold (%d)", strong);
    FastParams same = on;
    same.min_threshold = 20;  // not below the main threshold: off
    const auto c = DetectFast(img, same);
    FCHECK(c.size() == a.size() && std::equal(a.begin(), a.end(), c.begin(), [](const Keypoint &p, const Keypoint &q) {
             return p.x == q.x && p.y == q.y && p.score == q.score;
           }),
           "min_threshold >= threshold disables the fallback (identical result)");
    FastParams defaults;
    FCHECK(defaults.cell_size <= 0 && defaults.min_threshold <= 0, "the fallback is off by default");
  }
  return ftest::Finish("features_fast_test");
}
