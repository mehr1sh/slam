#include "features/fast.hpp"

#include <algorithm>

namespace features {

const int kFastCircle[16][2] = {{0, -3}, {1, -3},  {2, -2},  {3, -1},  {3, 0},  {3, 1},  {2, 2},  {1, 3},
                                {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

bool IsFastCorner(const GrayImage &img, int x, int y, int threshold, int arc_length) {
  const int c = img.at(x, y);
  int cls[16];  // +1 brighter, -1 darker, 0 similar
  for (int i = 0; i < 16; ++i) {
    const int v = img.at(x + kFastCircle[i][0], y + kFastCircle[i][1]);
    cls[i] = v > c + threshold ? 1 : (v < c - threshold ? -1 : 0);
  }
  // longest circular run of equal non-zero class
  for (int sign : {1, -1}) {
    int run = 0;
    for (int k = 0; k < 32; ++k) {  // two laps cover runs that wrap around
      run = cls[k & 15] == sign ? run + 1 : 0;
      if (run >= arc_length) return true;
    }
  }
  return false;
}

int FastScore(const GrayImage &img, int x, int y, int threshold) {
  const int c = img.at(x, y);
  int bright = 0, dark = 0;
  for (int i = 0; i < 16; ++i) {
    const int v = img.at(x + kFastCircle[i][0], y + kFastCircle[i][1]);
    if (v > c + threshold) bright += v - c - threshold;
    else if (v < c - threshold) dark += c - v - threshold;
  }
  return std::max(bright, dark);
}

std::vector<Keypoint> DetectFast(const GrayImage &img, const FastParams &params) {
  const int b = std::max(3, params.border);
  const int W = img.width, H = img.height;
  std::vector<Keypoint> out;
  if (W <= 2 * b || H <= 2 * b) return out;

  // score map: 0 = not a corner
  std::vector<int> score(size_t(W) * H, 0);
  for (int y = b; y < H - b; ++y)
    for (int x = b; x < W - b; ++x)
      if (IsFastCorner(img, x, y, params.threshold, params.arc_length))
        score[size_t(y) * W + x] = std::max(1, FastScore(img, x, y, params.threshold));

  // 3x3 non-maximum suppression; a neighbour with an equal score suppresses
  // the centre only if it comes earlier in raster order (deterministic).
  for (int y = b; y < H - b; ++y)
    for (int x = b; x < W - b; ++x) {
      const int s = score[size_t(y) * W + x];
      if (s == 0) continue;
      bool is_max = true;
      for (int dy = -1; dy <= 1 && is_max; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          const int n = score[size_t(y + dy) * W + (x + dx)];
          const bool earlier = dy < 0 || (dy == 0 && dx < 0);
          if (n > s || (n == s && earlier)) {
            is_max = false;
            break;
          }
        }
      if (is_max) {
        Keypoint k;
        k.x = float(x);
        k.y = float(y);
        k.score = s;
        out.push_back(k);
      }
    }

  std::sort(out.begin(), out.end(), [](const Keypoint &a, const Keypoint &c) {
    if (a.score != c.score) return a.score > c.score;
    if (a.y != c.y) return a.y < c.y;
    return a.x < c.x;
  });
  if (params.max_keypoints > 0 && int(out.size()) > params.max_keypoints) out.resize(params.max_keypoints);
  return out;
}

}  // namespace features
