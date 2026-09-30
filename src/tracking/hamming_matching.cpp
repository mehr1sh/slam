#include "tracking/hamming_matching.hpp"

#include <nmmintrin.h>

using namespace std;
using namespace cv;

int HammingDistance(const vector<uint32_t> &a, const vector<uint32_t> &b) {
  int dist = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dist += _mm_popcnt_u32(a[i] ^ b[i]);
  }
  return dist;
}

vector<DMatch> MatchHamming(const vector<vector<uint32_t>> &desc1,
                            const vector<vector<uint32_t>> &desc2, int max_distance) {
  vector<DMatch> matches;
  for (size_t i = 0; i < desc1.size(); ++i) {
    if (desc1[i].empty()) continue;
    int best_j = -1, best_dist = max_distance + 1;
    for (size_t j = 0; j < desc2.size(); ++j) {
      if (desc2[j].empty()) continue;
      int d = HammingDistance(desc1[i], desc2[j]);
      if (d < best_dist) {
        best_dist = d;
        best_j = (int)j;
      }
    }
    if (best_j >= 0 && best_dist <= max_distance) {
      matches.emplace_back((int)i, best_j, (float)best_dist);
    }
  }
  return matches;
}
