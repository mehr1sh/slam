#include "features/matcher.hpp"

#include <algorithm>

#include "features/brief.hpp"

namespace features {

std::vector<Match> MatchBruteForce(const std::vector<BriefDescriptor> &query,
                                   const std::vector<BriefDescriptor> &train) {
  std::vector<Match> out;
  if (query.empty() || train.empty()) return out;
  out.reserve(query.size());
  for (size_t q = 0; q < query.size(); ++q) {
    Match best;
    best.query = int(q);
    best.distance = 257;
    for (size_t t = 0; t < train.size(); ++t) {
      const int d = HammingDistance(query[q], train[t]);
      if (d < best.distance) {  // strict: the first (lowest-index) of equal distances is kept
        best.distance = d;
        best.train = int(t);
      }
    }
    out.push_back(best);
  }
  return out;
}

std::vector<Match> FilterMatchesByDistance(const std::vector<Match> &matches, int floor_distance, int *d_min_out,
                                           int *d_max_out) {
  int d_min = 10000, d_max = 0;
  for (const Match &m : matches) {
    d_min = std::min(d_min, m.distance);
    d_max = std::max(d_max, m.distance);
  }
  if (d_min_out) *d_min_out = d_min;
  if (d_max_out) *d_max_out = d_max;
  std::vector<Match> kept;
  const int limit = std::max(2 * d_min, floor_distance);
  for (const Match &m : matches)
    if (m.distance <= limit) kept.push_back(m);
  return kept;
}

}  // namespace features
