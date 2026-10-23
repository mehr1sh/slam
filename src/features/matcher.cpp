#include "features/matcher.hpp"

#include <algorithm>

#include "features/brief.hpp"

namespace features {

std::vector<Match> MatchBruteForce(const std::vector<BriefDescriptor> &query,
                                   const std::vector<BriefDescriptor> &train) {
  std::vector<Match> out;
  if (query.empty() || train.empty())
    return out;
  out.reserve(query.size());
  for (size_t q = 0; q < query.size(); ++q) {
    Match best;
    best.query = int(q);
    best.distance = 257;
    for (size_t t = 0; t < train.size(); ++t) {
      const int d = HammingDistance(query[q], train[t]);
      if (d < best.distance) { // strict: the first (lowest-index) of equal
                               // distances is kept
        best.distance = d;
        best.train = int(t);
      }
    }
    out.push_back(best);
  }
  return out;
}

std::vector<Match> FilterMatchesByDistance(const std::vector<Match> &matches,
                                           int floor_distance, int *d_min_out,
                                           int *d_max_out) {
  int d_min = 10000, d_max = 0;
  for (const Match &m : matches) {
    d_min = std::min(d_min, m.distance);
    d_max = std::max(d_max, m.distance);
  }
  if (d_min_out)
    *d_min_out = d_min;
  if (d_max_out)
    *d_max_out = d_max;
  std::vector<Match> kept;
  const int limit = std::max(2 * d_min, floor_distance);
  for (const Match &m : matches)
    if (m.distance <= limit)
      kept.push_back(m);
  return kept;
}

std::vector<Match> MatchMutualRatio(const std::vector<BriefDescriptor> &a,
                                    const std::vector<BriefDescriptor> &b,
                                    double ratio) {
  if (ratio <= 0 || ratio >= 1)
    return {};
  auto best = [&](const std::vector<BriefDescriptor> &q,
                  const std::vector<BriefDescriptor> &t) {
    std::vector<Match> out(q.size());
    for (size_t i = 0; i < q.size(); ++i) {
      int first = 257, second = 257, id = -1;
      for (size_t j = 0; j < t.size(); ++j) {
        int d = HammingDistance(q[i], t[j]);
        if (d < first) {
          second = first;
          first = d;
          id = j;
        } else if (d < second)
          second = d;
      }
      out[i] = {int(i), t.size() >= 2 && first < ratio * second ? id : -1,
                first};
    }
    return out;
  };
  auto ab = best(a, b), ba = best(b, a);
  std::vector<Match> out;
  for (auto m : ab)
    if (m.train >= 0 && ba[m.train].train == m.query)
      out.push_back(m);
  return out;
}
} // namespace features
