#include "features/matcher.hpp"

#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>

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

const char *MatchFateName(MatchFate f) {
  switch (f) {
    case MatchFate::kKept: return "kept";
    case MatchFate::kRemovedDistance: return "distance";
    case MatchFate::kRemovedMutual: return "mutual";
    case MatchFate::kRemovedRatio: return "ratio";
    default: return "level";
  }
}

namespace {
// distance / second_distance as an exact fraction (numerator, denominator)
void RatioFraction(const RankedMatch &m, long &num, long &den) {
  if (!m.has_second) num = 0, den = 1;
  else if (m.second_distance == 0) num = 1, den = 1;
  else num = m.distance, den = m.second_distance;
}
}  // namespace

double RankedMatch::ratio() const {
  long num, den;
  RatioFraction(*this, num, den);
  return double(num) / double(den);
}

bool QualityBefore(const RankedMatch &a, const RankedMatch &b) {
  long an, ad, bn, bd;
  RatioFraction(a, an, ad);
  RatioFraction(b, bn, bd);
  if (an * bd != bn * ad) return an * bd < bn * ad;
  if (a.distance != b.distance) return a.distance < b.distance;
  return a.query < b.query;
}

MatchSet MatchRanked(const std::vector<BriefDescriptor> &query, const std::vector<BriefDescriptor> &train,
                     const std::vector<float> &train_x, const std::vector<float> &train_y,
                     const std::vector<int> &query_level, const std::vector<int> &train_level,
                     const MatcherParams &params, const std::vector<bool> &query_valid,
                     const std::vector<bool> &train_valid, const std::vector<float> &query_x,
                     const std::vector<float> &query_y) {
  const size_t nq = query.size(), nt = train.size();
  if (train_x.size() != nt || train_y.size() != nt)
    throw std::invalid_argument("MatchRanked: train positions must match the train descriptors");
  if ((!query_level.empty() && query_level.size() != nq) || (!train_level.empty() && train_level.size() != nt) ||
      (!query_valid.empty() && query_valid.size() != nq) || (!train_valid.empty() && train_valid.size() != nt))
    throw std::invalid_argument("MatchRanked: level / validity lists must be empty or one per descriptor");
  const bool have_qpos = !query_x.empty();
  if (have_qpos ? (query_x.size() != nq || query_y.size() != nq) : !query_y.empty())
    throw std::invalid_argument("MatchRanked: query positions must be empty or one per query descriptor");
  if (params.mutual_location_aware && !have_qpos)
    throw std::invalid_argument("MatchRanked: mutual_location_aware needs the query positions");
  auto qvalid = [&](size_t q) { return query_valid.empty() || query_valid[q]; };
  auto tvalid = [&](size_t t) { return train_valid.empty() || train_valid[t]; };
  MatchSet out;
  MatchCounters &c = out.counters;
  c.queries = int(nq), c.trains = int(nt);
  for (size_t q = 0; q < nq; ++q) c.invalid_query += !qvalid(q);
  for (size_t t = 0; t < nt; ++t) c.invalid_train += !tvalid(t);
  if (c.invalid_train == int(nt)) return out;  // no valid train: no matches (also covers nt == 0)

  // forward nearest neighbour, location-aware second best, reverse nearest neighbour
  std::vector<int> row(nt);
  std::vector<int> col_best_d(nt, std::numeric_limits<int>::max()), col_best_q(nt, -1);
  const float r2 = params.same_location_radius_px * params.same_location_radius_px;
  for (size_t q = 0; q < nq; ++q) {
    if (!qvalid(q)) continue;
    RankedMatch m;
    m.query = int(q);
    m.distance = 257;
    for (size_t t = 0; t < nt; ++t) {
      if (!tvalid(t)) continue;
      const int d = HammingDistance(query[q], train[t]);
      row[t] = d;
      if (d < m.distance) m.distance = d, m.train = int(t);  // strict: lowest index among ties
      if (d < col_best_d[t]) col_best_d[t] = d, col_best_q[t] = int(q);  // strict: lowest query index
    }
    const float bx = train_x[m.train], by = train_y[m.train];
    for (size_t t = 0; t < nt; ++t) {
      if (!tvalid(t) || int(t) == m.train) continue;
      if (m.has_second && row[t] >= m.second_distance) continue;  // lowest index among equal second distances
      const float dx = train_x[t] - bx, dy = train_y[t] - by;
      if (dx * dx + dy * dy <= r2) continue;  // the same location (e.g. the same corner on another level)
      m.has_second = true, m.second_distance = row[t], m.second_train = int(t);
    }
    m.level_query = query_level.empty() ? 0 : query_level[q];
    m.level_train = train_level.empty() ? 0 : train_level[m.train];
    out.raw.push_back(m);
  }
  c.raw = int(out.raw.size());
  for (const RankedMatch &m : out.raw) c.d_min = std::min(c.d_min, m.distance), c.d_max = std::max(c.d_max, m.distance);
  const int limit = std::max(2 * c.d_min, params.floor_distance);

  for (RankedMatch &m : out.raw) {
    m.reverse_query = col_best_q[m.train];
    m.mutual = m.reverse_query == m.query;
    if (have_qpos) {
      const float dx = query_x[m.reverse_query] - query_x[m.query], dy = query_y[m.reverse_query] - query_y[m.query];
      m.mutual_location = dx * dx + dy * dy <= r2;
    }
    const bool mutual_ok = params.mutual_location_aware ? m.mutual_location : m.mutual;
    if (m.distance > limit) m.fate = MatchFate::kRemovedDistance, ++c.removed_distance;
    else if ((params.mutual || params.mutual_location_aware) && !mutual_ok) m.fate = MatchFate::kRemovedMutual, ++c.removed_mutual;
    else if (params.ratio > 0 && m.has_second && !(double(m.distance) < params.ratio * double(m.second_distance)))
      m.fate = MatchFate::kRemovedRatio, ++c.removed_ratio;
    else if (params.max_level_diff >= 0 && std::abs(m.level_query - m.level_train) > params.max_level_diff)
      m.fate = MatchFate::kRemovedLevel, ++c.removed_level;
    else m.fate = MatchFate::kKept;
  }
  std::vector<int> order;
  for (size_t k = 0; k < out.raw.size(); ++k)
    if (out.raw[k].fate == MatchFate::kKept) order.push_back(int(k));
  std::vector<int> ranked = order;
  std::sort(ranked.begin(), ranked.end(), [&](int a, int b) { return QualityBefore(out.raw[a], out.raw[b]); });
  for (size_t r = 0; r < ranked.size(); ++r) out.raw[ranked[r]].rank = int(r);
  for (int k : params.rank ? ranked : order) out.kept.push_back(out.raw[k]);
  c.kept = int(out.kept.size());
  return out;
}

} // namespace features
