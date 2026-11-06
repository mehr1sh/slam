// Self-checking test of features::MatchRanked (Stage 3): default-off equivalence with
// MatchBruteForce + FilterMatchesByDistance, many-to-one and mutual matching, tie-breaking,
// the location-aware ratio test (cross-level duplicates vs a genuine ambiguous candidate),
// invalid descriptors, zero-distance / missing-second cases, the level filter, the quality
// order and the removal counters.

#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

#include "features/brief.hpp"
#include "features/fast.hpp"
#include "features/matcher.hpp"
#include "features/multiscale.hpp"
#include "features/orientation.hpp"
#include "features/pyramid.hpp"
#include "feature_test_utils.hpp"

using namespace features;

namespace {

// A descriptor with bits [from, from + n) set: Hamming(Bits(0, a), Bits(0, b)) = |a - b|.
BriefDescriptor Bits(int from, int n) {
  BriefDescriptor d;
  for (int b = from; b < from + n; ++b) d.set(b);
  return d;
}

bool SameMatches(const std::vector<Match> &a, const std::vector<RankedMatch> &b) {
  if (a.size() != b.size()) return false;
  for (size_t k = 0; k < a.size(); ++k)
    if (a[k].query != b[k].query || a[k].train != b[k].train || a[k].distance != b[k].distance) return false;
  return true;
}

bool CountersAddUp(const MatchCounters &c) {
  return c.raw == c.kept + c.removed_distance + c.removed_mutual + c.removed_ratio + c.removed_level;
}

// Equivalence of the default MatchRanked with the original two-step matcher on given descriptors.
bool Equivalent(const std::vector<BriefDescriptor> &q, const std::vector<BriefDescriptor> &t, int floor_distance,
                std::string *why) {
  const std::vector<Match> raw = MatchBruteForce(q, t);
  int dmin = 0, dmax = 0;
  const std::vector<Match> kept = FilterMatchesByDistance(raw, floor_distance, &dmin, &dmax);
  std::vector<float> x(t.size()), y(t.size());
  for (size_t k = 0; k < t.size(); ++k) x[k] = float(k % 37) * 9.0f, y[k] = float(k / 37) * 9.0f;
  MatcherParams p;
  p.floor_distance = floor_distance;
  const MatchSet s = MatchRanked(q, t, x, y, {}, {}, p);
  if (!SameMatches(raw, s.raw)) return *why = "raw matches differ", false;
  if (!SameMatches(kept, s.kept)) return *why = "kept matches differ", false;
  if (!raw.empty() && (s.counters.d_min != dmin || s.counters.d_max != dmax)) return *why = "d_min / d_max differ", false;
  if (s.counters.kept != int(kept.size()) || s.counters.removed_distance != int(raw.size() - kept.size()))
    return *why = "counters differ", false;
  return true;
}

}  // namespace

int main() {
  std::printf("=== features/MatchRanked ===\n");
  std::string why;

  std::printf("\n-- default options reproduce MatchBruteForce + FilterMatchesByDistance --\n");
  {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 3; ++trial) {
      // few set bits -> many equal distances (ties) and a spread of distances around the floor
      std::vector<BriefDescriptor> q(300), t(350);
      for (auto *v : {&q, &t})
        for (auto &d : *v)
          for (int k = 0; k < 40; ++k) d.set(int(rng() % 96));
      why.clear();
      FCHECK(Equivalent(q, t, 30, &why), "random descriptors with ties (trial %d): identical raw / kept matches, "
             "distances, order, d_min / d_max%s", trial, why.empty() ? "" : (": " + why).c_str());
    }
    why.clear();
    FCHECK(Equivalent({Bits(0, 5)}, {Bits(0, 5)}, 30, &why), "a single exact match");
    FCHECK(MatchRanked({}, {Bits(0, 1)}, {0}, {0}, {}, {}).raw.empty() &&
               MatchRanked({Bits(0, 1)}, {}, {}, {}, {}, {}).raw.empty(),
           "empty query or train: no matches (as MatchBruteForce)");
    // a real multiscale extraction: textured image and a shifted copy
    const GrayImage a = ftest::Texture(320, 240, 11);
    GrayImage b(320, 240);
    for (int y = 0; y < 240; ++y)
      for (int x = 0; x < 320; ++x) b.at(x, y) = a.at(std::clamp(x - 5, 0, 319), std::clamp(y + 3, 0, 239));
    PyramidParams pp;
    FastParams fp;
    const MultiscaleFeatures fa = ExtractMultiscale(BuildPyramid(a, pp), fp), fb = ExtractMultiscale(BuildPyramid(b, pp), fp);
    std::vector<float> bx, by;
    std::vector<int> la, lb;
    for (const auto &k : fb.keypoints) bx.push_back(k.x), by.push_back(k.y), lb.push_back(k.level);
    for (const auto &k : fa.keypoints) la.push_back(k.level);
    const std::vector<Match> raw = MatchBruteForce(fa.descriptors, fb.descriptors);
    const std::vector<Match> kept = FilterMatchesByDistance(raw);
    const MatchSet s = MatchRanked(fa.descriptors, fb.descriptors, bx, by, la, lb, MatcherParams(), fa.valid, fb.valid);
    FCHECK(SameMatches(raw, s.raw) && SameMatches(kept, s.kept) && s.counters.invalid_query == 0 &&
               s.counters.invalid_train == 0,
           "8-level pyramid extraction (%zu / %zu keypoints): identical raw (%zu) and kept (%zu) matches",
           fa.keypoints.size(), fb.keypoints.size(), raw.size(), kept.size());
  }

  std::printf("\n-- many-to-one and mutual matching --\n");
  {
    // three queries whose nearest train is t0; t1 is nearest only to q3
    const std::vector<BriefDescriptor> q = {Bits(0, 10), Bits(0, 11), Bits(0, 12), Bits(100, 10)};
    const std::vector<BriefDescriptor> t = {Bits(0, 10), Bits(100, 10)};
    const std::vector<float> x = {10, 200}, y = {10, 200};
    const MatchSet def = MatchRanked(q, t, x, y, {}, {});
    FCHECK(def.kept.size() == 4 && def.kept[0].train == 0 && def.kept[1].train == 0 && def.kept[2].train == 0,
           "default: three queries share train 0 (many-to-one kept, as before)");
    FCHECK(def.raw[0].mutual && !def.raw[1].mutual && !def.raw[2].mutual && def.raw[3].mutual,
           "mutual flag: q0 <-> t0 and q3 <-> t1 are mutual; q1, q2 -> t0 are not");
    MatcherParams p;
    p.mutual = true;
    const MatchSet mu = MatchRanked(q, t, x, y, {}, {}, p);
    FCHECK(mu.kept.size() == 2 && mu.kept[0].query == 0 && mu.kept[1].query == 3 && mu.counters.removed_mutual == 2 &&
               mu.raw[1].fate == MatchFate::kRemovedMutual && CountersAddUp(mu.counters),
           "mutual: keeps q0 -> t0 and q3 -> t1, removes 2 (counted as mutual)");
    // asymmetric: q0 -> t0, but t0's nearest query is q1 (closer)
    const MatchSet as = MatchRanked({Bits(0, 12), Bits(0, 10)}, {Bits(0, 10), Bits(200, 5)}, {0, 50}, {0, 50}, {}, {}, p);
    FCHECK(as.kept.size() == 1 && as.kept[0].query == 1 && !as.raw[0].mutual,
           "asymmetric: q0 -> t0 while t0 -> q1: only q1 is kept");
  }

  std::printf("\n-- location-aware mutual matching --\n");
  {
    // q0 at (10, 10) and q1, the same corner on another level at (11, 10.5), both match t0; t0's reverse
    // nearest neighbour is q1 (closer descriptor). q2 at (200, 200) matches t0 too but is elsewhere.
    const std::vector<BriefDescriptor> q = {Bits(0, 12), Bits(0, 11), Bits(0, 13)}, t = {Bits(0, 10), Bits(100, 5)};
    const std::vector<float> tx = {50, 300}, ty = {50, 300}, qx = {10, 11, 200}, qy = {10, 10.5f, 200};
    MatcherParams strict;
    strict.mutual = true;
    const MatchSet s = MatchRanked(q, t, tx, ty, {}, {}, strict, {}, {}, qx, qy);
    FCHECK(s.raw[0].reverse_query == 1 && !s.raw[0].mutual && s.raw[0].mutual_location && s.kept.size() == 1 &&
               s.kept[0].query == 1,
           "strict: q0 loses t0 to q1 (the same corner on another level) and is removed; its location-aware flag is set");
    MatcherParams loc = strict;
    loc.mutual_location_aware = true;
    const MatchSet l = MatchRanked(q, t, tx, ty, {}, {}, loc, {}, {}, qx, qy);
    FCHECK(l.kept.size() == 2 && l.kept[0].query == 0 && l.kept[1].query == 1 && l.raw[2].fate == MatchFate::kRemovedMutual,
           "location-aware: q0 kept (reverse nearest neighbour 1.1 px away), q2 (elsewhere) still removed");
    MatcherParams only = MatcherParams();
    only.mutual_location_aware = true;  // implies mutual
    FCHECK(MatchRanked(q, t, tx, ty, {}, {}, only, {}, {}, qx, qy).kept.size() == 2,
           "mutual_location_aware alone implies mutual filtering");
    const MatchSet def = MatchRanked(q, t, tx, ty, {}, {}, MatcherParams(), {}, {}, qx, qy);
    FCHECK(def.kept.size() == 3 && SameMatches(MatchBruteForce(q, t), def.kept),
           "query positions given, options off: still identical to the original matcher");
    bool t1 = false, t2 = false;
    try {
      MatchRanked(q, t, tx, ty, {}, {}, loc);
    } catch (const std::invalid_argument &) {
      t1 = true;
    }
    try {
      MatchRanked(q, t, tx, ty, {}, {}, MatcherParams(), {}, {}, {1, 2}, {1, 2});
    } catch (const std::invalid_argument &) {
      t2 = true;
    }
    FCHECK(t1 && t2, "location-aware mutual without query positions, or positions of the wrong length: rejected");
  }

  std::printf("\n-- deterministic tie-breaking --\n");
  {
    // q0 is equidistant (1) from t0 and t2: the lowest train index wins, as MatchBruteForce
    const std::vector<BriefDescriptor> t = {Bits(0, 9), Bits(50, 20), Bits(0, 11)};
    const MatchSet s = MatchRanked({Bits(0, 10)}, t, {0, 40, 80}, {0, 40, 80}, {}, {});
    FCHECK(s.raw[0].train == 0 && s.raw[0].distance == 1 && MatchBruteForce({Bits(0, 10)}, t)[0].train == 0,
           "forward tie: lowest train index (0)");
    FCHECK(s.raw[0].has_second && s.raw[0].second_train == 2 && s.raw[0].second_distance == 1 && s.raw[0].ratio() == 1.0,
           "the tied train at another location is the second best: ratio exactly 1");
    // reverse tie: q0 and q1 identical -> both nearest to t0; t0's reverse NN is q0 (lowest index)
    MatcherParams p;
    p.mutual = true;
    const MatchSet r = MatchRanked({Bits(0, 10), Bits(0, 10)}, {Bits(0, 10)}, {0}, {0}, {}, {}, p);
    FCHECK(r.kept.size() == 1 && r.kept[0].query == 0, "reverse tie: the lowest query index is the mutual partner");
  }

  std::printf("\n-- location-aware ratio: cross-level duplicate vs genuine ambiguity --\n");
  {
    // t0 = the corner at level 0, t1 = the same corner at level 1 (1.1 px away, nearly the same descriptor),
    // t2 = a different place, much worse
    const std::vector<BriefDescriptor> q = {Bits(0, 20)};
    const std::vector<BriefDescriptor> t = {Bits(0, 21), Bits(0, 19), Bits(0, 40)};  // copy: distance 1 too
    const std::vector<float> x = {100.0f, 101.0f, 300.0f}, y = {100.0f, 100.5f, 300.0f};
    MatcherParams p;
    p.ratio = 0.8;
    const MatchSet loc = MatchRanked(q, t, x, y, {0}, {0, 1, 0}, p);
    FCHECK(loc.raw[0].second_train == 2 && loc.raw[0].second_distance == 20 && loc.kept.size() == 1,
           "location-aware (3 px): the level-1 copy is skipped, second = the other place (20): ratio %.3f, kept",
           loc.raw[0].ratio());
    MatcherParams naive = p;
    naive.same_location_radius_px = 0.0f;
    const MatchSet nv = MatchRanked(q, t, x, y, {0}, {0, 1, 0}, naive);
    FCHECK(nv.raw[0].second_train == 1 && nv.kept.empty() && nv.counters.removed_ratio == 1,
           "naive (radius 0): the copy of the same corner is the second best (ratio %.3f) and the match is lost",
           nv.raw[0].ratio());
    // genuine ambiguity: a different place almost as close
    const MatchSet amb = MatchRanked(q, {Bits(0, 21), Bits(0, 22), Bits(0, 19)}, {100, 101, 250}, {100, 100.5f, 40}, {}, {}, p);
    FCHECK(amb.raw[0].train == 0 && amb.raw[0].second_train == 2 && amb.raw[0].ratio() == 1.0 && amb.kept.empty() &&
               amb.raw[0].fate == MatchFate::kRemovedRatio,
           "a different place exactly as close (1 vs 1): ratio 1, removed as ambiguous");
    const MatchSet clear = MatchRanked(q, {Bits(0, 21), Bits(0, 22), Bits(0, 22)}, {100, 101, 250}, {100, 100.5f, 40}, {}, {}, p);
    FCHECK(clear.raw[0].second_train == 2 && clear.raw[0].ratio() == 0.5 && clear.kept.size() == 1,
           "a different place at 2 vs best 1: ratio 0.5 < 0.8, kept");
    const MatchSet amb2 = MatchRanked({Bits(0, 20)}, {Bits(0, 25), Bits(0, 26)}, {100, 250}, {100, 40}, {}, {}, p);
    FCHECK(amb2.raw[0].ratio() == 5.0 / 6.0 && amb2.kept.empty(), "best 5, other place 6: ratio 0.833 >= 0.8: removed");
  }

  std::printf("\n-- zero distance, missing second best --\n");
  {
    MatcherParams p;
    p.ratio = 0.8;
    const MatchSet z = MatchRanked({Bits(0, 10)}, {Bits(0, 10), Bits(0, 14)}, {0, 50}, {0, 50}, {}, {}, p);
    FCHECK(z.raw[0].distance == 0 && z.raw[0].second_distance == 4 && z.raw[0].ratio() == 0.0 && z.kept.size() == 1,
           "best 0, second 4: ratio 0, kept");
    const MatchSet zz = MatchRanked({Bits(0, 10)}, {Bits(0, 10), Bits(0, 10)}, {0, 50}, {0, 50}, {}, {}, p);
    FCHECK(zz.raw[0].distance == 0 && zz.raw[0].second_distance == 0 && zz.raw[0].ratio() == 1.0 && zz.kept.empty(),
           "best 0 and second 0 at another place: ratio 1 (fully ambiguous), removed");
    const MatchSet one = MatchRanked({Bits(0, 10)}, {Bits(0, 12)}, {5}, {5}, {}, {}, p);
    FCHECK(!one.raw[0].has_second && one.raw[0].second_distance == -1 && one.raw[0].ratio() == 0.0 && one.kept.size() == 1,
           "a single train: no second best, ratio 0, kept by the ratio test");
    const MatchSet near = MatchRanked({Bits(0, 10)}, {Bits(0, 12), Bits(0, 11)}, {5, 6}, {5, 7}, {}, {}, p);
    FCHECK(!near.raw[0].has_second && near.kept.size() == 1 && near.kept[0].train == 1,
           "every other train within 3 px of the best: no second best, kept");
  }

  std::printf("\n-- invalid descriptors --\n");
  {
    // t0 would be a perfect match but is invalid; q1 is invalid
    const std::vector<BriefDescriptor> q = {Bits(0, 10), Bits(0, 10)}, t = {Bits(0, 10), Bits(0, 13), Bits(60, 10)};
    const MatchSet s = MatchRanked(q, t, {0, 50, 90}, {0, 50, 90}, {}, {}, MatcherParams(), {true, false}, {false, true, true});
    FCHECK(s.counters.invalid_query == 1 && s.counters.invalid_train == 1 && s.raw.size() == 1 && s.raw[0].query == 0 &&
               s.raw[0].train == 1 && s.raw[0].distance == 3,
           "invalid query gets no match; invalid train is never a candidate (best = t1 at 3, not t0 at 0)");
    FCHECK(s.raw[0].has_second && s.raw[0].second_train == 2, "the invalid train is not the second best either");
    const MatchSet none = MatchRanked(q, t, {0, 50, 90}, {0, 50, 90}, {}, {}, MatcherParams(), {}, {false, false, false});
    FCHECK(none.raw.empty() && none.counters.invalid_train == 3, "all trains invalid: no matches");
    bool threw = false;
    try {
      MatchRanked(q, t, {0}, {0}, {}, {});
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    FCHECK(threw, "positions not one per train: std::invalid_argument");
  }

  std::printf("\n-- level filter and removal counters --\n");
  {
    // q0: kept; q1: distance 45 > 30; q2 -> t0 not mutual; q3: 5 vs 6 elsewhere (ratio 0.83); q4: level jump 3
    const std::vector<BriefDescriptor> q = {Bits(0, 30), Bits(200, 50), Bits(0, 31), Bits(100, 20), Bits(150, 10)};
    const std::vector<BriefDescriptor> t = {Bits(0, 30), Bits(100, 25), Bits(100, 26), Bits(150, 10), Bits(200, 5)};
    const std::vector<float> x = {0, 40, 80, 120, 160}, y = {0, 40, 80, 120, 160};
    MatcherParams p;
    p.mutual = true, p.ratio = 0.8, p.max_level_diff = 1;
    const MatchSet s = MatchRanked(q, t, x, y, {0, 0, 0, 0, 0}, {0, 0, 0, 3, 0}, p);
    const MatchCounters &c = s.counters;
    FCHECK(c.raw == 5 && c.kept == 1 && c.removed_distance == 1 && c.removed_mutual == 1 && c.removed_ratio == 1 &&
               c.removed_level == 1 && CountersAddUp(c),
           "one removal per filter, first failing filter counts (raw 5 = kept 1 + distance 1 + mutual 1 + ratio 1 + level 1)");
    FCHECK(s.raw[1].fate == MatchFate::kRemovedDistance && s.raw[2].fate == MatchFate::kRemovedMutual &&
               s.raw[3].fate == MatchFate::kRemovedRatio && s.raw[4].fate == MatchFate::kRemovedLevel &&
               std::string(MatchFateName(s.raw[3].fate)) == "ratio",
           "per-match fate recorded");
  }

  std::printf("\n-- quality order --\n");
  {
    std::mt19937 rng(3);
    std::vector<BriefDescriptor> q(120), t(160);
    for (auto *v : {&q, &t})
      for (auto &d : *v)
        for (int k = 0; k < 40; ++k) d.set(int(rng() % 96));
    std::vector<float> x(t.size()), y(t.size());
    for (size_t k = 0; k < t.size(); ++k) x[k] = float(rng() % 640), y[k] = float(rng() % 480);
    MatcherParams p;
    p.floor_distance = 60;
    const MatchSet byq = MatchRanked(q, t, x, y, {}, {}, p);
    p.rank = true;
    const MatchSet byr = MatchRanked(q, t, x, y, {}, {}, p), again = MatchRanked(q, t, x, y, {}, {}, p);
    bool sorted = true, contiguous = true, same = byr.kept.size() == again.kept.size();
    for (size_t k = 0; k < byr.kept.size(); ++k) {
      if (k && !QualityBefore(byr.kept[k - 1], byr.kept[k])) sorted = false;
      if (byr.kept[k].rank != int(k)) contiguous = false;
      if (same && (byr.kept[k].query != again.kept[k].query || byr.kept[k].rank != again.kept[k].rank)) same = false;
    }
    FCHECK(sorted && contiguous && byr.kept.size() == byq.kept.size(),
           "rank = true: %zu kept matches strictly ordered by (ratio, distance, query), ranks 0..n-1", byr.kept.size());
    bool ranks_equal = true;
    for (const RankedMatch &m : byq.kept)
      for (const RankedMatch &n : byr.kept)
        if (m.query == n.query && m.rank != n.rank) ranks_equal = false;
    FCHECK(ranks_equal, "rank = false keeps query order with the same rank values");
    FCHECK(same, "repeated call: identical order and ranks");
    RankedMatch a, b;
    a.query = 5, b.query = 2, a.distance = b.distance = 4, a.has_second = b.has_second = true;
    a.second_distance = 8, b.second_distance = 8;
    FCHECK(QualityBefore(b, a) && !QualityBefore(a, b), "equal ratio and distance: the lower query index first");
    a.second_distance = 10;  // 0.4 < 0.5
    FCHECK(QualityBefore(a, b), "lower ratio first (4/10 before 4/8)");
    RankedMatch c = b;
    c.has_second = false, c.distance = 30;
    FCHECK(QualityBefore(c, a), "no second best (ratio 0) ranks before any ratio > 0");
  }
  return ftest::Finish("features_matcher_ranked_test");
}
