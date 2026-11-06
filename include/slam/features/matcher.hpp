#pragma once

// Brute-force Hamming matching and the repository's existing match filter.

#include <string>
#include <vector>

#include "features/types.hpp"

namespace features {

// For every query descriptor, its nearest train descriptor by Hamming
// distance (ties: the lowest train index). One match per query descriptor,
// in query order; empty if either list is empty.
std::vector<Match> MatchBruteForce(const std::vector<BriefDescriptor> &query,
                                   const std::vector<BriefDescriptor> &train);

// The repository's distance filter (unchanged from the original
// find_feature_matches()): with d_min the smallest distance among `matches`,
// keep matches with distance <= max(2 * d_min, floor_distance).
// `d_min_out` / `d_max_out` (optional) receive the smallest/largest distance.
std::vector<Match> FilterMatchesByDistance(const std::vector<Match> &matches,
                                           int floor_distance = 30,
                                           int *d_min_out = nullptr,
                                           int *d_max_out = nullptr);

// Bidirectional nearest/second-nearest ratio tests plus mutual consistency.
std::vector<Match> MatchMutualRatio(const std::vector<BriefDescriptor> &a,
                                    const std::vector<BriefDescriptor> &b,
                                    double ratio = .87);

// ---------------------------------------------------------------------------
// Ranked matching (Stage 3). A separate entry point: MatchBruteForce and
// FilterMatchesByDistance above are unchanged and are what the default
// pipeline uses. With every option at its default, MatchRanked reproduces
// FilterMatchesByDistance(MatchBruteForce(q, t), floor_distance) exactly --
// the same matches, distances and order (tests: features_matcher_ranked_test).
//
// Semantics, per valid query q (invalid descriptors never take part):
//   best       nearest valid train by Hamming distance; ties -> lowest train index
//              (as MatchBruteForce)
//   second     the smallest distance among valid trains whose position is MORE than
//              same_location_radius_px from the best train's position, so that another
//              pyramid level's detection of the same corner is never the second best;
//              none -> has_second = false
//   ratio      distance / second_distance as an exact fraction: no second -> 0
//              (unambiguous); second_distance 0 (then distance is 0 too) -> 1
//   mutual     q is the reverse nearest neighbour of its best train among the valid
//              queries (ties -> lowest query index)
//   mutual_location  (needs query positions) the reverse nearest neighbour of q's best train
//              lies within same_location_radius_px of q -- q itself, or another pyramid
//              level's detection of the same corner in the query image. Implies `mutual`.
// Filters, applied in this order; a removed match is counted once, by the first
// filter it fails (so raw = kept + all removed counts):
//   distance   distance <= max(2 * d_min, floor_distance), d_min over the raw matches
//   mutual     (if enabled) mutual, or mutual_location if mutual_location_aware
//   ratio      (if ratio > 0) no second, or distance < ratio * second_distance
//   level      (if max_level_diff >= 0) |level_query - level_train| <= max_level_diff
// Quality order (always computed; `rank` field): ratio ascending, then distance
// ascending, then query index ascending -- a total, deterministic order compared in
// integer arithmetic. `kept` is in query order unless `rank` is set.

struct MatcherParams {
  int floor_distance = 30;               // the existing filter max(2 * d_min, floor_distance)
  bool mutual = false;
  bool mutual_location_aware = false;    // with `mutual`: accept a reverse nearest neighbour at the same location
  double ratio = 0.0;                    // 0 = off; e.g. 0.8
  float same_location_radius_px = 3.0f;  // the second best ignores trains this close to the best
  int max_level_diff = -1;               // -1 = off
  bool rank = false;                     // `kept` in quality order instead of query order
};

enum class MatchFate { kKept, kRemovedDistance, kRemovedMutual, kRemovedRatio, kRemovedLevel };
const char *MatchFateName(MatchFate f);  // "kept", "distance", "mutual", "ratio", "level"

struct RankedMatch {
  int query = -1, train = -1;
  int distance = 0;
  bool has_second = false;
  int second_distance = -1;  // -1 when has_second is false
  int second_train = -1;     // the train giving second_distance
  bool mutual = false;           // strict
  bool mutual_location = false;  // location-aware (false if query positions were not given)
  int reverse_query = -1;        // the reverse nearest neighbour of `train`
  int level_query = 0, level_train = 0;
  MatchFate fate = MatchFate::kKept;
  int rank = -1;             // position in the quality order among the kept matches (0 = best); -1 if removed
  double ratio() const;      // distance / second_distance with the conventions above
  Match match() const { return Match{query, train, distance}; }
};

struct MatchCounters {
  int queries = 0, trains = 0;
  int invalid_query = 0, invalid_train = 0;  // descriptors excluded from matching
  int raw = 0;                               // valid queries that received a nearest neighbour
  int removed_distance = 0, removed_mutual = 0, removed_ratio = 0, removed_level = 0;
  int kept = 0;
  int d_min = 10000, d_max = 0;              // over the raw matches (as FilterMatchesByDistance)
};

struct MatchSet {
  std::vector<RankedMatch> raw;   // one per valid query (if any valid train), in query order, with its fate
  std::vector<RankedMatch> kept;  // the kept matches: query order, or quality order if params.rank
  MatchCounters counters;
};

// Positions are original-image coordinates of the train keypoints (needed for the
// location-aware second best); levels may be empty (all 0). `query_valid` /
// `train_valid` may be empty (all valid). Query positions (may be empty) are needed only for
// mutual_location; params.mutual_location_aware without them is an error.
MatchSet MatchRanked(const std::vector<BriefDescriptor> &query, const std::vector<BriefDescriptor> &train,
                     const std::vector<float> &train_x, const std::vector<float> &train_y,
                     const std::vector<int> &query_level, const std::vector<int> &train_level,
                     const MatcherParams &params = MatcherParams(),
                     const std::vector<bool> &query_valid = {}, const std::vector<bool> &train_valid = {},
                     const std::vector<float> &query_x = {}, const std::vector<float> &query_y = {});

// The quality order: true if a ranks strictly before b.
bool QualityBefore(const RankedMatch &a, const RankedMatch &b);

} // namespace features
