#pragma once

// Brute-force Hamming matching and the repository's existing match filter.

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
std::vector<Match> FilterMatchesByDistance(const std::vector<Match> &matches, int floor_distance = 30,
                                           int *d_min_out = nullptr, int *d_max_out = nullptr);

}  // namespace features
