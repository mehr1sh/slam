#pragma once

// FAST corner detection (Rosten & Drummond): segment test on the 16-pixel
// Bresenham circle of radius 3, corner score, 3x3 non-maximum suppression,
// descriptor-safe border margin, strongest-N selection, optional per-cell
// threshold fallback for a more even spatial distribution. Deterministic.

#include <vector>

#include "features/types.hpp"

namespace features {

struct FastParams {
  int threshold = 30;       // |I(p) - I(c)| must exceed this for p to count as brighter/darker
  int arc_length = 9;       // contiguous circle pixels required (FAST-9)
  int border = 16;          // no keypoint closer than this to the image edge (>= descriptor/orientation radius + 1)
  int max_keypoints = 500;  // keep the strongest N after non-maximum suppression (<= 0: keep all)
  // Spatial distribution (off by default): the image is split into cell_size x
  // cell_size cells; a cell with no corner at `threshold` is searched again at
  // `min_threshold` (0 < min_threshold < threshold). Low-contrast regions then
  // contribute corners instead of the strongest edges taking every keypoint.
  int cell_size = 0;        // <= 0: off
  int min_threshold = 0;    // <= 0 or >= threshold: off
};

// The 16 circle offsets (dx, dy), clockwise from 12 o'clock.
extern const int kFastCircle[16][2];

// Segment test at (x, y); requires 3 <= x < width-3, 3 <= y < height-3.
bool IsFastCorner(const GrayImage &img, int x, int y, int threshold, int arc_length);

// Corner score: max(sum over brighter pixels of (I(p) - I(c) - t),
//                   sum over darker  pixels of (I(c) - I(p) - t))
// over all 16 circle pixels (Rosten & Drummond 2006, score "V").
int FastScore(const GrayImage &img, int x, int y, int threshold);

// Corners -> scores -> [per-cell fallback at min_threshold for cells without a
// corner] -> 3x3 non-maximum suppression (ties broken by raster order: the
// earlier pixel wins) -> sort by (score desc, y asc, x asc) -> first
// max_keypoints. Fallback corners are scored at min_threshold. Returned
// keypoints have angle = 0.
std::vector<Keypoint> DetectFast(const GrayImage &img, const FastParams &params = FastParams());

}  // namespace features
