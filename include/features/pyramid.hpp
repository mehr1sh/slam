#pragma once

// Image pyramid for multiscale feature detection (project-owned).
//
// Level 0 is the input image. Level l (l >= 1) is resampled from level l-1
// to size (round(W0 / s^l), round(H0 / s^l)), s = scale_factor, by bilinear
// interpolation with pixel-centre alignment:
//     dst(x, y) = src((x + 0.5) * rx - 0.5, (y + 0.5) * ry - 0.5)
// with rx = src.width / dst.width, ry = src.height / dst.height (coordinates
// clamped to the image). Resampling from the previous level (rather than from
// level 0) smooths progressively, which limits aliasing at the coarse levels.
//
// Coordinate mapping: because every step is centre-aligned, the composition
// is centre-aligned too, with the product of the ratios. A level-l pixel
// coordinate x maps to the level-0 coordinate
//     x0 = (x + 0.5) * scale_x[l] - 0.5,   scale_x[l] = W0 / W_l   (same for y)

#include <vector>

#include "features/types.hpp"

namespace features {

struct PyramidParams {
  int levels = 8;            // number of levels (1 = the input image only)
  double scale_factor = 1.2; // size ratio between consecutive levels
};

struct ImagePyramid {
  std::vector<GrayImage> levels;
  std::vector<double> scale_x, scale_y;  // level-0 size / level size, per axis
};

// Bilinear, pixel-centre-aligned resampling to (width, height).
GrayImage ResizeBilinear(const GrayImage &src, int width, int height);

ImagePyramid BuildPyramid(const GrayImage &base, const PyramidParams &params = PyramidParams());

// Level-l pixel coordinate -> level-0 (original image) coordinate.
inline float LevelToBaseX(const ImagePyramid &p, int level, float x) { return float((x + 0.5) * p.scale_x[level] - 0.5); }
inline float LevelToBaseY(const ImagePyramid &p, int level, float y) { return float((y + 0.5) * p.scale_y[level] - 0.5); }

}  // namespace features
