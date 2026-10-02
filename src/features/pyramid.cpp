#include "features/pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace features {

GrayImage ResizeBilinear(const GrayImage &src, int width, int height) {
  if (width <= 0 || height <= 0 || src.width <= 0 || src.height <= 0)
    throw std::invalid_argument("ResizeBilinear: empty image");
  if (width == src.width && height == src.height) return src;
  const double rx = double(src.width) / width, ry = double(src.height) / height;
  GrayImage dst(width, height);
  for (int y = 0; y < height; ++y) {
    const double fy = std::clamp((y + 0.5) * ry - 0.5, 0.0, double(src.height - 1));
    const int y0 = std::min(int(fy), src.height - 1), y1 = std::min(y0 + 1, src.height - 1);
    const double wy = fy - y0;
    for (int x = 0; x < width; ++x) {
      const double fx = std::clamp((x + 0.5) * rx - 0.5, 0.0, double(src.width - 1));
      const int x0 = std::min(int(fx), src.width - 1), x1 = std::min(x0 + 1, src.width - 1);
      const double wx = fx - x0;
      const double v = (1 - wy) * ((1 - wx) * src.at(x0, y0) + wx * src.at(x1, y0)) +
                       wy * ((1 - wx) * src.at(x0, y1) + wx * src.at(x1, y1));
      dst.at(x, y) = uint8_t(std::clamp(std::lround(v), 0L, 255L));
    }
  }
  return dst;
}

ImagePyramid BuildPyramid(const GrayImage &base, const PyramidParams &params) {
  if (params.levels < 1) throw std::invalid_argument("BuildPyramid: levels must be >= 1");
  ImagePyramid p;
  p.levels.push_back(base);
  p.scale_x.push_back(1.0);
  p.scale_y.push_back(1.0);
  for (int l = 1; l < params.levels; ++l) {
    const double s = std::pow(params.scale_factor, l);
    const int w = std::max(1, int(std::lround(base.width / s))), h = std::max(1, int(std::lround(base.height / s)));
    p.levels.push_back(ResizeBilinear(p.levels.back(), w, h));
    p.scale_x.push_back(double(base.width) / w);
    p.scale_y.push_back(double(base.height) / h);
  }
  return p;
}

}  // namespace features
