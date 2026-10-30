#include "features/types.hpp"

#include <stdexcept>

namespace features {

namespace {

// b, r: byte offsets of the blue and red channels within a pixel.
GrayImage ToGray(const uint8_t *data, int width, int height, size_t row_step, int channels, int b, int r) {
  if (channels != 1 && channels != 3 && channels != 4)
    throw std::invalid_argument("grayscale conversion: channels must be 1, 3 or 4");
  GrayImage g(width, height);
  for (int y = 0; y < height; ++y) {
    const uint8_t *row = data + size_t(y) * row_step;
    for (int x = 0; x < width; ++x) {
      const uint8_t *p = row + size_t(x) * channels;
      g.at(x, y) = channels == 1 ? p[0] : uint8_t((1868 * p[b] + 9617 * p[1] + 4899 * p[r] + 8192) >> 14);
    }
  }
  return g;
}

}  // namespace

GrayImage GrayFromInterleaved(const uint8_t *data, int width, int height, size_t row_step, int channels) {
  return ToGray(data, width, height, row_step, channels, 0, 2);
}

GrayImage GrayFromRGB(const uint8_t *data, int width, int height, size_t row_step, int channels) {
  return ToGray(data, width, height, row_step, channels, 2, 0);
}

}  // namespace features
