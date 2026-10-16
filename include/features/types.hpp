#pragma once

// Project-owned types of the feature pipeline (features/fast, orientation,
// brief, matcher). No third-party vision types.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace features {

// 8-bit single-channel image, row-major, tightly packed.
struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;  // width * height values

  GrayImage() = default;
  GrayImage(int w, int h, uint8_t fill = 0) : width(w), height(h), pixels(size_t(w) * h, fill) {}
  uint8_t at(int x, int y) const { return pixels[size_t(y) * width + x]; }
  uint8_t &at(int x, int y) { return pixels[size_t(y) * width + x]; }
  bool contains(int x, int y) const { return x >= 0 && y >= 0 && x < width && y < height; }
};

// Converts an interleaved 8-bit image to grey. `channels` = 1 (copied),
// 3 (B, G, R order) or 4 (B, G, R, A; alpha ignored); `row_step` is the
// number of bytes per row. ITU-R BT.601 luma weights in 14-bit fixed point:
//   Y = (1868 B + 9617 G + 4899 R + 8192) >> 14
GrayImage GrayFromInterleaved(const uint8_t *data, int width, int height, size_t row_step, int channels);

// Same conversion for R, G, B (, A) channel order, e.g. pixels decoded from PNG.
GrayImage GrayFromRGB(const uint8_t *data, int width, int height, size_t row_step, int channels);

struct Keypoint {
  float x = 0, y = 0;   // pixel coordinates (column, row)
  int score = 0;        // FAST corner score (larger = stronger)
  float angle = 0;      // orientation in radians, atan2(m01, m10); 0 until assigned
};

// 256-bit binary descriptor; bit i is the result of binary test i.
struct BriefDescriptor {
  std::array<uint64_t, 4> bits{};
  bool bit(int i) const { return (bits[i >> 6] >> (i & 63)) & 1u; }
  void set(int i) { bits[i >> 6] |= uint64_t(1) << (i & 63); }
};

struct Match {
  int query = -1;     // index into the first keypoint/descriptor list
  int train = -1;     // index into the second list
  int distance = 0;   // Hamming distance (0..256)
};

}  // namespace features
