#pragma once

// Helpers for the features/* unit tests: a CHECK macro that counts failures,
// and deterministic synthetic images. Test-only; no third-party vision code.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "features/types.hpp"

namespace ftest {

inline int &Failures() {
  static int n = 0;
  return n;
}

#define FCHECK(cond, ...)                                    \
  do {                                                       \
    const bool ok_ = (cond);                                 \
    std::printf("  [%s] ", ok_ ? "PASS" : "FAIL");           \
    std::printf(__VA_ARGS__);                                \
    std::printf("\n");                                       \
    if (!ok_) ++ftest::Failures();                           \
  } while (0)

inline int Finish(const char *name) {
  std::printf("\n%s: %s (%d failure%s)\n", name, ftest::Failures() ? "FAIL" : "PASS", ftest::Failures(),
              ftest::Failures() == 1 ? "" : "s");
  return ftest::Failures() ? 1 : 0;
}

// Uniform noise, fixed seed.
inline features::GrayImage Noise(int w, int h, uint32_t seed) {
  std::mt19937 rng(seed);
  features::GrayImage img(w, h);
  for (auto &p : img.pixels) p = uint8_t(rng() & 0xFF);
  return img;
}

// Smooth random texture (sum of a few oriented sinusoids + blobs), fixed seed:
// gives FAST corners and BRIEF structure without pure pixel noise.
inline features::GrayImage Texture(int w, int h, uint32_t seed) {
  std::mt19937 rng(seed);
  auto U = [&] { return (double(rng()) + 0.5) / 4294967296.0; };
  struct Wave { double kx, ky, ph, a; };
  Wave waves[6];
  for (auto &wv : waves) {
    const double ang = U() * 2 * M_PI, freq = 0.08 + 0.25 * U();
    wv = {freq * std::cos(ang), freq * std::sin(ang), U() * 2 * M_PI, 15 + 20 * U()};
  }
  features::GrayImage img(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double v = 128;
      for (const auto &wv : waves) v += wv.a * std::sin(wv.kx * x + wv.ky * y + wv.ph);
      img.at(x, y) = uint8_t(std::fmax(0, std::fmin(255, std::lround(v))));
    }
  // sharp-edged rectangles create proper corners
  for (int r = 0; r < 40; ++r) {
    const int x0 = int(U() * (w - 12)), y0 = int(U() * (h - 12)), sw = 4 + int(U() * 10), sh = 4 + int(U() * 10);
    const uint8_t val = uint8_t(U() * 255);
    for (int y = y0; y < std::min(h, y0 + sh); ++y)
      for (int x = x0; x < std::min(w, x0 + sw); ++x) img.at(x, y) = val;
  }
  return img;
}

// Exact 90-degree counter-clockwise rotation in image coordinates (x right,
// y down): out(x', y') = in(W-1-y', x'), i.e. a point (x, y) moves to
// (y, W-1-x). out has width H and height W.
inline features::GrayImage Rotate90(const features::GrayImage &in) {
  features::GrayImage out(in.height, in.width);
  for (int y = 0; y < out.height; ++y)
    for (int x = 0; x < out.width; ++x) out.at(x, y) = in.at(in.width - 1 - y, x);
  return out;
}

}  // namespace ftest
