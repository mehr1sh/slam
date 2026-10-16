// Self-checking test of features/orientation (intensity centroid).

#include <cmath>

#include "features/fast.hpp"
#include "features/orientation.hpp"
#include "feature_test_utils.hpp"

using namespace features;

namespace {

double AngleDiffDeg(double a, double b) {  // smallest signed difference, degrees
  double d = std::fmod((a - b) * 180.0 / M_PI, 360.0);
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return d;
}

}  // namespace

int main() {
  std::printf("=== features/orientation ===\n");

  // A bright blob placed at a known direction from the keypoint: the
  // intensity centroid must point at it.
  double worst = 0;
  for (int deg = -180; deg < 180; deg += 15) {
    GrayImage img(64, 64, 20);
    const double th = deg * M_PI / 180.0;
    const double bx = 32 + 9 * std::cos(th), by = 32 + 9 * std::sin(th);
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        if ((x - bx) * (x - bx) + (y - by) * (y - by) <= 16) img.at(x, y) = 230;
    worst = std::max(worst, std::abs(AngleDiffDeg(IntensityCentroidAngle(img, 32, 32), th)));
  }
  FCHECK(worst < 3.0, "blob at 24 known directions: max angle error %.2f deg", worst);

  // Image-axis convention: a blob straight below (+y) gives +90 degrees.
  {
    GrayImage img(64, 64, 0);
    for (int y = 40; y < 44; ++y)
      for (int x = 30; x < 35; ++x) img.at(x, y) = 255;
    FCHECK(std::abs(AngleDiffDeg(IntensityCentroidAngle(img, 32, 32), M_PI / 2)) < 1e-3,
           "blob below the keypoint -> +90 deg (x right, y down)");
  }

  // Exact 90-degree image rotation: a point (x, y) moves to (y, W-1-x), so
  // every offset (dx, dy) becomes (dy, -dx) and the angle drops by 90 degrees.
  {
    const GrayImage img = ftest::Texture(160, 160, 11);
    const GrayImage rot = ftest::Rotate90(img);
    std::vector<Keypoint> kps = DetectFast(img);
    AssignOrientations(img, kps);
    double max_err = 0;
    int n = 0;
    for (const Keypoint &k : kps) {
      const int rx = int(k.y), ry = img.width - 1 - int(k.x);
      if (rx < 16 || ry < 16 || rx >= rot.width - 16 || ry >= rot.height - 16) continue;
      max_err = std::max(max_err, std::abs(AngleDiffDeg(IntensityCentroidAngle(rot, rx, ry), k.angle - M_PI / 2)));
      ++n;
    }
    FCHECK(n > 50 && max_err < 1e-3, "%d keypoints: rotated image gives angle - 90 deg (max error %.2e deg)", n,
           max_err);
  }

  // Uniform patch: m10 = m01 = 0 -> atan2(0, 0) = 0, finite.
  {
    GrayImage flat(40, 40, 128);
    const float a = IntensityCentroidAngle(flat, 20, 20);
    FCHECK(std::isfinite(a) && a == 0.0f, "uniform patch -> angle 0 (finite)");
  }
  return ftest::Finish("features_orientation_test");
}
