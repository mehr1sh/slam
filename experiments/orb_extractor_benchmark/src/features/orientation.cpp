#include "features/orientation.hpp"

#include <cmath>
#include <stdexcept>

namespace features {

float IntensityCentroidAngle(const GrayImage &img, int x, int y, int radius) {
  if (x - radius < 0 || y - radius < 0 || x + radius >= img.width || y + radius >= img.height)
    throw std::out_of_range("IntensityCentroidAngle: patch leaves the image");
  long long m10 = 0, m01 = 0;
  const int r2 = radius * radius;
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx * dx + dy * dy > r2) continue;  // circular patch
      const int v = img.at(x + dx, y + dy);
      m10 += dx * v;
      m01 += dy * v;
    }
  return float(std::atan2(double(m01), double(m10)));
}

void AssignOrientations(const GrayImage &img, std::vector<Keypoint> &keypoints, int radius) {
  for (Keypoint &k : keypoints) k.angle = IntensityCentroidAngle(img, int(k.x), int(k.y), radius);
}

}  // namespace features
