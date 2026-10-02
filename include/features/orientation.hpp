#pragma once

// Keypoint orientation by the intensity centroid (Rosin 1999, as used by
// oriented FAST): over a circular patch of radius r around the keypoint,
//   m10 = sum x * I(x, y),  m01 = sum y * I(x, y)   (x, y relative to the keypoint)
//   angle = atan2(m01, m10)   [radians, image axes: x right, y down]

#include <vector>

#include "features/types.hpp"

namespace features {

constexpr int kOrientationRadius = 15;

// Requires the whole patch inside the image (DetectFast's border guarantees it).
float IntensityCentroidAngle(const GrayImage &img, int x, int y, int radius = kOrientationRadius);

// Sets keypoint.angle for every keypoint.
void AssignOrientations(const GrayImage &img, std::vector<Keypoint> &keypoints, int radius = kOrientationRadius);

}  // namespace features
