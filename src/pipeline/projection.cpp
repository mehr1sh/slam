#include "projection.hpp"

Eigen::Vector2d projectPoint(const Eigen::Vector3d &X, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                             const Intrinsics &K, double *depth) {
  const Eigen::Vector3d Xc = R * X + t;  // object frame -> camera frame
  if (depth) *depth = Xc.z();
  return Eigen::Vector2d(K.fx * Xc.x() / Xc.z() + K.cx,   // perspective division + intrinsics
                         K.fy * Xc.y() / Xc.z() + K.cy);
}
