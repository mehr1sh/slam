#pragma once

#include <Eigen/Core>

// Pinhole intrinsics (no lens distortion -- the synthetic renders have none).
struct Intrinsics {
  double fx, fy, cx, cy;
};

// Projects an object-frame 3D point X into the image of a camera with pose
// (R, t), where X_camera = R * X + t:
//   u = fx * Xc / Zc + cx,   v = fy * Yc / Zc + cy.
// If `depth` is non-null it receives Zc (the point is in front of the camera
// only if Zc > 0).
Eigen::Vector2d projectPoint(const Eigen::Vector3d &X, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                             const Intrinsics &K, double *depth = nullptr);
