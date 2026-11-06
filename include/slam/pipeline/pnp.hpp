#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "projection.hpp"

struct PnPResult {
  bool ok = false;
  std::string reason;              // why the solve failed (if !ok)
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  // diagnostics of the linear solve
  Eigen::Vector3d sv_left_block;   // singular values of the 3x3 block (lambda*R, ideally all equal)
  Eigen::Vector3d t_from_dlt = Eigen::Vector3d::Zero();  // m4 / lambda (before the t re-solve)
  double scale = 0;                // lambda = mean of those singular values
  double raw_rotation_deviation = 0;  // || M_3x3/lambda - R ||_F before projecting onto SO(3)
  double nullspace_ratio = 0;      // sigma_12 / sigma_11 of the DLT matrix (small = well-defined solution)
};

// Linear (DLT) PnP from n >= 6 correspondences, built from scratch:
//   X[k]  : 3D point in the object frame (here: camera-0 coordinates)
//   uv[k] : its observed pixel in the target image (here: frame 1)
// Returns (R, t) with X_camera = R * X + t. No RANSAC: every input is used.
PnPResult solvePnPDLT(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                      const Intrinsics &K);
