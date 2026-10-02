#pragma once

// Nonlinear PnP refinement from scratch (Eigen only): minimises the
// reprojection error of a FIXED set of correspondences (the RANSAC inliers)
// over the pose, starting from the linear (DLT) estimate.
//
//   cost(R, t) = 1/2 sum_k || pi(K (R X_k + t)) - uv_k ||^2      (pixels)
//
// Levenberg-Marquardt on SE(3) with a left perturbation
//   R <- Exp(phi) R,   t <- Exp(phi) t + rho,     xi = (rho, phi)
// so the Jacobian of one residual is
//   d pi / d Xc * [ I | -[Xc]_x ],  Xc = R X + t,
//   d pi / d Xc = [ fx/z  0  -fx x/z^2 ;  0  fy/z  -fy y/z^2 ].
// A step is accepted only if it lowers the cost (lambda /= 3), otherwise
// lambda *= 4 and the step is retried. The inlier set is not changed.

#include <vector>

#include <Eigen/Core>

#include "projection.hpp"

namespace scratch {

struct RefineParams {
  int max_iterations = 50;
  double lambda0 = 1e-3;       // initial damping, relative to the mean diagonal of J^T J
  double min_step = 1e-10;     // stop when |xi| falls below this
  double min_rel_decrease = 1e-12;  // stop when the cost decreases by less than this fraction
};

struct RefineResult {
  bool ok = false;               // false: fewer than 3 points, or a point behind the camera at the start
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  int iterations = 0;            // accepted LM steps
  double cost_initial = 0;       // sum of squared pixel residuals / 2, at the linear pose
  double cost_final = 0;
  double rms_initial_px = 0, rms_final_px = 0;
};

// X: 3D points (camera i), uv: pixels (frame i+1), use[k]: refine with point k.
RefineResult RefinePnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                       const std::vector<bool> &use, const Intrinsics &K, const Eigen::Matrix3d &R0,
                       const Eigen::Vector3d &t0, const RefineParams &params = RefineParams());

// Rotation vector -> rotation matrix (Rodrigues formula, from scratch).
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &phi);

}  // namespace scratch
