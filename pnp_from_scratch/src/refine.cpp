#include "refine.hpp"

#include <cmath>

#include <Eigen/Dense>

namespace scratch {

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &phi) {
  const double a = phi.norm();
  Eigen::Matrix3d W;
  W << 0, -phi.z(), phi.y(), phi.z(), 0, -phi.x(), -phi.y(), phi.x(), 0;
  if (a < 1e-12) return Eigen::Matrix3d::Identity() + W;
  return Eigen::Matrix3d::Identity() + std::sin(a) / a * W + (1 - std::cos(a)) / (a * a) * W * W;
}

namespace {

// Half the sum of squared residuals; false if a point is not in front of the camera.
bool Cost(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv, const std::vector<int> &idx,
          const Intrinsics &K, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double &cost) {
  cost = 0;
  for (int k : idx) {
    double z;
    const Eigen::Vector2d p = projectPoint(X[k], R, t, K, &z);
    if (!(z > 0)) return false;
    cost += 0.5 * (p - uv[k]).squaredNorm();
  }
  return true;
}

}  // namespace

RefineResult RefinePnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                       const std::vector<bool> &use, const Intrinsics &K, const Eigen::Matrix3d &R0,
                       const Eigen::Vector3d &t0, const RefineParams &params) {
  RefineResult out;
  out.R = R0;
  out.t = t0;
  std::vector<int> idx;
  for (size_t k = 0; k < X.size(); ++k)
    if (k < use.size() && use[k]) idx.push_back(int(k));
  if (idx.size() < 3 || !Cost(X, uv, idx, K, R0, t0, out.cost_initial)) {
    out.cost_final = out.cost_initial;
    return out;
  }
  Eigen::Matrix3d R = R0;
  Eigen::Vector3d t = t0;
  double cost = out.cost_initial, lambda = -1;
  for (int it = 0; it < params.max_iterations; ++it) {
    // normal equations of the linearised problem at (R, t)
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (int k : idx) {
      const Eigen::Vector3d Xc = R * X[k] + t;
      const double x = Xc.x(), y = Xc.y(), z = Xc.z(), iz = 1.0 / z;
      const Eigen::Vector2d e(K.fx * x * iz + K.cx - uv[k].x(), K.fy * y * iz + K.cy - uv[k].y());
      Eigen::Matrix<double, 2, 3> Jp;
      Jp << K.fx * iz, 0, -K.fx * x * iz * iz, 0, K.fy * iz, -K.fy * y * iz * iz;
      Eigen::Matrix<double, 3, 6> Jx;
      Jx.leftCols<3>().setIdentity();
      Jx.rightCols<3>() << 0, z, -y, -z, 0, x, y, -x, 0;  // -[Xc]_x
      const Eigen::Matrix<double, 2, 6> J = Jp * Jx;
      H += J.transpose() * J;
      g += J.transpose() * e;
    }
    if (lambda < 0) lambda = params.lambda0 * H.diagonal().mean();
    bool accepted = false;
    for (int tries = 0; tries < 10 && !accepted; ++tries) {
      Eigen::Matrix<double, 6, 6> A = H;
      A.diagonal().array() += lambda;
      const Eigen::Matrix<double, 6, 1> xi = A.ldlt().solve(-g);
      if (!xi.allFinite()) break;
      const Eigen::Matrix3d dR = ExpSO3(xi.tail<3>());
      const Eigen::Matrix3d Rn = dR * R;
      const Eigen::Vector3d tn = dR * t + xi.head<3>();
      double cn;
      if (Cost(X, uv, idx, K, Rn, tn, cn) && cn < cost) {
        const double rel = (cost - cn) / std::max(cost, 1e-300);
        R = Rn;
        t = tn;
        cost = cn;
        lambda = std::max(lambda / 3, 1e-12);
        accepted = true;
        ++out.iterations;
        if (xi.norm() < params.min_step || rel < params.min_rel_decrease) it = params.max_iterations;
      } else {
        lambda *= 4;
      }
    }
    if (!accepted) break;  // no decreasing step at any damping: converged
  }
  // re-orthonormalise (products of rotations drift by ~1e-16 per step)
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  out.R = svd.matrixU() * svd.matrixV().transpose();
  out.t = t;
  Cost(X, uv, idx, K, out.R, out.t, out.cost_final);
  out.rms_initial_px = std::sqrt(2 * out.cost_initial / idx.size());
  out.rms_final_px = std::sqrt(2 * out.cost_final / idx.size());
  out.ok = true;
  return out;
}

}  // namespace scratch
