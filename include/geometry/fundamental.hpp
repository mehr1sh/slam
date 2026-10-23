#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
namespace geometry_scratch {
// Pixel coordinates, x2^T F x1=0. Hartley normalization, rank-eight check,
// null-space solve, rank-two projection, then denormalization. Eigen only.
inline bool EstimateFundamental8Point(const std::vector<Eigen::Vector2d> &x1,
                                      const std::vector<Eigen::Vector2d> &x2,
                                      Eigen::Matrix3d &F) {
  if (x1.size() != x2.size() || x1.size() < 8)
    return false;
  auto normalize = [](const auto &points, Eigen::Matrix3d &T) {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (auto p : points) {
      if (!p.allFinite())
        return false;
      center += p;
    }
    center /= points.size();
    double d = 0;
    for (auto p : points)
      d += (p - center).norm();
    d /= points.size();
    if (d < 1e-12)
      return false;
    double s = std::sqrt(2.) / d;
    T = Eigen::Matrix3d::Identity();
    T(0, 0) = T(1, 1) = s;
    T(0, 2) = -s * center.x();
    T(1, 2) = -s * center.y();
    return true;
  };
  Eigen::Matrix3d T1, T2;
  if (!normalize(x1, T1) || !normalize(x2, T2))
    return false;
  Eigen::MatrixXd A(x1.size(), 9);
  for (size_t i = 0; i < x1.size(); ++i) {
    Eigen::Vector3d a = T1 * Eigen::Vector3d(x1[i].x(), x1[i].y(), 1),
                    b = T2 * Eigen::Vector3d(x2[i].x(), x2[i].y(), 1);
    A.row(i) << b.x() * a.x(), b.x() * a.y(), b.x(), b.y() * a.x(),
        b.y() * a.y(), b.y(), a.x(), a.y(), 1;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  if (svd.singularValues()[7] < 1e-10 * svd.singularValues()[0])
    return false;
  Eigen::VectorXd v = svd.matrixV().col(8);
  Eigen::Matrix3d M;
  M << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  Eigen::JacobiSVD<Eigen::Matrix3d> rank(M, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
  Eigen::Vector3d values = rank.singularValues();
  values[2] = 0;
  F = T2.transpose() * rank.matrixU() * values.asDiagonal() *
      rank.matrixV().transpose() * T1;
  if (!F.allFinite() || F.norm() < 1e-15)
    return false;
  F /= F.norm();
  return true;
}
} // namespace geometry_scratch
