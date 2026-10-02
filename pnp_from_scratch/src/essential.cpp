#include "essential.hpp"

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace scratch {

Eigen::Matrix3d Skew(const Eigen::Vector3d &t) {
  Eigen::Matrix3d S;
  S << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
  return S;
}

namespace {

// Similarity T with T [x 1]^T having zero mean and mean distance sqrt(2).
Eigen::Matrix3d HartleyNormalization(const Points2 &x) {
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  for (const auto &p : x) c += p;
  c /= double(x.size());
  double d = 0;
  for (const auto &p : x) d += (p - c).norm();
  d /= double(x.size());
  const double s = d > 0 ? std::sqrt(2.0) / d : 1.0;
  Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
  T(0, 0) = T(1, 1) = s;
  T(0, 2) = -s * c.x();
  T(1, 2) = -s * c.y();
  return T;
}

}  // namespace

Eigen::Matrix3d EnforceEssentialConstraints(const Eigen::Matrix3d &M) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d sv = svd.singularValues();
  const double s = 0.5 * (sv(0) + sv(1));
  return svd.matrixU() * Eigen::Vector3d(s, s, 0.0).asDiagonal() * svd.matrixV().transpose();
}

bool EstimateEssential8Point(const Points2 &x1, const Points2 &x2, Eigen::Matrix3d &E) {
  const int n = int(x1.size());
  if (n < 8 || x2.size() != x1.size()) return false;
  const Eigen::Matrix3d T1 = HartleyNormalization(x1), T2 = HartleyNormalization(x2);

  // x2^T E x1 = 0 is linear in the 9 entries of E (row-major e):
  //   [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] . e = 0
  Eigen::MatrixXd A(n, 9);
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d a = T1 * Eigen::Vector3d(x1[k].x(), x1[k].y(), 1.0);
    const Eigen::Vector3d b = T2 * Eigen::Vector3d(x2[k].x(), x2[k].y(), 1.0);
    A.row(k) << b.x() * a.x(), b.x() * a.y(), b.x(), b.y() * a.x(), b.y() * a.y(), b.y(), a.x(), a.y(), 1.0;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd e = svd.matrixV().col(8);
  Eigen::Matrix3d En;
  En << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

  // (T2 x2)^T En (T1 x1) = x2^T (T2^T En T1) x1
  E = EnforceEssentialConstraints(T2.transpose() * En * T1);
  const double norm = E.norm();
  if (!(norm > 0) || !std::isfinite(norm)) return false;
  E /= norm;
  return true;
}

std::array<PoseRt, 4> DecomposeEssential(const Eigen::Matrix3d &E) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
  // E is defined up to sign, so U and V may be replaced by their proper
  // (det = +1) versions; this keeps the candidate rotations proper.
  if (U.determinant() < 0) U.col(2) *= -1;
  if (V.determinant() < 0) V.col(2) *= -1;
  Eigen::Matrix3d W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Eigen::Matrix3d Ra = U * W * V.transpose(), Rb = U * W.transpose() * V.transpose();
  const Eigen::Vector3d u3 = U.col(2);  // unit null vector of E^T: the translation direction
  std::array<PoseRt, 4> c;
  c[0].R = Ra, c[0].t = u3;
  c[1].R = Ra, c[1].t = -u3;
  c[2].R = Rb, c[2].t = u3;
  c[3].R = Rb, c[3].t = -u3;
  return c;
}

bool TriangulateDepths(const PoseRt &P, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, double &d1, double &d2) {
  const Eigen::Vector3d r1 = P.R * Eigen::Vector3d(x1.x(), x1.y(), 1.0), r2(x2.x(), x2.y(), 1.0);
  // X_2 = R X_1 + t with X_1 = d1 [x1 1], X_2 = d2 [x2 1]  ->  [-r1  r2] (d1, d2)^T = t
  Eigen::Matrix<double, 3, 2> A;
  A.col(0) = -r1;
  A.col(1) = r2;
  const Eigen::Matrix2d N = A.transpose() * A;
  if (std::abs(N.determinant()) < 1e-12 * N.trace() * N.trace()) return false;  // parallel rays
  const Eigen::Vector2d d = N.ldlt().solve(A.transpose() * P.t);
  d1 = d(0);
  d2 = d(1);
  return true;
}

int CountInFront(const PoseRt &P, const Points2 &x1, const Points2 &x2) {
  int n = 0;
  for (size_t k = 0; k < x1.size(); ++k) {
    double d1, d2;
    if (TriangulateDepths(P, x1[k], x2[k], d1, d2) && d1 > 0 && d2 > 0) ++n;
  }
  return n;
}

bool RecoverPoseFromEssential(const Eigen::Matrix3d &E, const Points2 &x1, const Points2 &x2, RecoveredPose &out) {
  const auto cands = DecomposeEssential(E);
  out = RecoveredPose{};
  for (int c = 0; c < 4; ++c) {
    out.candidate_in_front[c] = CountInFront(cands[c], x1, x2);
    if (out.candidate_in_front[c] > out.in_front) {
      out.in_front = out.candidate_in_front[c];
      out.chosen = c;
    }
  }
  if (out.chosen < 0) return false;
  out.pose = cands[out.chosen];
  return true;
}

double SampsonDistance(const Eigen::Matrix3d &E, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2) {
  const Eigen::Vector3d a(x1.x(), x1.y(), 1.0), b(x2.x(), x2.y(), 1.0);
  const Eigen::Vector3d Ea = E * a, Etb = E.transpose() * b;
  const double r = b.dot(Ea);
  const double den = Ea.x() * Ea.x() + Ea.y() * Ea.y() + Etb.x() * Etb.x() + Etb.y() * Etb.y();
  return den > 0 ? r * r / den : 0.0;
}

}  // namespace scratch
