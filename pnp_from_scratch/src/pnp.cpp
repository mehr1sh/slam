// Eigen is used for matrix storage, products and SVD.

#include "pnp.hpp"

#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SVD>

PnPResult solvePnPDLT(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                      const Intrinsics &K) {
  PnPResult res;
  const int n = static_cast<int>(X.size());
  if (n < 6 || uv.size() != X.size()) {
    res.reason = "need at least 6 correspondences (got " + std::to_string(n) + ")";
    return res;
  }

  // 1. Remove the intrinsics: normalized image coordinates x = K^-1 [u v 1]^T.
  //    The unknown is then M = lambda * [R | t] directly (12 numbers).
  std::vector<Eigen::Vector2d> xn(n);
  for (int k = 0; k < n; ++k) xn[k] = Eigen::Vector2d((uv[k].x() - K.cx) / K.fx, (uv[k].y() - K.cy) / K.fy);

  // 2. Hartley normalization for numerical conditioning:
  //    2D: centroid -> 0, mean distance -> sqrt(2);  3D: centroid -> 0, mean distance -> sqrt(3).
  Eigen::Vector2d c2 = Eigen::Vector2d::Zero();
  Eigen::Vector3d c3 = Eigen::Vector3d::Zero();
  for (int k = 0; k < n; ++k) {
    c2 += xn[k];
    c3 += X[k];
  }
  c2 /= n;
  c3 /= n;
  double d2 = 0, d3 = 0;
  for (int k = 0; k < n; ++k) {
    d2 += (xn[k] - c2).norm();
    d3 += (X[k] - c3).norm();
  }
  const double s2 = std::sqrt(2.0) / (d2 / n), s3 = std::sqrt(3.0) / (d3 / n);
  Eigen::Matrix3d T2 = Eigen::Matrix3d::Identity();  // x~ = T2 * [x 1]
  T2(0, 0) = T2(1, 1) = s2;
  T2.block<2, 1>(0, 2) = -s2 * c2;
  Eigen::Matrix4d T3 = Eigen::Matrix4d::Identity();  // X~ = T3 * [X 1]
  T3.block<3, 3>(0, 0) *= s3;
  T3.block<3, 1>(0, 3) = -s3 * c3;

  // 3. Build the 2n x 12 DLT matrix. With m1, m2, m3 the rows of M~ and
  //    X~ the homogeneous normalized 3D point:
  //      x~ (m3 . X~) - m1 . X~ = 0   ->  [ X~^T   0^T   -x~ X~^T ]
  //      y~ (m3 . X~) - m2 . X~ = 0   ->  [ 0^T    X~^T  -y~ X~^T ]
  Eigen::MatrixXd A(2 * n, 12);
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d p = T2 * Eigen::Vector3d(xn[k].x(), xn[k].y(), 1.0);
    const Eigen::Vector4d P = T3 * Eigen::Vector4d(X[k].x(), X[k].y(), X[k].z(), 1.0);
    const double x = p.x() / p.z(), y = p.y() / p.z();
    A.row(2 * k) << P.transpose(), Eigen::RowVector4d::Zero(), -x * P.transpose();
    A.row(2 * k + 1) << Eigen::RowVector4d::Zero(), P.transpose(), -y * P.transpose();
  }

  // 4. Least-squares solution of A m = 0 with ||m|| = 1: the right singular
  //    vector of the smallest singular value.
  Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeFullV);
  const Eigen::VectorXd sA = svdA.singularValues();
  res.nullspace_ratio = sA(11) / sA(10);
  const Eigen::VectorXd m = svdA.matrixV().col(11);
  Eigen::Matrix<double, 3, 4> Mn;
  Mn << m.segment<4>(0).transpose(), m.segment<4>(4).transpose(), m.segment<4>(8).transpose();

  // 5. Undo the normalization: x~ = T2 x and X~ = T3 X, so
  //    x ~ T2^-1 Mn T3 X  ->  M = T2^-1 * Mn * T3  (= lambda [R | t], up to sign).
  Eigen::Matrix<double, 3, 4> M = T2.inverse() * Mn * T3;

  // 6. Fix the overall sign (m is only defined up to +-): the third row of
  //    M gives the depth lambda*Zc = m3 . [X 1]; choose the sign that puts the
  //    majority of points in front of the camera (cheirality).
  int front = 0;
  for (int k = 0; k < n; ++k) front += (M.row(2).dot(Eigen::Vector4d(X[k].x(), X[k].y(), X[k].z(), 1.0)) > 0);
  if (2 * front < n) M = -M;

  // 7. Project the 3x3 block onto SO(3): with SVD B = U S V^T, the closest
  //    rotation (Frobenius norm) is R = U diag(1, 1, det(U V^T)) V^T;
  //    lambda = mean singular value.
  const Eigen::Matrix3d B = M.leftCols<3>();
  Eigen::JacobiSVD<Eigen::Matrix3d> svdB(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d U = svdB.matrixU(), V = svdB.matrixV();
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (U * V.transpose()).determinant() < 0 ? -1.0 : 1.0;
  const Eigen::Matrix3d R = U * D * V.transpose();
  res.sv_left_block = svdB.singularValues();
  res.scale = res.sv_left_block.mean();
  if (!(res.scale > 0) || !std::isfinite(res.scale)) {
    res.reason = "degenerate scale lambda = " + std::to_string(res.scale);
    return res;
  }
  res.raw_rotation_deviation = (B / res.scale - R).norm();

  // 8. Re-solve t for the orthonormalized R (linear least squares, 3 unknowns).
  //    With Xr = R X:  x (Xr_z + t_z) = Xr_x + t_x  and  y (Xr_z + t_z) = Xr_y + t_y
  //    ->  [ -1  0  x ] t = Xr_x - x Xr_z
  //        [  0 -1  y ] t = Xr_y - y Xr_z
  //    (x, y = normalized image coordinates). Using t = m4 / lambda instead
  //    would mix the un-orthonormalized block with the corrected R.
  Eigen::MatrixXd At(2 * n, 3);
  Eigen::VectorXd bt(2 * n);
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d Xr = R * X[k];
    const double x = xn[k].x(), y = xn[k].y();
    At.row(2 * k) << -1, 0, x;
    At.row(2 * k + 1) << 0, -1, y;
    bt(2 * k) = Xr.x() - x * Xr.z();
    bt(2 * k + 1) = Xr.y() - y * Xr.z();
  }
  res.t_from_dlt = M.col(3) / res.scale;
  res.t = At.colPivHouseholderQr().solve(bt);
  res.R = R;
  res.ok = true;
  return res;
}
