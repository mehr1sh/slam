#pragma once
// Small helpers shared by the two programs (milestone 1 and the full sequence):
// pose type, quaternion <-> matrix (written out), the repository's error
// metrics, repository/CSV/ground-truth readers, reprojection statistics.
// No OpenCV.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "projection.hpp"

namespace scratch {

namespace fs = std::filesystem;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

struct Pose {  // X_b = R * X_a + t
  Matrix3d R = Matrix3d::Identity();
  Vector3d t = Vector3d::Zero();
};

// Unit quaternion (x, y, z, w) -> rotation matrix (written out, no library call).
inline Matrix3d QuatToR(double x, double y, double z, double w) {
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n, y /= n, z /= n, w /= n;
  Matrix3d R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
       2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
       2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
  return R;
}

// Same metrics as the repository (tests/report_utils.hpp):
// rotation error = acos((trace(R1^T R2) - 1) / 2), translation error = ||t1 - t2||.
inline double RotErrDeg(const Matrix3d &A, const Matrix3d &B) {
  const double c = std::clamp(((A.transpose() * B).trace() - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(c) * 180.0 / M_PI;
}
inline double RotAngleDeg(const Matrix3d &R) { return RotErrDeg(Matrix3d::Identity(), R); }

// Rotation matrix -> unit quaternion (x, y, z, w), Shepperd's method (written out).
inline Eigen::Vector4d RToQuat(const Matrix3d &R) {
  const double tr = R.trace();
  double x, y, z, w;
  if (tr > 0) {
    const double S = std::sqrt(tr + 1.0) * 2;
    w = 0.25 * S, x = (R(2, 1) - R(1, 2)) / S, y = (R(0, 2) - R(2, 0)) / S, z = (R(1, 0) - R(0, 1)) / S;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double S = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2;
    w = (R(2, 1) - R(1, 2)) / S, x = 0.25 * S, y = (R(0, 1) + R(1, 0)) / S, z = (R(0, 2) + R(2, 0)) / S;
  } else if (R(1, 1) > R(2, 2)) {
    const double S = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2;
    w = (R(0, 2) - R(2, 0)) / S, x = (R(0, 1) + R(1, 0)) / S, y = 0.25 * S, z = (R(1, 2) + R(2, 1)) / S;
  } else {
    const double S = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2;
    w = (R(1, 0) - R(0, 1)) / S, x = (R(0, 2) + R(2, 0)) / S, y = (R(1, 2) + R(2, 1)) / S, z = 0.25 * S;
  }
  Eigen::Vector4d q(x, y, z, w);
  if (q.w() < 0) q = -q;  // canonical sign
  return q / q.norm();
}

inline bool IsRepoRoot(const fs::path &p) { return fs::exists(p / "data" / "synthetic_bunny" / "intrinsics.txt"); }

inline fs::path FindRepo(const fs::path &start) {
  for (fs::path p = fs::absolute(start); !p.empty(); p = p.parent_path()) {
    if (IsRepoRoot(p)) return p;
    if (p == p.parent_path()) break;
  }
  return {};
}

// Minimal CSV reader: header -> column index.
struct Csv {
  std::map<std::string, int> col;
  std::vector<std::vector<std::string>> rows;
  bool load(const fs::path &path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::getline(f, line);
    std::stringstream hs(line);
    std::string h;
    for (int i = 0; std::getline(hs, h, ','); ++i) col[h] = i;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      std::vector<std::string> r;
      std::stringstream ss(line);
      std::string cell;
      while (std::getline(ss, cell, ',')) r.push_back(cell);
      if (!line.empty() && line.back() == ',') r.push_back("");
      rows.push_back(r);
    }
    return true;
  }
  double d(size_t r, const std::string &c) const { return std::stod(rows[r].at(col.at(c))); }
  int i(size_t r, const std::string &c) const { return std::stoi(rows[r].at(col.at(c))); }
};

inline std::vector<Pose> ReadGroundtruthTwc(const fs::path &path) {  // TUM: ts tx ty tz qx qy qz qw (T_wc)
  std::vector<Pose> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    double ts, tx, ty, tz, qx, qy, qz, qw;
    is >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Pose p;
    p.R = QuatToR(qx, qy, qz, qw);
    p.t = Vector3d(tx, ty, tz);
    out.push_back(p);
  }
  return out;
}

struct Reproj {
  double mean = NAN, median = NAN, max = NAN, rmse = NAN;
  int positive_depth = 0;
  std::vector<double> err;
};

inline Reproj Reprojection(const std::vector<Vector3d> &X, const std::vector<Vector2d> &uv, const Pose &T,
                    const Intrinsics &K) {
  Reproj r;
  double sum = 0, sq = 0;
  for (size_t k = 0; k < X.size(); ++k) {
    double z;
    const Vector2d p = projectPoint(X[k], T.R, T.t, K, &z);
    if (z > 0) ++r.positive_depth;
    const double e = (p - uv[k]).norm();
    r.err.push_back(e);
    sum += e;
    sq += e * e;
  }
  if (r.err.empty()) return r;
  std::vector<double> s = r.err;
  std::sort(s.begin(), s.end());
  r.mean = sum / s.size();
  r.median = s.size() % 2 ? s[s.size() / 2] : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);
  r.max = s.back();
  r.rmse = std::sqrt(sq / s.size());
  return r;
}

}  // namespace scratch
