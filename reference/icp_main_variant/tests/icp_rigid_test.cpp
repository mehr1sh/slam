// Self-checking test of the ICP alignment (src/tracking/icp.cpp): the robust
// rigid fit (EstimateRigidRansac / FitRigid) and its relative-pose convention.
// Points are generated with the pipeline's convention x_{i+1} = R x_i + t
// (T_{i+1<-i}); ICP's native result is T_{i<-i+1} (pts1 = R pts2 + t) and is
// inverted exactly once, as in slam_trajectory_test's RunIcpPair.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>

#include "tracking/icp.hpp"

namespace {

int failures = 0;
void Check(bool ok, const std::string &text) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", text.c_str());
  failures += !ok;
}
double RotErrDeg(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) {
  return std::acos(std::min(1.0, std::max(-1.0, ((A.transpose() * B).trace() - 1) / 2))) * 180 / M_PI;
}
std::string Fmt(const char *f, double a, double b = 0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b);
  return buf;
}

// Bunny-like points in camera i (0.45 - 0.58 m in front), moved into camera i+1.
void Make(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, int n, double outlier_frac, double noise, uint32_t seed,
          std::vector<cv::Point3f> &p1, std::vector<cv::Point3f> &p2, std::vector<bool> &is_outlier) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> N(0, 1);
  p1.clear(), p2.clear(), is_outlier.clear();
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d x(0.14 * (U(rng) - 0.5), 0.17 * (U(rng) - 0.5), 0.45 + 0.13 * U(rng));
    Eigen::Vector3d y = R * x + t + noise * Eigen::Vector3d(N(rng), N(rng), N(rng));
    const bool out = U(rng) < outlier_frac;
    if (out) y = Eigen::Vector3d(0.14 * (U(rng) - 0.5), 0.17 * (U(rng) - 0.5), 0.45 + 0.13 * U(rng));  // wrong match
    p1.emplace_back(float(x.x()), float(x.y()), float(x.z()));
    p2.emplace_back(float(y.x()), float(y.y()), float(y.z()));
    is_outlier.push_back(out);
  }
}

// RunIcpPair's arithmetic: native T_{i<-i+1} -> common T_{i+1<-i}, inverted once.
void ToCommon(const RigidRansacResult &r, Eigen::Matrix3d &R, Eigen::Vector3d &t) {
  R = r.R.transpose();
  t = -R * r.t;
}

}  // namespace

int main() {
  std::printf("=== ICP rigid alignment ===\n");
  std::vector<cv::Point3f> p1, p2;
  std::vector<bool> out;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  // identity motion
  Make(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 60, 0.0, 0.0, 1, p1, p2, out);
  RigidRansacResult r = EstimateRigidRansac(p1, p2);
  ToCommon(r, R, t);
  Check(r.ok && r.num_inliers == 60 && RotErrDeg(R, Eigen::Matrix3d::Identity()) < 1e-3 && t.norm() < 1e-6,
        Fmt("identity motion: rotation %.1e deg, translation %.1e m", RotErrDeg(R, Eigen::Matrix3d::Identity()), t.norm()));

  // known translation only
  const Eigen::Vector3d tk(-0.0869, -0.0018, 0.0074);
  Make(Eigen::Matrix3d::Identity(), tk, 60, 0.0, 0.0, 2, p1, p2, out);
  r = EstimateRigidRansac(p1, p2);
  ToCommon(r, R, t);
  Check(r.ok && RotErrDeg(R, Eigen::Matrix3d::Identity()) < 1e-3 && (t - tk).norm() < 1e-6,
        Fmt("known translation: rotation %.1e deg, translation error %.1e m", RotErrDeg(R, Eigen::Matrix3d::Identity()),
            (t - tk).norm()));

  // known rotation only (10 deg about the orbit's axis)
  const Eigen::Matrix3d Rk = Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(0, 0.97, 0.24).normalized()).matrix();
  Make(Rk, Eigen::Vector3d::Zero(), 60, 0.0, 0.0, 3, p1, p2, out);
  r = EstimateRigidRansac(p1, p2);
  ToCommon(r, R, t);
  Check(r.ok && RotErrDeg(R, Rk) < 1e-3 && t.norm() < 1e-6,
        Fmt("known rotation (10 deg): rotation error %.1e deg, translation %.1e m", RotErrDeg(R, Rk), t.norm()));

  // relative-pose convention: x_{i+1} = R x_i + t is recovered after exactly one inversion
  Make(Rk, tk, 60, 0.0, 0.0, 4, p1, p2, out);
  r = EstimateRigidRansac(p1, p2);
  ToCommon(r, R, t);
  const bool once = RotErrDeg(R, Rk) < 1e-3 && (t - tk).norm() < 1e-6;
  const bool native_is_inverse = RotErrDeg(r.R, Rk.transpose()) < 1e-3 && (r.t + Rk.transpose() * tk).norm() < 1e-6;
  Check(once && native_is_inverse, "convention: native result = T_{i<-i+1} (pts1 = R pts2 + t); one inversion gives "
        "T_{i+1<-i}");
  Eigen::Matrix3d R2 = R.transpose();  // a second inversion would be wrong
  Check(RotErrDeg(R2, Rk) > 15, Fmt("a second inversion would give %.1f deg error (detectable)", RotErrDeg(R2, Rk)));

  // agreement with the book SVD + refinement on clean data
  {
    std::ostringstream sink;
    std::streambuf *old = std::cout.rdbuf(sink.rdbuf());  // the book code prints
    cv::Mat Rb, tb;
    pose_estimation_3d3d(p1, p2, Rb, tb);
    std::cout.rdbuf(old);
    Eigen::Matrix3d Rbe;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) Rbe(a, b) = Rb.at<double>(a, b);
    Check(RotErrDeg(Rbe, r.R) < 1e-3, Fmt("same result as the book SVD on clean data (%.1e deg apart)", RotErrDeg(Rbe, r.R)));
  }

  // gross mismatches: 35 % wrong correspondences, 1 mm noise
  Make(Rk, tk, 120, 0.35, 0.001, 5, p1, p2, out);
  r = EstimateRigidRansac(p1, p2);
  ToCommon(r, R, t);
  int tp = 0, fp = 0, n_out = 0;
  for (size_t k = 0; k < out.size(); ++k) tp += !out[k] && r.inliers[k], fp += out[k] && r.inliers[k], n_out += out[k];
  Check(r.ok && RotErrDeg(R, Rk) < 0.5 && (t - tk).norm() < 0.005,
        Fmt("35 %% mismatches: rotation error %.3f deg, translation error %.4f m", RotErrDeg(R, Rk), (t - tk).norm()));
  Check(fp == 0 && tp > 0.9 * (int(out.size()) - n_out),
        Fmt("inliers: %.0f correct kept, %.0f wrong accepted", tp, fp));
  {
    std::vector<int> all(p1.size());
    for (size_t k = 0; k < all.size(); ++k) all[k] = int(k);
    Eigen::Matrix3d Ra;
    Eigen::Vector3d ta;
    FitRigid(p1, p2, all, Ra, ta);
    Check(RotErrDeg(Ra.transpose(), Rk) > 3 * RotErrDeg(R, Rk) + 1,
          Fmt("without inlier selection the same data gives %.1f deg (the failure fixed)", RotErrDeg(Ra.transpose(), Rk)));
  }
  const RigidRansacResult r2 = EstimateRigidRansac(p1, p2);
  Check(r2.R == r.R && r2.t == r.t && r2.inliers == r.inliers, "deterministic: identical result on a second run");

  // degenerate input
  Check(!EstimateRigidRansac(std::vector<cv::Point3f>(p1.begin(), p1.begin() + 2),
                             std::vector<cv::Point3f>(p2.begin(), p2.begin() + 2)).ok,
        "fewer than 3 correspondences: not ok");
  std::printf("\nicp_rigid_test: %s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
