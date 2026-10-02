// Self-checking test of the scratch RANSAC around the linear PnP and the
// 8-point essential matrix: deliberate outliers, recovery of the true pose,
// inlier-mask quality, determinism.

#include <algorithm>
#include <cmath>
#include <random>

#include <Eigen/Geometry>

#include "check.hpp"
#include "pnp.hpp"
#include "ransac.hpp"

using namespace scratch;

namespace {

struct Data {
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
  std::vector<bool> is_outlier;
};

// Bunny-like data: points 0.45-0.58 m in front of camera i, true motion
// (R, t), pixel noise, and a fraction of gross mismatches (random pixels).
Data Make(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Intrinsics &K, int n, double outlier_frac,
          double noise_px, uint32_t seed) {
  std::mt19937 rng(seed);
  auto U = [&] { return (double(rng()) + 0.5) / 4294967296.0; };
  auto N = [&] { return std::sqrt(-2 * std::log(U())) * std::cos(2 * M_PI * U()); };
  Data d;
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d X(0.14 * (U() - 0.5), 0.17 * (U() - 0.5), 0.45 + 0.13 * U());
    Eigen::Vector2d p = projectPoint(X, R, t, K) + noise_px * Eigen::Vector2d(N(), N());
    const bool out = U() < outlier_frac;
    if (out) p = Eigen::Vector2d(260 + 130 * U(), 170 + 170 * U());  // a random pixel on the object
    d.X.push_back(X);
    d.uv.push_back(p);
    d.is_outlier.push_back(out);
  }
  return d;
}

}  // namespace

int main() {
  std::printf("=== scratch RANSAC ===\n");
  const Intrinsics K{520.9, 521.0, 325.1, 249.7};
  const Eigen::Matrix3d R = Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(0, 0.97, 0.24).normalized()).matrix();
  const Eigen::Vector3d t(-0.0869, -0.0018, 0.0074);

  // --- sampler ---
  {
    std::mt19937 a(7), b(7);
    const auto s1 = SampleIndices(a, 100, 6), s2 = SampleIndices(b, 100, 6);
    auto sorted = s1;
    std::sort(sorted.begin(), sorted.end());
    CHECK(s1 == s2 && std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end() &&
              sorted.front() >= 0 && sorted.back() < 100,
          "sampler: 6 distinct indices in [0, 100), reproducible");
  }

  // --- PnP with 30 % gross outliers ---
  const Data d = Make(R, t, K, 160, 0.30, 0.5, 11);
  const int n_out = int(std::count(d.is_outlier.begin(), d.is_outlier.end(), true));
  const PnPResult plain = solvePnPDLT(d.X, d.uv, K);
  CHECK(plain.ok && RotErrDeg(R, plain.R) > 5.0,
        "without RANSAC the linear PnP breaks: rotation error %.1f deg (%d / 160 outliers)", RotErrDeg(R, plain.R), n_out);
  const RansacParams params;  // 300 iterations, 8 px, seed 12345
  const RansacPnPResult r = RansacPnP(d.X, d.uv, K, params);
  CHECK(r.ok, "RANSAC PnP returns a pose (%s)", r.reason.c_str());
  // Reference: the same linear solver on exactly the true inliers (an oracle
  // RANSAC cannot beat; its error is the DLT's own noise sensitivity).
  std::vector<Eigen::Vector3d> Xi;
  std::vector<Eigen::Vector2d> ui;
  for (size_t k = 0; k < d.X.size(); ++k)
    if (!d.is_outlier[k]) Xi.push_back(d.X[k]), ui.push_back(d.uv[k]);
  const PnPResult oracle = solvePnPDLT(Xi, ui, K);
  CHECK(RotErrDeg(R, r.R) < 1.0 && (t - r.t).norm() < 0.01,
        "true pose recovered: rotation error %.3f deg, translation error %.4f m", RotErrDeg(R, r.R), (t - r.t).norm());
  CHECK(std::abs(RotErrDeg(R, r.R) - RotErrDeg(R, oracle.R)) < 0.15,
        "as accurate as the linear solver on the true inliers only (oracle: %.3f deg, %.4f m)", RotErrDeg(R, oracle.R),
        (t - oracle.t).norm());
  int tp = 0, fp = 0;
  for (size_t k = 0; k < d.X.size(); ++k) {
    tp += !d.is_outlier[k] && r.inlier_mask[k];
    fp += d.is_outlier[k] && r.inlier_mask[k];
  }
  CHECK(tp >= 0.97 * (160 - n_out) && fp <= 2,
        "inlier mask: %d / %d true inliers kept, %d / %d outliers accepted (outliers inside 8 px can pass)", tp,
        160 - n_out, fp, n_out);
  CHECK(r.num_inliers == tp + fp && int(r.inlier_mask.size()) == 160 && r.refit_used,
        "final mask from the refit pose: %d inliers (best minimal sample: %d)", r.num_inliers, r.best_sample_inliers);
  CHECK(r.models_tried > 250, "%d of %d minimal samples gave a valid model", r.models_tried, params.iterations);

  const RansacPnPResult r2 = RansacPnP(d.X, d.uv, K, params);
  CHECK(r2.R == r.R && r2.t == r.t && r2.inlier_mask == r.inlier_mask, "deterministic: same seed -> identical result");
  RansacParams other = params;
  other.seed = 999;
  const RansacPnPResult r3 = RansacPnP(d.X, d.uv, K, other);
  CHECK(r3.ok && RotErrDeg(R, r3.R) < 1.0, "another seed also recovers the pose (%.3f deg)", RotErrDeg(R, r3.R));
  {
    RansacParams few = params;
    few.iterations = 0;
    CHECK(!RansacPnP(d.X, d.uv, K, few).ok, "0 iterations -> failure is reported");
    std::vector<Eigen::Vector3d> X5(d.X.begin(), d.X.begin() + 5);
    std::vector<Eigen::Vector2d> u5(d.uv.begin(), d.uv.begin() + 5);
    CHECK(!RansacPnP(X5, u5, K).ok, "5 correspondences -> failure is reported");
  }

  // --- essential matrix with 30 % outliers ---
  // The minimal 8-point solver is ill-conditioned under noise for a narrow
  // field of view and a small baseline (the Bunny's case; see README). The
  // RANSAC logic is therefore tested on a wide, deep scene where minimal
  // samples are informative, with more iterations.
  {
    std::mt19937 rng(21);
    auto U = [&] { return (double(rng()) + 0.5) / 4294967296.0; };
    auto Nn = [&] { return std::sqrt(-2 * std::log(U())) * std::cos(2 * M_PI * U()); };
    Points2 x1, x2;
    std::vector<bool> outl;
    for (int k = 0; k < 200; ++k) {
      const Eigen::Vector3d X(1.2 * (U() - 0.5), 0.9 * (U() - 0.5), 0.6 + 1.0 * U());
      Eigen::Vector2d p = projectPoint(X, R, t, K) + 0.3 * Eigen::Vector2d(Nn(), Nn());
      outl.push_back(U() < 0.3);
      if (outl.back()) p = Eigen::Vector2d(40 + 560 * U(), 40 + 400 * U());
      x1.emplace_back(X.x() / X.z(), X.y() / X.z());
      x2.emplace_back((p.x() - K.cx) / K.fx, (p.y() - K.cy) / K.fy);
    }
    RansacParams ep = params;
    ep.threshold_px = 1.5;
    ep.iterations = 3000;
    const RansacEssentialResult e = RansacEssential(x1, x2, K.fx, ep);
    const double dir = std::acos(std::clamp(e.pose.pose.t.dot(t.normalized()), -1.0, 1.0)) * 180 / M_PI;
    int tp = 0, fp = 0;
    for (size_t k = 0; k < outl.size(); ++k) tp += !outl[k] && e.inlier_mask[k], fp += outl[k] && e.inlier_mask[k];
    CHECK(e.ok && RotErrDeg(R, e.pose.pose.R) < 1.0 && dir < 10.0,
          "RANSAC essential (wide scene, 30%% outliers): rotation error %.3f deg, t direction error %.2f deg",
          RotErrDeg(R, e.pose.pose.R), dir);
    CHECK(tp > 0.8 * std::count(outl.begin(), outl.end(), false) && fp < 0.1 * tp,
          "essential inlier mask: %d true inliers kept, %d outliers accepted (outliers near their epipolar line pass)",
          tp, fp);
  }
  return scratch_test::Finish("test_ransac");
}
