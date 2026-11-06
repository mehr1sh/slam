// Self-checking test of the scratch nonlinear PnP refinement (refine.hpp):
// the rotation exponential, the analytic Jacobian (against finite
// differences), exact convergence on noise-free data, the improvement over
// the linear PnP on noisy data, and that only the selected points are used.

#include <cmath>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "check.hpp"
#include "pnp.hpp"
#include "refine.hpp"

using namespace scratch;

namespace {

struct Data {
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
};

// Bunny-like data as in test_ransac: points 0.45-0.58 m in front of camera i,
// projected with the true motion (R, t) plus Gaussian pixel noise.
Data Make(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Intrinsics &K, int n, double noise_px,
          uint32_t seed) {
  std::mt19937 rng(seed);
  auto U = [&] { return (double(rng()) + 0.5) / 4294967296.0; };
  auto N = [&] { return std::sqrt(-2 * std::log(U())) * std::cos(2 * M_PI * U()); };
  Data d;
  for (int k = 0; k < n; ++k) {
    const Eigen::Vector3d X(0.14 * (U() - 0.5), 0.17 * (U() - 0.5), 0.45 + 0.13 * U());
    d.X.push_back(X);
    d.uv.push_back(projectPoint(X, R, t, K) + noise_px * Eigen::Vector2d(N(), N()));
  }
  return d;
}

double Rms(const Data &d, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Intrinsics &K) {
  double s = 0;
  for (size_t k = 0; k < d.X.size(); ++k) s += (projectPoint(d.X[k], R, t, K) - d.uv[k]).squaredNorm();
  return std::sqrt(s / d.X.size());
}

}  // namespace

int main() {
  std::printf("=== scratch nonlinear PnP refinement ===\n");
  const Intrinsics K{520.9, 521.0, 325.1, 249.7};
  const Eigen::Matrix3d R = Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(0, 0.97, 0.24).normalized()).matrix();
  const Eigen::Vector3d t(-0.0869, -0.0018, 0.0074);

  // --- rotation exponential ---
  {
    const Eigen::Vector3d axis = Eigen::Vector3d(1, -2, 0.5).normalized();
    double worst = 0;
    for (double a : {0.0, 1e-9, 0.1, 1.0, 3.0}) {
      const Eigen::Matrix3d E = ExpSO3(a * axis), Ref = Eigen::AngleAxisd(a, axis).matrix();
      worst = std::max(worst, (E - Ref).norm());
    }
    CHECK(worst < 1e-12, "ExpSO3 equals the axis-angle rotation (max |diff| %.1e)", worst);
  }

  // --- the Gauss-Newton direction uses the right Jacobian: one step from a pose
  //     0.5 deg / 5 mm off on noise-free data lands within 1e-4 of the truth ---
  const Data exact = Make(R, t, K, 120, 0.0, 3);
  const std::vector<bool> all(exact.X.size(), true);
  {
    const Eigen::Matrix3d R0 = ExpSO3(Eigen::Vector3d(0.004, -0.006, 0.003)) * R;
    const Eigen::Vector3d t0 = t + Eigen::Vector3d(0.003, -0.004, 0.002);
    RefineParams one;
    one.max_iterations = 1;
    one.lambda0 = 1e-12;
    const RefineResult f = RefinePnP(exact.X, exact.uv, all, K, R0, t0, one);
    CHECK(f.ok && f.iterations == 1 && RotErrDeg(R, f.R) < 1e-2 && (f.t - t).norm() < 1e-4,
          "one undamped step: %.2f deg / %.1f mm -> %.1e deg / %.1e m (rms %.2f -> %.1e px)", RotErrDeg(R, R0),
          1e3 * (t0 - t).norm(), RotErrDeg(R, f.R), (f.t - t).norm(), f.rms_initial_px, f.rms_final_px);
  }

  // --- noise-free data, a poor start (5 deg, 3 cm): converges to the exact pose ---
  {
    const Eigen::Matrix3d R0 = ExpSO3(Eigen::Vector3d(0.05, -0.06, 0.04)) * R;
    const Eigen::Vector3d t0 = t + Eigen::Vector3d(0.02, -0.015, 0.015);
    const RefineResult f = RefinePnP(exact.X, exact.uv, all, K, R0, t0);
    CHECK(f.ok && RotErrDeg(R, f.R) < 1e-6 && (f.t - t).norm() < 1e-8 && f.rms_final_px < 1e-6,
          "noise-free: from %.2f deg / %.3f m to %.1e deg / %.1e m in %d steps (rms %.1f -> %.1e px)",
          RotErrDeg(R, R0), (t0 - t).norm(), RotErrDeg(R, f.R), (f.t - t).norm(), f.iterations, f.rms_initial_px,
          f.rms_final_px);
    CHECK(std::abs(f.R.determinant() - 1) < 1e-12 && (f.R.transpose() * f.R - Eigen::Matrix3d::Identity()).norm() < 1e-12,
          "the refined R is a proper rotation");
  }

  // --- noisy data (1 px): refinement from the linear PnP, 40 trials ---
  {
    int better_rms = 0, never_worse = 0, n = 40;
    double lin_rot = 0, ref_rot = 0, lin_tr = 0, ref_tr = 0;
    for (int s = 0; s < n; ++s) {
      const Data d = Make(R, t, K, 150, 1.0, 100 + s);
      const PnPResult lin = solvePnPDLT(d.X, d.uv, K);
      const RefineResult f = RefinePnP(d.X, d.uv, std::vector<bool>(d.X.size(), true), K, lin.R, lin.t);
      better_rms += f.ok && Rms(d, f.R, f.t, K) < Rms(d, lin.R, lin.t, K);
      never_worse += f.cost_final <= f.cost_initial;
      lin_rot += RotErrDeg(R, lin.R) / n, ref_rot += RotErrDeg(R, f.R) / n;
      lin_tr += (lin.t - t).norm() / n, ref_tr += (f.t - t).norm() / n;
    }
    CHECK(never_worse == n && better_rms == n, "the reprojection error decreases in %d / %d trials", better_rms, n);
    CHECK(ref_rot < 0.8 * lin_rot && ref_tr < 0.8 * lin_tr,
          "pose error with 1 px noise, mean of %d trials: linear %.3f deg / %.2f mm -> refined %.3f deg / %.2f mm", n,
          lin_rot, 1e3 * lin_tr, ref_rot, 1e3 * ref_tr);
  }

  // --- only the selected points are used: gross outliers outside the mask change nothing ---
  {
    Data d = Make(R, t, K, 150, 1.0, 77);
    std::vector<bool> use(d.X.size(), true);
    Data in = d;
    for (int k = 0; k < 30; ++k) {
      d.X.push_back(Eigen::Vector3d(0.01 * k, 0.0, 0.5));
      d.uv.push_back(Eigen::Vector2d(10 + 5 * k, 400));  // wrong pixels
      use.push_back(false);
    }
    const PnPResult lin = solvePnPDLT(in.X, in.uv, K);
    const RefineResult a = RefinePnP(d.X, d.uv, use, K, lin.R, lin.t),
                       b = RefinePnP(in.X, in.uv, std::vector<bool>(in.X.size(), true), K, lin.R, lin.t);
    CHECK(a.ok && (a.R - b.R).norm() < 1e-12 && (a.t - b.t).norm() < 1e-12,
          "30 masked-out outliers: identical result to refining the 150 inliers alone");
  }

  // --- failures ---
  {
    const std::vector<Eigen::Vector3d> X2(exact.X.begin(), exact.X.begin() + 2);
    const std::vector<Eigen::Vector2d> u2(exact.uv.begin(), exact.uv.begin() + 2);
    const RefineResult few = RefinePnP(X2, u2, {true, true}, K, R, t);
    const RefineResult behind = RefinePnP(exact.X, exact.uv, all, K, R, t + Eigen::Vector3d(0, 0, -1.0));
    CHECK(!few.ok && few.R == R && !behind.ok, "fewer than 3 points, or points behind the camera: not refined, "
          "input pose returned");
  }

  // --- deterministic ---
  {
    const Data d = Make(R, t, K, 150, 1.0, 5);
    const PnPResult lin = solvePnPDLT(d.X, d.uv, K);
    const std::vector<bool> u(d.X.size(), true);
    const RefineResult a = RefinePnP(d.X, d.uv, u, K, lin.R, lin.t), b = RefinePnP(d.X, d.uv, u, K, lin.R, lin.t);
    CHECK(a.R == b.R && a.t == b.t && a.iterations == b.iterations, "deterministic (%d steps)", a.iterations);
  }
  return scratch_test::Finish("test_refine");
}
