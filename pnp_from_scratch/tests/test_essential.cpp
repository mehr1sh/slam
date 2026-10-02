// Self-checking test of the scratch essential-matrix module: 8-point
// estimate, constraint enforcement, decomposition, cheirality, pose
// recovery. Synthetic scenes with a known motion X_2 = R X_1 + t.

#include <algorithm>
#include <cmath>
#include <random>

#include <Eigen/Geometry>

#include "check.hpp"
#include "essential.hpp"

using namespace scratch;

namespace {

double RotErr(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) { return RotErrDeg(A, B); }

double DirErrDeg(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)) * 180.0 / M_PI;
}

struct Scene {
  PoseRt truth;
  Points2 x1, x2;
};

// n random points 0.4-0.6 m in front of camera 1, seen by camera 2 = (R, t).
Scene MakeScene(int n, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double noise, uint32_t seed) {
  std::mt19937 rng(seed);
  auto U = [&] { return (double(rng()) + 0.5) / 4294967296.0; };
  auto N = [&] { return std::sqrt(-2 * std::log(U())) * std::cos(2 * M_PI * U()); };
  Scene s;
  s.truth.R = R;
  s.truth.t = t;
  while (int(s.x1.size()) < n) {
    const Eigen::Vector3d X1(0.3 * (U() - 0.5), 0.3 * (U() - 0.5), 0.4 + 0.2 * U());
    const Eigen::Vector3d X2 = R * X1 + t;
    if (X2.z() <= 0.05) continue;
    s.x1.emplace_back(X1.x() / X1.z() + noise * N(), X1.y() / X1.z() + noise * N());
    s.x2.emplace_back(X2.x() / X2.z() + noise * N(), X2.y() / X2.z() + noise * N());
  }
  return s;
}

// E up to scale and sign vs [t]_x R
double EssentialDistance(const Eigen::Matrix3d &E, const PoseRt &P) {
  const Eigen::Matrix3d T = Skew(P.t) * P.R;
  const Eigen::Matrix3d a = E / E.norm(), b = T / T.norm();
  return std::min((a - b).norm(), (a + b).norm());
}

}  // namespace

int main() {
  std::printf("=== scratch essential matrix ===\n");
  // the Bunny-like motion: 10 deg about a tilted axis, 8.7 cm baseline
  const Eigen::Matrix3d R = Eigen::AngleAxisd(10 * M_PI / 180, Eigen::Vector3d(0, 0.97, 0.24).normalized()).matrix();
  const Eigen::Vector3d t(-0.0869, -0.0018, 0.0074);

  // --- noise-free: exact recovery ---
  const Scene s = MakeScene(60, R, t, 0.0, 1);
  Eigen::Matrix3d E;
  CHECK(EstimateEssential8Point(s.x1, s.x2, E), "8-point estimate from 60 exact correspondences");
  CHECK(EssentialDistance(E, s.truth) < 1e-9, "E equals [t]x R up to scale/sign (distance %.1e)", EssentialDistance(E, s.truth));
  {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
    const auto sv = svd.singularValues();
    CHECK(std::abs(sv(0) - sv(1)) < 1e-12 && std::abs(sv(2)) < 1e-12, "singular values (s, s, 0): %.3g %.3g %.1e", sv(0),
          sv(1), sv(2));
  }
  double worst_epi = 0;
  for (size_t k = 0; k < s.x1.size(); ++k)
    worst_epi = std::max(worst_epi, std::abs(Eigen::Vector3d(s.x2[k].x(), s.x2[k].y(), 1).dot(
                                            E * Eigen::Vector3d(s.x1[k].x(), s.x1[k].y(), 1))));
  CHECK(worst_epi < 1e-12, "epipolar constraint x2^T E x1 = 0 (max |residual| %.1e)", worst_epi);
  {
    Eigen::Matrix3d noisy = E + 0.01 * Eigen::Matrix3d::Random();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(EnforceEssentialConstraints(noisy));
    CHECK(std::abs(svd.singularValues()(0) - svd.singularValues()(1)) < 1e-12 && svd.singularValues()(2) < 1e-12,
          "constraint enforcement maps any 3x3 matrix to (s, s, 0)");
  }

  // --- decomposition ---
  const auto cands = DecomposeEssential(E);
  int matches_truth = -1;
  bool proper = true, consistent = true;
  for (int c = 0; c < 4; ++c) {
    proper &= std::abs(cands[c].R.determinant() - 1) < 1e-12 && std::abs(cands[c].t.norm() - 1) < 1e-12;
    consistent &= EssentialDistance(E, cands[c]) < 1e-9;
    if (RotErr(cands[c].R, R) < 1e-6 && DirErrDeg(cands[c].t, t) < 1e-6) matches_truth = c;
  }
  CHECK(proper, "4 candidates: det R = +1, |t| = 1");
  CHECK(consistent, "every candidate reproduces E (up to scale/sign)");
  CHECK(matches_truth >= 0, "the true (R, t/|t|) is one of the 4 candidates (#%d)", matches_truth);

  // --- cheirality ---
  RecoveredPose rp;
  CHECK(RecoverPoseFromEssential(E, s.x1, s.x2, rp), "pose recovered");
  CHECK(rp.chosen == matches_truth && rp.in_front == 60, "cheirality picks the true candidate: in front %d / %d / %d / %d",
        rp.candidate_in_front[0], rp.candidate_in_front[1], rp.candidate_in_front[2], rp.candidate_in_front[3]);
  int others_max = 0;
  for (int c = 0; c < 4; ++c)
    if (c != rp.chosen) others_max = std::max(others_max, rp.candidate_in_front[c]);
  CHECK(others_max < 30, "the other three candidates put most points behind a camera (best of them: %d / 60)", others_max);
  CHECK(RotErr(rp.pose.R, R) < 1e-6 && DirErrDeg(rp.pose.t, t) < 1e-6,
        "recovered R error %.1e deg, t direction error %.1e deg", RotErr(rp.pose.R, R), DirErrDeg(rp.pose.t, t));
  {
    double d1, d2;
    TriangulateDepths(s.truth, s.x1[0], s.x2[0], d1, d2);
    const Eigen::Vector3d X1 = d1 * Eigen::Vector3d(s.x1[0].x(), s.x1[0].y(), 1);
    CHECK((R * X1 + t - d2 * Eigen::Vector3d(s.x2[0].x(), s.x2[0].y(), 1)).norm() < 1e-12 && d1 > 0.39 && d1 < 0.61,
          "triangulated depths satisfy X_2 = R X_1 + t (depth %.3f m)", d1);
  }

  // --- minimal and degenerate inputs ---
  {
    const Scene m = MakeScene(8, R, t, 0.0, 2);
    Eigen::Matrix3d E8;
    CHECK(EstimateEssential8Point(m.x1, m.x2, E8) && EssentialDistance(E8, m.truth) < 1e-8,
          "exactly 8 points suffice");
    Points2 a(m.x1.begin(), m.x1.begin() + 7), b(m.x2.begin(), m.x2.begin() + 7);
    CHECK(!EstimateEssential8Point(a, b, E8), "7 points are rejected");
  }

  // --- noise: ~1 px at f = 520 ---
  {
    const Scene n = MakeScene(200, R, t, 1.0 / 520.0, 3);
    Eigen::Matrix3d En;
    RecoveredPose rn;
    const bool ok = EstimateEssential8Point(n.x1, n.x2, En) && RecoverPoseFromEssential(En, n.x1, n.x2, rn);
    CHECK(ok && RotErr(rn.pose.R, R) < 1.0 && DirErrDeg(rn.pose.t, t) < 10.0,
          "1 px noise, 200 points: rotation error %.3f deg, t direction error %.2f deg", RotErr(rn.pose.R, R),
          DirErrDeg(rn.pose.t, t));
    double s_in = 0;
    for (size_t k = 0; k < n.x1.size(); ++k) s_in += SampsonDistance(En, n.x1[k], n.x2[k]);
    s_in = std::sqrt(s_in / n.x1.size()) * 520.0;
    CHECK(s_in < 3.0, "RMS Sampson distance %.2f px (noise-level)", s_in);
    // The motion is mostly along x, so epipolar lines are nearly horizontal: a
    // mismatch ACROSS them (along y) is detectable, one ALONG them is not.
    const double across = std::sqrt(SampsonDistance(En, n.x1[0], n.x2[0] + Eigen::Vector2d(0, 30.0 / 520))) * 520;
    const double along = std::sqrt(SampsonDistance(En, n.x1[0], n.x2[0] + Eigen::Vector2d(30.0 / 520, 0))) * 520;
    CHECK(across > 10.0 && along < across, "a 30 px shift across the epipolar line: Sampson distance %.1f px "
          "(along the line: %.1f px)", across, along);
  }
  return scratch_test::Finish("test_essential");
}
