#include "ransac.hpp"

#include <cmath>
#include <limits>
#include <numeric>

#include "pnp.hpp"

namespace scratch {

std::vector<int> SampleIndices(std::mt19937 &rng, int n, int k) {
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  for (int i = 0; i < k; ++i) {
    const int j = i + int(rng() % uint32_t(n - i));
    std::swap(idx[i], idx[j]);
  }
  idx.resize(k);
  return idx;
}

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// One RANSAC hypothesis evaluation: inlier mask, count and residual sum.
struct Score {
  int inliers = -1;
  double sum = kInf;
  bool better_than(const Score &o) const { return inliers > o.inliers || (inliers == o.inliers && sum < o.sum); }
};

Score Evaluate(const std::vector<double> &res, double thr, std::vector<bool> *mask = nullptr) {
  Score s;
  s.inliers = 0;
  s.sum = 0;
  if (mask) mask->assign(res.size(), false);
  for (size_t k = 0; k < res.size(); ++k)
    if (res[k] < thr) {
      ++s.inliers;
      s.sum += res[k];
      if (mask) (*mask)[k] = true;
    }
  return s;
}

std::vector<double> ReprojectionResiduals(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                                          const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Intrinsics &K) {
  std::vector<double> r(X.size());
  for (size_t k = 0; k < X.size(); ++k) {
    double z;
    const Eigen::Vector2d p = projectPoint(X[k], R, t, K, &z);
    r[k] = z > 0 ? (p - uv[k]).norm() : kInf;  // behind the camera: never an inlier
  }
  return r;
}

template <typename V>
V Select(const V &v, const std::vector<int> &idx) {
  V out;
  out.reserve(idx.size());
  for (int i : idx) out.push_back(v[i]);
  return out;
}

std::vector<int> MaskIndices(const std::vector<bool> &mask) {
  std::vector<int> idx;
  for (size_t k = 0; k < mask.size(); ++k)
    if (mask[k]) idx.push_back(int(k));
  return idx;
}

}  // namespace

RansacPnPResult RansacPnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                          const Intrinsics &K, const RansacParams &params) {
  constexpr int kSample = 6;  // minimal sample of the linear (DLT) PnP
  RansacPnPResult out;
  const int n = int(X.size());
  if (n < kSample || uv.size() != X.size()) {
    out.reason = "need at least 6 correspondences (got " + std::to_string(n) + ")";
    return out;
  }
  std::mt19937 rng(params.seed);
  Score best;
  Eigen::Matrix3d bestR;
  Eigen::Vector3d bestt;
  std::vector<bool> best_mask;
  for (int it = 0; it < params.iterations; ++it) {
    const std::vector<int> s = SampleIndices(rng, n, kSample);
    const PnPResult m = solvePnPDLT(Select(X, s), Select(uv, s), K);
    if (!m.ok) continue;
    ++out.models_tried;
    std::vector<bool> mask;
    const Score sc = Evaluate(ReprojectionResiduals(X, uv, m.R, m.t, K), params.threshold_px, &mask);
    if (sc.better_than(best)) {
      best = sc;
      bestR = m.R;
      bestt = m.t;
      best_mask = mask;
    }
  }
  if (best.inliers < kSample) {
    out.reason = "no model with at least 6 inliers (best " + std::to_string(std::max(0, best.inliers)) + ")";
    return out;
  }
  out.best_sample_inliers = best.inliers;
  // refit the scratch DLT on all inliers of the best model
  const std::vector<int> in = MaskIndices(best_mask);
  const PnPResult refit = solvePnPDLT(Select(X, in), Select(uv, in), K);
  out.refit_used = refit.ok;
  out.R = refit.ok ? refit.R : bestR;
  out.t = refit.ok ? refit.t : bestt;
  out.residual_px = ReprojectionResiduals(X, uv, out.R, out.t, K);
  out.num_inliers = Evaluate(out.residual_px, params.threshold_px, &out.inlier_mask).inliers;
  out.ok = true;
  return out;
}

RansacEssentialResult RansacEssential(const Points2 &x1, const Points2 &x2, double focal_px,
                                      const RansacParams &params) {
  constexpr int kSample = 8;
  RansacEssentialResult out;
  const int n = int(x1.size());
  if (n < kSample || x2.size() != x1.size()) {
    out.reason = "need at least 8 correspondences (got " + std::to_string(n) + ")";
    return out;
  }
  auto residuals = [&](const Eigen::Matrix3d &E) {
    std::vector<double> r(n);
    for (int k = 0; k < n; ++k) r[k] = std::sqrt(SampsonDistance(E, x1[k], x2[k])) * focal_px;  // ~pixels
    return r;
  };
  std::mt19937 rng(params.seed);
  Score best;
  Eigen::Matrix3d bestE;
  std::vector<bool> best_mask;
  for (int it = 0; it < params.iterations; ++it) {
    const std::vector<int> s = SampleIndices(rng, n, kSample);
    Eigen::Matrix3d E;
    if (!EstimateEssential8Point(Select(x1, s), Select(x2, s), E)) continue;
    ++out.models_tried;
    std::vector<bool> mask;
    const Score sc = Evaluate(residuals(E), params.threshold_px, &mask);
    if (sc.better_than(best)) {
      best = sc;
      bestE = E;
      best_mask = mask;
    }
  }
  if (best.inliers < kSample) {
    out.reason = "no model with at least 8 inliers";
    return out;
  }
  out.best_sample_inliers = best.inliers;
  const std::vector<int> in = MaskIndices(best_mask);
  Eigen::Matrix3d E;
  if (!EstimateEssential8Point(Select(x1, in), Select(x2, in), E)) E = bestE;
  out.E = E;
  out.num_inliers = Evaluate(residuals(E), params.threshold_px, &out.inlier_mask).inliers;
  const std::vector<int> fin = MaskIndices(out.inlier_mask);
  if (!RecoverPoseFromEssential(E, Select(x1, fin), Select(x2, fin), out.pose)) {
    out.reason = "cheirality: no candidate has points in front of both cameras";
    return out;
  }
  out.ok = true;
  return out;
}

}  // namespace scratch
