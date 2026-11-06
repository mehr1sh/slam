#include "experiment.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <random>

#include <Eigen/Eigenvalues>

#include "pnp.hpp"
#include "projection.hpp"
#include "ransac.hpp"

namespace scratch::experiment {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<double> Residuals(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                              const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Intrinsics &K) {
  std::vector<double> r(X.size());
  for (size_t k = 0; k < X.size(); ++k) {
    double z;
    const Eigen::Vector2d p = projectPoint(X[k], R, t, K, &z);
    r[k] = z > 0 ? (p - uv[k]).norm() : kInf;
  }
  return r;
}

template <typename V>
V Select(const V &v, const std::vector<int> &idx) {
  V out;
  for (int i : idx) out.push_back(v[i]);
  return out;
}
}  // namespace

// ---------------------------------------------------------------- presets
const std::vector<std::string> &MatcherConfigIds() {
  static const std::vector<std::string> ids = {"M0", "M1", "M2", "M2n", "M3", "M4", "M5", "M1L", "M3L"};
  return ids;
}

bool ApplyMatcherConfig(const std::string &id, FrontendParams &f, std::string *error) {
  features::MatcherParams m;  // defaults: everything off, radius 3 px
  f.ranked_matcher = id != "M0";
  if (id == "M0") {
  } else if (id == "M1") {
    m.mutual = true;
  } else if (id == "M2") {
    m.ratio = 0.8;
  } else if (id == "M2n") {
    m.ratio = 0.8, m.same_location_radius_px = 0.0f;
  } else if (id == "M3") {
    m.mutual = true, m.ratio = 0.8;
  } else if (id == "M4") {
    m.mutual = true, m.ratio = 0.8, m.max_level_diff = 1;
  } else if (id == "M5") {
    m.mutual = true, m.ratio = 0.8, m.rank = true;
  } else if (id == "M1L") {
    m.mutual = m.mutual_location_aware = true;
  } else if (id == "M3L") {
    m.mutual = m.mutual_location_aware = true, m.ratio = 0.8;
  } else {
    if (error) *error = "unknown matcher configuration " + id;
    return false;
  }
  f.matcher = m;
  return true;
}

std::string DescribeMatcherConfig(const std::string &id) {
  static const std::map<std::string, std::string> d = {
      {"M0", "original matcher (MatchBruteForce + distance filter)"},
      {"M1", "mutual nearest neighbours (strict)"},
      {"M2", "location-aware ratio 0.8"},
      {"M2n", "naive ratio 0.8 (comparison only)"},
      {"M3", "mutual + location-aware ratio 0.8"},
      {"M4", "M3 + |level difference| <= 1"},
      {"M5", "M3 + quality order, ordinary uniform RANSAC"},
      {"M1L", "location-aware mutual nearest neighbours"},
      {"M3L", "location-aware mutual + location-aware ratio 0.8"}};
  const auto it = d.find(id);
  return it == d.end() ? "" : it->second;
}

// ---------------------------------------------------------------- integration configurations
const std::vector<std::string> &IntegrationConfigIds() {
  static const std::vector<std::string> ids = {
      "M0", "M1", "M2", "M2n", "M3", "M4", "M5", "M1L", "M3L", "L0", "T0", "T0G", "T0R", "T0M", "T0Q", "T0GR",
      "T0E", "T0H", "T0-harris", "T0-budget", "T0-fast20", "T0-mr", "T0-reselect", "T0-ransac", "T0C", "T0CG"};
  return ids;
}

bool ApplyIntegrationConfig(const std::string &id, PipelineParams &p, std::string *error) {
  p = PipelineParams{};
  if (!id.empty() && id[0] == 'M') {  // Stage-3C presets on the Stage-2 front end of main
    ApplyStage2Frontend(p);
    p.validity.enabled = true;
    return ApplyMatcherConfig(id, p.frontend, error);
  }
  if (id == "L0") return true;
  if (id.rfind("T0", 0) != 0) {
    if (error) *error = "unknown configuration " + id;
    return false;
  }
  ApplyImprovedDefaults(p);
  auto ranked = [&](bool mutual, double ratio) {
    p.frontend.ranked_matcher = true;
    p.frontend.matcher = features::MatcherParams();
    p.frontend.matcher.mutual = mutual, p.frontend.matcher.ratio = ratio;
  };
  if (id == "T0") {
  } else if (id == "T0G") {
    p.validity.enabled = true;
  } else if (id == "T0R") {
    ranked(true, 0.8);
  } else if (id == "T0M") {
    ranked(true, 0.0);
  } else if (id == "T0Q") {
    ranked(false, 0.8);
  } else if (id == "T0GR") {
    ranked(true, 0.8);
    p.validity.enabled = true;
  } else if (id == "T0E") {
    p.frontend.depth_edge_m = 0;
  } else if (id == "T0H") {
    p.refine_params.huber_delta_px = 0;
  } else if (id == "T0-harris") {
    p.frontend.selection.harris = false;
  } else if (id == "T0-budget") {
    p.frontend.selection.total_features = 0;
  } else if (id == "T0-fast20") {
    p.frontend.fast.threshold = 20;
  } else if (id == "T0-mr") {
    p.frontend.mutual_ratio = false;
  } else if (id == "T0-reselect") {
    p.reselect_inliers = false;
  } else if (id == "T0-ransac") {
    p.pnp_ransac.iterations = 300, p.pnp_ransac.threshold_px = 8.0;
  } else if (id == "T0C" || id == "T0CG") {  // candidate suggested by the two most robust ablations
    p.pnp_ransac.iterations = 300, p.pnp_ransac.threshold_px = 8.0;
    p.frontend.selection.total_features = 0;
    p.validity.enabled = id == "T0CG";
  } else {
    if (error) *error = "unknown configuration " + id;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- RANSAC trace
RansacTrace TraceRansacPnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                           const Intrinsics &K, const RansacParams &params) {
  // the algorithm of RansacPnP (ransac.cpp), step by step
  constexpr int kSample = 6;
  RansacTrace tr;
  const int n = int(X.size());
  if (n < kSample) return tr;
  std::mt19937 rng(params.seed);
  int best_inl = -1;
  double best_sum = kInf;
  std::vector<bool> best_mask;
  for (int it = 0; it < params.iterations; ++it) {
    const std::vector<int> s = SampleIndices(rng, n, kSample);
    const PnPResult m = solvePnPDLT(Select(X, s), Select(uv, s), K);
    if (!m.ok) continue;
    const std::vector<double> res = Residuals(X, uv, m.R, m.t, K);
    int inl = 0;
    double sum = 0;
    std::vector<bool> mask(n, false);
    for (int k = 0; k < n; ++k)
      if (res[k] < params.threshold_px) ++inl, sum += res[k], mask[k] = true;
    if (inl > best_inl || (inl == best_inl && sum < best_sum)) {
      best_inl = inl, best_sum = sum, best_mask = mask;
      tr.R_sample = m.R, tr.t_sample = m.t, tr.sample = s;
    }
  }
  if (best_inl < kSample) return tr;
  tr.ok = true;
  tr.sample_inliers = best_inl;
  std::vector<int> in;
  for (int k = 0; k < n; ++k)
    if (best_mask[k]) in.push_back(k);
  const PnPResult refit = solvePnPDLT(Select(X, in), Select(uv, in), K);
  tr.refit_ok = refit.ok;
  tr.R_refit = refit.ok ? refit.R : tr.R_sample;
  tr.t_refit = refit.ok ? refit.t : tr.t_sample;
  const std::vector<double> res = Residuals(X, uv, tr.R_refit, tr.t_refit, K);
  tr.refit_mask.assign(n, false);
  for (int k = 0; k < n; ++k)
    if (res[k] < params.threshold_px) tr.refit_mask[k] = true, ++tr.refit_inliers;
  return tr;
}

bool TraceMatches(const RansacTrace &t, const PairResult &r) {
  // RANSAC-stage success: ransac_mask is set exactly when RansacPnP returned a pose (pnp_ok can be cleared
  // afterwards by the teammate's inlier re-selection, which is not part of the RANSAC stage).
  const bool ransac_ok = !r.ransac_mask.empty();
  if (t.ok != ransac_ok) return false;
  if (!t.ok) return true;
  int ransac_inliers = 0;
  for (bool b : r.ransac_mask) ransac_inliers += b;
  return t.R_refit == r.R_linear && t.t_refit == r.t_linear && t.refit_mask == r.ransac_mask &&
         t.refit_inliers == ransac_inliers && t.sample_inliers == r.pnp_best_sample_inliers;
}

// ---------------------------------------------------------------- diagnostics
Eigen::Vector3d CovarianceEigenvalues(const std::vector<Eigen::Vector3d> &X) {
  if (X.size() < 2) return Eigen::Vector3d::Zero();
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto &x : X) mean += x;
  mean /= double(X.size());
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  for (const auto &x : X) C += (x - mean) * (x - mean).transpose();
  C /= double(X.size());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
  Eigen::Vector3d e = es.eigenvalues();  // ascending
  for (int k = 0; k < 3; ++k) e(k) = std::max(0.0, e(k));
  return e;
}

int CountLocationPairGroups(const std::vector<Eigen::Vector2d> &src, const std::vector<Eigen::Vector2d> &dst,
                            double radius) {
  const int n = int(src.size());
  std::vector<int> parent(n), order(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return src[a].x() < src[b].x(); });
  auto find = [&](int a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  };
  int groups = n;
  for (int ia = 0; ia < n; ++ia)
    for (int ib = ia + 1; ib < n; ++ib) {
      const int a = order[ia], b = order[ib];
      if (src[b].x() - src[a].x() > radius) break;
      if ((src[a] - src[b]).norm() <= radius && (dst[a] - dst[b]).norm() <= radius) {
        const int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb, --groups;
      }
    }
  return groups;
}

GateDiagnostics DiagnoseGate(const PairResult &r, const ValidityParams &P, double thr) {
  GateDiagnostics d;
  const PoseValidity &v = r.validity;
  d.v = v;
  d.c_finite = v.finite;
  d.c_so3 = v.rotation_valid;
  d.c_rows = v.inlier_rows >= P.min_inlier_rows;
  d.c_kp_src = v.distinct_src >= P.min_distinct_keypoints, d.c_kp_dst = v.distinct_dst >= P.min_distinct_keypoints;
  d.c_loc_src = v.distinct_loc_src >= P.min_distinct_locations;
  d.c_loc_dst = v.distinct_loc_dst >= P.min_distinct_locations;
  d.c_ratio = v.inlier_ratio >= P.min_inlier_ratio;
  d.c_cells_src = v.cells_src >= P.min_coverage_cells, d.c_cells_dst = v.cells_dst >= P.min_coverage_cells;
  d.c_median = v.reproj_median_px <= P.max_reproj_median_factor * thr;
  d.c_p90 = v.reproj_p90_px <= P.max_reproj_p90_factor * thr;

  std::vector<Eigen::Vector2d> is, id, ns, nd;
  std::vector<Eigen::Vector3d> Xin;
  std::map<std::pair<long, long>, int> cell_rows;
  for (size_t k = 0; k < r.pnp_index.size(); ++k) {
    const Correspondence &c = r.correspondences[r.pnp_index[k]];
    is.push_back(c.uv_i), id.push_back(c.uv_j);
    if (r.pnp_ok && r.pnp_mask[k]) {
      ns.push_back(c.uv_i), nd.push_back(c.uv_j), Xin.push_back(c.X_i);
      ++cell_rows[{long(std::floor(c.uv_j.x() / P.coverage_cell_px)), long(std::floor(c.uv_j.y() / P.coverage_cell_px))}];
    }
  }
  d.input_loc_src = CountLocationGroups(is, P.location_radius_px);
  d.input_loc_dst = CountLocationGroups(id, P.location_radius_px);
  d.input_pairs = CountLocationPairGroups(is, id, P.location_radius_px);
  d.inlier_pairs = CountLocationPairGroups(ns, nd, P.location_radius_px);
  d.input_cells_dst = CountOccupiedCells(id, P.coverage_cell_px);
  d.ratio_rows = v.inlier_ratio;
  d.ratio_pairs = d.input_pairs ? double(d.inlier_pairs) / d.input_pairs : 0;
  d.ratio_loc_dst = d.input_loc_dst ? double(v.distinct_loc_dst) / d.input_loc_dst : 0;
  d.ratio_cells_dst = d.input_cells_dst ? double(v.cells_dst) / d.input_cells_dst : 0;
  d.rows_per_location = v.distinct_loc_dst ? double(v.inlier_rows) / v.distinct_loc_dst : 0;
  int largest = 0;
  for (const auto &kv : cell_rows) largest = std::max(largest, kv.second);
  d.largest_cell_share = v.inlier_rows ? double(largest) / v.inlier_rows : 0;
  const Eigen::Vector3d e = CovarianceEigenvalues(Xin);
  d.eig1 = e(0), d.eig2 = e(1), d.eig3 = e(2);
  d.condition = e(0) > 0 ? std::sqrt(e(2) / e(0)) : kInf;
  d.planarity = e(2) > 0 ? e(0) / e(2) : 0;
  if (!Xin.empty()) {
    std::vector<double> z;
    for (const auto &x : Xin) z.push_back(x.z());
    d.depth_min = *std::min_element(z.begin(), z.end()), d.depth_max = *std::max_element(z.begin(), z.end());
    d.depth_median = Percentile(z, 50);
  }
  return d;
}

// ---------------------------------------------------------------- truth (offline only)
TruthEval EvaluateWithTruth(const PairResult &r, const RansacTrace &trace, const Pose &G, const Intrinsics &K,
                            double thr) {
  TruthEval t;
  if (r.pnp_ok) t.rot_err_deg = RotErrDeg(G.R, r.R), t.trans_err_m = (G.t - r.t).norm();
  if (trace.ok) {
    t.sample_rot_err_deg = RotErrDeg(G.R, trace.R_sample), t.sample_trans_err_m = (G.t - trace.t_sample).norm();
    t.refit_rot_err_deg = RotErrDeg(G.R, trace.R_refit), t.refit_trans_err_m = (G.t - trace.t_refit).norm();
  }
  int inl = 0;
  for (size_t k = 0; k < r.pnp_index.size(); ++k) {
    const Correspondence &c = r.correspondences[r.pnp_index[k]];
    double z;
    const Eigen::Vector2d p = projectPoint(c.X_i, G.R, G.t, K, &z);
    const bool ok = z > 0 && (p - c.uv_j).norm() < thr;
    t.correct_inputs += ok;
    if (r.pnp_ok && r.pnp_mask[k]) ++inl, t.correct_inliers += ok;
  }
  if (!r.pnp_index.empty()) t.correct_input_share = double(t.correct_inputs) / r.pnp_index.size();
  if (inl) t.correct_inlier_share = double(t.correct_inliers) / inl;
  return t;
}

}  // namespace scratch::experiment
