#include "validity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <set>
#include <utility>

#include <Eigen/Dense>

namespace scratch {

const char *PoseStatusName(PoseStatus s) {
  switch (s) {
    case PoseStatus::kAccepted: return "accepted";
    case PoseStatus::kRejected: return "rejected";
    default: return "solver_failed";
  }
}

int CountDistinctIds(std::vector<int> ids) {
  std::sort(ids.begin(), ids.end());
  return int(std::unique(ids.begin(), ids.end()) - ids.begin());
}

int CountLocationGroups(const std::vector<Eigen::Vector2d> &pts, double radius) {
  const int n = int(pts.size());
  std::vector<int> parent(n), order(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pts[a].x() < pts[b].x(); });
  auto find = [&](int a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  };
  int groups = n;
  for (int ia = 0; ia < n; ++ia)
    for (int ib = ia + 1; ib < n; ++ib) {
      const int a = order[ia], b = order[ib];
      if (pts[b].x() - pts[a].x() > radius) break;  // sorted by x: no later point can be within the radius
      if ((pts[a] - pts[b]).norm() <= radius) {
        const int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb, --groups;
      }
    }
  return groups;
}

int CountOccupiedCells(const std::vector<Eigen::Vector2d> &pts, int cell_px) {
  std::set<std::pair<long, long>> cells;
  for (const auto &p : pts)
    cells.insert({long(std::floor(p.x() / cell_px)), long(std::floor(p.y() / cell_px))});
  return int(cells.size());
}

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * double(v.size() - 1);
  const size_t lo = size_t(std::floor(pos)), hi = std::min(lo + 1, v.size() - 1);
  const double frac = pos - double(lo);
  if (frac == 0 || v[hi] == v[lo]) return v[lo];  // also avoids 0 * inf and inf - inf (residuals behind the camera)
  if (std::isinf(v[hi])) return v[hi];
  return v[lo] + frac * (v[hi] - v[lo]);
}

PoseValidity EvaluatePose(bool solver_ok, const std::string &solver_reason, const Eigen::Matrix3d &R,
                          const Eigen::Vector3d &t, const std::vector<ValidityRow> &rows, double threshold_px,
                          const ValidityParams &params) {
  PoseValidity v;
  v.pnp_inputs = int(rows.size());
  if (!solver_ok) {
    v.status = PoseStatus::kSolverFailed;
    v.reason = solver_reason.empty() ? "solver returned no pose" : solver_reason;
    return v;
  }
  // --- the evidence ---
  v.finite = R.allFinite() && t.allFinite();
  if (v.finite) {
    v.orthogonality_error = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();
    v.det_error = std::abs(R.determinant() - 1.0);
    v.rotation_valid = v.orthogonality_error <= params.rotation_tolerance && v.det_error <= params.rotation_tolerance;
  }
  std::vector<int> src, dst;
  std::vector<Eigen::Vector2d> uv_src, uv_dst;
  std::vector<double> res;
  for (const ValidityRow &r : rows) {
    if (!r.inlier) continue;
    src.push_back(r.src), dst.push_back(r.dst);
    uv_src.push_back(r.uv_src), uv_dst.push_back(r.uv_dst);
    res.push_back(std::isfinite(r.residual_px) ? r.residual_px : std::numeric_limits<double>::infinity());
  }
  v.inlier_rows = int(src.size());
  v.distinct_src = CountDistinctIds(src), v.distinct_dst = CountDistinctIds(dst);
  v.distinct_loc_src = CountLocationGroups(uv_src, params.location_radius_px);
  v.distinct_loc_dst = CountLocationGroups(uv_dst, params.location_radius_px);
  v.inlier_ratio = rows.empty() ? 0.0 : double(v.inlier_rows) / double(rows.size());
  v.cells_src = CountOccupiedCells(uv_src, params.coverage_cell_px);
  v.cells_dst = CountOccupiedCells(uv_dst, params.coverage_cell_px);
  v.reproj_median_px = Percentile(res, 50), v.reproj_p90_px = Percentile(res, 90);

  // --- the decision ---
  std::vector<std::string> fail;
  char buf[160];
  auto need = [&](bool ok, const char *fmt, double have, double limit) {
    if (ok) return;
    std::snprintf(buf, sizeof(buf), fmt, have, limit);
    fail.push_back(buf);
  };
  if (!v.finite) fail.push_back("non-finite pose");
  else need(v.rotation_valid, "R not in SO(3) (orthogonality error %.3g, tolerance %.3g)",
            std::max(v.orthogonality_error, v.det_error), params.rotation_tolerance);
  need(v.inlier_rows >= params.min_inlier_rows, "%.0f inlier rows < %.0f", v.inlier_rows, params.min_inlier_rows);
  need(v.distinct_src >= params.min_distinct_keypoints, "%.0f distinct frame-i inlier keypoints < %.0f",
       v.distinct_src, params.min_distinct_keypoints);
  need(v.distinct_dst >= params.min_distinct_keypoints, "%.0f distinct frame-j inlier keypoints < %.0f",
       v.distinct_dst, params.min_distinct_keypoints);
  need(v.distinct_loc_src >= params.min_distinct_locations, "%.0f distinct frame-i inlier locations < %.0f",
       v.distinct_loc_src, params.min_distinct_locations);
  need(v.distinct_loc_dst >= params.min_distinct_locations, "%.0f distinct frame-j inlier locations < %.0f",
       v.distinct_loc_dst, params.min_distinct_locations);
  need(v.inlier_ratio >= params.min_inlier_ratio, "inlier ratio %.4f < %.4f", v.inlier_ratio, params.min_inlier_ratio);
  need(v.cells_src >= params.min_coverage_cells, "frame-i inliers cover %.0f cells < %.0f", v.cells_src,
       params.min_coverage_cells);
  need(v.cells_dst >= params.min_coverage_cells, "frame-j inliers cover %.0f cells < %.0f", v.cells_dst,
       params.min_coverage_cells);
  const double med_limit = params.max_reproj_median_factor * threshold_px,
               p90_limit = params.max_reproj_p90_factor * threshold_px;
  need(v.reproj_median_px <= med_limit, "inlier reprojection median %.3f px > %.3f px", v.reproj_median_px, med_limit);
  need(v.reproj_p90_px <= p90_limit, "inlier reprojection p90 %.3f px > %.3f px", v.reproj_p90_px, p90_limit);

  for (size_t k = 0; k < fail.size(); ++k) v.reason += (k ? "; " : "") + fail[k];
  v.status = fail.empty() ? PoseStatus::kAccepted : PoseStatus::kRejected;
  return v;
}

}  // namespace scratch
