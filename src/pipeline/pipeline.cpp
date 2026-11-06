#include "pipeline.hpp"
#include "geometry/fundamental.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scratch {

namespace {

double Median(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// mean / median / max of the residuals selected by the mask
void InlierStats(const std::vector<double> &res, const std::vector<bool> &mask,
                 double &mean, double &median, double &max) {
  std::vector<double> v;
  for (size_t k = 0; k < res.size(); ++k)
    if (mask[k])
      v.push_back(res[k]);
  if (v.empty())
    return;
  double s = 0;
  for (double x : v)
    s += x;
  mean = s / v.size();
  median = Median(v);
  max = *std::max_element(v.begin(), v.end());
}

} // namespace

void ApplyImprovedDefaults(PipelineParams &params) {
  params.frontend.selection = {true, 1400};
  params.frontend.fast.threshold = 10;
  params.frontend.mutual_ratio = true;
  params.frontend.depth_edge_m = .012;
  params.refine_params.huber_delta_px = 2.;
  params.reselect_inliers = true;
  params.pnp_ransac.iterations = 650;
  params.pnp_ransac.threshold_px = 3.;
}

void ApplyStage2Frontend(PipelineParams &params) {
  params = PipelineParams{};
  params.frontend.fast.cell_size = 30;
  params.frontend.fast.min_threshold = 7;
}

PairResult ProcessPair(int i, int j, const FrameFeatures &fi,
                       const FrameFeatures &fj, const RgbdFrame &frame_i,
                       const RgbdFrame &frame_j, const Intrinsics &K,
                       const PipelineParams &params) {
  PairResult r;
  r.i = i;
  r.j = j;
  r.keypoints_i = int(fi.keypoints.size());
  r.keypoints_j = int(fj.keypoints.size());
  PairMatches pm =
      MatchFrames(fi, fj, frame_i.depth, frame_j.depth, K, params.frontend);
  r.raw_matches = int(pm.raw.size());
  r.filtered_matches = int(pm.filtered.size());
  r.d_min = pm.d_min;
  r.d_max = pm.d_max;
  r.correspondences = std::move(pm.correspondences);
  r.ranked_matcher = pm.ranked;
  r.match_details = std::move(pm.ranked_raw);
  r.match_counters = pm.counters;

  // --- two-view stage: essential matrix on all filtered matches ---
  Points2 x1, x2;
  for (const Correspondence &c : r.correspondences) {
    x1.emplace_back((c.uv_i.x() - K.cx) / K.fx, (c.uv_i.y() - K.cy) / K.fy);
    x2.emplace_back((c.uv_j.x() - K.cx) / K.fx, (c.uv_j.y() - K.cy) / K.fy);
  }
  Points2 pixels1, pixels2;
  for (const auto &c : r.correspondences) {
    pixels1.push_back(c.uv_i);
    pixels2.push_back(c.uv_j);
  }
  r.fundamental_ok =
      geometry_scratch::EstimateFundamental8Point(pixels1, pixels2, r.F);
  const RansacEssentialResult e =
      RansacEssential(x1, x2, K.fx, params.essential_ransac);
  r.essential_ok = e.ok;
  r.essential_reason = e.reason;
  r.essential_inliers = e.num_inliers;
  r.essential_mask = e.inlier_mask;
  if (e.ok) {
    r.R_essential = e.pose.pose.R;
    r.t_essential_dir = e.pose.pose.t;
    r.essential_in_front = e.pose.in_front;
  }

  // --- metric stage: RANSAC around the scratch linear PnP ---
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
  for (size_t k = 0; k < r.correspondences.size(); ++k)
    if (r.correspondences[k].has_3d) {
      r.pnp_index.push_back(int(k));
      X.push_back(r.correspondences[k].X_i);
      uv.push_back(r.correspondences[k].uv_j);
    }
  const RansacPnPResult p = RansacPnP(X, uv, K, params.pnp_ransac);
  r.pnp_ok = p.ok;
  r.pnp_reason = p.reason;
  if (!p.ok) {
    r.validity = EvaluatePose(false, p.reason, r.R, r.t, {},
                              params.pnp_ransac.threshold_px, params.validity);
    r.validity.pnp_inputs = int(X.size());
    return r;
  }
  r.R = r.R_linear = p.R;
  r.t = r.t_linear = p.t;
  r.pnp_inliers = p.num_inliers;
  r.pnp_best_sample_inliers = p.best_sample_inliers;
  r.pnp_mask =
      p.inlier_mask; // the inlier set; the refinement does not change it
  r.ransac_mask = p.inlier_mask;
  r.pnp_residual_px = p.residual_px;
  InlierStats(p.residual_px, p.inlier_mask, r.linear_reproj_inlier_mean,
              r.linear_reproj_inlier_median, r.linear_reproj_inlier_max);
  if (params.refine) {
    const RefineResult f =
        RefinePnP(X, uv, p.inlier_mask, K, p.R, p.t, params.refine_params);
    r.refine_rms_initial_px = f.rms_initial_px;
    r.refine_rms_final_px = f.rms_final_px;
    if (f.ok) {
      r.refined = true;
      r.refine_iterations = f.iterations;
      r.R = f.R;
      r.t = f.t;
      for (size_t k = 0; k < X.size(); ++k) {
        double z;
        const Eigen::Vector2d q = projectPoint(X[k], r.R, r.t, K, &z);
        r.pnp_residual_px[k] = z > 0 ? (q - uv[k]).norm()
                                     : std::numeric_limits<double>::infinity();
      }
    }
  }
  if (params.reselect_inliers && r.refined) {
    for (size_t i = 0; i < r.pnp_mask.size(); ++i)
      r.pnp_mask[i] = r.pnp_residual_px[i] < params.pnp_ransac.threshold_px;
    auto refined =
        RefinePnP(X, uv, r.pnp_mask, K, r.R, r.t, params.refine_params);
    if (refined.ok) {
      r.R = refined.R;
      r.t = refined.t;
      r.refine_iterations += refined.iterations;
    }
    r.pnp_inliers = 0;
    for (size_t i = 0; i < X.size(); ++i) {
      double z;
      auto q = projectPoint(X[i], r.R, r.t, K, &z);
      r.pnp_residual_px[i] = z > 0 ? (q - uv[i]).norm() : INFINITY;
      r.pnp_mask[i] = r.pnp_residual_px[i] < params.pnp_ransac.threshold_px;
      r.pnp_inliers += r.pnp_mask[i];
    }
    if (r.pnp_inliers < 6) {
      r.pnp_ok = false;
      r.pnp_reason = "Insufficient refined support";
    }
  }
  InlierStats(r.pnp_residual_px, r.pnp_mask, r.reproj_inlier_mean,
              r.reproj_inlier_median, r.reproj_inlier_max);
  std::vector<double> all;
  for (double v : r.pnp_residual_px)
    if (std::isfinite(v))
      all.push_back(v);
  r.reproj_all_median = Median(all);

  // --- pose-validity gate (ported from main): the FINAL pose and mask (after inlier
  // re-selection), the correspondences, nothing else ---
  std::vector<ValidityRow> rows(r.pnp_index.size());
  for (size_t k = 0; k < rows.size(); ++k) {
    const Correspondence &c = r.correspondences[r.pnp_index[k]];
    rows[k].src = c.query, rows[k].dst = c.train;
    rows[k].uv_src = c.uv_i, rows[k].uv_dst = c.uv_j;
    rows[k].residual_px = r.pnp_residual_px[k];
    rows[k].inlier = r.pnp_mask[k];
  }
  r.validity = EvaluatePose(r.pnp_ok, r.pnp_reason, r.R, r.t, rows,
                            params.pnp_ransac.threshold_px, params.validity);
  r.validity.pnp_inputs = int(rows.size());
  r.pose_accepted = r.pnp_ok && (!params.validity.enabled || r.validity.accepted());
  return r;
}

} // namespace scratch
