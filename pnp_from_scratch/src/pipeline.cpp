#include "pipeline.hpp"

#include <algorithm>
#include <cmath>

namespace scratch {

namespace {

double Median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

}  // namespace

PairResult ProcessPair(int i, int j, const FrameFeatures &fi, const FrameFeatures &fj, const RgbdFrame &frame_i,
                       const RgbdFrame &frame_j, const Intrinsics &K, const PipelineParams &params) {
  PairResult r;
  r.i = i;
  r.j = j;
  r.keypoints_i = int(fi.keypoints.size());
  r.keypoints_j = int(fj.keypoints.size());
  PairMatches pm = MatchFrames(fi, fj, frame_i.depth, frame_j.depth, K, params.frontend);
  r.raw_matches = int(pm.raw.size());
  r.filtered_matches = int(pm.filtered.size());
  r.d_min = pm.d_min;
  r.d_max = pm.d_max;
  r.correspondences = std::move(pm.correspondences);

  // --- two-view stage: essential matrix on all filtered matches ---
  Points2 x1, x2;
  for (const Correspondence &c : r.correspondences) {
    x1.emplace_back((c.uv_i.x() - K.cx) / K.fx, (c.uv_i.y() - K.cy) / K.fy);
    x2.emplace_back((c.uv_j.x() - K.cx) / K.fx, (c.uv_j.y() - K.cy) / K.fy);
  }
  const RansacEssentialResult e = RansacEssential(x1, x2, K.fx, params.essential_ransac);
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
  if (!p.ok) return r;
  r.R = p.R;
  r.t = p.t;
  r.pnp_inliers = p.num_inliers;
  r.pnp_best_sample_inliers = p.best_sample_inliers;
  r.pnp_mask = p.inlier_mask;
  r.pnp_residual_px = p.residual_px;
  std::vector<double> inl, all;
  for (size_t k = 0; k < X.size(); ++k) {
    if (std::isfinite(p.residual_px[k])) all.push_back(p.residual_px[k]);
    if (p.inlier_mask[k]) inl.push_back(p.residual_px[k]);
  }
  if (!inl.empty()) {
    double s = 0;
    for (double v : inl) s += v;
    r.reproj_inlier_mean = s / inl.size();
    r.reproj_inlier_median = Median(inl);
    r.reproj_inlier_max = *std::max_element(inl.begin(), inl.end());
  }
  r.reproj_all_median = Median(all);
  return r;
}

}  // namespace scratch
