#pragma once

// PnP pose-validity gate (Stage 2). Decides, from the image correspondences and
// the estimated pose only, whether a RANSAC PnP result may be accumulated into
// the trajectory. Ground truth is not an input.
//
// A pair ends in one of three states:
//   solver_failed  RANSAC PnP returned no pose
//   rejected       a pose was returned but fails at least one criterion below
//   accepted       a pose was returned and passes every enabled criterion
//
// Criteria (each failure is recorded; the reason lists all of them):
//   finite         R and t contain no NaN / inf
//   SO(3)          ||R^T R - I||_F <= rotation_tolerance and |det R - 1| <= rotation_tolerance
//   inlier rows    >= min_inlier_rows
//   distinct keypoints among the inliers, by index, in frame i AND in frame j: >= min_distinct_keypoints
//   distinct locations among the inliers (single-linkage groups within location_radius_px; the same corner
//                  detected on several pyramid levels is one location), each frame: >= min_distinct_locations
//   inlier ratio   inlier rows / PnP inputs >= min_inlier_ratio (0 = recorded only)
//   coverage       occupied coverage_cell_px x coverage_cell_px image cells, each frame: >= min_coverage_cells
//   reprojection   inlier residuals under the FINAL pose: median <= max_reproj_median_factor * RANSAC threshold,
//                  90th percentile <= max_reproj_p90_factor * RANSAC threshold
//
// Default values and their derivation: README "Pose validity gate".

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace scratch {

struct ValidityParams {
  bool enabled = true;               // false: every returned pose is accepted (Stage-1 behaviour); still evaluated
  int min_inlier_rows = 20;
  int min_distinct_keypoints = 20;
  int min_distinct_locations = 20;   // minimal sample (6) + chance support of a wrong pose (<= 12), rounded up
  double location_radius_px = 2.0;
  double min_inlier_ratio = 0.0;     // recorded; not enforced by default (README)
  int coverage_cell_px = 40;
  int min_coverage_cells = 7;        // more cells than a 6-point minimal sample can occupy
  double max_reproj_median_factor = 0.5;
  double max_reproj_p90_factor = 1.0;
  double rotation_tolerance = 1e-6;
};

// One PnP input (a 3D->2D correspondence) as the gate sees it.
struct ValidityRow {
  int src = -1, dst = -1;  // keypoint index in frame i / frame j
  Eigen::Vector2d uv_src = Eigen::Vector2d::Zero(), uv_dst = Eigen::Vector2d::Zero();  // pixels
  double residual_px = 0;  // reprojection error under the final pose
  bool inlier = false;     // RANSAC inlier
};

enum class PoseStatus { kAccepted, kRejected, kSolverFailed };
const char *PoseStatusName(PoseStatus s);  // "accepted", "rejected", "solver_failed"

struct PoseValidity {
  PoseStatus status = PoseStatus::kSolverFailed;
  std::string reason;  // solver reason, or every failed criterion ("; "-separated); "" when accepted
  bool finite = false, rotation_valid = false;
  double orthogonality_error = NAN, det_error = NAN;  // ||R^T R - I||_F, |det R - 1|
  int pnp_inputs = 0, inlier_rows = 0;
  int distinct_src = 0, distinct_dst = 0;          // by keypoint index
  int distinct_loc_src = 0, distinct_loc_dst = 0;  // by location
  double inlier_ratio = 0;
  int cells_src = 0, cells_dst = 0;
  double reproj_median_px = NAN, reproj_p90_px = NAN;
  bool accepted() const { return status == PoseStatus::kAccepted; }
};

// threshold_px: the RANSAC inlier threshold (the reprojection limits are relative to it).
PoseValidity EvaluatePose(bool solver_ok, const std::string &solver_reason, const Eigen::Matrix3d &R,
                          const Eigen::Vector3d &t, const std::vector<ValidityRow> &rows, double threshold_px,
                          const ValidityParams &params = ValidityParams());

// helpers (exposed for the tests)
int CountDistinctIds(std::vector<int> ids);
int CountLocationGroups(const std::vector<Eigen::Vector2d> &pts, double radius);  // single linkage
int CountOccupiedCells(const std::vector<Eigen::Vector2d> &pts, int cell_px);
double Percentile(std::vector<double> v, double q);  // linear interpolation between order statistics; NaN if empty

}  // namespace scratch
