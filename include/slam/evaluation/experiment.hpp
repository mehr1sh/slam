#pragma once

// Stage-3C experiment support (not used by scratch_pipeline's estimate):
//   * matcher presets M0..M5 (+ M2n, M1L, M3L) on top of features::MatchRanked;
//   * a trace of RansacPnP -- the same algorithm re-run step by step so that the best
//     minimal-sample model and the DLT refit are visible; every trace is checked to
//     reproduce the production RansacPnP result exactly (TraceMatches);
//   * offline gate diagnostics: every Stage-2 check separately, candidate RELATIVE
//     support metrics and 3D conditioning of the inlier points -- all computed from the
//     correspondences and the pose only (no ground truth);
//   * EvaluateWithTruth: the only function here that takes ground truth; evaluation only.
// The production gate (validity.hpp) is not changed or called with other thresholds here.

#include <string>
#include <vector>

#include "common.hpp"
#include "pipeline.hpp"

namespace scratch::experiment {

// ---- matcher presets ----
// M0 original matcher | M1 mutual | M2 location-aware ratio 0.8 | M2n naive ratio 0.8 (comparison only) |
// M3 mutual + ratio | M4 M3 + |level difference| <= 1 | M5 M3 + quality order (ordinary RANSAC) |
// M1L / M3L: M1 / M3 with location-aware mutual matching
const std::vector<std::string> &MatcherConfigIds();
bool ApplyMatcherConfig(const std::string &id, FrontendParams &frontend, std::string *error = nullptr);
std::string DescribeMatcherConfig(const std::string &id);

// ---- integration configurations (branch integration/teammate) ----
// Complete pipeline configurations:
//   M0..M3L  the Stage-3C presets on the Stage-2 front end of main (ApplyStage2Frontend + gate on +
//            ApplyMatcherConfig): reproduce the Stage-3C experiments
//   L0       PipelineParams{} (the teammate's --legacy)
//   T0       the teammate defaults (ApplyImprovedDefaults), gate off
//   T0G      T0 + Stage-2 gate             T0R  T0 + MatchRanked mutual + location-aware ratio 0.8
//   T0M      T0 + MatchRanked mutual only  T0Q  T0 + MatchRanked location-aware ratio 0.8 only
//   T0GR     T0 + gate + MatchRanked M3
//   T0E      T0, depth-edge rejection off  T0H  T0, Huber off (least squares)
//   leave-one-out from T0: T0-harris (no Harris ranking), T0-budget (500 per level instead of 1400 total),
//            T0-fast20 (FAST threshold 20), T0-mr (original NN + distance filter instead of mutual/ratio),
//            T0-reselect (no inlier re-selection), T0-ransac (300 iterations, 8 px)
//   T0C      candidate combining the two most robust ablations: T0 with RANSAC 300 it / 8 px and 500 features
//            per level (no total budget); T0CG = T0C + Stage-2 gate (at 8 px the gate is in its derivation regime)
// MatchRanked keeps the existing distance floor max(2 d_min, 30); the teammate matcher has none.
const std::vector<std::string> &IntegrationConfigIds();
bool ApplyIntegrationConfig(const std::string &id, PipelineParams &params, std::string *error = nullptr);

// ---- RANSAC trace ----
struct RansacTrace {
  bool ok = false;
  Eigen::Matrix3d R_sample = Eigen::Matrix3d::Identity();  // best minimal-sample model
  Eigen::Vector3d t_sample = Eigen::Vector3d::Zero();
  int sample_inliers = 0;
  std::vector<int> sample;                                  // its 6 input indices
  bool refit_ok = false;
  Eigen::Matrix3d R_refit = Eigen::Matrix3d::Identity();    // DLT on the sample model's inliers (= RansacPnP's pose)
  Eigen::Vector3d t_refit = Eigen::Vector3d::Zero();
  int refit_inliers = 0;
  std::vector<bool> refit_mask;
};
RansacTrace TraceRansacPnP(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv,
                           const Intrinsics &K, const RansacParams &params);
// The trace reproduces the production result: same pose (bitwise), RANSAC mask and inlier count.
// (Compared with PairResult::ransac_mask: with inlier re-selection the final mask differs.)
bool TraceMatches(const RansacTrace &t, const PairResult &r);

// ---- offline gate diagnostics (no ground truth) ----
struct GateDiagnostics {
  PoseValidity v;  // the production evaluation, as ProcessPair computed it
  // every Stage-2 check (true = passes)
  bool c_finite = false, c_so3 = false, c_rows = false, c_kp_src = false, c_kp_dst = false, c_loc_src = false,
       c_loc_dst = false, c_ratio = false, c_cells_src = false, c_cells_dst = false, c_median = false, c_p90 = false;
  // the inputs' own structure
  int input_loc_src = 0, input_loc_dst = 0;  // distinct locations among all PnP inputs
  int input_pairs = 0, inlier_pairs = 0;     // location-PAIR groups (both ends within the radius)
  int input_cells_dst = 0;
  // candidate relative-support metrics
  double ratio_rows = 0;        // inlier rows / PnP inputs (= v.inlier_ratio)
  double ratio_pairs = 0;       // inlier location pairs / input location pairs
  double ratio_loc_dst = 0;     // distinct inlier destination locations / input destination locations
  double ratio_cells_dst = 0;   // inlier cells / input cells (destination image)
  double rows_per_location = 0; // inlier rows / distinct inlier destination locations (duplicate inflation)
  double largest_cell_share = 0;  // largest share of inlier rows in one coverage cell (destination)
  // 3D support: covariance of the inlier points X_i (camera i, metres)
  double eig1 = 0, eig2 = 0, eig3 = 0;  // ascending
  double condition = 0;                 // sqrt(eig3 / eig1)
  double planarity = 0;                 // eig1 / eig3 (0 = planar)
  double depth_min = 0, depth_max = 0, depth_median = 0;
};
GateDiagnostics DiagnoseGate(const PairResult &r, const ValidityParams &params, double threshold_px);

// covariance eigenvalues (ascending) of a point set; zeros if fewer than 2 points
Eigen::Vector3d CovarianceEigenvalues(const std::vector<Eigen::Vector3d> &X);
// single-linkage groups of correspondences whose source AND destination ends are within `radius`
int CountLocationPairGroups(const std::vector<Eigen::Vector2d> &src, const std::vector<Eigen::Vector2d> &dst,
                            double radius);

// ---- evaluation with ground truth (offline only) ----
struct TruthEval {
  double rot_err_deg = NAN, trans_err_m = NAN;              // final pose
  double sample_rot_err_deg = NAN, sample_trans_err_m = NAN;  // best minimal-sample model
  double refit_rot_err_deg = NAN, refit_trans_err_m = NAN;    // DLT refit (before LM)
  int correct_inputs = 0;          // PnP inputs that reproject under the TRUE motion within the threshold
  double correct_input_share = NAN;
  int correct_inliers = 0;
  double correct_inlier_share = NAN;
};
TruthEval EvaluateWithTruth(const PairResult &r, const RansacTrace &trace, const Pose &G, const Intrinsics &K,
                            double threshold_px);

}  // namespace scratch::experiment
