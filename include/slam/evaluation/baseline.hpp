#pragma once

// Stage-1 measurement and regression harness for the scratch pipeline.
//
// A fixture is one frame pair i -> j with its RGB-D files and intrinsics. The
// harness runs the UNCHANGED pipeline stages on it (ExtractFeatures on both
// frames, ProcessPair with default PipelineParams, i.e. exactly what
// scratch_pipeline does for that pair) and measures what happened at every stage:
//
//   keypoints -> cross-level near-duplicate keypoints -> raw / filtered matches
//   -> unique source / destination keypoints -> 3D->2D inputs -> RANSAC inlier rows
//   -> distinct inlier keypoints (by index and by location) -> solver status
//   -> diagnostic pose validity -> error vs ground truth
//
// Ground-truth isolation: LoadFixtures returns the ground-truth poses in a
// separate list (FixtureGroundTruth). The estimation entry point MeasureFixture
// receives only the Fixture (images + intrinsics) and cannot see them;
// ScoreMeasurement adds the errors afterwards and changes no estimated value.
//
// Pose validity is recorded separately from solver success and is a
// DIAGNOSTIC only (nothing in the pipeline or the trajectory uses it). The rule
// is the reference evaluator's AcceptPnp (tests/tum/tum_trajectory_eval.cpp):
// the inliers contain at least `min_distinct` distinct keypoints in frame i AND
// in frame j; additionally R must be a finite proper rotation and t finite.
//
// Standard library + Eigen + the scratch library only.

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "pipeline.hpp"

namespace scratch::baseline {

// ---- fixtures ----
struct Fixture {
  std::string name, dataset;  // dataset: "bunny" or "tum" (informative)
  int frame_i = 0, frame_j = 0;
  // repository-relative paths; "data/rgbd_dataset_*" paths are read from $SLAM_TUM_ROOT when it is set
  // (LoadFixtures; the result is then absolute, and repo / path keeps it)
  std::string rgb_i, depth_i, rgb_j, depth_j;
  Intrinsics K{0, 0, 0, 0};
  // optional: an existing scratch_pipeline output directory (repository-relative)
  // that contains this pair as pairs.csv row `export_pair`; used to show that the
  // harness reproduces the reported numbers. Empty / -1 = none.
  std::string export_dir;
  int export_pair = -1;
};
struct FixtureGroundTruth {  // scoring only, never passed to the estimator
  std::string name;
  Pose Twc_i, Twc_j;  // T_wc (camera -> world) of the two frames
};

// Fixture CSV (one row per pair), columns:
//   name,dataset,frame_i,frame_j,rgb_i,depth_i,rgb_j,depth_j,fx,fy,cx,cy,
//   gt_i_x,gt_i_y,gt_i_z,gt_i_qx,gt_i_qy,gt_i_qz,gt_i_qw, gt_j_... (same),export_dir,export_pair
bool LoadFixtures(const fs::path &csv, std::vector<Fixture> &fixtures, std::vector<FixtureGroundTruth> &gt,
                  std::string *error = nullptr);

// ---- counting helpers (exposed for the tests) ----
// Number of distinct values.
int CountDistinct(std::vector<int> ids);
// Number of groups of points connected by distances <= radius (single linkage).
int CountDistinctLocations(const std::vector<Eigen::Vector2d> &pts, double radius);
// Keypoints that have a keypoint from ANOTHER pyramid level within `radius` px
// (original-image coordinates): the same corner detected on several levels.
int CountCrossLevelNearDuplicates(const FrameFeatures &f, double radius);
// Matches whose destination (train) keypoint is also the destination of another match.
int CountSharedDestinationMatches(const std::vector<int> &train);

// ---- measurement ----
struct PairMeasurement {
  std::string name;
  int frame_i = 0, frame_j = 0;
  // front end
  int keypoints_i = 0, keypoints_j = 0;
  int cross_level_dup_i = 0, cross_level_dup_j = 0;  // CountCrossLevelNearDuplicates, kDuplicateRadiusPx
  int raw_matches = 0, filtered_matches = 0;
  int unique_src_filtered = 0, unique_dst_filtered = 0;  // distinct frame-i / frame-j keypoints of the filtered matches
  int shared_dst_filtered = 0;                           // filtered matches sharing their frame-j keypoint
  // PnP
  int pnp_inputs = 0;  // 3D->2D correspondences (filtered matches with frame-i depth)
  int unique_src_pnp = 0, unique_dst_pnp = 0;
  int inlier_rows = 0;
  int distinct_inlier_i = 0, distinct_inlier_j = 0;  // by keypoint index
  int distinct_inlier_loc_i = 0, distinct_inlier_loc_j = 0;  // by location (kDuplicateRadiusPx)
  int shared_dst_inliers = 0;  // inlier rows sharing their frame-j keypoint
  bool solver_ok = false;      // RANSAC PnP returned a pose (PairResult::pnp_ok; what the trajectory uses)
  std::string solver_reason;
  bool pose_valid = false;     // diagnostic validity (see top of file); never used by the pipeline
  std::string validity_reason; // "" if valid
  bool refined = false;
  int refine_iterations = 0;
  double reproj_inlier_mean_px = NAN;
  // estimate T_{j<-i} (X_j = R X_i + t)
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  double est_rotation_deg = NAN, est_translation_m = NAN;
  // Stage-2 pose-validity gate, as ProcessPair evaluated it (PairResult::validity / pose_accepted)
  PoseValidity gate;
  bool pose_accepted = false;
  // scoring (ScoreMeasurement only)
  bool scored = false;
  double gt_motion_deg = NAN, gt_motion_m = NAN, rot_err_deg = NAN, trans_err_m = NAN;
};

constexpr double kDuplicateRadiusPx = 2.0;
constexpr int kMinDistinctInliers = 6;  // the reference evaluator's --pnp-min-distinct-inliers default

// GT-free: measures a pipeline result and the two frames' features.
PairMeasurement Measure(const std::string &name, const PairResult &r, const FrameFeatures &fi,
                        const FrameFeatures &fj);

// Loads the fixture's frames (relative to `repo`), runs ExtractFeatures + ProcessPair, measures.
// `result` (optional) receives the full PairResult. False if a file cannot be loaded.
bool MeasureFixture(const Fixture &fx, const fs::path &repo, const PipelineParams &params, PairMeasurement &m,
                    std::string *error = nullptr, PairResult *result = nullptr);

// T_{j<-i} = T_wc[j]^-1 T_wc[i]  (the convention of scratch_pipeline and of PairResult::R, t).
Pose RelativeGroundTruth(const Pose &Twc_i, const Pose &Twc_j);
// Fills gt_motion_* and the errors (rotation: angle of R_gt^T R; translation: |t_gt - t|, the
// metrics of scratch_pipeline's pairs.csv). Changes no estimated field.
void ScoreMeasurement(PairMeasurement &m, const FixtureGroundTruth &gt);

// ---- report / baseline lock ----
// Every locked field as (name, value); integers and flags are exact doubles.
std::vector<std::pair<std::string, double>> NumericFields(const PairMeasurement &m);
// Stage-2 gate fields (locked separately in baseline_expected_gate.csv; the Stage-1 field set is unchanged):
// gate_status (0 accepted, 1 rejected, 2 solver_failed), pose_accepted and every gate quantity.
std::vector<std::pair<std::string, double>> GateFields(const PairMeasurement &m);
// Absolute tolerance of a field (0 = exact). See TOLERANCES in baseline.cpp.
double FieldTolerance(const std::string &field);

struct Expected {  // locked values of one fixture, by field name
  std::string name;
  std::map<std::string, double> values;
};
bool LoadExpected(const fs::path &csv, std::vector<Expected> &out, std::string *error = nullptr);
// One CSV row per measurement, header = "name," + NumericFields names; 17 significant digits.
std::string ExpectedCsv(const std::vector<PairMeasurement> &ms);
std::string GateExpectedCsv(const std::vector<PairMeasurement> &ms);  // the same for GateFields

struct Difference {
  std::string field;
  double expected = NAN, actual = NAN, tolerance = 0;
};
// Fields of `m` that differ from `e` by more than their tolerance; a field missing on either side counts.
std::vector<Difference> Compare(const PairMeasurement &m, const Expected &e);
std::vector<Difference> CompareGate(const PairMeasurement &m, const Expected &e);

// Cross-check against an existing scratch_pipeline export (fixture export_dir / export_pair):
// counts exactly, errors within the export's printed precision. Empty `problems` = reproduced.
struct ExportCheck {
  bool available = false;
  std::vector<std::string> problems;
  std::vector<std::string> compared;  // field names compared
};
ExportCheck CheckAgainstExport(const PairMeasurement &m, const Fixture &fx, const fs::path &repo);

std::string ToJson(const PairMeasurement &m, int indent = 2);

// ---- multi-seed evaluation (Stage 3) ----
// The same fixture pair is processed once per RANSAC seed: features are extracted ONCE; ProcessPair is then
// called with only PipelineParams::pnp_ransac.seed changed. Matching and correspondence generation are
// deterministic and seed-independent, and every run's PnP inputs are checked to be identical to the first
// run's (inputs_identical). Ground truth scores the poses afterwards and influences nothing.
std::vector<uint32_t> DefaultSeedSweep(int n);  // {12345 (the pipeline default), 1, 2, ..., n - 1}
// "12345,1,2,7-9" -> {12345, 1, 2, 7, 8, 9}; duplicates rejected; false on malformed input.
bool ParseSeedList(const std::string &s, std::vector<uint32_t> &out, std::string *error = nullptr);

struct SeedRun {
  std::string name;
  uint32_t seed = 0;
  bool inputs_identical = true;  // PnP inputs equal to the first seed's
  bool solver_ok = false, accepted = false;
  PoseStatus gate = PoseStatus::kSolverFailed;
  std::string reason;            // gate / solver reason
  int pnp_inputs = 0, inlier_rows = 0, distinct_loc_dst = 0;
  double rot_err_deg = NAN, trans_err_m = NAN;  // of the returned pose (NaN if the solver failed)
  double seconds = 0;            // wall time of ProcessPair (matching + essential diagnostic + PnP + gate)
};
struct SweepStat {
  int n = 0;
  double median = NAN, p95 = NAN, max = NAN, stddev = NAN;  // p95: linear interpolation; stddev: population
};
SweepStat Summarize(std::vector<double> v);  // non-finite values are ignored
struct SeedSummary {
  std::string name;
  int seeds = 0, solver_ok = 0, accepted = 0, rejected = 0, solver_failed = 0, inputs_changed = 0;
  int returned_over_2deg = 0;
  SweepStat rot_returned, trans_returned;  // every returned pose (accepted or rejected)
  SweepStat rot_accepted, trans_accepted;  // accepted poses only (what a trajectory would use)
  SweepStat seconds;
};
bool SweepFixture(const Fixture &fx, const FixtureGroundTruth &gt, const fs::path &repo, const PipelineParams &params,
                  const std::vector<uint32_t> &seeds, std::vector<SeedRun> &runs, std::string *error = nullptr);
SeedSummary SummarizeSweep(const std::string &name, const std::vector<SeedRun> &runs);

// ---- trajectory-level ATE of an existing run (no re-estimation) ----
// Raw ATE RMSE (no alignment) of <run>/trajectory.csv positions against the ground-truth
// positions: <run>/frames.csv (TUM runs) or data/synthetic_bunny/groundtruth.txt (Bunny runs).
struct AteResult {
  bool ok = false;
  std::string error;
  int frames = 0;
  double ate_raw_rmse_m = NAN, final_error_m = NAN;
};
AteResult RawAte(const fs::path &run_dir, const fs::path &repo);

}  // namespace scratch::baseline
