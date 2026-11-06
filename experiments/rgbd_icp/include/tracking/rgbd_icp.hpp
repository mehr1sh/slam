#pragma once
#include <Eigen/Core>
#include "features/types.hpp"
#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
namespace rgbd_icp {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
struct Pose {
  M3 R = M3::Identity();
  V3 t = V3::Zero();
  V3 apply(const V3 &p) const { return R * p + t; }
  Pose inverse() const { return {R.transpose(), -R.transpose() * t}; }
  Pose operator*(const Pose &b) const { return {R * b.R, R * b.t + t}; }
};
struct Camera {
  int width = 0, height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;
};
struct Config {
  double depth_scale = 5000, ratio = .87, depth_edge = .012,
         ransac_threshold = .009;
  int features = 1400, fast_threshold = 10, ransac_iterations = 650,
      min_inliers = 12;
  int stride = 3, max_points = 2800, min_pairs = 70, point_iterations = 18,
      plane_iterations = 16;
  double max_distance = .024, trim = .83, huber = .004, max_angle_deg = 13,
         max_translation = .07;
  unsigned seed = 42;
};
// Common representation lets benchmarks change only feature extraction.
struct ImageFeatures {
  std::vector<Eigen::Vector2d> pixels;
  std::vector<features::BriefDescriptor> descriptors;
  // Read-only metadata for reproducible keypoint overlays.
  std::vector<int> levels;
  std::vector<float> angles_rad;
};
struct Frame {
  cv::Mat rgb, depth;
};
// Read-only diagnostics: errors below use the same fixed pairs before/after
// an accepted update. The next iteration independently refreshes NN pairs.
struct IterationInfo {
  int iteration = 0, pairs = 0;
  double rmse_before = 0, rmse_after = 0;
  double normal_before = 0, normal_after = 0;
  double objective_before = 0, objective_after = 0;
  double rotation_step_deg = 0, translation_step_m = 0;
  std::uint64_t assignment_hash = 0;
};
struct SolverInfo {
  int iterations = 0, pairs = 0;
  double rmse = std::numeric_limits<double>::infinity();
  bool accepted = false, converged = false;
  std::string reason;
  std::vector<IterationInfo> history;
};
struct PairClouds {
  std::vector<V3> source, target, normals;
};
struct PairResult {
  bool accepted = false;
  std::string status = "INSUFFICIENT_FEATURES";
  int matches = 0, valid3d = 0, inliers = 0;
  double inlier_ratio = 0;
  Pose baseline, robust, point, plane;
  SolverInfo point_info, plane_info;
};
struct Edge {
  int i, j;
  Pose measurement;
  bool loop = false;
};
struct GraphResult {
  std::vector<Pose> poses;
  bool converged = false;
  int iterations = 0;
  double initial_cost = 0, final_cost = 0;
};
M3 skew(const V3 &x);
M3 expSO3(const V3 &w);
// Left pose increment: (R,t) <- (Exp(w) R, Exp(w) t + v).
// This is an SO(3) x R^3 retraction, not a full SE(3) exponential.
Pose leftIncrement(const Pose &, const Eigen::Matrix<double, 6, 1> &);
Eigen::Matrix<double, 3, 6> pointJacobian(const V3 &transformed);
V3 logSO3(const M3 &R);
double angleDeg(const M3 &R);
Pose rigidSVD(const std::vector<V3> &source, const std::vector<V3> &target);
bool ransac(const std::vector<V3> &source, const std::vector<V3> &target,
            const Config &, Pose &, std::vector<int> &inliers, unsigned seed);
Pose pointICP(const std::vector<V3> &source, const std::vector<V3> &target,
              const Pose &, const Config &, SolverInfo &);
Pose planeICP(const std::vector<V3> &source, const std::vector<V3> &target,
              const std::vector<V3> &normals, const Pose &, const Config &,
              SolverInfo &);
Camera readCamera(const std::string &directory);
Frame readFrame(const std::string &directory, int index, const Camera &);
void matchedFeaturePoints(const Frame &, const Frame &, const Camera &,
    const Config &, const ImageFeatures &, const ImageFeatures &,
    std::vector<V3> &target, std::vector<V3> &source, int &match_count);
ImageFeatures extractFeatures(const cv::Mat &, const Config &);
PairResult registerFramesWithFeatures(const Frame &target, const Frame &source,
    const Camera &, const Config &, const ImageFeatures &target_features,
    const ImageFeatures &source_features, unsigned seed = 42,
    PairClouds *diagnostics = nullptr);
PairResult registerFrames(const Frame &target, const Frame &source,
                          const Camera &, const Config &, unsigned seed = 42,
                          PairClouds *diagnostics = nullptr);
std::vector<Edge> findLoops(const std::string &directory, const Camera &,
                            const Config &, const std::vector<Pose> &poses);
GraphResult optimizeGraph(const std::vector<Pose> &initial,
                          const std::vector<Edge> &edges, int iterations = 25);
} // namespace rgbd_icp
