#include "tracking/rgbd_icp.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace rgbd_icp;
void require(bool x, const char *message) {
  if (!x)
    throw std::runtime_error(message);
}
int main() {
  try {
    std::mt19937 rng(5);
    std::normal_distribution<double> d;
    std::vector<V3> s, t, n;
    Pose truth{expSO3(V3(.2, -.1, .4)), V3(.3, -.2, .5)};
    for (int i = 0; i < 100; ++i) {
      s.emplace_back(d(rng), d(rng), d(rng));
      t.push_back(truth.apply(s.back()));
      n.emplace_back(d(rng), d(rng), d(rng));
      n.back().normalize();
    }
    Pose fit = rigidSVD(s, t);
    require(angleDeg(fit.R * truth.R.transpose()) < 1e-5 &&
                (fit.t - truth.t).norm() < 1e-10,
            "SVD known transform");
    require((fit.inverse() * fit).t.norm() < 1e-10, "Inverse composition");
    auto reflected = s;
    for (auto &p : reflected)
      p.x() *= -1;
    Pose reflection = rigidSVD(s, reflected);
    require(std::abs(reflection.R.determinant() - 1) < 1e-10,
            "Reflection correction must keep det R=+1");
    bool invalid = false;
    try {
      rigidSVD({}, {});
    } catch (const std::invalid_argument &) {
      invalid = true;
    }
    require(invalid, "Empty SVD must fail");
    auto corrupted = t;
    for (int i = 0; i < 30; ++i)
      corrupted[i] += V3(d(rng), d(rng), d(rng));
    Config c;
    c.ransac_threshold = .003;
    c.min_inliers = 20;
    std::vector<int> in;
    Pose robust;
    require(ransac(s, corrupted, c, robust, in, 42) && in.size() == 70,
            "RANSAC outlier rejection");
    require((robust.t - truth.t).norm() < 1e-9, "RANSAC pose recovery");
    SolverInfo info;
    c.max_distance = .1;
    c.trim = 1;
    c.min_pairs = 20;
    Pose aligned = pointICP(s, t, truth, c, info);
    require(info.accepted && (aligned.t - truth.t).norm() < 1e-9,
            "Aligned ICP invariance");
    require(int(info.history.size()) == info.iterations && !info.history.empty(),
            "Point ICP logs every performed update");
    for (const auto &row : info.history)
      require(row.objective_after <= row.objective_before + 1e-12,
              "GN must decrease fixed-pair squared distance");
    // Central finite differences catch skew sign and left/right convention errors.
    V3 testPoint(.3, -.4, 1.2);
    auto analytic = pointJacobian(truth.apply(testPoint));
    for (int axis = 0; axis < 6; ++axis) {
      Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
      dx[axis] = 1e-6;
      V3 numerical = (leftIncrement(truth, dx).apply(testPoint) -
                      leftIncrement(truth, -dx).apply(testPoint)) / 2e-6;
      require((numerical - analytic.col(axis)).norm() < 1e-8,
              "Point GN analytic Jacobian must match finite differences");
      require(std::abs(n[0].dot(numerical) - n[0].dot(analytic.col(axis))) < 1e-8,
              "Plane GN Jacobian must match finite differences");
    }
    Pose small{expSO3(V3(.005, -.008, .004)), V3(.003, -.002, .004)};
    t.clear();
    for (auto p : s)
      t.push_back(small.apply(p));
    Pose gn = pointICP(s, t, Pose{}, c, info);
    require(info.accepted && angleDeg(gn.R * small.R.transpose()) < 1e-4 &&
                (gn.t - small.t).norm() < 1e-6,
            "Point GN must recover nonzero rotation and translation");
    for (const auto &row : info.history)
      require(row.objective_after <= row.objective_before + 1e-12,
              "Point GN fixed-pair line search must descend");
    Pose plane = planeICP(s, t, n, Pose{}, c, info);
    require(info.accepted && angleDeg(plane.R * small.R.transpose()) < 1e-4 &&
                (plane.t - small.t).norm() < 1e-6,
            "Point-plane nonzero recovery");
    require(int(info.history.size()) == info.iterations && !info.history.empty(),
            "Plane ICP logs every performed update");
    for (const auto &row : info.history)
      require(row.objective_after <= row.objective_before + 1e-12,
              "Plane ICP must decrease fixed-pair Huber objective");
    // Partial overlap with spurious source points exercises NN refresh/filtering.
    auto partial = s;
    for (int i = 0; i < 15; ++i) partial.push_back(V3(20+i, 30, 40));
    pointICP(partial, t, Pose{}, c, info);
    require(info.accepted && info.pairs >= 80, "Partial-overlap GN support");
    // A large required update must respect the motion gate.
    Config tight = c;
    tight.max_translation = 1e-8; tight.max_angle_deg = 1e-8;
    pointICP(s, t, Pose{}, tight, info);
    require(!info.accepted && info.reason == "NO_DESCENT", "GN motion gate rejection");
    c.point_iterations = 0;
    pointICP(s, t, Pose{}, c, info);
    require(!info.accepted && info.iterations == 0,
            "Skipped ICP must not succeed");
    c.point_iterations = 18;
    std::vector<V3> far = t;
    for (auto &p : far)
      p += V3(10, 10, 10);
    pointICP(s, far, Pose{}, c, info);
    require(!info.accepted, "No overlap rejection");
    Camera camera{640, 480, 520.9, 521, 325.1, 249.7};
    Frame blank{cv::Mat::zeros(480, 640, CV_8UC3),
                cv::Mat::zeros(480, 640, CV_16UC1)};
    auto emptyPair = registerFrames(blank, blank, camera, Config{});
    require(!emptyPair.accepted && emptyPair.status == "INSUFFICIENT_FEATURES",
            "Textureless/invalid-depth frame must be rejected");
    std::vector<Pose> poses(4);
    for (int i = 0; i < 4; ++i)
      poses[i].t = V3(.1 * i, 0, 0);
    std::vector<Edge> edges;
    for (int i = 0; i < 3; ++i)
      edges.push_back({i, i + 1, poses[i].inverse() * poses[i + 1], false});
    edges.push_back({0, 3, poses[0].inverse() * poses[3], true});
    poses[1].t += V3(.01, .003, 0);
    poses[2].R = expSO3(V3(0, 0, .01));
    auto graph = optimizeGraph(poses, edges);
    require(graph.converged && graph.final_cost < graph.initial_cost &&
                graph.final_cost < 1e-6,
            "Pose graph loop correction");
    require(graph.poses[0].t.norm() == 0, "Fixed graph anchor");
    auto disconnected =
        optimizeGraph(poses, {{2, 3, edges[2].measurement, false}});
    require(disconnected.converged &&
                disconnected.final_cost <= disconnected.initial_cost,
            "Disconnected graph gauges");
    std::cout << "PASS: SVD, reflection, inverse, invalid input, RANSAC, point "
                 "GN ICP, plane GN, analytic Jacobians, motion gate, partial overlap, skipped/no-overlap rejection, pose graph and "
                 "disconnected graph\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
