#include "tracking/icp.hpp"

#include "optimization/g2o_types.hpp"
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/SVD>
#include <iostream>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>

using namespace std;
using namespace cv;

// Book code starts
void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W=" << W << endl;

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {
    R_ = -R_;
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) { return true; }

  bool write(ostream &out) const { return true; }

protected:
  Eigen::Vector3d _point;
};

void bundleAdjustment(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t) {
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(true);

  VertexPose *pose = new VertexPose();
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  for (size_t i = 0; i < pts1.size(); i++) {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    edge->setVertex(0, pose);
    edge->setMeasurement(Eigen::Vector3d(
      pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}
// Book code ends

// ---------------------------------------------------------------------------
// Project code (not from the book): robust 3D-3D alignment.

bool FitRigid(const vector<Point3f> &pts1, const vector<Point3f> &pts2, const vector<int> &idx, Eigen::Matrix3d &R,
              Eigen::Vector3d &t) {
  if (idx.size() < 3) return false;
  Eigen::Vector3d c1 = Eigen::Vector3d::Zero(), c2 = Eigen::Vector3d::Zero();
  for (int k : idx) {
    c1 += Eigen::Vector3d(pts1[k].x, pts1[k].y, pts1[k].z);
    c2 += Eigen::Vector3d(pts2[k].x, pts2[k].y, pts2[k].z);
  }
  c1 /= double(idx.size());
  c2 /= double(idx.size());
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int k : idx)
    W += (Eigen::Vector3d(pts1[k].x, pts1[k].y, pts1[k].z) - c1) *
         (Eigen::Vector3d(pts2[k].x, pts2[k].y, pts2[k].z) - c2).transpose();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d s = svd.singularValues();
  if (s(1) <= 1e-12 * std::max(1.0, s(0))) return false;  // collinear points: rotation undetermined
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) D(2, 2) = -1;  // proper rotation
  R = svd.matrixU() * D * svd.matrixV().transpose();
  t = c1 - R * c2;
  return true;
}

RigidRansacResult EstimateRigidRansac(const vector<Point3f> &pts1, const vector<Point3f> &pts2, double threshold_m,
                                      int iterations, uint32_t seed) {
  RigidRansacResult out;
  const int n = int(pts1.size());
  if (n < 3 || pts2.size() != pts1.size()) return out;
  auto residual = [&](const Eigen::Matrix3d &R, const Eigen::Vector3d &t, int k) {
    return (Eigen::Vector3d(pts1[k].x, pts1[k].y, pts1[k].z) -
            (R * Eigen::Vector3d(pts2[k].x, pts2[k].y, pts2[k].z) + t)).norm();
  };
  auto score = [&](const Eigen::Matrix3d &R, const Eigen::Vector3d &t, vector<int> &in, double &sum) {
    in.clear();
    sum = 0;
    for (int k = 0; k < n; ++k) {
      const double r = residual(R, t, k);
      if (r < threshold_m) in.push_back(k), sum += r;
    }
  };
  std::mt19937 rng(seed);
  vector<int> order(n), best_in, in;
  double best_sum = std::numeric_limits<double>::infinity(), sum;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  for (int it = 0; it < iterations; ++it) {
    std::iota(order.begin(), order.end(), 0);
    for (int i = 0; i < 3; ++i) std::swap(order[i], order[i + int(rng() % uint32_t(n - i))]);
    if (!FitRigid(pts1, pts2, vector<int>(order.begin(), order.begin() + 3), R, t)) continue;
    score(R, t, in, sum);
    if (in.size() > best_in.size() || (in.size() == best_in.size() && sum < best_sum)) {
      best_in = in;
      best_sum = sum;
      out.R = R;
      out.t = t;
    }
  }
  if (best_in.size() < 3) return out;
  if (FitRigid(pts1, pts2, best_in, R, t)) {  // least-squares refit on the inliers, kept if it loses none
    score(R, t, in, sum);
    if (in.size() >= best_in.size()) out.R = R, out.t = t;
  }
  score(out.R, out.t, in, sum);
  out.inliers.assign(n, false);
  for (int k : in) out.inliers[k] = true;
  out.num_inliers = int(in.size());
  out.ok = out.num_inliers >= 3;
  return out;
}
