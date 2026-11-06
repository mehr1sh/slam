#pragma once

#include <g2o/core/base_vertex.h>
#include <sophus/se3.hpp>

// Book code starts
// VertexPose: defined identically (character-for-character) in both
// pose_estimation_3d2d.cpp and pose_estimation_3d3d.cpp -- extracted once.
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(std::istream &in) override { return true; }  // unused; book left empty, warns on -Wreturn-type

  virtual bool write(std::ostream &out) const override { return true; }
};
// Book code ends
