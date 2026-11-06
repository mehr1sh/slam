#include "render/trajectory.hpp"

#include <cmath>

#include <Eigen/Geometry>  // cross_impl specialization used by .cross() below

namespace {

// Any axis not nearly parallel to `up`, used to seed the in-plane orbit basis.
Eigen::Vector3f PickReferenceAxis(const Eigen::Vector3f &up) {
  Eigen::Vector3f x_axis(1, 0, 0), z_axis(0, 0, 1);
  return std::abs(up.dot(x_axis)) < 0.9f ? x_axis : z_axis;
}

}  // namespace

std::vector<Sophus::SE3d> GenerateOrbitTrajectory(const OrbitTrajectoryParams &params) {
  std::vector<Sophus::SE3d> poses;
  poses.reserve(params.num_keyframes);

  Eigen::Vector3f up = params.world_up.normalized();
  Eigen::Vector3f ref = PickReferenceAxis(up);
  Eigen::Vector3f a = up.cross(ref).normalized();
  Eigen::Vector3f b = up.cross(a).normalized();

  for (int i = 0; i < params.num_keyframes; ++i) {
    float frac = params.num_keyframes > 1 ? (float)i / (float)params.num_keyframes : 0.f;
    float deg = params.start_deg + frac * (params.end_deg - params.start_deg);
    float theta = deg * (float)M_PI / 180.0f;

    Eigen::Vector3f C = params.center + params.height * up +
                        params.radius * (std::cos(theta) * a + std::sin(theta) * b);

    Eigen::Vector3f forward = (params.center - C).normalized();
    Eigen::Vector3f right = forward.cross(up).normalized();
    Eigen::Vector3f cam_up = right.cross(forward).normalized();

    // Columns = camera's local axes expressed in world coordinates, i.e. the
    // rotation from camera frame to world frame (R_cw) -- exactly what
    // Sophus::SE3d(R_cw, C) needs to represent T_world_camera. Camera +Y
    // points down (CV convention), so the world-up-aligned axis is negated.
    Eigen::Matrix3f R_cw;
    R_cw.col(0) = right;
    R_cw.col(1) = -cam_up;
    R_cw.col(2) = forward;

    Eigen::Quaterniond q(Eigen::Matrix3d(R_cw.cast<double>()));
    poses.emplace_back(q, C.cast<double>());
  }
  return poses;
}
