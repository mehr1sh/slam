#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

struct OrbitTrajectoryParams {
  Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0);  // look-at target
  float radius = 0.5f;                                 // metres, in the plane perp. to world_up
  float height = 0.1f;                                  // metres, offset along world_up above center
  Eigen::Vector3f world_up = Eigen::Vector3f(0, 1, 0);
  int num_keyframes = 60;
  float start_deg = 0.f;
  float end_deg = 360.f;  // num_keyframes samples over [start_deg, end_deg)
};

// Returns T_world_camera[i] (camera-to-world) for each keyframe: translation
// is the camera center in world coordinates, rotation maps camera-frame
// vectors into world-frame vectors. Invert to get the world-to-camera (R,t)
// this repo's rendering/SLAM code expects (x_cam = R*x_world + t).
std::vector<Sophus::SE3d> GenerateOrbitTrajectory(const OrbitTrajectoryParams &params);
