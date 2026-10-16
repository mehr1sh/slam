#pragma once

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "camera/camera.hpp"
#include "render/mesh.hpp"

struct RenderTarget {
  cv::Mat rgb;      // CV_8UC3, BGR (matches cv::imread/imwrite convention used elsewhere)
  cv::Mat depth_m;  // CV_32FC1, camera-space Z in metres; background = +inf
};

struct Light {
  Eigen::Vector3f direction_world = Eigen::Vector3f(-0.3f, -1.0f, -0.5f);
  // Ambient kept fairly high (rather than a physically low value) because
  // the light direction is fixed while the camera orbits 360 degrees --
  // without this, frames on the far side of the orbit render almost black.
  float ambient = 0.4f;
  float diffuse = 0.6f;
};

struct RenderOptions {
  cv::Vec3b background_color = cv::Vec3b(30, 30, 30);
  cv::Vec3b albedo = cv::Vec3b(180, 180, 180);
  double near = 0.01;  // metres; triangles with any vertex at/behind this are skipped (no clipping)
  bool backface_cull = true;
};

// R_wc, t_wc: world -> camera extrinsics, x_cam = R_wc * x_world + t_wc (this
// repo's standard (R,t) convention, see ARCHITECTURE.md).
RenderTarget RenderMesh(const Mesh &mesh, const PinholeCamera &cam, int width, int height,
                        const cv::Mat &R_wc, const cv::Mat &t_wc, const Light &light,
                        const RenderOptions &opts = RenderOptions());

// pixel = round(depth_metres * depth_scale), 0 = invalid -- matches the
// /5000.0 TUM depth convention already hardcoded in tests/pnp_test.cpp and
// tests/icp_test.cpp.
cv::Mat DepthMetersToTUM16U(const cv::Mat &depth_m, double depth_scale = 5000.0);
