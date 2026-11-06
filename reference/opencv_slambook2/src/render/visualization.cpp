#include "render/visualization.hpp"

#include <cmath>

#include <Eigen/Geometry>  // cross_impl specialization used by .cross() below

void AppendLineAsQuad(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &faces,
                      std::vector<Rgb8> &colors, const Eigen::Vector3f &a,
                      const Eigen::Vector3f &b, float thickness, const Rgb8 &color) {
  Eigen::Vector3f dir = b - a;
  float len = dir.norm();
  if (len < 1e-9f) return;
  dir /= len;

  Eigen::Vector3f arbitrary =
      std::abs(dir.dot(Eigen::Vector3f(0, 1, 0))) < 0.9f ? Eigen::Vector3f(0, 1, 0)
                                                          : Eigen::Vector3f(1, 0, 0);
  Eigen::Vector3f side = dir.cross(arbitrary).normalized() * (thickness * 0.5f);

  int base = (int)vertices.size();
  vertices.push_back(a - side);
  vertices.push_back(a + side);
  vertices.push_back(b + side);
  vertices.push_back(b - side);
  for (int i = 0; i < 4; ++i) colors.push_back(color);

  faces.emplace_back(base + 0, base + 1, base + 2);
  faces.emplace_back(base + 0, base + 2, base + 3);
}

void AppendFrustumWireframe(std::vector<Eigen::Vector3f> &vertices,
                            std::vector<Eigen::Vector3i> &faces, std::vector<Rgb8> &colors,
                            const Sophus::SE3d &T_wc, const PinholeCamera &cam, int width,
                            int height, float vis_near, float thickness, const Rgb8 &color) {
  Eigen::Vector3f center = T_wc.translation().cast<float>();

  auto corner_world = [&](double u, double v) {
    double x = (u - cam.cx) / cam.fx * vis_near;
    double y = (v - cam.cy) / cam.fy * vis_near;
    Eigen::Vector3d p_cam(x, y, vis_near);
    return (T_wc * p_cam).cast<float>();
  };

  Eigen::Vector3f c00 = corner_world(0, 0);
  Eigen::Vector3f c10 = corner_world(width, 0);
  Eigen::Vector3f c01 = corner_world(0, height);
  Eigen::Vector3f c11 = corner_world(width, height);

  AppendLineAsQuad(vertices, faces, colors, center, c00, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, center, c10, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, center, c01, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, center, c11, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, c00, c10, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, c10, c11, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, c11, c01, thickness, color);
  AppendLineAsQuad(vertices, faces, colors, c01, c00, thickness, color);
}

void AppendTrajectoryPolyline(std::vector<Eigen::Vector3f> &vertices,
                              std::vector<Eigen::Vector3i> &faces, std::vector<Rgb8> &colors,
                              const std::vector<Sophus::SE3d> &T_wc_list, float thickness,
                              const Rgb8 &color_start, const Rgb8 &color_end) {
  size_t n = T_wc_list.size();
  for (size_t i = 0; i + 1 < n; ++i) {
    float frac = n > 1 ? (float)i / (float)(n - 1) : 0.f;
    Rgb8 color{(unsigned char)(color_start.r + frac * (color_end.r - color_start.r)),
               (unsigned char)(color_start.g + frac * (color_end.g - color_start.g)),
               (unsigned char)(color_start.b + frac * (color_end.b - color_start.b))};
    Eigen::Vector3f a = T_wc_list[i].translation().cast<float>();
    Eigen::Vector3f b = T_wc_list[i + 1].translation().cast<float>();
    AppendLineAsQuad(vertices, faces, colors, a, b, thickness, color);
  }
}
