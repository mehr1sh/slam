#pragma once

#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "camera/camera.hpp"
#include "render/ply_mesh_io.hpp"

// PLY "edge" elements aren't reliably supported across viewers, so every
// line segment is appended as a thin quad (2 triangles) into the same
// vertex+face buffers the bunny mesh itself is written with.
void AppendLineAsQuad(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &faces,
                      std::vector<Rgb8> &colors, const Eigen::Vector3f &a,
                      const Eigen::Vector3f &b, float thickness, const Rgb8 &color);

// 8 segments (4 center->corner + 4 corner-rectangle edges) for one keyframe's
// view frustum, sized by `vis_near` (a visualization-only distance,
// independent of the rasterizer's near-plane).
void AppendFrustumWireframe(std::vector<Eigen::Vector3f> &vertices,
                            std::vector<Eigen::Vector3i> &faces, std::vector<Rgb8> &colors,
                            const Sophus::SE3d &T_wc, const PinholeCamera &cam, int width,
                            int height, float vis_near, float thickness, const Rgb8 &color);

// Connects consecutive camera centers, colored as a gradient from
// color_start (first keyframe) to color_end (last).
void AppendTrajectoryPolyline(std::vector<Eigen::Vector3f> &vertices,
                              std::vector<Eigen::Vector3i> &faces, std::vector<Rgb8> &colors,
                              const std::vector<Sophus::SE3d> &T_wc_list, float thickness,
                              const Rgb8 &color_start, const Rgb8 &color_end);
