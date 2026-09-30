#include "render/rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

RenderTarget RenderMesh(const Mesh &mesh, const PinholeCamera &cam, int width, int height,
                        const cv::Mat &R_wc, const cv::Mat &t_wc, const Light &light,
                        const RenderOptions &opts) {
  RenderTarget target;
  target.rgb = cv::Mat(height, width, CV_8UC3,
                        cv::Scalar(opts.background_color[0], opts.background_color[1],
                                   opts.background_color[2]));
  target.depth_m =
      cv::Mat(height, width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::infinity()));

  cv::Mat K = cam.K();

  Eigen::Matrix3d R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R(i, j) = R_wc.at<double>(i, j);
  Eigen::Vector3d t(t_wc.at<double>(0), t_wc.at<double>(1), t_wc.at<double>(2));

  Eigen::Vector3d light_dir_cam = (R * light.direction_world.normalized().cast<double>()).normalized();

  std::vector<Eigen::Vector3d> cam_pts(mesh.vertices.size());
  for (size_t i = 0; i < mesh.vertices.size(); ++i)
    cam_pts[i] = R * mesh.vertices[i].cast<double>() + t;

  for (size_t fidx = 0; fidx < mesh.faces.size(); ++fidx) {
    const Eigen::Vector3i &face = mesh.faces[fidx];
    const Eigen::Vector3d &p0 = cam_pts[face[0]];
    const Eigen::Vector3d &p1 = cam_pts[face[1]];
    const Eigen::Vector3d &p2 = cam_pts[face[2]];

    // No near-plane clipping: skip a triangle entirely if any vertex is
    // at/behind `near`, rather than projecting a point with z<=0 (which
    // would divide-by-near-zero/flip sign and rasterize garbage). Acceptable
    // since orbit trajectories keep the camera well outside the mesh.
    if (p0.z() <= opts.near || p1.z() <= opts.near || p2.z() <= opts.near) continue;

    Eigen::Vector3d n_cam = (R * mesh.face_normals[fidx].cast<double>()).normalized();
    Eigen::Vector3d centroid = (p0 + p1 + p2) / 3.0;
    // Camera is at the origin of its own frame, so `centroid` is already the
    // view direction from the camera to the triangle.
    if (opts.backface_cull && n_cam.dot(centroid) >= 0) continue;

    cv::Point2d uv0 = cam2pixel(cv::Point3d(p0.x(), p0.y(), p0.z()), K);
    cv::Point2d uv1 = cam2pixel(cv::Point3d(p1.x(), p1.y(), p1.z()), K);
    cv::Point2d uv2 = cam2pixel(cv::Point3d(p2.x(), p2.y(), p2.z()), K);

    double min_u = std::min({uv0.x, uv1.x, uv2.x});
    double max_u = std::max({uv0.x, uv1.x, uv2.x});
    double min_v = std::min({uv0.y, uv1.y, uv2.y});
    double max_v = std::max({uv0.y, uv1.y, uv2.y});
    int x0 = std::max(0, (int)std::floor(min_u));
    int x1 = std::min(width - 1, (int)std::ceil(max_u));
    int y0 = std::max(0, (int)std::floor(min_v));
    int y1 = std::min(height - 1, (int)std::ceil(max_v));
    if (x0 > x1 || y0 > y1) continue;

    // Standard screen-space barycentric weights (Real-Time Rendering formula);
    // sign-consistent regardless of the triangle's winding order.
    double denom = (uv1.y - uv2.y) * (uv0.x - uv2.x) + (uv2.x - uv1.x) * (uv0.y - uv2.y);
    if (std::abs(denom) < 1e-9) continue;

    double inv_z0 = 1.0 / p0.z(), inv_z1 = 1.0 / p1.z(), inv_z2 = 1.0 / p2.z();

    double intensity = light.ambient + light.diffuse * std::max(0.0, -n_cam.dot(light_dir_cam));
    intensity = std::min(intensity, 1.0);
    cv::Vec3b color((uchar)std::min(255.0, opts.albedo[0] * intensity),
                     (uchar)std::min(255.0, opts.albedo[1] * intensity),
                     (uchar)std::min(255.0, opts.albedo[2] * intensity));

    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        double px = x + 0.5, py = y + 0.5;
        double w0 = ((uv1.y - uv2.y) * (px - uv2.x) + (uv2.x - uv1.x) * (py - uv2.y)) / denom;
        double w1 = ((uv2.y - uv0.y) * (px - uv2.x) + (uv0.x - uv2.x) * (py - uv2.y)) / denom;
        double w2 = 1.0 - w0 - w1;
        if (w0 < 0 || w1 < 0 || w2 < 0) continue;

        // Perspective-correct depth: interpolate 1/z (affine in screen
        // space for a pinhole projection), then invert.
        double inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2;
        float z = (float)(1.0 / inv_z);

        float &depth_ref = target.depth_m.at<float>(y, x);
        if (z < depth_ref) {
          depth_ref = z;
          target.rgb.at<cv::Vec3b>(y, x) = color;
        }
      }
    }
  }

  return target;
}

cv::Mat DepthMetersToTUM16U(const cv::Mat &depth_m, double depth_scale) {
  cv::Mat out(depth_m.rows, depth_m.cols, CV_16UC1, cv::Scalar(0));
  for (int y = 0; y < depth_m.rows; ++y) {
    for (int x = 0; x < depth_m.cols; ++x) {
      float z = depth_m.at<float>(y, x);
      if (!std::isfinite(z) || z <= 0) continue;
      double val = std::min(65535.0, std::round(z * depth_scale));
      out.at<ushort>(y, x) = (ushort)val;
    }
  }
  return out;
}
