// Renders the Stanford Bunny along a circular orbit trajectory, producing a
// synthetic RGB-D dataset laid out as a drop-in structural match for
// data/tum_sample/ (same 640x480 shapes, same /5000.0 depth scale), plus a
// TUM-format groundtruth.txt and a combined mesh+trajectory+frustum PLY
// visualization.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "camera/camera.hpp"
#include "render/mesh.hpp"
#include "render/ply_mesh_io.hpp"
#include "render/rasterizer.hpp"
#include "render/trajectory.hpp"
#include "render/visualization.hpp"
#include "../report_utils.hpp"

using namespace std;

namespace {

cv::Mat ToCvMat3(const Eigen::Matrix3d &R) {
  cv::Mat m(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m.at<double>(i, j) = R(i, j);
  return m;
}

cv::Mat ToCvVec3(const Eigen::Vector3d &v) {
  return (cv::Mat_<double>(3, 1) << v(0), v(1), v(2));
}

string ZeroPad(int i, int width) {
  ostringstream oss;
  oss << setw(width) << setfill('0') << i;
  return oss.str();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    cout << "usage: render_bunny_test <mesh.ply> <output_dir> [--num-frames N] "
            "[--width W] [--height H] [--radius-scale R] [--height-scale H]"
         << endl;
    return 1;
  }
  string mesh_path = argv[1];
  string output_dir_arg = argv[2];
  int num_frames = 36, width = 640, height = 480;
  float radius_scale = 3.0f, height_scale = 0.5f;
  for (int i = 3; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--num-frames" && i + 1 < argc) num_frames = atoi(argv[++i]);
    else if (arg == "--width" && i + 1 < argc) width = atoi(argv[++i]);
    else if (arg == "--height" && i + 1 < argc) height = atoi(argv[++i]);
    else if (arg == "--radius-scale" && i + 1 < argc) radius_scale = atof(argv[++i]);
    else if (arg == "--height-scale" && i + 1 < argc) height_scale = atof(argv[++i]);
  }

  Mesh mesh;
  string err;
  if (!ReadPLYMesh(mesh_path, mesh, &err)) {
    cerr << "failed to load mesh: " << err << endl;
    return 1;
  }
  cout << "loaded mesh: " << mesh.vertices.size() << " vertices, " << mesh.faces.size()
       << " faces" << endl;

  Aabb box = ComputeAabb(mesh);
  Eigen::Vector3f center = box.Center();
  float bound_radius = box.BoundingSphereRadius();

  OrbitTrajectoryParams traj_params;
  traj_params.center = center;
  traj_params.radius = radius_scale * bound_radius;
  traj_params.height = height_scale * bound_radius;
  traj_params.num_keyframes = num_frames;
  vector<Sophus::SE3d> T_wc_list = GenerateOrbitTrajectory(traj_params);

  PinholeCamera cam{520.9, 521.0, 325.1, 249.7};

  filesystem::path output_dir = output_dir_arg;
  filesystem::create_directories(output_dir);

  ofstream gt(output_dir / "groundtruth.txt");
  ofstream intr(output_dir / "intrinsics.txt");
  intr << width << " " << height << " " << cam.fx << " " << cam.fy << " " << cam.cx << " "
       << cam.cy << endl;

  Light light;
  vector<double> render_times_ms;

  for (int i = 0; i < num_frames; ++i) {
    auto t0 = chrono::steady_clock::now();

    // Trajectory stores T_world_camera; rendering needs the inverse
    // (world -> camera) per this repo's (R,t) convention.
    Sophus::SE3d T_wc = T_wc_list[i];
    Sophus::SE3d T_cw = T_wc.inverse();
    cv::Mat R_cw = ToCvMat3(T_cw.rotationMatrix());
    cv::Mat t_cw = ToCvVec3(T_cw.translation());

    RenderTarget rt = RenderMesh(mesh, cam, width, height, R_cw, t_cw, light);
    cv::Mat depth16 = DepthMetersToTUM16U(rt.depth_m);

    auto t1 = chrono::steady_clock::now();
    render_times_ms.push_back(chrono::duration<double, milli>(t1 - t0).count());

    string idx = ZeroPad(i, 6);
    cv::imwrite((output_dir / (idx + ".png")).string(), rt.rgb);
    cv::imwrite((output_dir / (idx + "_depth.png")).string(), depth16);

    Eigen::Quaterniond q = T_wc.unit_quaternion();
    Eigen::Vector3d tr = T_wc.translation();
    gt << fixed << setprecision(6) << (double)i << " " << tr.x() << " " << tr.y() << " "
       << tr.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
  gt.close();
  intr.close();

  Stats render_stats = ComputeStats(render_times_ms);

  vector<Eigen::Vector3f> vis_vertices = mesh.vertices;
  vector<Eigen::Vector3i> vis_faces = mesh.faces;
  vector<Rgb8> vis_colors(mesh.vertices.size(), Rgb8{180, 180, 180});

  float line_thickness = bound_radius * 0.01f;
  AppendTrajectoryPolyline(vis_vertices, vis_faces, vis_colors, T_wc_list, line_thickness,
                           Rgb8{255, 0, 0}, Rgb8{255, 255, 0});

  int frustum_stride = max(1, num_frames / 12);
  for (int i = 0; i < num_frames; i += frustum_stride) {
    AppendFrustumWireframe(vis_vertices, vis_faces, vis_colors, T_wc_list[i], cam, width, height,
                          bound_radius * 0.3f, line_thickness, Rgb8{0, 255, 255});
  }

  filesystem::path vis_dir = "../output/synthetic_bunny";
  filesystem::create_directories(vis_dir);
  filesystem::path vis_path = vis_dir / "scene_visualization.ply";
  WritePLYMesh(vis_path.string(), vis_vertices, vis_faces, vis_colors);

  cout << "\n=== BUNNY RENDER SUMMARY ===" << endl;
  cout << "frames: " << num_frames << " (" << width << "x" << height << ")" << endl;
  PrintStats("render time (ms)", render_stats);
  cout << "wrote dataset to: " << output_dir.string() << endl;
  cout << "wrote visualization to: " << vis_path.string() << endl;

  return 0;
}
