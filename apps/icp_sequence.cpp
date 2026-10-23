#include "tracking/rgbd_icp.hpp"
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
using namespace rgbd_icp;
namespace fs = std::filesystem;
void writeCloudAudit(const std::string &out, int i, const PairClouds &clouds,
                    const PairResult &result) {
  fs::path dest = fs::path(out) / ("audit_pair_" + std::to_string(i) + "_" +
                                 std::to_string(i + 1));
  fs::create_directories(dest);
  std::ofstream cloud(dest / "clouds.csv"), poses(dest / "stage_poses.csv");
  cloud << "role,index,x,y,z,nx,ny,nz\n" << std::setprecision(15);
  for (auto role : {"source", "target"}) {
    bool target = std::string(role) == "target";
    const auto &points = target ? clouds.target : clouds.source;
    for (size_t k = 0; k < points.size(); ++k) {
      V3 normal = target ? clouds.normals[k] : V3::Zero();
      cloud << role << ',' << k;
      for (int a = 0; a < 3; ++a) cloud << ',' << points[k][a];
      for (int a = 0; a < 3; ++a) cloud << ',' << normal[a];
      cloud << '\n';
    }
  }
  poses << "stage,r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n"
        << std::setprecision(15);
  std::vector<std::pair<std::string, Pose>> stages{
      {"unaligned", Pose{}}, {"baseline", result.baseline},
      {"robust", result.robust}, {"point", result.point},
      {"plane", result.plane}};
  for (const auto &[name, pose] : stages) {
    poses << name;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) poses << ',' << pose.R(r, c);
    for (int r = 0; r < 3; ++r) poses << ',' << pose.t[r];
    poses << '\n';
  }
  if (!cloud || !poses) throw std::runtime_error("Cannot write cloud audit");
}
std::vector<Pose> readGT(const std::string &dir) {
  std::ifstream f(dir + "/groundtruth.txt");
  if (!f)
    throw std::runtime_error("Cannot open ground truth for evaluation");
  std::vector<Pose> poses;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    double stamp, tx, ty, tz, qx, qy, qz, qw;
    std::istringstream row(line);
    if (!(row >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
      throw std::runtime_error("Malformed GT row");
    Eigen::Quaterniond q(qw, qx, qy, qz);
    if (q.norm() < 1e-10)
      throw std::runtime_error("Invalid quaternion");
    poses.push_back({q.normalized().toRotationMatrix(), V3(tx, ty, tz)});
  }
  return poses;
}
int main(int argc, char **argv) {
  try {
    std::string dir = argc > 1 ? argv[1] : "data/synthetic_bunny",
                out = argc > 2 ? argv[2] : "icp_cpp_results";
    Config c;
    std::cout << "ICP solvers: point-to-point damped Gauss-Newton; "
                 "point-to-plane Huber-weighted damped Gauss-Newton\n";
    Camera k = readCamera(dir);
    fs::create_directories(out);
    // Invalidate completion before starting: a failed run cannot be displayed
    // as fresh.
    fs::remove(out + "/run_complete.txt");
    fs::remove(out + "/selected_trajectory.txt");
    fs::remove(out + "/pose_graph_trajectory.txt");
    // Frame availability, not ground truth, determines the inference sequence.
    int count = 0;
    for (;; ++count) {
      std::ostringstream path;
      path << dir << '/' << std::setw(6) << std::setfill('0') << count
           << ".png";
      if (!fs::exists(path.str()))
        break;
    }
    if (count < 2)
      throw std::runtime_error("Need >=2 consecutive RGB-D frames");
    std::map<std::string, std::vector<Pose>> trajectories;
    std::map<std::string, Pose> velocity;
    for (auto name : {"baseline", "robust", "point", "plane"})
      trajectories[name] = {Pose{}};
    std::vector<Edge> odometry;
    int failures = 0;
    std::ofstream metadata(out + "/solver_metadata.txt");
    metadata << "point_solver=damped_gauss_newton\nplane_solver=huber_damped_gauss_newton\n"
                "initialization=feature_3d_ransac_svd\nupdate=left_SO3_R3_retraction\n";
    std::ofstream pairs(out + "/pair_metrics.csv");
    std::ofstream history(out + "/iteration_history.csv");
    history << "pair,stage,iteration,pairs,nn_rmse_before_m,nn_rmse_after_m,"
               "normal_rmse_before_m,normal_rmse_after_m,objective_before_m2,"
               "objective_after_m2,rotation_step_deg,translation_step_m,"
               "assignment_hash\n" << std::setprecision(15);
    pairs
        << "pair,status,matches,valid_3d,inliers,inlier_ratio,point_iterations,"
           "plane_iterations,point_pairs,plane_pairs,point_rmse_m,plane_rmse_m,"
           "point_converged,plane_converged,point_stop,plane_stop\n";
    pairs << std::setprecision(12);
    for (int i = 0; i < count - 1; ++i) {
      PairClouds clouds;
      auto r = registerFrames(readFrame(dir, i, k), readFrame(dir, i + 1, k), k,
                              c, c.seed + i, i == 15 ? &clouds : nullptr);
      if (i == 15 && r.accepted) writeCloudAudit(out, i, clouds, r);
      for (auto stage : {"point", "plane"}) {
        const auto &info = std::string(stage) == "point" ? r.point_info : r.plane_info;
        for (const auto &row : info.history)
          history << i << ',' << stage << ',' << row.iteration << ',' << row.pairs
                  << ',' << row.rmse_before << ',' << row.rmse_after
                  << ',' << row.normal_before << ',' << row.normal_after
                  << ',' << row.objective_before << ',' << row.objective_after
                  << ',' << row.rotation_step_deg << ',' << row.translation_step_m
                  << ',' << row.assignment_hash << '\n';
      }
      if (!r.accepted)
        ++failures;
      else
        odometry.push_back({i, i + 1, r.plane, false});
      std::map<std::string, Pose> relative{{"baseline", r.baseline},
                                           {"robust", r.robust},
                                           {"point", r.point},
                                           {"plane", r.plane}};
      for (auto &[name, poses] : trajectories) {
        Pose step = r.accepted ? relative[name] : velocity[name];
        if (r.accepted)
          velocity[name] = step;
        poses.push_back(poses.back() * step);
      }
      pairs << i << ',' << r.status << ',' << r.matches << ',' << r.valid3d
            << ',' << r.inliers << ',' << r.inlier_ratio << ','
            << r.point_info.iterations << ',' << r.plane_info.iterations << ','
            << r.point_info.pairs << ',' << r.plane_info.pairs << ','
            << r.point_info.rmse << ',' << r.plane_info.rmse << ','
            << r.point_info.converged << ',' << r.plane_info.converged << ','
            << r.point_info.reason << ',' << r.plane_info.reason << '\n';
      std::cout << i << "->" << i + 1 << ": " << r.status
                << " inliers=" << r.inliers
                << " plane RMSE=" << r.plane_info.rmse << '\n';
    }
    auto loops = findLoops(dir, k, c, trajectories["plane"]);
    std::ofstream loopFile(out + "/loop_edges.csv");
    loopFile << "i,j\n";
    for (auto e : loops) {
      odometry.push_back(e);
      loopFile << e.i << ',' << e.j << '\n';
    }
    std::cout << "Accepted loops: " << loops.size() << '\n';
    if (!loops.empty()) {
      auto graph = optimizeGraph(trajectories["plane"], odometry);
      std::cout << "Graph converged=" << graph.converged
                << " iterations=" << graph.iterations
                << " cost=" << graph.initial_cost << " -> " << graph.final_cost
                << '\n';
      if (graph.converged && graph.final_cost <= graph.initial_cost)
        trajectories["pose_graph"] = graph.poses;
      else
        std::cerr << "Graph did not converge; retaining odometry\n";
    }
    // Inference above never reads GT. Evaluate only after all estimation is
    // complete.
    bool hasGT = fs::exists(dir + "/groundtruth.txt");
    std::vector<Pose> gt;
    if (hasGT) {
      gt = readGT(dir);
      if (int(gt.size()) != count)
        throw std::runtime_error("GT/frame count mismatch");
    }
    std::ofstream summary(out + "/summary_metrics.csv");
    summary << "method,ATE_RMSE_m,mean_RPE_rotation_deg,mean_RPE_translation_m,"
               "final_position_error_m,final_rotation_error_deg,accepted_pairs,"
               "failed_pairs\n"
            << std::setprecision(12);
    for (auto &[name, poses] : trajectories) {
      std::ofstream file(out + "/" + name + "_trajectory.txt");
      file << "# frame tx ty tz qx qy qz qw (T_wc; frame-zero world if no GT)\n"
           << std::setprecision(12);
      double sum = 0, rot = 0, trans = 0;
      std::vector<Pose> aligned;
      for (int i = 0; i < count; ++i) {
        Pose world = hasGT ? gt[0] * poses[i] : poses[i];
        aligned.push_back(world);
        Eigen::Quaterniond q(world.R);
        file << i << ' ' << world.t.transpose() << ' ' << q.x() << ' ' << q.y()
             << ' ' << q.z() << ' ' << q.w() << '\n';
        if (hasGT)
          sum += (world.t - gt[i].t).squaredNorm();
        if (hasGT && i) {
          Pose err = (gt[i - 1].inverse() * gt[i]).inverse() *
                     (aligned[i - 1].inverse() * world);
          rot += angleDeg(err.R);
          trans += err.t.norm();
        }
      }
      if (hasGT) {
        double ate = std::sqrt(sum / count);
        summary << name << ',' << ate << ',' << rot / (count - 1) << ','
                << trans / (count - 1) << ','
                << (aligned.back().t - gt.back().t).norm() << ','
                << angleDeg(aligned.back().R * gt.back().R.transpose()) << ','
                << count - 1 - failures << ',' << failures << '\n';
        std::cout << name << " ATE RMSE=" << ate << " m\n";
      }
    }
    // Blender reads this explicit current-run choice, never an old leftover
    // file.
    {
      std::ofstream selected(out + "/selected_trajectory.txt");
      selected << (trajectories.count("pose_graph") ? "pose_graph" : "plane")
               << '\n';
      if (!selected)
        throw std::runtime_error("Failed to write trajectory selection");
    }
    {
      std::ofstream done(out + "/run_complete.txt");
      done << "frames=" << count << " failed_pairs=" << failures << '\n';
      if (!done)
        throw std::runtime_error("Failed to write completion marker");
    }
    std::cout << "Accepted " << count - 1 - failures << '/' << count - 1
              << " pairs. Results: " << out << '\n';
    return failures ? 2 : 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
