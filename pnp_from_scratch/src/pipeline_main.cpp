// scratch_pipeline: the complete, self-contained scratch pipeline over the
// synthetic Bunny sequence -- RGB-D PNGs in, a 36-pose trajectory out.
// Standard library + Eigen + the project-owned feature modules only.
//
// Usage: scratch_pipeline [--repo DIR] [--frames N] [--fast-threshold T]
//                         [--pnp-iterations I] [--pnp-threshold PX] [--seed S]
// Output (generated, gitignored): pnp_from_scratch/results/pipeline/
//   pairs.csv                      one row per pair (counts, inliers, poses, errors)
//   trajectory.csv / trajectory.txt   36 poses T_wc (CSV with errors; TUM format)
//   correspondences/pair_<i>_<j>.csv  every filtered match with depth, 3D point,
//                                     essential and PnP inlier flags, residual
//   summary.txt                    the summary printed at the end

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "pipeline.hpp"

using namespace scratch;

namespace {

Pose Inverse(const Pose &T) {
  Pose r;
  r.R = T.R.transpose();
  r.t = -r.R * T.t;
  return r;
}
Pose Compose(const Pose &A, const Pose &B) {
  Pose r;
  r.R = A.R * B.R;
  r.t = A.R * B.t + A.t;
  return r;
}
std::string F(double v, int p = 4) {
  if (!std::isfinite(v)) return "";
  std::ostringstream o;
  o << std::fixed << std::setprecision(p) << v;
  return o.str();
}
double DirErrDeg(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  if (a.norm() == 0 || b.norm() == 0) return NAN;
  return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)) * 180.0 / M_PI;
}
struct Stat {
  double mean = NAN, median = NAN, max = NAN;
};
Stat Stats(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  s.max = v.back();
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  PipelineParams params;
  fs::path repo;
  int max_frames = 0;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto next = [&] { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
    if (s == "--repo") repo = next();
    else if (s == "--frames") max_frames = std::stoi(next());
    else if (s == "--fast-threshold") params.frontend.fast.threshold = std::stoi(next());
    else if (s == "--pnp-iterations") params.pnp_ransac.iterations = std::stoi(next());
    else if (s == "--pnp-threshold") params.pnp_ransac.threshold_px = std::stod(next());
    else if (s == "--seed") params.pnp_ransac.seed = uint32_t(std::stoul(next()));
    else {
      std::cerr << "unknown option " << s << "\n";
      return 2;
    }
  }
  if (repo.empty()) repo = FindRepo(fs::current_path());
  if (repo.empty()) repo = FindRepo(fs::path(argv[0]).parent_path());
  if (repo.empty()) {
    std::cerr << "cannot find the repository root; pass --repo DIR\n";
    return 2;
  }
  const fs::path ds = repo / "data" / "synthetic_bunny";
  Intrinsics K;
  {
    std::ifstream f(ds / "intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  const std::vector<Pose> Twc_gt = ReadGroundtruthTwc(ds / "groundtruth.txt");
  int N = int(Twc_gt.size());
  if (max_frames > 1) N = std::min(N, max_frames);

  std::cout << "scratch pipeline: FAST threshold " << params.frontend.fast.threshold << ", strongest "
            << params.frontend.fast.max_keypoints << ", match filter max(2 d_min, " << params.frontend.match_floor
            << "); PnP RANSAC " << params.pnp_ransac.iterations << " it, " << params.pnp_ransac.threshold_px
            << " px, seed " << params.pnp_ransac.seed << "; essential RANSAC " << params.essential_ransac.iterations
            << " it, " << params.essential_ransac.threshold_px << " px\n";

  // ---- per pair ----
  std::vector<PairResult> pairs;
  RgbdFrame prev_frame, cur_frame;
  FrameFeatures prev_feat, cur_feat;
  std::string err;
  if (!LoadRgbdFrame(ds.string(), 0, prev_frame, &err)) {
    std::cerr << err << "\n";
    return 1;
  }
  prev_feat = ExtractFeatures(prev_frame.rgb, params.frontend);
  for (int i = 0; i + 1 < N; ++i) {
    if (!LoadRgbdFrame(ds.string(), i + 1, cur_frame, &err)) {
      std::cerr << err << "\n";
      return 1;
    }
    cur_feat = ExtractFeatures(cur_frame.rgb, params.frontend);
    pairs.push_back(ProcessPair(i, i + 1, prev_feat, cur_feat, prev_frame, cur_frame, K, params));
    std::swap(prev_frame, cur_frame);
    std::swap(prev_feat, cur_feat);
  }

  // ---- trajectory: T_wc[0] = T_gt[0], T_wc[i+1] = T_wc[i] * T_{i+1<-i}^-1 (failed pair: previous pose held) ----
  std::vector<Pose> Twc(N);
  std::vector<std::string> source(N, "scratch_pipeline");
  Twc[0] = Twc_gt[0];
  source[0] = "gt_anchor";
  for (int i = 0; i + 1 < N; ++i) {
    if (pairs[i].pnp_ok) {
      Pose rel;
      rel.R = pairs[i].R;
      rel.t = pairs[i].t;
      Twc[i + 1] = Compose(Twc[i], Inverse(rel));
    } else {
      Twc[i + 1] = Twc[i];
      source[i + 1] = "held_previous_pose_pair_failed";
    }
  }

  // ---- errors vs ground truth ----
  std::vector<double> rot_err(pairs.size(), NAN), trans_err(pairs.size(), NAN), e_rot(pairs.size(), NAN),
      e_dir(pairs.size(), NAN);
  for (size_t k = 0; k < pairs.size(); ++k) {
    const Pose G = Compose(Inverse(Twc_gt[k + 1]), Twc_gt[k]);  // T_{k+1<-k}
    if (pairs[k].pnp_ok) {
      rot_err[k] = RotErrDeg(G.R, pairs[k].R);
      trans_err[k] = (G.t - pairs[k].t).norm();
    }
    if (pairs[k].essential_ok) {
      e_rot[k] = RotErrDeg(G.R, pairs[k].R_essential);
      e_dir[k] = DirErrDeg(G.t, pairs[k].t_essential_dir);
    }
  }
  std::vector<double> abs_t(N), abs_r(N);
  for (int k = 0; k < N; ++k) {
    abs_t[k] = (Twc_gt[k].t - Twc[k].t).norm();
    abs_r[k] = RotErrDeg(Twc_gt[k].R, Twc[k].R);
  }

  // ---- outputs ----
  const fs::path out = repo / "pnp_from_scratch" / "results" / "pipeline";
  fs::create_directories(out / "correspondences");
  {
    std::ofstream f(out / "pairs.csv");
    f << "pair,frame_i,frame_j,keypoints_i,keypoints_j,raw_matches,filtered_matches,d_min,correspondences_3d2d,"
         "pnp_ok,pnp_inliers,pnp_best_sample_inliers,reproj_inlier_mean_px,reproj_inlier_median_px,"
         "reproj_inlier_max_px,reproj_all_median_px,rot_err_deg,trans_err_m,tx,ty,tz,"
         "r00,r01,r02,r10,r11,r12,r20,r21,r22,est_rotation_deg,"
         "essential_ok,essential_inliers,essential_in_front,essential_rot_err_deg,essential_t_dir_err_deg,"
         "pnp_vs_essential_rot_deg\n";
    for (size_t k = 0; k < pairs.size(); ++k) {
      const PairResult &p = pairs[k];
      f << k << "," << p.i << "," << p.j << "," << p.keypoints_i << "," << p.keypoints_j << "," << p.raw_matches
        << "," << p.filtered_matches << "," << p.d_min << "," << p.pnp_index.size() << "," << int(p.pnp_ok) << ","
        << p.pnp_inliers << "," << p.pnp_best_sample_inliers << "," << F(p.reproj_inlier_mean) << ","
        << F(p.reproj_inlier_median) << "," << F(p.reproj_inlier_max) << "," << F(p.reproj_all_median) << ","
        << F(rot_err[k], 6) << "," << F(trans_err[k], 6) << "," << F(p.t.x(), 9) << "," << F(p.t.y(), 9) << ","
        << F(p.t.z(), 9) << ",";
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) f << F(p.R(a, b), 9) << ",";
      f << F(RotAngleDeg(p.R), 6) << "," << int(p.essential_ok) << "," << p.essential_inliers
        << "," << p.essential_in_front << "," << F(e_rot[k], 6) << "," << F(e_dir[k], 6) << ","
        << (p.pnp_ok && p.essential_ok ? F(RotErrDeg(p.R, p.R_essential), 6) : "") << "\n";
      std::ofstream c(out / "correspondences" / ("pair_" + std::to_string(p.i) + "_" + std::to_string(p.j) + ".csv"));
      c << "match,query,train,hamming,u_i,v_i,u_j,v_j,depth_raw_i,depth_raw_j,X_i,Y_i,Z_i,essential_inlier,"
           "pnp_used,pnp_inlier,pnp_residual_px\n";
      std::vector<int> pnp_of(p.correspondences.size(), -1);
      for (size_t q = 0; q < p.pnp_index.size(); ++q) pnp_of[p.pnp_index[q]] = int(q);
      for (size_t m = 0; m < p.correspondences.size(); ++m) {
        const Correspondence &cc = p.correspondences[m];
        const int q = pnp_of[m];
        c << m << "," << cc.query << "," << cc.train << "," << cc.hamming << "," << cc.uv_i.x() << "," << cc.uv_i.y()
          << "," << cc.uv_j.x() << "," << cc.uv_j.y() << "," << cc.depth_raw_i << "," << cc.depth_raw_j << ",";
        if (cc.has_3d) c << F(cc.X_i.x(), 6) << "," << F(cc.X_i.y(), 6) << "," << F(cc.X_i.z(), 6) << ",";
        else c << ",,,";
        c << (m < p.essential_mask.size() ? int(p.essential_mask[m]) : 0) << "," << int(q >= 0) << ","
          << (q >= 0 && p.pnp_ok ? int(p.pnp_mask[q]) : 0) << ","
          << (q >= 0 && p.pnp_ok ? F(p.pnp_residual_px[q], 4) : "") << "\n";
      }
    }
  }
  {
    std::ofstream f(out / "trajectory.csv"), t(out / "trajectory.txt");
    f << "# Scratch pipeline trajectory, T_wc (camera -> world); FAST + rotated BRIEF + Hamming + "
         "RANSAC(linear PnP), all project-owned\n"
      << "# accumulation: T_wc[0] = T_gt[0], T_wc[i+1] = T_wc[i] * T_{i+1<-i}^-1 (failed pair: previous pose held)\n"
      << "frame,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22,qx,qy,qz,qw,rotation_error_deg,translation_error_m,"
         "pnp_inliers,pose_source\n";
    t << "# Scratch pipeline trajectory (FAST + rotated BRIEF + Hamming + RANSAC linear PnP)\n"
      << "# timestamp tx ty tz qx qy qz qw  (T_wc: camera -> world)\n";
    for (int k = 0; k < N; ++k) {
      const Eigen::Vector4d q = RToQuat(Twc[k].R);
      f << k << "," << F(Twc[k].t.x(), 9) << "," << F(Twc[k].t.y(), 9) << "," << F(Twc[k].t.z(), 9);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) f << "," << F(Twc[k].R(a, b), 9);
      f << "," << F(q(0), 9) << "," << F(q(1), 9) << "," << F(q(2), 9) << "," << F(q(3), 9) << "," << F(abs_r[k], 6)
        << "," << F(abs_t[k], 9) << "," << (k ? std::to_string(pairs[k - 1].pnp_inliers) : "") << "," << source[k]
        << "\n";
      t << std::fixed << std::setprecision(6) << double(k) << " " << Twc[k].t.x() << " " << Twc[k].t.y() << " "
        << Twc[k].t.z() << " " << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << "\n";
    }
  }

  // ---- summary ----
  std::vector<double> kp, filt, c3, inl, rep;
  int failed = 0, e_failed = 0;
  for (const auto &p : pairs) {
    kp.push_back(p.keypoints_i);
    filt.push_back(p.filtered_matches);
    c3.push_back(double(p.pnp_index.size()));
    if (p.pnp_ok) inl.push_back(p.pnp_inliers), rep.push_back(p.reproj_inlier_mean);
    failed += !p.pnp_ok;
    e_failed += !p.essential_ok;
  }
  kp.push_back(pairs.back().keypoints_j);
  const Stat sr = Stats(std::vector<double>(rot_err.begin(), rot_err.end())),
             st = Stats(std::vector<double>(trans_err.begin(), trans_err.end()));
  std::vector<double> er, ed;
  for (size_t k = 0; k < pairs.size(); ++k)
    if (pairs[k].essential_ok) er.push_back(e_rot[k]), ed.push_back(e_dir[k]);
  size_t worst = 0;
  for (size_t k = 1; k < pairs.size(); ++k)
    if (rot_err[k] > rot_err[worst]) worst = k;
  std::ostringstream o;
  o << "========================================\n"
    << "SCRATCH PIPELINE -- FULL SEQUENCE\n"
    << "========================================\n"
    << "Pairs: " << pairs.size() << " attempted, " << pairs.size() - failed << " solved by RANSAC PnP, " << failed
    << " failed (previous pose held)\n"
    << "Poses: " << N << "\n\n"
    << "Front end (per frame / pair, mean): keypoints " << F(Stats(kp).mean, 1) << ", filtered matches "
    << F(Stats(filt).mean, 1) << ", 3D->2D correspondences " << F(Stats(c3).mean, 1) << "\n"
    << "RANSAC PnP inliers (mean / min): " << F(Stats(inl).mean, 1) << " / "
    << F(*std::min_element(inl.begin(), inl.end()), 0) << "; inlier reprojection error mean " << F(Stats(rep).mean, 2)
    << " px\n\n"
    << "Per-pair PnP error vs ground truth:\n"
    << "  rotation    mean " << F(sr.mean, 3) << " deg, median " << F(sr.median, 3) << " deg, max " << F(sr.max, 3)
    << " deg (pair " << worst << "->" << worst + 1 << ")\n"
    << "  translation mean " << F(st.mean, 4) << " m, median " << F(st.median, 4) << " m, max " << F(st.max, 4)
    << " m\n"
    << "Per-pair essential-matrix estimate (scale-free, diagnostic): " << pairs.size() - e_failed << " solved;\n"
    << "  rotation error mean " << F(Stats(er).mean, 3) << " deg, median " << F(Stats(er).median, 3)
    << " deg; translation-direction error median " << F(Stats(ed).median, 2) << " deg\n\n"
    << "Trajectory (frame " << N - 1 << "): position error " << F(abs_t[N - 1], 4) << " m, rotation error "
    << F(abs_r[N - 1], 3) << " deg\n\n"
    << "Pair 0->1: " << pairs[0].keypoints_i << "/" << pairs[0].keypoints_j << " keypoints, "
    << pairs[0].filtered_matches << " filtered matches, " << pairs[0].pnp_index.size() << " 3D->2D, "
    << pairs[0].pnp_inliers << " RANSAC inliers; rotation error " << F(rot_err[0], 3) << " deg, translation error "
    << F(trans_err[0], 4) << " m, inlier reprojection mean/median/max " << F(pairs[0].reproj_inlier_mean, 2) << "/"
    << F(pairs[0].reproj_inlier_median, 2) << "/" << F(pairs[0].reproj_inlier_max, 2) << " px\n"
    << "========================================\n";
  std::cout << "\npair   kp_i kp_j filt 3d2d inl  rot_err  trans_err  reproj  | E: inl rot_err t_dir_err\n";
  for (size_t k = 0; k < pairs.size(); ++k) {
    const PairResult &p = pairs[k];
    std::cout << std::setw(2) << p.i << "->" << std::left << std::setw(3) << p.j << std::right << std::setw(5)
              << p.keypoints_i << std::setw(5) << p.keypoints_j << std::setw(5) << p.filtered_matches << std::setw(5)
              << p.pnp_index.size() << std::setw(4) << p.pnp_inliers << std::setw(9) << F(rot_err[k], 3)
              << std::setw(11) << F(trans_err[k], 4) << std::setw(8) << F(p.reproj_inlier_mean, 2) << "  |"
              << std::setw(6) << p.essential_inliers << std::setw(8) << F(e_rot[k], 2) << std::setw(10)
              << F(e_dir[k], 1) << (p.pnp_ok ? "" : "  PnP FAILED: " + p.pnp_reason) << "\n";
  }
  std::cout << "\n" << o.str();
  std::ofstream(out / "summary.txt") << o.str();
  std::cout << "wrote pnp_from_scratch/results/pipeline/{pairs.csv,trajectory.csv,trajectory.txt,correspondences/,"
               "summary.txt}\n";
  return failed == int(pairs.size()) ? 1 : 0;
}
