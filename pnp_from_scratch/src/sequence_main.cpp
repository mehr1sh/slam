// Milestone 2: the SAME scratch linear PnP (src/pnp.cpp, unchanged) on ALL 35
// consecutive pairs of the synthetic Bunny sequence, using every raw 3D->2D
// correspondence, then accumulated into a 36-pose trajectory.
//
// Standard library + Eigen only; no RANSAC result of the reference pipeline is read: a
// correspondence is used iff the pipeline could back-project it (valid depth
// in frame i), exactly how the existing pipeline builds its PnP input.
//
// Usage: pnp_full_sequence [--repo DIR] [--corr-dir DIR]
//   --corr-dir DIR  directory with pair_<i>_<i+1>_correspondences.csv
//                   (default: the frozen baseline export,
//                    <repo>/docs/migration/baseline/correspondences; a fresh export
//                    of the reference pipeline: pnp_from_scratch/export_correspondences.sh)

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "pnp.hpp"
#include "projection.hpp"

using namespace scratch;

namespace {

Pose Inverse(const Pose &T) {
  Pose r;
  r.R = T.R.transpose();
  r.t = -r.R * T.t;
  return r;
}
Pose Compose(const Pose &A, const Pose &B) {  // (A * B) x = A (B x)
  Pose r;
  r.R = A.R * B.R;
  r.t = A.R * B.t + A.t;
  return r;
}

struct PairRow {
  int pair = 0, filtered_matches = 0, n = 0, positive_depth = 0;
  bool success = false;       // solver returned a finite, proper rotation
  bool cheirality_ok = false; // majority of points in front of camera i+1 (else a physically invalid pose)
  std::string reason;
  Pose T;  // T_{i+1<-i}
  double est_rot = NAN, gt_rot = NAN, rot_err = NAN, trans_err = NAN, gt_tnorm = NAN;
  Reproj rp;
  double sigma_ratio = NAN;
  Vector3d block_sv = Vector3d::Constant(NAN);
};

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

std::string F(double v, int p = 3) {
  if (!std::isfinite(v)) return "";
  std::ostringstream o;
  o << std::fixed << std::setprecision(p) << v;
  return o.str();
}

}  // namespace

int main(int argc, char **argv) {
  fs::path repo, corr_dir;
  for (int a = 1; a < argc; ++a) {
    std::string s = argv[a];
    if (s == "--repo" && a + 1 < argc) repo = argv[++a];
    else if (s == "--corr-dir" && a + 1 < argc) corr_dir = argv[++a];
  }
  if (repo.empty()) repo = FindRepo(fs::current_path());
  if (repo.empty()) repo = FindRepo(fs::path(argv[0]).parent_path());
  if (repo.empty()) {
    std::cerr << "cannot find the repository root; pass --repo DIR\n";
    return 2;
  }
  if (corr_dir.empty()) corr_dir = repo / "docs" / "migration" / "baseline" / "correspondences";
  const fs::path ds = repo / "data" / "synthetic_bunny";
  Intrinsics K;
  {
    std::ifstream f(ds / "intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  const std::vector<Pose> Twc_gt = ReadGroundtruthTwc(ds / "groundtruth.txt");
  const int N = static_cast<int>(Twc_gt.size());
  if (!fs::exists(corr_dir / "pair_0_1_correspondences.csv")) {
    std::cerr << "missing " << corr_dir / "pair_0_1_correspondences.csv" << "\n"
              << "Create the raw correspondences of all pairs with:  pnp_from_scratch/export_correspondences.sh\n";
    return 2;
  }

  // ---------------------------------------------------------------- pairs --
  std::vector<PairRow> pairs;
  for (int i = 0; i + 1 < N; ++i) {
    const int j = i + 1;
    PairRow r;
    r.pair = i;
    Csv c;
    const fs::path path = corr_dir / ("pair_" + std::to_string(i) + "_" + std::to_string(j) + "_correspondences.csv");
    if (!c.load(path)) {
      r.reason = "missing " + path.filename().string();
      pairs.push_back(r);
      continue;
    }
    r.filtered_matches = static_cast<int>(c.rows.size());
    // Raw 3D->2D correspondences: every filtered match whose frame-i pixel has
    // depth (the pipeline's back-projection rule). The 3D point is re-checked
    // against (u_i, v_i, depth) with intrinsics.txt; no inlier column is read.
    std::vector<Vector3d> X;
    std::vector<Vector2d> uv;
    int inconsistent = 0;
    for (size_t k = 0; k < c.rows.size(); ++k) {
      const int d = c.i(k, "depth_raw_i");
      if (d <= 0) continue;
      const Vector3d P(c.d(k, "X_i"), c.d(k, "Y_i"), c.d(k, "Z_i"));
      const double Z = d / 5000.0;
      const Vector3d B((c.d(k, "u_i") - K.cx) / K.fx * Z, (c.d(k, "v_i") - K.cy) / K.fy * Z, Z);
      if ((P - B).norm() > 2e-5) ++inconsistent;
      X.push_back(P);
      uv.push_back(Vector2d(c.d(k, "u_j"), c.d(k, "v_j")));
    }
    r.n = static_cast<int>(X.size());
    if (inconsistent) {
      std::cerr << "pair " << i << ": " << inconsistent << " 3D points disagree with the back-projection -- aborting\n";
      return 1;
    }
    const PnPResult s = solvePnPDLT(X, uv, K);  // the unchanged milestone-1 solver
    const Pose G = Compose(Inverse(Twc_gt[j]), Twc_gt[i]);  // T_{j<-i} = T_wc[j]^-1 T_wc[i]
    r.gt_rot = RotAngleDeg(G.R);
    r.gt_tnorm = G.t.norm();
    r.sigma_ratio = s.nullspace_ratio;
    r.block_sv = s.sv_left_block;
    if (!s.ok) {
      r.reason = s.reason;
      pairs.push_back(r);
      continue;
    }
    r.T.R = s.R;
    r.T.t = s.t;
    r.rp = Reprojection(X, uv, r.T, K);
    r.positive_depth = r.rp.positive_depth;
    r.est_rot = RotAngleDeg(r.T.R);
    r.rot_err = RotErrDeg(G.R, r.T.R);
    r.trans_err = (G.t - r.T.t).norm();
    r.success = std::isfinite(r.rot_err) && std::isfinite(r.trans_err) && std::isfinite(r.rp.mean) &&
                (r.T.R.transpose() * r.T.R - Matrix3d::Identity()).norm() < 1e-9 &&
                std::abs(r.T.R.determinant() - 1) < 1e-9;
    if (!r.success) r.reason = "non-finite or invalid rotation";
    r.cheirality_ok = 2 * r.positive_depth > r.n;
    if (r.success && !r.cheirality_ok)
      r.reason = "pose returned; only " + std::to_string(r.positive_depth) + "/" + std::to_string(r.n) +
                 " points in front of the camera";
    pairs.push_back(r);
  }

  // ---------------------------------------------------------- trajectory --
  // Existing project convention (tests/synthetic/slam_trajectory_test.cpp):
  //   T_wc[0] = T_gt[0];  T_wc[i+1] = T_wc[i] * T_{i+1<-i}^-1;
  //   a failed pair keeps the previous pose (and is flagged).
  std::vector<Pose> Twc(N);
  std::vector<std::string> source(N);
  Twc[0] = Twc_gt[0];
  source[0] = "gt_anchor";
  for (int i = 0; i + 1 < N; ++i) {
    if (pairs[i].success) {
      Twc[i + 1] = Compose(Twc[i], Inverse(pairs[i].T));
      source[i + 1] = "scratch_linear_pnp";
    } else {
      Twc[i + 1] = Twc[i];
      source[i + 1] = "held_previous_pose_pair_failed";
    }
  }
  std::vector<double> abs_t(N), abs_r(N);
  for (int k = 0; k < N; ++k) {
    abs_t[k] = (Twc_gt[k].t - Twc[k].t).norm();  // camera-centre distance
    abs_r[k] = RotErrDeg(Twc_gt[k].R, Twc[k].R);
  }

  // -------------------------------------------------------------- output --
  const fs::path out = repo / "pnp_from_scratch" / "results";
  fs::create_directories(out);
  const std::string note =
      "Pure linear (DLT) PnP on ALL raw 3D->2D correspondences of every pair (filtered ORB matches with "
      "frame-i depth); no RANSAC, no reference-pipeline inlier selection or pose";
  {
    std::ofstream f(out / "scratch_pnp_pair_results.csv");
    f << "pair,frame_i,frame_j,filtered_matches,num_correspondences,success,cheirality_ok,reason,num_positive_depth,"
         "est_rotation_deg,est_tx,est_ty,est_tz,est_translation_norm_m,gt_rotation_deg,gt_translation_norm_m,"
         "rotation_error_deg,translation_error_m,mean_reprojection_px,median_reprojection_px,max_reprojection_px,"
         "rmse_reprojection_px,dlt_sigma12_over_sigma11,block_sv0,block_sv1,block_sv2\n";
    for (const auto &r : pairs)
      f << r.pair << "," << r.pair << "," << r.pair + 1 << "," << r.filtered_matches << "," << r.n << ","
        << int(r.success) << "," << int(r.cheirality_ok) << "," << r.reason << "," << r.positive_depth << "," << F(r.est_rot, 6) << ","
        << F(r.T.t.x(), 9) << "," << F(r.T.t.y(), 9) << "," << F(r.T.t.z(), 9) << "," << F(r.T.t.norm(), 9) << ","
        << F(r.gt_rot, 6) << "," << F(r.gt_tnorm, 9) << "," << F(r.rot_err, 6) << "," << F(r.trans_err, 9) << ","
        << F(r.rp.mean, 6) << "," << F(r.rp.median, 6) << "," << F(r.rp.max, 6) << "," << F(r.rp.rmse, 6) << ","
        << F(r.sigma_ratio, 6) << "," << F(r.block_sv(0), 6) << "," << F(r.block_sv(1), 6) << ","
        << F(r.block_sv(2), 6) << "\n";
  }
  {
    std::ofstream f(out / "scratch_pnp_trajectory.csv"), t(out / "scratch_pnp_trajectory.txt");
    f << "# Scratch linear PnP trajectory, T_wc (camera -> world), written by pnp_full_sequence\n"
      << "# note: " << note << "\n"
      << "# hud_note: pure linear PnP on raw matches, no RANSAC\n"
      << "# accumulation: T_wc[0] = T_gt[0], T_wc[i+1] = T_wc[i] * T_{i+1<-i}^-1 (failed pair: previous pose held)\n"
      << "# per-frame pair columns (reprojection, counts, success) describe pair (frame-1 -> frame)\n"
      << "frame,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22,qx,qy,qz,qw,rotation_error_deg,translation_error_m,"
         "mean_reprojection_px,median_reprojection_px,max_reprojection_px,num_correspondences,num_positive_depth,"
         "success,cheirality_ok,pose_source\n";
    t << "# Scratch linear PnP trajectory; " << note << "\n# timestamp tx ty tz qx qy qz qw  (T_wc: camera -> world)\n";
    for (int k = 0; k < N; ++k) {
      const Matrix3d &R = Twc[k].R;
      const Eigen::Vector4d q = RToQuat(R);
      const Vector3d &p = Twc[k].t;
      f << k << "," << F(p.x(), 9) << "," << F(p.y(), 9) << "," << F(p.z(), 9);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) f << "," << F(R(a, b), 9);
      f << "," << F(q(0), 9) << "," << F(q(1), 9) << "," << F(q(2), 9) << "," << F(q(3), 9) << ","
        << F(abs_r[k], 6) << "," << F(abs_t[k], 9);
      if (k == 0) {
        f << ",,,,,,1,1," << source[k] << "\n";
      } else {
        const PairRow &r = pairs[k - 1];
        f << "," << F(r.rp.mean, 6) << "," << F(r.rp.median, 6) << "," << F(r.rp.max, 6) << "," << r.n << ","
          << r.positive_depth << "," << int(r.success) << "," << int(r.cheirality_ok) << "," << source[k] << "\n";
      }
      t << std::fixed << std::setprecision(6) << double(k) << " " << p.x() << " " << p.y() << " " << p.z() << " "
        << q(0) << " " << q(1) << " " << q(2) << " " << q(3) << "\n";
    }
  }

  // ------------------------------------------------------------- summary --
  std::vector<double> re, te;
  int ok = 0;
  const PairRow *worst = nullptr;
  for (const auto &r : pairs) {
    if (!r.success) continue;
    ++ok;
    re.push_back(r.rot_err);
    te.push_back(r.trans_err);
    if (!worst || r.rot_err > worst->rot_err) worst = &r;
  }
  const Stat sr = Stats(re), st = Stats(te);
  std::string bad;
  int n_bad = 0;
  for (const auto &r : pairs)
    if (r.success && !r.cheirality_ok) {
      ++n_bad;
      bad += (bad.empty() ? "" : ", ") + std::to_string(r.pair) + "->" + std::to_string(r.pair + 1) + " (" +
             std::to_string(r.positive_depth) + "/" + std::to_string(r.n) + ")";
    }
  std::ostringstream o;
  o << "========================================\n"
    << "SCRATCH LINEAR PNP — FULL SEQUENCE\n"
    << "========================================\n\n"
    << "Pairs attempted: " << pairs.size() << "\n"
    << "Pairs successful: " << ok << "\n"
    << "Pairs failed: " << pairs.size() - ok << "\n"
    << "Pairs whose pose puts most points BEHIND the camera (physically invalid, still used as returned): " << n_bad
    << (n_bad ? "\n  " + bad : "") << "\n\n"
    << "Total poses: " << N << "\n\n"
    << "Per-pair rotation error:\n"
    << "  mean:   " << F(sr.mean) << " deg\n  median: " << F(sr.median) << " deg\n  max:    " << F(sr.max)
    << " deg" << (worst ? "  (pair " + std::to_string(worst->pair) + "->" + std::to_string(worst->pair + 1) + ")" : "")
    << "\n\nPer-pair translation error:\n"
    << "  mean:   " << F(st.mean, 4) << " m\n  median: " << F(st.median, 4) << " m\n  max:    " << F(st.max, 4)
    << " m\n\nTrajectory:\n"
    << "  final position error: " << F(abs_t[N - 1], 4) << " m\n"
    << "  final rotation error: " << F(abs_r[N - 1], 3) << " deg\n\n"
    << "Pair 0->1: " << pairs[0].n << " correspondences, rotation error " << F(pairs[0].rot_err) << " deg, "
    << "translation error " << F(pairs[0].trans_err, 4) << " m, mean reprojection " << F(pairs[0].rp.mean) << " px\n\n"
    << "NOTE:\nThis is PURE LINEAR PNP on RAW CORRESPONDENCES.\n"
    << "No RANSAC or reference-pipeline inlier selection was used.\n"
    << "========================================\n";
  std::cout << "pair  corr  ok  +depth  est_rot  rot_err  trans_err  reproj_mean  reproj_med  sigma12/11\n";
  for (const auto &r : pairs)
    std::cout << std::setw(2) << r.pair << "->" << std::left << std::setw(3) << r.pair + 1 << std::right
              << std::setw(4) << r.n << std::setw(4) << (r.success ? "Y" : "N") << std::setw(8) << r.positive_depth
              << std::setw(9) << F(r.est_rot, 2) << std::setw(9) << F(r.rot_err, 2) << std::setw(11)
              << F(r.trans_err, 4) << std::setw(13) << F(r.rp.mean, 2) << std::setw(12) << F(r.rp.median, 2)
              << std::setw(12) << F(r.sigma_ratio, 3)
              << (!r.success ? "  FAILED: " + r.reason : (!r.cheirality_ok ? "  INVALID: " + r.reason : "")) << "\n";
  std::cout << "\n" << o.str();
  std::ofstream(out / "scratch_pnp_full_sequence_summary.txt") << o.str();
  std::cout << "wrote pnp_from_scratch/results/scratch_pnp_trajectory.{csv,txt}, scratch_pnp_pair_results.csv, "
               "scratch_pnp_full_sequence_summary.txt\n";
  return 0;
}
