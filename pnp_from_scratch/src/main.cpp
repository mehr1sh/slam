// Milestone 1: from-scratch linear PnP on the existing synthetic-Bunny
// frame 0 -> 1 correspondences. No OpenCV is linked or used.
//
// Usage: pnp_from_scratch [--repo DIR] [--corr FILE]
//   --repo DIR   repository root (default: searched upward from the current
//                directory and from the executable's directory)
//   --corr FILE  correspondence export of slam_trajectory_test
//                (default: <repo>/results/data/cpp_export/pair_0_1_correspondences.csv)

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "common.hpp"
#include "pnp.hpp"
#include "projection.hpp"

using namespace scratch;

namespace {

int g_failures = 0;
bool Check(const std::string &what, bool ok, const std::string &value) {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << std::left << std::setw(46) << what << value << "\n";
  if (!ok) ++g_failures;
  return ok;
}
std::string Fmt(double v, int prec = 3, bool sci = false) {
  std::ostringstream o;
  if (sci) o << std::scientific;
  else o << std::fixed;
  o << std::setprecision(prec) << v;
  return o.str();
}

// --------------------------------------------------------------- self-tests
void SelfTests(const Intrinsics &K, const std::vector<Vector3d> &X, const Pose &G) {
  std::cout << "\n== Self-tests ==\n";
  // (a) projection with a hand-computed answer
  {
    const Vector2d p = projectPoint(Vector3d(0.1, -0.05, 0.5), Matrix3d::Identity(), Vector3d::Zero(), K);
    const Vector2d expect(K.fx * 0.2 + K.cx, K.fy * -0.1 + K.cy);
    Check("projectPoint, identity pose", (p - expect).norm() < 1e-12, "|diff| = " + Fmt((p - expect).norm(), 1, true));
    Matrix3d Ry;  // 90 deg about +Y: X_c = Ry * (0,0,1) + (0,0,2) = (1, 0, 2)
    Ry << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    const Vector2d q = projectPoint(Vector3d(0, 0, 1), Ry, Vector3d(0, 0, 2), K);
    const Vector2d expect2(K.fx * 0.5 + K.cx, K.cy);
    Check("projectPoint, rotated + translated pose", (q - expect2).norm() < 1e-12,
          "|diff| = " + Fmt((q - expect2).norm(), 1, true));
  }
  // (b) DLT on exact projections of the real 3D points under the GT motion
  {
    std::vector<Vector2d> uv;
    for (const auto &x : X) uv.push_back(projectPoint(x, G.R, G.t, K));
    const PnPResult r = solvePnPDLT(X, uv, K);
    Check("DLT recovers GT from noise-free projections", r.ok && RotErrDeg(G.R, r.R) < 1e-4 && (G.t - r.t).norm() < 1e-8,
          "rot err " + Fmt(RotErrDeg(G.R, r.R), 1, true) + " deg, trans err " + Fmt((G.t - r.t).norm(), 1, true) + " m");
  }
}

struct RunResult {
  std::string name;
  PnPResult pnp;
  Pose T;
  Reproj rp;
  double rot_err = NAN, trans_err = NAN;
  int n = 0;
};

RunResult Run(const std::string &name, const std::vector<Vector3d> &X, const std::vector<Vector2d> &uv,
              const Intrinsics &K, const Pose &G) {
  RunResult r;
  r.name = name;
  r.n = static_cast<int>(X.size());
  r.pnp = solvePnPDLT(X, uv, K);
  if (!r.pnp.ok) return r;
  r.T.R = r.pnp.R;
  r.T.t = r.pnp.t;
  r.rp = Reprojection(X, uv, r.T, K);
  r.rot_err = RotErrDeg(G.R, r.T.R);
  r.trans_err = (G.t - r.T.t).norm();
  return r;
}

void Validate(const RunResult &r, double baseline, bool counts = true) {
  std::cout << "\n== Sanity checks: " << r.name << " ==\n";
  if (!counts)
    std::cout << "  (informational: linear PnP has no outlier rejection, so gross mismatches in the input\n"
                 "   are expected to break it until RANSAC is added; not counted in the exit status)\n";
  const int before = g_failures;
  if (!Check("solver returned a pose", r.pnp.ok, r.pnp.ok ? "" : r.pnp.reason)) return;
  const double orth = (r.T.R.transpose() * r.T.R - Matrix3d::Identity()).norm();
  Check("R^T R = I", orth < 1e-9, "||R^T R - I||_F = " + Fmt(orth, 2, true));
  Check("det(R) = +1", std::abs(r.T.R.determinant() - 1) < 1e-9, "det = " + Fmt(r.T.R.determinant(), 12));
  Check("points in front of camera 1 (Zc > 0)", r.rp.positive_depth >= 0.95 * r.n,
        std::to_string(r.rp.positive_depth) + " / " + std::to_string(r.n));
  Check("reprojection error finite", std::isfinite(r.rp.mean) && std::isfinite(r.rp.max),
        "mean " + Fmt(r.rp.mean) + " px, max " + Fmt(r.rp.max) + " px");
  Check("rotation within 5 deg of GT", r.rot_err < 5.0, Fmt(r.rot_err) + " deg");
  Check("translation within 40% of the 0->1 baseline", r.trans_err < 0.4 * baseline,
        Fmt(r.trans_err, 4) + " m (baseline " + Fmt(baseline, 4) + " m)");
  if (!counts) g_failures = before;
}

}  // namespace

int main(int argc, char **argv) {
  fs::path repo, corr_path;
  for (int a = 1; a < argc; ++a) {
    std::string s = argv[a];
    if (s == "--repo" && a + 1 < argc) repo = argv[++a];
    else if (s == "--corr" && a + 1 < argc) corr_path = argv[++a];
  }
  if (repo.empty()) repo = FindRepo(fs::current_path());
  if (repo.empty()) repo = FindRepo(fs::path(argv[0]).parent_path());
  if (repo.empty() || !IsRepoRoot(repo)) {
    std::cerr << "cannot find the repository root (data/synthetic_bunny/intrinsics.txt); pass --repo DIR\n";
    return 2;
  }
  if (corr_path.empty()) corr_path = repo / "results" / "data" / "cpp_export" / "pair_0_1_correspondences.csv";
  const fs::path metrics_path = corr_path.parent_path() / "pair_metrics.csv";
  const fs::path ds = repo / "data" / "synthetic_bunny";

  // ---- inputs ----
  Intrinsics K;
  {
    std::ifstream f(ds / "intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  Csv corr;
  if (!corr.load(corr_path)) {
    std::cerr << "missing " << corr_path << "\n"
              << "Create it (export-only flags; trajectory outputs are unchanged) with, from the repo root:\n"
              << "  pixi run -e results results\n"
              << "or\n"
              << "  cd build && ./slam_trajectory_test ../data/synthetic_bunny --export-dir "
                 "../results/data/cpp_export --export-pair 0\n";
    return 2;
  }
  // PnP correspondences exactly as the C++ pipeline built them: rows with
  // pnp_used = 1, 3D point (X_i, Y_i, Z_i) in camera-0 coordinates, 2D (u_j, v_j) in frame 1.
  std::vector<Vector3d> X_all, X_inl;
  std::vector<Vector2d> uv_all, uv_inl;
  for (size_t r = 0; r < corr.rows.size(); ++r) {
    if (corr.i(r, "pnp_used") != 1) continue;
    const Vector3d X(corr.d(r, "X_i"), corr.d(r, "Y_i"), corr.d(r, "Z_i"));
    const Vector2d uv(corr.d(r, "u_j"), corr.d(r, "v_j"));
    X_all.push_back(X);
    uv_all.push_back(uv);
    if (corr.i(r, "pnp_inlier") == 1) {
      X_inl.push_back(X);
      uv_inl.push_back(uv);
    }
  }
  // Ground truth T_{1<-0} = T_wc[1]^-1 * T_wc[0]  (same formula as slam_trajectory_test)
  const std::vector<Pose> Twc = ReadGroundtruthTwc(ds / "groundtruth.txt");
  Pose G;
  G.R = Twc[1].R.transpose() * Twc[0].R;
  G.t = Twc[1].R.transpose() * (Twc[0].t - Twc[1].t);
  const double baseline = (Twc[1].t - Twc[0].t).norm();

  std::cout << std::fixed;
  std::cout << "Repository        : " << repo.filename().string() << " (auto-detected)\n"
            << "Correspondences   : " << fs::relative(corr_path, repo).string() << "\n"
            << "Intrinsics        : fx " << K.fx << " fy " << K.fy << " cx " << K.cx << " cy " << K.cy << "\n"
            << "GT motion 0->1    : rotation " << Fmt(RotAngleDeg(G.R)) << " deg, |t| " << Fmt(G.t.norm(), 4)
            << " m  (T_{1<-0} = T_wc[1]^-1 T_wc[0])\n"
            << "3D->2D pairs      : " << X_all.size() << " (pnp_used = 1), of which " << X_inl.size()
            << " are inliers of the existing OpenCV RANSAC\n";

  SelfTests(K, X_all, G);

  // ---- the scratch solve: ALL correspondences, no RANSAC ----
  RunResult A = Run("scratch DLT, all correspondences", X_all, uv_all, K, G);
  // ---- reference subset: same scratch solver on the points the EXISTING run marked as inliers ----
  RunResult B = Run("scratch DLT, existing-RANSAC inlier subset", X_inl, uv_inl, K, G);

  for (const RunResult *r : {&A, &B}) {
    std::cout << "\n== " << r->name << " ==\n";
    if (!r->pnp.ok) {
      std::cout << "  FAILED: " << r->pnp.reason << "\n";
      continue;
    }
    std::cout << "Scratch PnP:\n"
              << "  correspondences:   " << r->n << "\n"
              << "  rotation error:    " << Fmt(r->rot_err) << " deg   (estimated rotation " << Fmt(RotAngleDeg(r->T.R))
              << " deg vs GT " << Fmt(RotAngleDeg(G.R)) << ")\n"
              << "  translation error: " << Fmt(r->trans_err, 4) << " m\n"
              << "Reprojection:\n"
              << "  mean:   " << Fmt(r->rp.mean) << " px\n"
              << "  median: " << Fmt(r->rp.median) << " px\n"
              << "  max:    " << Fmt(r->rp.max) << " px\n"
              << "  RMSE:   " << Fmt(r->rp.rmse) << " px\n"
              << "Linear-solve diagnostics:\n"
              << "  singular values of lambda*R block: " << Fmt(r->pnp.sv_left_block(0), 4) << ", "
              << Fmt(r->pnp.sv_left_block(1), 4) << ", " << Fmt(r->pnp.sv_left_block(2), 4) << "\n"
              << "  ||M_3x3/lambda - R||_F before SO(3) projection: " << Fmt(r->pnp.raw_rotation_deviation, 4) << "\n"
              << "  t from m4/lambda (before re-solve): error " << Fmt((G.t - r->pnp.t_from_dlt).norm(), 4) << " m\n"
              << "  DLT sigma_12 / sigma_11: " << Fmt(r->pnp.nullspace_ratio, 4) << "\n";
  }

  // ---- existing OpenCV result (reference only; read from the export, not recomputed) ----
  Csv met;
  bool have_existing = met.load(metrics_path) && !met.rows.empty();
  RunResult E;
  int existing_inliers = -1;
  if (have_existing) {
    size_t row = 0;
    for (; row < met.rows.size(); ++row)
      if (met.i(row, "pair") == 0) break;
    E.T.R = QuatToR(met.d(row, "pnp_rel_qx"), met.d(row, "pnp_rel_qy"), met.d(row, "pnp_rel_qz"), met.d(row, "pnp_rel_qw"));
    E.T.t = Vector3d(met.d(row, "pnp_rel_tx"), met.d(row, "pnp_rel_ty"), met.d(row, "pnp_rel_tz"));
    E.rp = Reprojection(X_all, uv_all, E.T, K);
    E.rot_err = RotErrDeg(G.R, E.T.R);
    E.trans_err = (G.t - E.T.t).norm();
    existing_inliers = met.i(row, "pnp_inliers");
  }

  std::cout << "\n== Comparison (frame 0 -> 1) ==\n";
  std::cout << std::left << std::setw(26) << "" << std::setw(18) << "Scratch (all)" << std::setw(25)
            << "Scratch (inl. subset)" << "Existing OpenCV PnP\n";
  auto row3 = [&](const std::string &label, double a, double b, double e, int prec, const std::string &unit) {
    std::cout << std::left << std::setw(26) << label << std::setw(18) << (Fmt(a, prec) + unit) << std::setw(25)
              << (Fmt(b, prec) + unit) << (have_existing ? Fmt(e, prec) + unit : "n/a") << "\n";
  };
  Reproj B_all = B.pnp.ok ? Reprojection(X_all, uv_all, B.T, K) : Reproj{};
  row3("Rotation error", A.rot_err, B.rot_err, E.rot_err, 3, " deg");
  row3("Translation error", A.trans_err, B.trans_err, E.trans_err, 4, " m");
  const Reproj A_inl = A.pnp.ok ? Reprojection(X_inl, uv_inl, A.T, K) : Reproj{};
  const Reproj E_inl = have_existing ? Reprojection(X_inl, uv_inl, E.T, K) : Reproj{};
  row3("Mean reproj. (121 inl.)", A_inl.mean, B.rp.mean, E_inl.mean, 3, " px");
  row3("Median reproj. (121 inl.)", A_inl.median, B.rp.median, E_inl.median, 3, " px");
  row3("Mean reproj. (all 136)", A.rp.mean, B_all.mean, E.rp.mean, 3, " px");
  row3("Median reproj. (all 136)", A.rp.median, B_all.median, E.rp.median, 3, " px");
  std::cout << std::left << std::setw(26) << "Inliers" << std::setw(18) << "--" << std::setw(25) << "--"
            << (have_existing ? std::to_string(existing_inliers) + " (RANSAC, 8 px)" : "n/a") << "\n";
  std::cout << "(Inlier subset = which points the existing OpenCV RANSAC kept; the pose itself is still\n"
               " computed only by the scratch DLT.)\n";

  Validate(A, baseline, /*counts=*/false);
  Validate(B, baseline);

  // ---- write results ----
  const fs::path out_dir = repo / "pnp_from_scratch" / "results";
  fs::create_directories(out_dir);
  {
    std::ofstream f(out_dir / "frame_0_1_reprojection.csv");
    f << "index,X,Y,Z,u_obs,v_obs,existing_ransac_inlier,"
         "u_inlier_subset,v_inlier_subset,err_inlier_subset_px,u_all,v_all,err_all_px\n"
      << std::setprecision(6);
    for (size_t r = 0, k = 0; r < corr.rows.size(); ++r) {
      if (corr.i(r, "pnp_used") != 1) continue;
      const Vector2d pb = projectPoint(X_all[k], B.T.R, B.T.t, K), pa = projectPoint(X_all[k], A.T.R, A.T.t, K);
      f << k << "," << X_all[k].x() << "," << X_all[k].y() << "," << X_all[k].z() << "," << uv_all[k].x() << ","
        << uv_all[k].y() << "," << corr.i(r, "pnp_inlier") << "," << pb.x() << "," << pb.y() << ","
        << (pb - uv_all[k]).norm() << "," << pa.x() << "," << pa.y() << "," << A.rp.err[k] << "\n";
      ++k;
    }
  }
  {
    std::ofstream f(out_dir / "frame_0_1_pose.txt");
    f << std::setprecision(9) << "# T_{1<-0}: X_cam1 = R * X_cam0 + t\n"
      << "# scratch DLT on the " << B.n << " existing-RANSAC inliers (valid pose)\nR =\n" << B.T.R
      << "\nt = " << B.T.t.transpose() << "\n"
      << "# scratch DLT on all " << A.n << " correspondences (no outlier rejection; fails)\nR =\n" << A.T.R
      << "\nt = " << A.T.t.transpose() << "\n"
      << "# ground truth (T_wc[1]^-1 T_wc[0])\nR =\n" << G.R << "\nt = " << G.t.transpose() << "\n";
  }
  std::cout << "\nwrote pnp_from_scratch/results/frame_0_1_reprojection.csv, frame_0_1_pose.txt\n";
  std::cout << (g_failures == 0 ? "All counted sanity checks passed (scratch DLT on the inlier subset is a valid pose).\n"
                                : std::to_string(g_failures) + " sanity check(s) FAILED (see above).\n");
  return g_failures == 0 ? 0 : 1;
}
