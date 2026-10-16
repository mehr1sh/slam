// Diagnostic-only companion of slam_trajectory_test. It changes NOTHING in
// the pipeline: for every frame pair it repeats the exact same calls with the
// exact same arguments (find_feature_matches, truncated depth lookup,
// pixel2cam back-projection, solvePnPRansac(..., false, 100, 8.0f, 0.99, ...),
// pose_estimation_3d3d + bundleAdjustment), checks that it reproduces the
// committed trajectories, and then measures the intermediate quantities
// against ground truth:
//   - GT replay of the accumulation formula (plus two negative controls)
//   - per-correspondence GT reprojection / 3D residuals, depth consistency,
//     distance to the bunny silhouette
//   - ICP residuals before/after SVD and g2o, and at the GT motion
//   - "perfect correspondence" sanity runs of the same solvers
//   - diagnostic re-solves on correspondence subsets (interior-only,
//     GT-consistent) and with a half-pixel principal-point offset. These are
//     MEASUREMENTS of the solvers' sensitivity, not pipeline changes.
//
// Usage: trajectory_diagnostics_test <dataset_dir> <out_dir>
// Writes <out_dir>/{gt_replay.csv, gt_replay_controls.csv,
//   correspondence_diagnostics.csv, pair_solver_checks.csv,
//   reaccumulation_check.csv}.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sophus/se3.hpp>

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "tracking/icp.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

namespace {

struct Intrinsics {
  int width, height;
  double fx, fy, cx, cy;
};

Intrinsics ReadIntrinsics(const string &path) {
  ifstream f(path);
  Intrinsics k;
  f >> k.width >> k.height >> k.fx >> k.fy >> k.cx >> k.cy;
  return k;
}

vector<Sophus::SE3d> ReadTUM(const string &path) {
  vector<Sophus::SE3d> poses;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    double ts, tx, ty, tz, qx, qy, qz, qw;
    iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    poses.emplace_back(Eigen::Quaterniond(qw, qx, qy, qz).normalized(), Eigen::Vector3d(tx, ty, tz));
  }
  return poses;
}

string PadIndex(int i) {
  ostringstream oss;
  oss << setw(6) << setfill('0') << i;
  return oss.str();
}

Sophus::SE3d CvToSE3(const Mat &R, const Mat &t) {
  Eigen::Matrix3d Re;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) Re(r, c) = R.at<double>(r, c);
  return Sophus::SE3d(Eigen::Quaterniond(Re), Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
}

// The same two error functions slam_trajectory_test uses (report_utils.hpp).
double RotErr(const Sophus::SE3d &A, const Sophus::SE3d &B) {
  return RotationAngleDeg(EigenRotToCv(A.rotationMatrix()), EigenRotToCv(B.rotationMatrix()));
}
double TransErr(const Sophus::SE3d &A, const Sophus::SE3d &B) {
  return TranslationDiff(EigenVecToCv(A.translation()), EigenVecToCv(B.translation()));
}
double AngleDeg(const Sophus::SE3d &T) { return T.so3().log().norm() * 180.0 / M_PI; }

// Chebyshev distance (pixels) from (x, y) to the nearest zero-depth pixel,
// capped at `cap`. 0 means the pixel itself has no depth.
int SilhouetteDistance(const Mat &depth, int x, int y, int cap = 10) {
  for (int r = 0; r <= cap; ++r)
    for (int dy = -r; dy <= r; ++dy)
      for (int dx = -r; dx <= r; ++dx) {
        if (max(abs(dx), abs(dy)) != r) continue;
        int xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= depth.cols || yy >= depth.rows) return r;
        if (depth.at<unsigned short>(yy, xx) == 0) return r;
      }
  return cap + 1;
}

// Largest |depth difference| (m) between the pixel and its 8 neighbours that
// have depth -- a depth-discontinuity indicator.
double DepthJump(const Mat &depth, int x, int y) {
  double c = depth.at<unsigned short>(y, x) / 5000.0, m = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= depth.cols || yy >= depth.rows) continue;
      unsigned short d = depth.at<unsigned short>(yy, xx);
      if (d == 0) continue;
      m = max(m, fabs(d / 5000.0 - c));
    }
  return m;
}

Point2d Project(const Eigen::Vector3d &p, const Intrinsics &k) {
  return Point2d(k.fx * p.x() / p.z() + k.cx, k.fy * p.y() / p.z() + k.cy);
}

struct PnpOut {
  bool ok = false;
  int inliers = 0;
  Sophus::SE3d T;  // T_{j<-i}
};

// Identical call to slam_trajectory_test's RunPnpPair.
PnpOut RunPnp(const vector<Point3f> &p3, const vector<Point2f> &p2, const Mat &K,
              vector<int> *inl = nullptr) {
  PnpOut o;
  if (p3.size() < 4) return o;
  Mat r, t, R;
  vector<int> inliers;
  if (!solvePnPRansac(p3, p2, K, Mat(), r, t, false, 100, 8.0f, 0.99, inliers)) return o;
  Rodrigues(r, R);
  o.ok = true;
  o.inliers = (int)inliers.size();
  o.T = CvToSE3(R, t);
  if (inl) *inl = inliers;
  return o;
}

// Identical calls to slam_trajectory_test's RunIcpPair. Returns native
// x_i = R*x_j + t results (SVD-only and after g2o); conversion to the
// common T_{j<-i} is done by the caller exactly as the pipeline does.
struct IcpOut {
  bool ok = false;
  Sophus::SE3d T_svd_native, T_g2o_native;
};

IcpOut RunIcp(const vector<Point3f> &p1, const vector<Point3f> &p2) {
  IcpOut o;
  if (p1.size() < 3) return o;
  Mat R, t;
  // the book functions print to stdout; silence them
  streambuf *old = cout.rdbuf();
  ostringstream sink;
  cout.rdbuf(sink.rdbuf());
  pose_estimation_3d3d(p1, p2, R, t);
  o.T_svd_native = CvToSE3(R, t);
  bundleAdjustment(p1, p2, R, t);
  o.T_g2o_native = CvToSE3(R, t);
  cout.rdbuf(old);
  o.ok = true;
  return o;
}

Sophus::SE3d NativeToCommon(const Sophus::SE3d &T_native) {
  // same arithmetic as the pipeline: R' = R^T, t' = -R^T t
  Eigen::Matrix3d R = T_native.rotationMatrix().transpose();
  Eigen::Vector3d t = -R * T_native.translation();
  return Sophus::SE3d(Eigen::Quaterniond(R), t);
}

// residuals |p1 - T*p2| for T in native ICP form
vector<double> Residuals(const vector<Point3f> &p1, const vector<Point3f> &p2, const Sophus::SE3d &T) {
  vector<double> r;
  for (size_t k = 0; k < p1.size(); ++k) {
    Eigen::Vector3d a(p1[k].x, p1[k].y, p1[k].z), b(p2[k].x, p2[k].y, p2[k].z);
    r.push_back((a - T * b).norm());
  }
  return r;
}

double Mean(const vector<double> &v) {
  if (v.empty()) return NAN;
  double s = 0;
  for (double x : v) s += x;
  return s / v.size();
}
double Median(vector<double> v) {
  if (v.empty()) return NAN;
  sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
double Rms(const vector<double> &v) {
  if (v.empty()) return NAN;
  double s = 0;
  for (double x : v) s += x * x;
  return sqrt(s / v.size());
}

void WritePose(ofstream &f, const Sophus::SE3d &T) {
  Eigen::Vector3d t = T.translation();
  Eigen::Quaterniond q = T.unit_quaternion();
  f << "," << t.x() << "," << t.y() << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z()
    << "," << q.w();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    cerr << "usage: trajectory_diagnostics_test <dataset_dir> <out_dir>" << endl;
    return 1;
  }
  string ds = argv[1], out = argv[2];
  Intrinsics intr = ReadIntrinsics(ds + "/intrinsics.txt");
  Mat K = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);
  Mat K_half = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx - 0.5, 0, intr.fy, intr.cy - 0.5, 0, 0, 1);
  vector<Sophus::SE3d> T_gt = ReadTUM(ds + "/groundtruth.txt");
  vector<Sophus::SE3d> T_pnp_file = ReadTUM(ds + "/pnp_trajectory.txt");
  vector<Sophus::SE3d> T_icp_file = ReadTUM(ds + "/icp_trajectory.txt");
  const int N = (int)T_gt.size();
  cout << fixed << setprecision(9);

  // ===== 1. GT replay: the pipeline's accumulation applied to GT motions ====
  {
    ofstream f(out + "/gt_replay.csv");
    f << "frame,translation_error,rotation_error_deg\n" << scientific << setprecision(6);
    ofstream c(out + "/gt_replay_controls.csv");
    c << "frame,right_mul_no_inverse_trans_m,right_mul_no_inverse_rot_deg,"
         "left_mul_inverse_trans_m,left_mul_inverse_rot_deg\n" << fixed << setprecision(9);
    Sophus::SE3d T = T_gt[0], A = T_gt[0], B = T_gt[0];
    double max_t = 0, max_r = 0;
    for (int k = 0; k < N; ++k) {
      if (k > 0) {
        Sophus::SE3d rel = T_gt[k].inverse() * T_gt[k - 1];  // T_{k<-k-1}, as the pipeline
        T = T * rel.inverse();                                // the pipeline's formula
        A = A * rel;                                          // control: no inverse
        B = rel.inverse() * B;                                // control: wrong side
      }
      double te = TransErr(T_gt[k], T), re = RotErr(T_gt[k], T);
      max_t = max(max_t, te);
      max_r = max(max_r, re);
      f << k << "," << te << "," << re << "\n";
      c << k << "," << TransErr(T_gt[k], A) << "," << RotErr(T_gt[k], A) << ","
        << TransErr(T_gt[k], B) << "," << RotErr(T_gt[k], B) << "\n";
    }
    cout << "GT_REPLAY max_translation_error=" << scientific << max_t
         << " max_rotation_error_deg=" << max_r << fixed << endl;
  }

  // ===== 2. per pair =====
  ofstream fc(out + "/correspondence_diagnostics.csv");
  fc << "pair,match,u_i,v_i,u_j,v_j,hamming,depth_raw_i,depth_raw_j,Z_i,Z_j,"
        "sil_dist_i,sil_dist_j,depth_jump_i,depth_jump_j,pnp_used,pnp_inlier,icp_used,"
        "gt_pred_u_j,gt_pred_v_j,gt_reproj_err_px,gt_reproj_dx,gt_reproj_dy,est_reproj_err_px,"
        "gt_pred_Z_j,depth_consistency_m,icp_gt_residual_m,icp_est_residual_m,icp_identity_residual_m\n";
  fc << fixed << setprecision(6);
  ofstream fs(out + "/pair_solver_checks.csv");
  fs << "pair,keypoints_i,keypoints_j,raw_matches,filtered_matches,pnp_correspondences,pnp_inliers,"
        "icp_correspondences,gt_rot_deg,gt_trans_m,"
        "pnp_rot_deg,pnp_trans_m,pnp_rot_err_deg,pnp_trans_err_m,"
        "icp_svd_rot_deg,icp_svd_trans_m,icp_svd_rot_err_deg,icp_svd_trans_err_m,"
        "icp_g2o_rot_deg,icp_g2o_trans_m,icp_g2o_rot_err_deg,icp_g2o_trans_err_m,"
        "svd_vs_g2o_rot_deg,svd_vs_g2o_trans_m,"
        "icp_res_identity_mean,icp_res_identity_median,icp_res_svd_mean,icp_res_svd_median,icp_res_svd_rms,"
        "icp_res_g2o_mean,icp_res_g2o_median,icp_res_g2o_rms,icp_res_gt_mean,icp_res_gt_median,icp_res_gt_rms,"
        "pnp_reproj_all_mean,pnp_reproj_all_median,pnp_reproj_all_p90,pnp_reproj_all_max,"
        "pnp_reproj_inl_mean,pnp_reproj_inl_median,pnp_reproj_inl_p90,pnp_reproj_inl_max,"
        "pnp_gt_reproj_all_mean,pnp_gt_reproj_all_median,pnp_gt_reproj_inl_mean,pnp_gt_reproj_inl_median,"
        "perfect_pnp_rot_err_deg,perfect_pnp_trans_err_m,"
        "perfect_icp_rot_err_deg,perfect_icp_trans_err_m,perfect_icp_svd_rot_err_deg,perfect_icp_svd_trans_err_m,"
        "half_px_pnp_rot_err_deg,half_px_pnp_trans_err_m,half_px_pnp_rot_deg,"
        "n_pnp_interior,interior_pnp_rot_err_deg,interior_pnp_trans_err_m,interior_pnp_rot_deg,"
        "n_pnp_gtconsistent,gtcons_pnp_rot_err_deg,gtcons_pnp_trans_err_m,gtcons_pnp_rot_deg,"
        "n_icp_interior,interior_icp_rot_err_deg,interior_icp_trans_err_m,interior_icp_rot_deg,"
        "n_icp_gtconsistent,gtcons_icp_rot_err_deg,gtcons_icp_trans_err_m,gtcons_icp_rot_deg,"
        "pnp_rel_tx,pnp_rel_ty,pnp_rel_tz,pnp_rel_qx,pnp_rel_qy,pnp_rel_qz,pnp_rel_qw,"
        "icp_rel_tx,icp_rel_ty,icp_rel_tz,icp_rel_qx,icp_rel_qy,icp_rel_qz,icp_rel_qw\n";
  fs << fixed << setprecision(9);

  vector<Sophus::SE3d> pnp_rel(N - 1), icp_rel(N - 1);
  for (int i = 0; i < N - 1; ++i) {
    int j = i + 1;
    cerr << "pair " << i << "->" << j << endl;
    Mat img_i = imread(ds + "/" + PadIndex(i) + ".png", IMREAD_COLOR);
    Mat img_j = imread(ds + "/" + PadIndex(j) + ".png", IMREAD_COLOR);
    Mat dep_i = imread(ds + "/" + PadIndex(i) + "_depth.png", IMREAD_UNCHANGED);
    Mat dep_j = imread(ds + "/" + PadIndex(j) + "_depth.png", IMREAD_UNCHANGED);
    Sophus::SE3d G = T_gt[j].inverse() * T_gt[i];  // T_{j<-i}

    vector<KeyPoint> kp_i, kp_j;
    vector<DMatch> matches, raw;
    {
      streambuf *old = cout.rdbuf();
      ostringstream sink;
      cout.rdbuf(sink.rdbuf());
      find_feature_matches(img_i, img_j, kp_i, kp_j, matches, &raw);
      cout.rdbuf(old);
    }

    // --- same correspondence construction as the pipeline ---
    vector<Point3f> pnp3;
    vector<Point2f> pnp2;
    vector<int> pnp_m;
    vector<Point3f> icp1, icp2;
    vector<int> icp_m;
    for (size_t mi = 0; mi < matches.size(); ++mi) {
      const auto &a = kp_i[matches[mi].queryIdx].pt, &b = kp_j[matches[mi].trainIdx].pt;
      ushort d1 = dep_i.ptr<unsigned short>((int)a.y)[(int)a.x];
      ushort d2 = dep_j.ptr<unsigned short>((int)b.y)[(int)b.x];
      if (d1 != 0) {
        float dd = d1 / 5000.0f;
        Point2d p = pixel2cam(a, K);
        pnp3.push_back(Point3f(p.x * dd, p.y * dd, dd));
        pnp2.push_back(b);
        pnp_m.push_back((int)mi);
      }
      if (d1 != 0 && d2 != 0) {
        Point2d p1 = pixel2cam(a, K), p2 = pixel2cam(b, K);
        float dd1 = float(d1) / 5000.0f, dd2 = float(d2) / 5000.0f;
        icp1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        icp2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
        icp_m.push_back((int)mi);
      }
    }

    // --- PnP as the pipeline ---
    vector<int> inl;
    PnpOut pnp = RunPnp(pnp3, pnp2, K, &inl);
    vector<char> is_inl(pnp3.size(), 0);
    for (int x : inl) is_inl[x] = 1;
    pnp_rel[i] = pnp.T;

    // --- ICP as the pipeline ---
    IcpOut icp = RunIcp(icp1, icp2);
    Sophus::SE3d icp_svd = NativeToCommon(icp.T_svd_native);
    Sophus::SE3d icp_g2o = NativeToCommon(icp.T_g2o_native);
    icp_rel[i] = icp_g2o;
    Sophus::SE3d Ginv = G.inverse();  // native ICP direction T_{i<-j}

    // --- per-correspondence diagnostics ---
    vector<double> reproj_all, reproj_inl, gt_reproj_all, gt_reproj_inl;
    vector<int> pnp_corr_of_m(matches.size(), -1), icp_corr_of_m(matches.size(), -1);
    for (size_t c = 0; c < pnp_m.size(); ++c) pnp_corr_of_m[pnp_m[c]] = (int)c;
    for (size_t c = 0; c < icp_m.size(); ++c) icp_corr_of_m[icp_m[c]] = (int)c;
    vector<int> sil_i_of_m(matches.size()), sil_j_of_m(matches.size());
    vector<double> gt_err_of_m(matches.size(), NAN), icp_gt_res_of_m(matches.size(), NAN);
    for (size_t mi = 0; mi < matches.size(); ++mi) {
      const auto &a = kp_i[matches[mi].queryIdx].pt, &b = kp_j[matches[mi].trainIdx].pt;
      int xi = (int)a.x, yi = (int)a.y, xj = (int)b.x, yj = (int)b.y;
      ushort d1 = dep_i.at<unsigned short>(yi, xi), d2 = dep_j.at<unsigned short>(yj, xj);
      int si = SilhouetteDistance(dep_i, xi, yi), sj = SilhouetteDistance(dep_j, xj, yj);
      sil_i_of_m[mi] = si;
      sil_j_of_m[mi] = sj;
      int pc = pnp_corr_of_m[mi], ic = icp_corr_of_m[mi];
      fc << i << "," << mi << "," << a.x << "," << a.y << "," << b.x << "," << b.y << ","
         << matches[mi].distance << "," << d1 << "," << d2 << "," << d1 / 5000.0 << "," << d2 / 5000.0
         << "," << si << "," << sj << "," << (d1 ? DepthJump(dep_i, xi, yi) : NAN) << ","
         << (d2 ? DepthJump(dep_j, xj, yj) : NAN) << "," << (pc >= 0) << ","
         << (pc >= 0 ? (int)is_inl[pc] : 0) << "," << (ic >= 0);
      if (pc >= 0) {
        Eigen::Vector3d X(pnp3[pc].x, pnp3[pc].y, pnp3[pc].z);
        Eigen::Vector3d Xg = G * X, Xe = pnp.T * X;
        Point2d ug = Project(Xg, intr), ue = Project(Xe, intr);
        double eg = hypot(ug.x - b.x, ug.y - b.y), ee = hypot(ue.x - b.x, ue.y - b.y);
        gt_err_of_m[mi] = eg;
        reproj_all.push_back(ee);
        gt_reproj_all.push_back(eg);
        if (is_inl[pc]) {
          reproj_inl.push_back(ee);
          gt_reproj_inl.push_back(eg);
        }
        fc << "," << ug.x << "," << ug.y << "," << eg << "," << b.x - ug.x << "," << b.y - ug.y << ","
           << ee << "," << Xg.z() << "," << (d2 ? d2 / 5000.0 - Xg.z() : NAN);
      } else {
        fc << ",,,,,,,,";
      }
      if (ic >= 0) {
        Eigen::Vector3d p(icp1[ic].x, icp1[ic].y, icp1[ic].z), q(icp2[ic].x, icp2[ic].y, icp2[ic].z);
        double rg = (p - Ginv * q).norm();
        icp_gt_res_of_m[mi] = rg;
        fc << "," << rg << "," << (p - icp.T_g2o_native * q).norm() << "," << (p - q).norm();
      } else {
        fc << ",,,";
      }
      fc << "\n";
    }

    // --- perfect-correspondence sanity runs (same solver calls) ---
    vector<Point2f> perfect2;
    for (const auto &X : pnp3) {
      Point2d u = Project(G * Eigen::Vector3d(X.x, X.y, X.z), intr);
      perfect2.push_back(Point2f((float)u.x, (float)u.y));
    }
    PnpOut pnp_perfect = RunPnp(pnp3, perfect2, K);
    vector<Point3f> perfect_q;
    for (const auto &p : icp1) {
      Eigen::Vector3d q = G * Eigen::Vector3d(p.x, p.y, p.z);  // the same point in camera j
      perfect_q.push_back(Point3f((float)q.x(), (float)q.y(), (float)q.z()));
    }
    IcpOut icp_perfect = RunIcp(icp1, perfect_q);

    // --- half-pixel principal-point variant (renderer samples pixel centres
    //     at x+0.5; OpenCV puts pixel centres at integer coordinates) ---
    vector<Point3f> pnp3_half;
    for (size_t c = 0; c < pnp3.size(); ++c) {
      const auto &a = kp_i[matches[pnp_m[c]].queryIdx].pt;
      float dd = pnp3[c].z;
      Point2d p = pixel2cam(a, K_half);
      pnp3_half.push_back(Point3f(p.x * dd, p.y * dd, dd));
    }
    PnpOut pnp_half = RunPnp(pnp3_half, pnp2, K_half);

    // --- diagnostic subsets ---
    vector<Point3f> s3, g3;
    vector<Point2f> s2, g2;
    for (size_t c = 0; c < pnp3.size(); ++c) {
      int mi = pnp_m[c];
      if (sil_i_of_m[mi] >= 4 && sil_j_of_m[mi] >= 4) { s3.push_back(pnp3[c]); s2.push_back(pnp2[c]); }
      if (gt_err_of_m[mi] < 1.0) { g3.push_back(pnp3[c]); g2.push_back(pnp2[c]); }
    }
    PnpOut pnp_int = RunPnp(s3, s2, K), pnp_gc = RunPnp(g3, g2, K);
    vector<Point3f> si1, si2, gi1, gi2;
    for (size_t c = 0; c < icp1.size(); ++c) {
      int mi = icp_m[c];
      if (sil_i_of_m[mi] >= 4 && sil_j_of_m[mi] >= 4) { si1.push_back(icp1[c]); si2.push_back(icp2[c]); }
      if (icp_gt_res_of_m[mi] < 0.005) { gi1.push_back(icp1[c]); gi2.push_back(icp2[c]); }
    }
    IcpOut icp_int = RunIcp(si1, si2), icp_gc = RunIcp(gi1, gi2);

    // --- residual summaries ---
    vector<double> r_id = Residuals(icp1, icp2, Sophus::SE3d());
    vector<double> r_svd = Residuals(icp1, icp2, icp.T_svd_native);
    vector<double> r_g2o = Residuals(icp1, icp2, icp.T_g2o_native);
    vector<double> r_gt = Residuals(icp1, icp2, Ginv);
    auto pct = [](vector<double> v, double q) {
      if (v.empty()) return (double)NAN;
      sort(v.begin(), v.end());
      return v[min(v.size() - 1, (size_t)floor(q * (v.size() - 1) + 0.5))];
    };
    auto mx = [](const vector<double> &v) { return v.empty() ? (double)NAN : *max_element(v.begin(), v.end()); };
    auto pe = [&](const PnpOut &o, ofstream &f) {
      if (o.ok) f << "," << RotErr(G, o.T) << "," << TransErr(G, o.T);
      else f << ",,";
    };
    auto pe3 = [&](const PnpOut &o, ofstream &f) {
      pe(o, f);
      f << "," << (o.ok ? AngleDeg(o.T) : NAN);
    };
    auto ie3 = [&](const IcpOut &o, ofstream &f) {
      if (o.ok) {
        Sophus::SE3d T = NativeToCommon(o.T_g2o_native);
        f << "," << RotErr(G, T) << "," << TransErr(G, T) << "," << AngleDeg(T);
      } else f << ",,,";
    };

    fs << i << "," << kp_i.size() << "," << kp_j.size() << "," << raw.size() << "," << matches.size()
       << "," << pnp3.size() << "," << pnp.inliers << "," << icp1.size() << "," << AngleDeg(G) << ","
       << G.translation().norm() << "," << AngleDeg(pnp.T) << "," << pnp.T.translation().norm() << ","
       << RotErr(G, pnp.T) << "," << TransErr(G, pnp.T) << "," << AngleDeg(icp_svd) << ","
       << icp_svd.translation().norm() << "," << RotErr(G, icp_svd) << "," << TransErr(G, icp_svd) << ","
       << AngleDeg(icp_g2o) << "," << icp_g2o.translation().norm() << "," << RotErr(G, icp_g2o) << ","
       << TransErr(G, icp_g2o) << "," << RotErr(icp_svd, icp_g2o) << "," << TransErr(icp_svd, icp_g2o)
       << "," << Mean(r_id) << "," << Median(r_id) << "," << Mean(r_svd) << "," << Median(r_svd) << ","
       << Rms(r_svd) << "," << Mean(r_g2o) << "," << Median(r_g2o) << "," << Rms(r_g2o) << ","
       << Mean(r_gt) << "," << Median(r_gt) << "," << Rms(r_gt) << "," << Mean(reproj_all) << ","
       << Median(reproj_all) << "," << pct(reproj_all, 0.9) << "," << mx(reproj_all) << ","
       << Mean(reproj_inl) << "," << Median(reproj_inl) << "," << pct(reproj_inl, 0.9) << ","
       << mx(reproj_inl) << "," << Mean(gt_reproj_all) << "," << Median(gt_reproj_all) << ","
       << Mean(gt_reproj_inl) << "," << Median(gt_reproj_inl);
    pe(pnp_perfect, fs);
    {
      Sophus::SE3d Tp = NativeToCommon(icp_perfect.T_g2o_native), Ts = NativeToCommon(icp_perfect.T_svd_native);
      fs << "," << RotErr(G, Tp) << "," << TransErr(G, Tp) << "," << RotErr(G, Ts) << "," << TransErr(G, Ts);
    }
    pe3(pnp_half, fs);
    fs << "," << s3.size();
    pe3(pnp_int, fs);
    fs << "," << g3.size();
    pe3(pnp_gc, fs);
    fs << "," << si1.size();
    ie3(icp_int, fs);
    fs << "," << gi1.size();
    ie3(icp_gc, fs);
    WritePose(fs, pnp.T);
    WritePose(fs, icp_g2o);
    fs << "\n";
  }

  // ===== 3. re-accumulate the re-run motions and compare with the files ====
  {
    ofstream f(out + "/reaccumulation_check.csv");
    f << "frame,pnp_vs_file_trans_m,pnp_vs_file_rot_deg,icp_vs_file_trans_m,icp_vs_file_rot_deg\n";
    f << scientific << setprecision(6);
    Sophus::SE3d A = T_gt[0], B = T_gt[0];
    double mt = 0;
    for (int k = 0; k < N; ++k) {
      if (k > 0) {
        A = A * pnp_rel[k - 1].inverse();
        B = B * icp_rel[k - 1].inverse();
      }
      double a = TransErr(A, T_pnp_file[k]), b = TransErr(B, T_icp_file[k]);
      mt = max(mt, max(a, b));
      f << k << "," << a << "," << RotErr(A, T_pnp_file[k]) << "," << b << "," << RotErr(B, T_icp_file[k])
        << "\n";
    }
    cout << "REACCUMULATION max_translation_diff_vs_files=" << scientific << mt << endl;
  }
  cout << "wrote " << out << "/{gt_replay,gt_replay_controls,correspondence_diagnostics,"
       << "pair_solver_checks,reaccumulation_check}.csv" << endl;
  return 0;
}
