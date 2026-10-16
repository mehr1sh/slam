// Rigorously validates the trajectory-composition math already used by
// tests/synthetic/slam_trajectory_test.cpp. Does NOT modify PnP, ICP,
// feature matching, the synthetic renderer, or ground-truth generation --
// this file only reads existing data/code and independently re-derives or
// re-checks numbers. RunPnpPairOnce()/RunIcpPairOnce() below are an
// intentional, read-only DUPLICATE of slam_trajectory_test.cpp's
// RunPnpPair()/RunIcpPair() (byte-for-byte the same book-code calls),
// copied rather than shared via a header so the original "source of truth"
// file is left completely untouched by this validation work.
//
// === Transform conventions, quoted from the actual current source ===
//
// T_wc[k] (groundtruth.txt / slam_trajectory.csv's gt_ columns): camera-to-
// world for frame k. FROM: camera-k's local frame. TO: world frame.
// Translation = camera center in world; rotation maps camera-local vectors
// to world vectors. (render_bunny_test.cpp / trajectory.hpp's own doc
// comment: "Returns T_world_camera[i] (camera-to-world)".)
//
// PnP relative pose (src/tracking/pnp.cpp via tests/*/pnp_test.cpp's own
// call pattern: 3D points backprojected from frame i's depth, 2D targets
// are frame j's pixels; solvePnP/solvePnPRansac returns (R,t) satisfying
// x_j = R*x_i + t). FROM: camera-i local frame. TO: camera-j local frame.
// This is T_{j<-i} directly (Sophus::SE3d(R,t)*p == R*p+t, matching
// Sophus's own semantics with zero reinterpretation needed).
//
// ICP relative pose (src/tracking/icp.cpp's pose_estimation_3d3d(pts1=
// frame i, pts2=frame j, R, t): "t_ = centroid(pts1) - R*centroid(pts2)",
// i.e. solves x_i = R*x_j + t). FROM: camera-j local frame. TO: camera-i
// local frame -- T_{i<-j}, the OPPOSITE direction from PnP. Inverting once
// (R' = R^T, t' = -R^T t) gives T_{j<-i}, the same form/direction PnP
// already produces natively.
//
// T_rel (either, once normalized to the common T_{j<-i} direction) as a
// 4x4 homogeneous matrix: [[R, t], [0, 1]], mapping camera-i points to
// camera-j points: x_j_homog = T_rel * x_i_homog.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <sophus/se3.hpp>

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "tracking/icp.hpp"
#include "tracking/pnp.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

namespace {

struct Intrinsics { int width, height; double fx, fy, cx, cy; };

Intrinsics ReadIntrinsics(const string &path) {
  ifstream f(path);
  Intrinsics k;
  f >> k.width >> k.height >> k.fx >> k.fy >> k.cx >> k.cy;
  return k;
}

vector<Sophus::SE3d> ReadGroundtruth(const string &path) {
  vector<Sophus::SE3d> poses;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    double ts, tx, ty, tz, qx, qy, qz, qw;
    iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    poses.emplace_back(q, Eigen::Vector3d(tx, ty, tz));
  }
  return poses;
}

// Reads a TUM-format trajectory file (pnp_trajectory.txt / icp_trajectory.txt),
// also returning the raw timestamps for the frame-ordering check in
// section 9/10.
struct FileTrajectory {
  vector<double> timestamps;
  vector<Sophus::SE3d> poses;
};

FileTrajectory ReadTrajectoryFile(const string &path) {
  FileTrajectory traj;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    double ts, tx, ty, tz, qx, qy, qz, qw;
    iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    traj.timestamps.push_back(ts);
    Eigen::Quaterniond q(qw, qx, qy, qz);
    traj.poses.emplace_back(q, Eigen::Vector3d(tx, ty, tz));
  }
  return traj;
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
  Eigen::Vector3d te(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  return Sophus::SE3d(Eigen::Quaterniond(Re), te);
}

// THE accumulation formula under test, isolated as its own function so
// every check below (zero-motion, perfect-motion, identity-anchor) uses
// the exact same code path -- verified, not just asserted, to be
// T_wc[i+1] = T_wc[i] * T_rel(i,i+1)^-1, matching
// tests/synthetic/slam_trajectory_test.cpp lines 259/265 verbatim.
Sophus::SE3d AccumulateStep(const Sophus::SE3d &T_wc_i, const Sophus::SE3d &T_rel_i_to_j) {
  return T_wc_i * T_rel_i_to_j.inverse();
}

void PrintPose(const string &label, const Sophus::SE3d &T) {
  cout << label << ":\n";
  cout << "R =\n" << EigenRotToCv(T.rotationMatrix()) << endl;
  cout << "t = " << EigenVecToCv(T.translation()).t() << endl;
}

// --- Duplicated (read-only, byte-for-byte) from slam_trajectory_test.cpp,
// purely so this validation file can exercise the REAL PnP/ICP code on
// real frame pairs without modifying or depending on internals of the
// original "source of truth" file. ---

Sophus::SE3d RunPnpPairOnce(const string &dataset_dir, const Mat &K, int i, int j) {
  Mat img_i = imread(dataset_dir + "/" + PadIndex(i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);

  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);

  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (const DMatch &m : matches) {
    ushort d = depth_i.ptr<unsigned short>((int)keypoints_i[m.queryIdx].pt.y)
                      [(int)keypoints_i[m.queryIdx].pt.x];
    if (d == 0) continue;
    float dd = d / 5000.0f;
    Point2d p = pixel2cam(keypoints_i[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p.x * dd, p.y * dd, dd));
    pts_2d.push_back(keypoints_j[m.trainIdx].pt);
  }

  Mat r, t_cv, R_cv;
  vector<int> inliers;
  solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t_cv, false, 100, 8.0f, 0.99, inliers);
  Rodrigues(r, R_cv);
  return CvToSE3(R_cv, t_cv);
}

Sophus::SE3d RunIcpPairOnce(const string &dataset_dir, const Mat &K, int i, int j) {
  Mat img_i = imread(dataset_dir + "/" + PadIndex(i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  Mat depth_j = imread(dataset_dir + "/" + PadIndex(j) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);

  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);

  vector<Point3f> pts1, pts2;
  for (const DMatch &m : matches) {
    ushort d1 = depth_i.ptr<unsigned short>((int)keypoints_i[m.queryIdx].pt.y)
                       [(int)keypoints_i[m.queryIdx].pt.x];
    ushort d2 = depth_j.ptr<unsigned short>((int)keypoints_j[m.trainIdx].pt.y)
                       [(int)keypoints_j[m.trainIdx].pt.x];
    if (d1 == 0 || d2 == 0) continue;
    Point2d p1 = pixel2cam(keypoints_i[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_j[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0f;
    float dd2 = float(d2) / 5000.0f;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  Mat R_icp, t_icp;
  pose_estimation_3d3d(pts1, pts2, R_icp, t_icp);
  bundleAdjustment(pts1, pts2, R_icp, t_icp);  // existing book code, unmodified

  // The ONE and ONLY convention-normalizing inversion: native T_{i<-j} -> common T_{j<-i}.
  Mat R_common = R_icp.t();
  Mat t_common = -R_common * t_icp;
  return CvToSE3(R_common, t_common);
}

}  // namespace

int main(int argc, char **argv) {
  string dataset_dir = argc > 1 ? argv[1] : "../data/synthetic_bunny";
  Intrinsics intr = ReadIntrinsics(dataset_dir + "/intrinsics.txt");
  Mat K = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);
  vector<Sophus::SE3d> T_gt = ReadGroundtruth(dataset_dir + "/groundtruth.txt");
  int N = (int)T_gt.size();

  cout << "\n========================================" << endl;
  cout << "SECTION 6: ZERO-MOTION TEST" << endl;
  cout << "========================================" << endl;
  {
    double max_err = 0;
    for (int k : {0, 5, 20, 35}) {
      Sophus::SE3d result = AccumulateStep(T_gt[k], Sophus::SE3d());  // T_rel = Identity
      double err = (result.translation() - T_gt[k].translation()).norm() +
                   RotationAngleDeg(EigenRotToCv(T_gt[k].rotationMatrix()),
                                    EigenRotToCv(result.rotationMatrix()));
      max_err = max(max_err, err);
      cout << "  anchor frame " << k << ": |result - T_wc[" << k << "]| = " << err << endl;
    }
    cout << (max_err < 1e-9 ? "PASS" : "FAIL")
         << ": T_rel=Identity => T_wc[i+1] == T_wc[i], max combined error = " << max_err << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 7: PERFECT-MOTION TEST (ground-truth relative poses through the "
          "SAME accumulation function)"
       << endl;
  cout << "========================================" << endl;
  {
    vector<Sophus::SE3d> T_recomputed(N);
    T_recomputed[0] = T_gt[0];
    for (int i = 0; i < N - 1; ++i) {
      Sophus::SE3d T_gt_rel = T_gt[i + 1].inverse() * T_gt[i];  // exact GT relative, common convention
      T_recomputed[i + 1] = AccumulateStep(T_recomputed[i], T_gt_rel);
    }
    double max_t_err = 0, max_r_err = 0;
    for (int k = 0; k < N; ++k) {
      double t_err = (T_recomputed[k].translation() - T_gt[k].translation()).norm();
      double r_err = RotationAngleDeg(EigenRotToCv(T_gt[k].rotationMatrix()),
                                      EigenRotToCv(T_recomputed[k].rotationMatrix()));
      max_t_err = max(max_t_err, t_err);
      max_r_err = max(max_r_err, r_err);
    }
    cout << "max translation error over all " << N << " frames: " << max_t_err << " m" << endl;
    cout << "max rotation error over all " << N << " frames: " << max_r_err << " deg" << endl;
    cout << ((max_t_err < 1e-6 && max_r_err < 1e-4) ? "PASS" : "FAIL")
         << ": perfect ground-truth relative poses reproduce the ground-truth trajectory "
            "through the accumulation formula"
         << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 8: IDENTITY-ANCHOR TEST" << endl;
  cout << "========================================" << endl;
  {
    vector<Sophus::SE3d> T_recomputed(N);
    T_recomputed[0] = Sophus::SE3d();  // Identity, not T_gt[0]
    for (int i = 0; i < N - 1; ++i) {
      Sophus::SE3d T_gt_rel = T_gt[i + 1].inverse() * T_gt[i];
      T_recomputed[i + 1] = AccumulateStep(T_recomputed[i], T_gt_rel);
    }
    double max_t_err = 0, max_r_err = 0;
    for (int k = 0; k < N; ++k) {
      Sophus::SE3d T_gt_rel_to_frame0 = T_gt[0].inverse() * T_gt[k];  // GT expressed relative to frame 0
      double t_err = (T_recomputed[k].translation() - T_gt_rel_to_frame0.translation()).norm();
      double r_err = RotationAngleDeg(EigenRotToCv(T_gt_rel_to_frame0.rotationMatrix()),
                                      EigenRotToCv(T_recomputed[k].rotationMatrix()));
      max_t_err = max(max_t_err, t_err);
      max_r_err = max(max_r_err, r_err);
    }
    cout << "max translation error vs GT-relative-to-frame0 over all " << N
         << " frames: " << max_t_err << " m" << endl;
    cout << "max rotation error vs GT-relative-to-frame0 over all " << N
         << " frames: " << max_r_err << " deg" << endl;
    cout << ((max_t_err < 1e-6 && max_r_err < 1e-4) ? "PASS" : "FAIL")
         << ": identity-anchored accumulation reproduces GT expressed relative to frame 0"
         << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 4: ONE-STEP VALIDATION (frame 0 -> 1), REAL PnP(RANSAC)" << endl;
  cout << "========================================" << endl;
  {
    Sophus::SE3d T_gt_rel = T_gt[1].inverse() * T_gt[0];
    Sophus::SE3d T_pnp_rel = RunPnpPairOnce(dataset_dir, K, 0, 1);

    PrintPose("GT relative (0->1)", T_gt_rel);
    PrintPose("PnP relative (0->1)", T_pnp_rel);
    double rot_err = RotationAngleDeg(EigenRotToCv(T_gt_rel.rotationMatrix()),
                                      EigenRotToCv(T_pnp_rel.rotationMatrix()));
    double t_err = TranslationDiff(EigenVecToCv(T_gt_rel.translation()),
                                   EigenVecToCv(T_pnp_rel.translation()));
    cout << "R error = " << rot_err << " deg" << endl;
    cout << "t error = " << t_err << " m" << endl;

    Sophus::SE3d T_wc1_expected = AccumulateStep(T_gt[0], T_pnp_rel);
    cout << "\nAccumulated frame-1 pose from T_wc[0] + PnP relative (via the accumulation formula):"
         << endl;
    PrintPose("  T_wc[1] (expected, from composition)", T_wc1_expected);
    PrintPose("  T_wc[1] (ground truth)", T_gt[1]);
    cout << "position error: " << (T_wc1_expected.translation() - T_gt[1].translation()).norm()
         << " m" << endl;
    cout << "rotation error: "
         << RotationAngleDeg(EigenRotToCv(T_gt[1].rotationMatrix()),
                              EigenRotToCv(T_wc1_expected.rotationMatrix()))
         << " deg" << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 5: TWO-STEP VALIDATION (0 -> 1 -> 2), REAL PnP(RANSAC)" << endl;
  cout << "========================================" << endl;
  {
    Sophus::SE3d T_rel01 = RunPnpPairOnce(dataset_dir, K, 0, 1);
    Sophus::SE3d T_rel12 = RunPnpPairOnce(dataset_dir, K, 1, 2);
    Sophus::SE3d T_pnp0 = T_gt[0];
    Sophus::SE3d T_pnp1 = AccumulateStep(T_pnp0, T_rel01);
    Sophus::SE3d T_pnp2 = AccumulateStep(T_pnp1, T_rel12);

    for (int k = 0; k < 3; ++k) {
      const Sophus::SE3d &pnp_pose = (k == 0) ? T_pnp0 : (k == 1 ? T_pnp1 : T_pnp2);
      cout << "Frame " << k << ":" << endl;
      cout << "  GT position:  " << EigenVecToCv(T_gt[k].translation()).t() << endl;
      cout << "  PnP position: " << EigenVecToCv(pnp_pose.translation()).t() << endl;
      cout << "  rotation error: "
           << RotationAngleDeg(EigenRotToCv(T_gt[k].rotationMatrix()),
                                EigenRotToCv(pnp_pose.rotationMatrix()))
           << " deg" << endl;
    }
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 10 (ICP variant of section 4): ONE-STEP VALIDATION, REAL ICP" << endl;
  cout << "========================================" << endl;
  {
    Sophus::SE3d T_gt_rel = T_gt[1].inverse() * T_gt[0];
    Sophus::SE3d T_icp_rel = RunIcpPairOnce(dataset_dir, K, 0, 1);
    PrintPose("GT relative (0->1)", T_gt_rel);
    PrintPose("ICP relative, converted to common convention (0->1)", T_icp_rel);
    cout << "R error = "
         << RotationAngleDeg(EigenRotToCv(T_gt_rel.rotationMatrix()), EigenRotToCv(T_icp_rel.rotationMatrix()))
         << " deg" << endl;
    cout << "t error = "
         << TranslationDiff(EigenVecToCv(T_gt_rel.translation()), EigenVecToCv(T_icp_rel.translation()))
         << " m" << endl;

    Sophus::SE3d T_wc1_expected = AccumulateStep(T_gt[0], T_icp_rel);
    cout << "position error of accumulated frame-1 vs GT: "
         << (T_wc1_expected.translation() - T_gt[1].translation()).norm() << " m" << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 9: pnp_trajectory.txt FILE VERIFICATION" << endl;
  cout << "========================================" << endl;
  {
    FileTrajectory pnp_file = ReadTrajectoryFile(dataset_dir + "/pnp_trajectory.txt");
    cout << "frames in file: " << pnp_file.poses.size() << " (expected " << N << ")" << endl;
    bool order_ok = true;
    for (int k = 0; k < (int)pnp_file.timestamps.size(); ++k)
      if (std::abs(pnp_file.timestamps[k] - (double)k) > 1e-6) order_ok = false;
    cout << "timestamp == frame index for every row: " << (order_ok ? "PASS" : "FAIL") << endl;
    double frame0_diff = (pnp_file.poses[0].translation() - T_gt[0].translation()).norm();
    cout << "frame 0 matches GT frame 0: " << frame0_diff << " m ("
         << (frame0_diff < 1e-6 ? "PASS" : "FAIL") << ")" << endl;
    bool quat_ok = true;
    for (const auto &p : pnp_file.poses)
      if (std::abs(p.unit_quaternion().norm() - 1.0) > 1e-6) quat_ok = false;
    cout << "all quaternions unit-norm: " << (quat_ok ? "PASS" : "FAIL") << endl;
    // Spot-check: file's own consecutive-frame relative pose (0->1) should
    // match a fresh, independent PnP run on that same pair.
    Sophus::SE3d file_rel01 = pnp_file.poses[1].inverse() * pnp_file.poses[0];
    Sophus::SE3d fresh_rel01 = RunPnpPairOnce(dataset_dir, K, 0, 1);
    double spot_check_err = (file_rel01.translation() - fresh_rel01.translation()).norm();
    cout << "spot-check: file's implied 0->1 relative pose vs fresh PnP re-run: "
         << spot_check_err << " m (" << (spot_check_err < 1e-6 ? "PASS, self-consistent" : "FAIL") << ")"
         << endl;
  }

  cout << "\n========================================" << endl;
  cout << "SECTION 10: icp_trajectory.txt FILE VERIFICATION" << endl;
  cout << "========================================" << endl;
  {
    FileTrajectory icp_file = ReadTrajectoryFile(dataset_dir + "/icp_trajectory.txt");
    cout << "frames in file: " << icp_file.poses.size() << " (expected " << N << ")" << endl;
    bool order_ok = true;
    for (int k = 0; k < (int)icp_file.timestamps.size(); ++k)
      if (std::abs(icp_file.timestamps[k] - (double)k) > 1e-6) order_ok = false;
    cout << "timestamp == frame index for every row: " << (order_ok ? "PASS" : "FAIL") << endl;
    double frame0_diff = (icp_file.poses[0].translation() - T_gt[0].translation()).norm();
    cout << "frame 0 matches GT frame 0: " << frame0_diff << " m ("
         << (frame0_diff < 1e-6 ? "PASS" : "FAIL") << ")" << endl;
    Sophus::SE3d file_rel01 = icp_file.poses[1].inverse() * icp_file.poses[0];
    Sophus::SE3d fresh_rel01 = RunIcpPairOnce(dataset_dir, K, 0, 1);
    double spot_check_err = (file_rel01.translation() - fresh_rel01.translation()).norm();
    cout << "spot-check: file's implied 0->1 relative pose vs fresh ICP re-run: "
         << spot_check_err << " m (" << (spot_check_err < 1e-6 ? "PASS, self-consistent" : "FAIL") << ")"
         << endl;
  }

  cout << "\n=== VALIDATION COMPLETE ===" << endl;
  return 0;
}
