// Builds complete global PnP(RANSAC) and ICP trajectories over the full
// synthetic bunny sequence, using the EXISTING pose-estimation code
// (src/tracking/pnp.cpp's solvePnP call pattern, extended here to
// solvePnPRansac per an explicit instruction to use the robust variant for
// this trajectory experiment; src/tracking/icp.cpp's pose_estimation_3d3d +
// bundleAdjustment, completely unmodified). No changes to any existing
// algorithm source file.
//
// === Convention derivation (not assumed) ===
// groundtruth.txt stores T_wc[k] (camera-to-world): translation = camera
// center in world, rotation maps camera-local vectors to world vectors.
//
// PnP (RANSAC) recovers, for a pair (i, i+1), (R,t) such that
// x_{i+1} = R*x_i + t -- i.e. this IS T_{i+1<-i} directly (Sophus::SE3d(R,t)
// applied to a point already does R*p+t, matching Sophus's own semantics).
//
// ICP's pose_estimation_3d3d(pts1=frame i, pts2=frame i+1, R, t) solves the
// OPPOSITE direction, x_i = R*x_{i+1} + t, i.e. T_{i<-i+1}. Inverting once
// (R' = R^T, t' = -R^T t) gives the same T_{i+1<-i} form PnP already uses.
//
// To accumulate T_{i+1<-i} into the global camera-to-world convention:
//   T_wc[i+1] * p_{i+1} = world point
//   T_{i+1<-i} maps camera-i points to camera-(i+1) points, so its inverse
//   T_{i<-i+1} maps camera-(i+1) points to camera-i points, and
//   T_wc[i] * T_{i<-i+1} * p_{i+1} = T_wc[i] * p_i = world point.
//   => T_wc[i+1] = T_wc[i] * T_{i+1<-i}.inverse()
// This is NOT "T_next = T_relative * T_current" -- verified algebraically
// above, not assumed. Frame 0 is initialized to the ground-truth frame-0
// pose (this repo has no independent absolute-pose source, so frame 0 is
// the natural common anchor for both estimated trajectories).

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

struct PairwisePnpResult {
  int matches = 0, corr = 0, inliers = -1;
  double rot_err = -1, trans_err = -1;
  bool ok = false;
  Sophus::SE3d T_rel;  // x_{i+1} = R*x_i + t  (T_{i+1<-i})
};

PairwisePnpResult RunPnpPair(const string &dataset_dir, const Mat &K,
                              const vector<Sophus::SE3d> &T_gt, int i, int j) {
  PairwisePnpResult res;
  Mat img_i = imread(dataset_dir + "/" + PadIndex(i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  if (img_i.empty() || img_j.empty() || depth_i.empty()) return res;

  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);
  res.matches = (int)matches.size();

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
  res.corr = (int)pts_3d.size();
  if (res.corr < 4) return res;

  Mat r, t_cv, R_cv;
  vector<int> inliers;
  bool ransac_ok = solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t_cv, false, 100, 8.0f, 0.99, inliers);
  if (!ransac_ok) return res;
  Rodrigues(r, R_cv);
  res.inliers = (int)inliers.size();
  res.T_rel = CvToSE3(R_cv, t_cv);

  Sophus::SE3d T_gt_rel = T_gt[j].inverse() * T_gt[i];
  res.rot_err = RotationAngleDeg(EigenRotToCv(T_gt_rel.rotationMatrix()), R_cv);
  res.trans_err = TranslationDiff(EigenVecToCv(T_gt_rel.translation()), t_cv);
  res.ok = true;
  return res;
}

struct PairwiseIcpResult {
  int matches = 0, corr = 0;
  double rot_err = -1, trans_err = -1;
  bool ok = false;
  Sophus::SE3d T_rel;  // converted to common convention: x_{i+1} = R*x_i + t
};

PairwiseIcpResult RunIcpPair(const string &dataset_dir, const Mat &K,
                              const vector<Sophus::SE3d> &T_gt, int i, int j) {
  PairwiseIcpResult res;
  Mat img_i = imread(dataset_dir + "/" + PadIndex(i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  Mat depth_j = imread(dataset_dir + "/" + PadIndex(j) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  if (img_i.empty() || img_j.empty() || depth_i.empty() || depth_j.empty()) return res;

  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);
  res.matches = (int)matches.size();

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
  res.corr = (int)pts1.size();
  if (res.corr < 3) return res;

  Mat R_icp, t_icp;
  pose_estimation_3d3d(pts1, pts2, R_icp, t_icp);
  bundleAdjustment(pts1, pts2, R_icp, t_icp);  // existing book code, unmodified

  // Convert ICP's native x_i = R*x_j + t into the common x_j = R*x_i + t.
  Mat R_common = R_icp.t();
  Mat t_common = -R_common * t_icp;
  res.T_rel = CvToSE3(R_common, t_common);

  Sophus::SE3d T_gt_rel = T_gt[j].inverse() * T_gt[i];
  res.rot_err = RotationAngleDeg(EigenRotToCv(T_gt_rel.rotationMatrix()), R_common);
  res.trans_err = TranslationDiff(EigenVecToCv(T_gt_rel.translation()), t_common);
  res.ok = true;
  return res;
}

void WriteTrajectoryTUM(const string &path, const string &convention_comment,
                        const vector<Sophus::SE3d> &T_wc) {
  ofstream f(path);
  f << "# " << convention_comment << "\n";
  f << "# timestamp tx ty tz qx qy qz qw  (T_wc: camera -> world)\n";
  f << fixed << setprecision(6);
  for (size_t i = 0; i < T_wc.size(); ++i) {
    Eigen::Quaterniond q = T_wc[i].unit_quaternion();
    Eigen::Vector3d t = T_wc[i].translation();
    f << (double)i << " " << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
      << q.y() << " " << q.z() << " " << q.w() << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  string dataset_dir = argc > 1 ? argv[1] : "../data/synthetic_bunny";

  Intrinsics intr = ReadIntrinsics(dataset_dir + "/intrinsics.txt");
  Mat K = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);
  vector<Sophus::SE3d> T_gt = ReadGroundtruth(dataset_dir + "/groundtruth.txt");
  int N = (int)T_gt.size();
  cout << "Loaded " << N << " ground-truth frames from " << dataset_dir << "/groundtruth.txt" << endl;

  vector<PairwisePnpResult> pnp_pairs(N - 1);
  vector<PairwiseIcpResult> icp_pairs(N - 1);
  for (int i = 0; i < N - 1; ++i) {
    cout << "processing pair " << i << " -> " << i + 1 << "..." << endl;
    pnp_pairs[i] = RunPnpPair(dataset_dir, K, T_gt, i, i + 1);
    icp_pairs[i] = RunIcpPair(dataset_dir, K, T_gt, i, i + 1);
  }

  cout << "\n=== PER-PAIR RELATIVE POSE RESULTS ===\n" << endl;
  cout << left << setw(8) << "Pair" << setw(9) << "Matches" << setw(7) << "PnP3D2D"
       << setw(10) << "PnPInl" << setw(12) << "PnPRotErr" << setw(13) << "PnPTransErr"
       << setw(9) << "ICP3D3D" << setw(12) << "ICPRotErr" << setw(13) << "ICPTransErr" << endl;
  for (int i = 0; i < N - 1; ++i) {
    const auto &p = pnp_pairs[i];
    const auto &c = icp_pairs[i];
    ostringstream label;
    label << i << "->" << i + 1;
    cout << left << setw(8) << label.str() << setw(9) << p.matches << setw(7) << p.corr
         << setw(10) << (p.ok ? p.inliers : -1) << fixed << setprecision(3)
         << setw(12) << (p.ok ? p.rot_err : -1.0) << setw(13) << (p.ok ? p.trans_err : -1.0)
         << setw(9) << c.corr << setw(12) << (c.ok ? c.rot_err : -1.0)
         << setw(13) << (c.ok ? c.trans_err : -1.0) << endl;
  }

  // === Accumulate global trajectories (see file header for the derivation
  // of this exact composition order). Frame 0 anchored to ground truth. ===
  vector<Sophus::SE3d> T_pnp(N), T_icp(N);
  T_pnp[0] = T_gt[0];
  T_icp[0] = T_gt[0];
  int pnp_failures = 0, icp_failures = 0;
  for (int i = 0; i < N - 1; ++i) {
    if (pnp_pairs[i].ok) {
      T_pnp[i + 1] = T_pnp[i] * pnp_pairs[i].T_rel.inverse();
    } else {
      T_pnp[i + 1] = T_pnp[i];  // fallback: hold position, flagged below
      pnp_failures++;
    }
    if (icp_pairs[i].ok) {
      T_icp[i + 1] = T_icp[i] * icp_pairs[i].T_rel.inverse();
    } else {
      T_icp[i + 1] = T_icp[i];
      icp_failures++;
    }
  }

  vector<double> pnp_t_err(N), pnp_r_err(N), icp_t_err(N), icp_r_err(N);
  for (int k = 0; k < N; ++k) {
    pnp_t_err[k] = TranslationDiff(EigenVecToCv(T_gt[k].translation()),
                                   EigenVecToCv(T_pnp[k].translation()));
    pnp_r_err[k] = RotationAngleDeg(EigenRotToCv(T_gt[k].rotationMatrix()),
                                    EigenRotToCv(T_pnp[k].rotationMatrix()));
    icp_t_err[k] = TranslationDiff(EigenVecToCv(T_gt[k].translation()),
                                   EigenVecToCv(T_icp[k].translation()));
    icp_r_err[k] = RotationAngleDeg(EigenRotToCv(T_gt[k].rotationMatrix()),
                                    EigenRotToCv(T_icp[k].rotationMatrix()));
  }

  cout << "\n=== PER-FRAME ACCUMULATED TRAJECTORY ERROR ===\n" << endl;
  cout << left << setw(7) << "Frame" << setw(30) << "GT position" << setw(30) << "PnP position"
       << setw(30) << "ICP position" << setw(11) << "PnPTErr" << setw(11) << "ICPTErr"
       << setw(11) << "PnPRErr" << setw(11) << "ICPRErr" << endl;
  auto fmt_pos = [](const Eigen::Vector3d &p) {
    ostringstream o;
    o << fixed << setprecision(3) << "(" << p.x() << "," << p.y() << "," << p.z() << ")";
    return o.str();
  };
  for (int k = 0; k < N; ++k) {
    cout << left << setw(7) << k << setw(30) << fmt_pos(T_gt[k].translation())
         << setw(30) << fmt_pos(T_pnp[k].translation()) << setw(30) << fmt_pos(T_icp[k].translation())
         << fixed << setprecision(4) << setw(11) << pnp_t_err[k] << setw(11) << icp_t_err[k]
         << setw(11) << pnp_r_err[k] << setw(11) << icp_r_err[k] << endl;
  }

  auto stat = [](vector<double> v) {
    Stats s = ComputeStats(v);
    return s;
  };
  Stats pnp_t_stats = stat(pnp_t_err), pnp_r_stats = stat(pnp_r_err);
  Stats icp_t_stats = stat(icp_t_err), icp_r_stats = stat(icp_r_err);
  int pnp_successful_pairs = (N - 1) - pnp_failures;
  int icp_successful_pairs = (N - 1) - icp_failures;

  cout << "\n========================================" << endl;
  cout << "SYNTHETIC BUNNY SLAM TRAJECTORY" << endl;
  cout << "========================================\n" << endl;
  cout << "Frames: " << N << "\n" << endl;

  cout << "PNP (RANSAC)" << endl;
  cout << "-------------" << endl;
  cout << "Successful pairs: " << pnp_successful_pairs << " / " << (N - 1) << endl;
  cout << "Mean rotation error: " << pnp_r_stats.mean << " deg" << endl;
  cout << "Median rotation error: " << pnp_r_stats.median << " deg" << endl;
  cout << "Max rotation error: " << pnp_r_stats.max << " deg" << endl;
  cout << "Mean translation error: " << pnp_t_stats.mean << " m" << endl;
  cout << "Median translation error: " << pnp_t_stats.median << " m" << endl;
  cout << "Max translation error: " << pnp_t_stats.max << " m" << endl;
  cout << "Final translation error: " << pnp_t_err[N - 1] << " m" << endl;
  cout << "Final rotation error: " << pnp_r_err[N - 1] << " deg\n" << endl;

  cout << "ICP" << endl;
  cout << "---" << endl;
  cout << "Successful pairs: " << icp_successful_pairs << " / " << (N - 1) << endl;
  cout << "Mean rotation error: " << icp_r_stats.mean << " deg" << endl;
  cout << "Median rotation error: " << icp_r_stats.median << " deg" << endl;
  cout << "Max rotation error: " << icp_r_stats.max << " deg" << endl;
  cout << "Mean translation error: " << icp_t_stats.mean << " m" << endl;
  cout << "Median translation error: " << icp_t_stats.median << " m" << endl;
  cout << "Max translation error: " << icp_t_stats.max << " m" << endl;
  cout << "Final translation error: " << icp_t_err[N - 1] << " m" << endl;
  cout << "Final rotation error: " << icp_r_err[N - 1] << " deg" << endl;
  cout << "\n========================================" << endl;

  // === Save outputs ===
  WriteTrajectoryTUM(dataset_dir + "/pnp_trajectory.txt",
                     "PnP(RANSAC) global trajectory, accumulated via "
                     "T_wc[i+1]=T_wc[i]*T_rel(i,i+1)^-1, frame 0 anchored to groundtruth.txt",
                     T_pnp);
  WriteTrajectoryTUM(dataset_dir + "/icp_trajectory.txt",
                     "ICP global trajectory (converted from native x_i=R*x_j+t to common "
                     "x_j=R*x_i+t before accumulation), frame 0 anchored to groundtruth.txt",
                     T_icp);
  cout << "\nwrote " << dataset_dir << "/pnp_trajectory.txt" << endl;
  cout << "wrote " << dataset_dir << "/icp_trajectory.txt" << endl;

  ofstream csv(dataset_dir + "/slam_trajectory.csv");
  csv << "frame,gt_x,gt_y,gt_z,pnp_x,pnp_y,pnp_z,icp_x,icp_y,icp_z,"
      << "gt_qx,gt_qy,gt_qz,gt_qw,pnp_qx,pnp_qy,pnp_qz,pnp_qw,icp_qx,icp_qy,icp_qz,icp_qw,"
      << "pnp_translation_error,pnp_rotation_error,icp_translation_error,icp_rotation_error\n";
  csv << fixed << setprecision(6);
  for (int k = 0; k < N; ++k) {
    Eigen::Vector3d gt_t = T_gt[k].translation(), pnp_t = T_pnp[k].translation(),
                    icp_t = T_icp[k].translation();
    Eigen::Quaterniond gt_q = T_gt[k].unit_quaternion(), pnp_q = T_pnp[k].unit_quaternion(),
                       icp_q = T_icp[k].unit_quaternion();
    csv << k << "," << gt_t.x() << "," << gt_t.y() << "," << gt_t.z() << "," << pnp_t.x() << ","
        << pnp_t.y() << "," << pnp_t.z() << "," << icp_t.x() << "," << icp_t.y() << "," << icp_t.z()
        << "," << gt_q.x() << "," << gt_q.y() << "," << gt_q.z() << "," << gt_q.w() << ","
        << pnp_q.x() << "," << pnp_q.y() << "," << pnp_q.z() << "," << pnp_q.w() << "," << icp_q.x()
        << "," << icp_q.y() << "," << icp_q.z() << "," << icp_q.w() << "," << pnp_t_err[k] << ","
        << pnp_r_err[k] << "," << icp_t_err[k] << "," << icp_r_err[k] << "\n";
  }
  cout << "wrote " << dataset_dir << "/slam_trajectory.csv" << endl;

  return 0;
}
