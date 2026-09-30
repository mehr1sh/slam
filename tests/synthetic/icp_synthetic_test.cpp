// Tests the EXISTING, unmodified ICP implementation (src/tracking/icp.cpp,
// via the identical call pattern tests/icp_test.cpp already uses) on one
// frame pair from the synthetic bunny dataset.
//
// Convention (verified from src/tracking/icp.cpp and tests/icp_test.cpp,
// not assumed): pts1 is backprojected from frame i's depth, pts2 from
// frame j's depth. pose_estimation_3d3d(pts1, pts2, R, t) computes
// t = centroid(pts1) - R*centroid(pts2), i.e. it solves for x1 = R*x2 + t
// -- the OPPOSITE direction from PnP's x_j = R*x_i + t. So ICP's raw
// output here is T_{i<-j} (maps frame-j 3D points into frame-i's frame),
// while the ground truth / PnP convention used elsewhere in this project
// is T_{j<-i}. The two are related by simple inversion: T_{j<-i} =
// (T_{i<-j})^-1, i.e. R_common = R_icp^T, t_common = -R_icp^T * t_icp.
// This conversion happens once, right here, at the point ICP's result
// meets the common convention -- ICP's own math is never touched.

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
    if (line.empty()) continue;
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

// Inverts an (R,t): from x_a = R*x_b + t to x_b = R'*x_a + t'.
void InvertPose(const Mat &R, const Mat &t, Mat &R_inv, Mat &t_inv) {
  R_inv = R.t();
  t_inv = -R_inv * t;
}

}  // namespace

int main(int argc, char **argv) {
  string dataset_dir = argc > 1 ? argv[1] : "../data/synthetic_bunny";
  int frame_i = argc > 2 ? atoi(argv[2]) : 0;
  int frame_j = argc > 3 ? atoi(argv[3]) : 1;

  Intrinsics intr = ReadIntrinsics(dataset_dir + "/intrinsics.txt");
  Mat K = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);

  Mat img_i = imread(dataset_dir + "/" + PadIndex(frame_i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(frame_j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(frame_i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  Mat depth_j = imread(dataset_dir + "/" + PadIndex(frame_j) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  if (img_i.empty() || img_j.empty() || depth_i.empty() || depth_j.empty()) {
    cerr << "failed to load frame " << frame_i << "/" << frame_j << " from " << dataset_dir << endl;
    return 1;
  }

  int valid_depth_i = countNonZero(depth_i);
  int valid_depth_j = countNonZero(depth_j);

  // Book-code-identical feature matching (src/features/features.cpp).
  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);

  // Book-code-identical 3D-3D correspondence construction (see
  // tests/icp_test.cpp): pts1 from frame i's depth, pts2 from frame j's
  // depth, both required to have valid depth at the matched keypoint.
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

  cout << "=== ICP SYNTHETIC TEST ===\n" << endl;
  cout << "Frame " << frame_i << " -> Frame " << frame_j << "\n" << endl;

  vector<Sophus::SE3d> T_wc = ReadGroundtruth(dataset_dir + "/groundtruth.txt");
  Sophus::SE3d T_gt = T_wc[frame_j].inverse() * T_wc[frame_i];  // common convention: x_j = R*x_i + t
  Mat R_gt = EigenRotToCv(T_gt.rotationMatrix());
  Mat t_gt = EigenVecToCv(T_gt.translation());
  cout << "Ground Truth:" << endl;
  cout << "R =\n" << R_gt << endl;
  cout << "t = " << t_gt.t() << endl;

  if ((int)pts1.size() < 3) {
    cout << "\nFATAL: only " << pts1.size()
         << " valid 3D-3D correspondences -- ICP needs at least 3. Not running ICP." << endl;
    return 1;
  }

  // Existing book code, unmodified: closed-form SVD (pose_estimation_3d3d),
  // exactly as tests/icp_test.cpp calls it.
  Mat R_svd, t_svd;
  pose_estimation_3d3d(pts1, pts2, R_svd, t_svd);
  Mat R_svd_copy = R_svd.clone(), t_svd_copy = t_svd.clone();

  // Existing book code, unmodified: g2o refinement (bundleAdjustment),
  // overwrites R_svd/t_svd in place exactly as tests/icp_test.cpp does.
  bundleAdjustment(pts1, pts2, R_svd, t_svd);

  cout << "\nICP raw output (x_i = R*x_j + t, this repo's ICP convention):" << endl;
  cout << "R =\n" << R_svd << endl;
  cout << "t = " << t_svd.t() << endl;

  // Convert to the common (PnP/ground-truth) convention by inversion.
  Mat R_common, t_common;
  InvertPose(R_svd, t_svd, R_common, t_common);
  cout << "\nICP converted to common convention (x_j = R*x_i + t):" << endl;
  cout << "R =\n" << R_common << endl;
  cout << "t = " << t_common.t() << endl;

  double rot_err = RotationAngleDeg(R_gt, R_common);
  double t_err = TranslationDiff(t_gt, t_common);
  cout << "\nRotation error: " << rot_err << " degrees" << endl;
  cout << "Translation error: " << t_err << " metres" << endl;

  // Point-to-point alignment residual (same diagnostic tests/icp_test.cpp
  // already reports), using the FINAL (post-bundleAdjustment) R,t in ICP's
  // own raw convention -- p1 ~= R*p2+t.
  vector<double> errs;
  for (size_t k = 0; k < pts1.size(); ++k) {
    Mat p2 = (Mat_<double>(3, 1) << pts2[k].x, pts2[k].y, pts2[k].z);
    Mat p1_est = R_svd * p2 + t_svd;
    Mat p1 = (Mat_<double>(3, 1) << pts1[k].x, pts1[k].y, pts1[k].z);
    errs.push_back(norm(p1_est - p1));
  }
  Stats err_stats = ComputeStats(errs);

  cout << "\nNumber of valid depth pixels (frame " << frame_i << "): " << valid_depth_i << endl;
  cout << "Number of valid depth pixels (frame " << frame_j << "): " << valid_depth_j << endl;
  cout << "Number of feature matches: " << matches.size() << endl;
  cout << "Number of 3D-3D correspondences: " << pts1.size() << endl;
  cout << "ICP iterations: 10 (fixed, hardcoded in bundleAdjustment(), see src/tracking/icp.cpp)"
       << endl;
  cout << "Final point-to-point alignment error (m) mean/median/rms/max: "
       << err_stats.mean << " / " << err_stats.median << " / " << err_stats.rms << " / "
       << err_stats.max << endl;

  return 0;
}
