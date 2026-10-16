// Runs the EXISTING, unmodified PnP implementation (src/tracking/pnp.cpp's
// plain solvePnP call, via the identical pattern tests/pnp_test.cpp and
// tests/pnp_synthetic_test.cpp already use) across several consecutive
// synthetic-bunny frame pairs, to see whether the frame-0->1 outlier
// sensitivity was a one-off or a repeated pattern. No algorithm code is
// touched -- this only calls existing functions in a loop and tabulates.

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

struct PairResult {
  int i, j;
  int feat_i = 0, feat_j = 0, matches = 0, corr3d2d = 0;
  double plain_rot_err = -1, plain_trans_err = -1;
  double ransac_rot_err = -1, ransac_trans_err = -1;
  int ransac_inliers = -1;
  bool ok = false;
};

PairResult RunPair(const string &dataset_dir, const Mat &K,
                    const vector<Sophus::SE3d> &T_wc, int i, int j) {
  PairResult res;
  res.i = i;
  res.j = j;

  Mat img_i = imread(dataset_dir + "/" + PadIndex(i) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat img_j = imread(dataset_dir + "/" + PadIndex(j) + ".png", CV_LOAD_IMAGE_COLOR);
  Mat depth_i = imread(dataset_dir + "/" + PadIndex(i) + "_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  if (img_i.empty() || img_j.empty() || depth_i.empty()) return res;

  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches);
  res.feat_i = (int)keypoints_i.size();
  res.feat_j = (int)keypoints_j.size();
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
  res.corr3d2d = (int)pts_3d.size();
  if (res.corr3d2d < 4) return res;

  Sophus::SE3d T_gt = T_wc[j].inverse() * T_wc[i];
  Mat R_gt = EigenRotToCv(T_gt.rotationMatrix());
  Mat t_gt = EigenVecToCv(T_gt.translation());

  // Existing book code, unmodified.
  Mat r, t_cv, R_cv;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t_cv, false);
  Rodrigues(r, R_cv);
  res.plain_rot_err = RotationAngleDeg(R_gt, R_cv);
  res.plain_trans_err = TranslationDiff(t_gt, t_cv);

  // Diagnostic only.
  Mat r_ransac, t_ransac, R_ransac;
  vector<int> inliers;
  bool ransac_ok = solvePnPRansac(pts_3d, pts_2d, K, Mat(), r_ransac, t_ransac,
                                  false, 100, 8.0f, 0.99, inliers);
  if (ransac_ok) {
    Rodrigues(r_ransac, R_ransac);
    res.ransac_rot_err = RotationAngleDeg(R_gt, R_ransac);
    res.ransac_trans_err = TranslationDiff(t_gt, t_ransac);
    res.ransac_inliers = (int)inliers.size();
  }

  res.ok = true;
  return res;
}

}  // namespace

int main(int argc, char **argv) {
  string dataset_dir = argc > 1 ? argv[1] : "../data/synthetic_bunny";
  int start = argc > 2 ? atoi(argv[2]) : 0;
  int end = argc > 3 ? atoi(argv[3]) : 10;  // inclusive pair end index (pairs start..start+1 .. end-1..end)

  Intrinsics intr = ReadIntrinsics(dataset_dir + "/intrinsics.txt");
  Mat K = (Mat_<double>(3, 3) << intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);
  vector<Sophus::SE3d> T_wc = ReadGroundtruth(dataset_dir + "/groundtruth.txt");

  vector<PairResult> results;
  for (int i = start; i < end; ++i) {
    cout << "processing pair " << i << " -> " << i + 1 << "..." << endl;
    results.push_back(RunPair(dataset_dir, K, T_wc, i, i + 1));
  }

  cout << "\n=== PNP MULTI-FRAME TEST (frames " << start << " -> " << end << ") ===\n" << endl;
  cout << left
       << setw(8) << "Pair" << setw(11) << "Features" << setw(9) << "Matches"
       << setw(7) << "3D-2D" << setw(15) << "PlainRotErr" << setw(16) << "PlainTransErr"
       << setw(13) << "RansacRotErr" << setw(15) << "RansacTransErr" << setw(9) << "Inliers"
       << endl;
  for (const auto &r : results) {
    ostringstream pair_label;
    pair_label << r.i << "->" << r.j;
    ostringstream feat_label;
    feat_label << r.feat_i << "/" << r.feat_j;
    cout << left << setw(8) << pair_label.str() << setw(11) << feat_label.str()
         << setw(9) << r.matches << setw(7) << r.corr3d2d;
    if (!r.ok) {
      cout << "FAILED (insufficient correspondences)" << endl;
      continue;
    }
    cout << fixed << setprecision(3)
         << setw(15) << r.plain_rot_err << setw(16) << r.plain_trans_err
         << setw(13) << r.ransac_rot_err << setw(15) << r.ransac_trans_err
         << setw(9) << r.ransac_inliers << endl;
  }

  int plain_bad = 0, ransac_bad = 0;
  for (const auto &r : results) {
    if (!r.ok) continue;
    if (r.plain_rot_err > 5.0 || r.plain_trans_err > 0.05) plain_bad++;
    if (r.ransac_rot_err > 5.0 || r.ransac_trans_err > 0.05) ransac_bad++;
  }
  cout << "\n=== SUMMARY ===" << endl;
  cout << "pairs tested: " << results.size() << endl;
  cout << "plain solvePnP pairs with rot_err>5deg or trans_err>5cm: " << plain_bad
       << " / " << results.size() << endl;
  cout << "RANSAC diagnostic pairs with rot_err>5deg or trans_err>5cm: " << ransac_bad
       << " / " << results.size() << endl;

  return 0;
}
