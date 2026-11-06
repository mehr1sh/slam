// Tests the EXISTING, unmodified PnP implementation (src/tracking/pnp.cpp,
// via the identical find_feature_matches()/solvePnP() call pattern
// tests/pnp_test.cpp already uses) on one frame pair from the synthetic
// bunny dataset, comparing its recovered relative pose against the known
// ground truth in data/synthetic_bunny/groundtruth.txt.
//
// Convention (verified from tests/pnp_test.cpp, not assumed): 3D points are
// backprojected from frame i's depth map; 2D targets are frame j's pixel
// keypoints. The recovered (R,t) therefore maps frame-i camera coordinates
// into frame-j camera coordinates: x_j = R*x_i + t.

#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_UNCHANGED
#include <sophus/se3.hpp>

#include "camera/camera.hpp"
#include "features/features.hpp"
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
  if (img_i.empty() || img_j.empty() || depth_i.empty()) {
    cerr << "failed to load frame " << frame_i << "/" << frame_j << " from " << dataset_dir << endl;
    return 1;
  }

  // Book-code-identical feature matching (src/features/features.cpp).
  vector<KeyPoint> keypoints_i, keypoints_j;
  vector<DMatch> matches, all_matches;
  find_feature_matches(img_i, img_j, keypoints_i, keypoints_j, matches, &all_matches);

  // Book-code-identical 3D-2D correspondence construction (see
  // tests/pnp_test.cpp): 3D points backprojected from frame i's depth via
  // pixel2cam(); 2D targets are frame j's matched pixel keypoints.
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (const DMatch &m : matches) {
    ushort d = depth_i.ptr<unsigned short>((int)keypoints_i[m.queryIdx].pt.y)
                      [(int)keypoints_i[m.queryIdx].pt.x];
    if (d == 0) continue;  // bad depth
    float dd = d / 5000.0f;
    Point2d p = pixel2cam(keypoints_i[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p.x * dd, p.y * dd, dd));
    pts_2d.push_back(keypoints_j[m.trainIdx].pt);
  }

  cout << "=== PNP SYNTHETIC TEST ===\n" << endl;
  cout << "Frame " << frame_i << " -> Frame " << frame_j << "\n" << endl;

  vector<Sophus::SE3d> T_wc = ReadGroundtruth(dataset_dir + "/groundtruth.txt");
  // Same derivation as tests/synthetic_pose_eval_test.cpp's T_ji: maps
  // frame-i camera coords -> frame-j camera coords, matching PnP's own
  // x_j = R*x_i + t convention exactly.
  Sophus::SE3d T_gt = T_wc[frame_j].inverse() * T_wc[frame_i];
  Mat R_gt = EigenRotToCv(T_gt.rotationMatrix());
  Mat t_gt = EigenVecToCv(T_gt.translation());

  cout << "Ground Truth:" << endl;
  cout << "R =\n" << R_gt << endl;
  cout << "t = " << t_gt.t() << endl;

  if (pts_3d.size() < 4) {
    cout << "\nFATAL: only " << pts_3d.size()
         << " valid 3D-2D correspondences -- solvePnP needs at least 4. Not running PnP." << endl;
    cout << "\nNumber of detected features (frame " << frame_i << "): " << keypoints_i.size() << endl;
    cout << "Number of detected features (frame " << frame_j << "): " << keypoints_j.size() << endl;
    cout << "Number of feature matches: " << matches.size() << endl;
    cout << "Number of valid 3D-2D correspondences: " << pts_3d.size() << endl;
    return 1;
  }

  // Existing book code, unmodified: identical solvePnP call to tests/pnp_test.cpp.
  Mat r, t_cv, R_cv;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t_cv, false);
  Rodrigues(r, R_cv);

  cout << "\nPnP:" << endl;
  cout << "R =\n" << R_cv << endl;
  cout << "t = " << t_cv.t() << endl;

  double rot_err = RotationAngleDeg(R_gt, R_cv);
  double t_err = TranslationDiff(t_gt, t_cv);
  cout << "\nRotation error: " << rot_err << " degrees" << endl;
  cout << "Translation error: " << t_err << " metres" << endl;

  // Diagnostic-only, does NOT change the R,t reported above: plain
  // solvePnP (the book code used above) has no built-in outlier rejection,
  // so solvePnPRansac is run separately purely to report an inlier count.
  Mat r_ransac, t_ransac;
  vector<int> inliers;
  bool ransac_ok = solvePnPRansac(pts_3d, pts_2d, K, Mat(), r_ransac, t_ransac,
                                  false, 100, 8.0f, 0.99, inliers);
  Mat R_ransac;
  if (ransac_ok) {
    Rodrigues(r_ransac, R_ransac);
    cout << "\nPnP (RANSAC, diagnostic only):" << endl;
    cout << "R =\n" << R_ransac << endl;
    cout << "t = " << t_ransac.t() << endl;
    cout << "Rotation error vs GT: " << RotationAngleDeg(R_gt, R_ransac) << " degrees" << endl;
    cout << "Translation error vs GT: " << TranslationDiff(t_gt, t_ransac) << " metres" << endl;
  }

  cout << "\nNumber of detected features (frame " << frame_i << "): " << keypoints_i.size() << endl;
  cout << "Number of detected features (frame " << frame_j << "): " << keypoints_j.size() << endl;
  cout << "Number of feature matches: " << matches.size() << endl;
  cout << "Number of valid 3D-2D correspondences: " << pts_3d.size() << endl;
  cout << "Number of PnP inliers (solvePnPRansac, diagnostic only, 8.0px threshold): "
       << (ransac_ok ? (int)inliers.size() : -1) << " / " << pts_3d.size() << endl;

  return 0;
}
