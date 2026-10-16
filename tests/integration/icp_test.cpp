// Adapted from pose_estimation_3d3d.cpp's main(): find matches, back-project
// both frames' keypoints to 3D via their depth maps, solve ICP (closed-form
// SVD), then refine with the g2o bundle adjustment.

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_UNCHANGED

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "tracking/icp.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "usage: icp_test img1 img2 depth1 depth2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "matches: " << matches.size() << endl;

  // Book code starts
  //-- 建立3D点
  Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts1, pts2;

  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;
  Mat R, t;
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "ICP via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;

  // Not book code: bundleAdjustment() below overwrites R,t in place with the
  // g2o-refined result, so the SVD solution must be saved first to allow a
  // before/after comparison.
  Mat R_svd = R.clone();
  Mat t_svd = t.clone();

  cout << "calling bundle adjustment" << endl;

  bundleAdjustment(pts1, pts2, R, t);

  // verify p1 = R * p2 + t
  for (int i = 0; i < 5; i++) {
    cout << "p1 = " << pts1[i] << endl;
    cout << "p2 = " << pts2[i] << endl;
    cout << "(R*p2+t) = " <<
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
         << endl;
    cout << endl;
  }
  // Book code ends

  // Not book code: quantitative reporting for the presentation.
  auto alignment_errors = [&](const Mat &Rm, const Mat &tm) {
    vector<double> errs;
    for (size_t i = 0; i < pts1.size(); ++i) {
      Mat p2 = (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z);
      Mat p1_est = Rm * p2 + tm;
      Mat p1 = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, pts1[i].z);
      errs.push_back(cv::norm(p1_est - p1));
    }
    return ComputeStats(errs);
  };
  Stats err_before = alignment_errors(R_svd, t_svd);
  Stats err_after = alignment_errors(R, t);

  double rot_diff = RotationAngleDeg(R_svd, R);
  double t_diff = TranslationDiff(t_svd, t);

  cout << "\n=== ICP SUMMARY ===" << endl;
  cout << "3D-3D correspondences used: " << pts1.size() << endl;
  cout << "rotation diff SVD-vs-g2o (deg): " << rot_diff << endl;
  cout << "translation diff SVD-vs-g2o (norm): " << t_diff << endl;
  cout << "point-to-point error before optimization (SVD) mean/median/rms/max: "
       << err_before.mean << " / " << err_before.median << " / " << err_before.rms << " / "
       << err_before.max << endl;
  cout << "point-to-point error after optimization (g2o) mean/median/rms/max: "
       << err_after.mean << " / " << err_after.median << " / " << err_after.rms << " / "
       << err_after.max << endl;
  cout << "(the check p1 ~= R*p2+t above uses the final R,t -- this is exactly the "
       << "'after optimization' statistic over the full correspondence set, not just "
       << "the 5 printed samples)" << endl;

  return 0;
}
