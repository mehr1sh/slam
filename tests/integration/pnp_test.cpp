// Adapted from pose_estimation_3d2d.cpp's main(): find matches, back-project
// image-1 keypoints to 3D via depth, solve PnP (OpenCV), then refine with
// the two from-scratch BA implementations (Gauss-Newton, g2o).

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_UNCHANGED
#include <chrono>

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "tracking/pnp.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 5) {
    cout << "usage: pnp_test img1 img2 depth1 depth2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "matches: " << matches.size() << endl;

  // Book code starts
  //-- 建立3D点
  Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (DMatch m:matches) {
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;

  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }

  cout << "calling bundle adjustment by gauss newton" << endl;
  Sophus::SE3d pose_gn;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

  cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
  // Book code ends

  // Not book code: quantitative reporting for the presentation.
  Mat R_gn = EigenRotToCv(pose_gn.rotationMatrix());
  Mat t_gn = EigenVecToCv(pose_gn.translation());
  Mat R_g2o = EigenRotToCv(pose_g2o.rotationMatrix());
  Mat t_g2o = EigenVecToCv(pose_g2o.translation());

  auto reprojection_stats = [&](const Mat &Rm, const Mat &tm) {
    vector<double> errs;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
      Mat X = (Mat_<double>(3, 1) << pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
      Mat Xc = Rm * X + tm;
      Mat uvw = K * Xc;
      double u = uvw.at<double>(0) / uvw.at<double>(2);
      double v = uvw.at<double>(1) / uvw.at<double>(2);
      double du = u - pts_2d[i].x, dv = v - pts_2d[i].y;
      errs.push_back(std::sqrt(du * du + dv * dv));
    }
    return ComputeStats(errs);
  };
  Stats err_cv = reprojection_stats(R, t);
  Stats err_gn = reprojection_stats(R_gn, t_gn);
  Stats err_g2o = reprojection_stats(R_g2o, t_g2o);

  double rot_diff_gn_cv = RotationAngleDeg(R, R_gn);
  double rot_diff_g2o_cv = RotationAngleDeg(R, R_g2o);
  double rot_diff_gn_g2o = RotationAngleDeg(R_gn, R_g2o);
  double t_diff_gn_cv = TranslationDiff(t, t_gn);
  double t_diff_g2o_cv = TranslationDiff(t, t_g2o);
  double t_diff_gn_g2o = TranslationDiff(t_gn, t_g2o);

  cout << "\n=== PNP SUMMARY ===" << endl;
  cout << "3D-2D correspondences used: " << pts_3d.size() << " (rejected for bad depth: "
       << matches.size() - pts_3d.size() << " of " << matches.size() << ")" << endl;
  cout << "rotation diff (deg) solvePnP-GN / solvePnP-g2o / GN-g2o: " << rot_diff_gn_cv << " / "
       << rot_diff_g2o_cv << " / " << rot_diff_gn_g2o << endl;
  cout << "translation diff (norm) solvePnP-GN / solvePnP-g2o / GN-g2o: " << t_diff_gn_cv
       << " / " << t_diff_g2o_cv << " / " << t_diff_gn_g2o << endl;
  cout << "reprojection error (px) solvePnP mean/median/rms/max: " << err_cv.mean << " / "
       << err_cv.median << " / " << err_cv.rms << " / " << err_cv.max << endl;
  cout << "reprojection error (px) Gauss-Newton mean/median/rms/max: " << err_gn.mean << " / "
       << err_gn.median << " / " << err_gn.rms << " / " << err_gn.max << endl;
  cout << "reprojection error (px) g2o mean/median/rms/max: " << err_g2o.mean << " / "
       << err_g2o.median << " / " << err_g2o.rms << " / " << err_g2o.max << endl;

  return 0;
}
