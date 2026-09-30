// From-scratch normalized 8-point Essential Matrix + pose recovery test.
// Synthetic 3D points, known relative (R,t), projected into two cameras,
// then E is estimated and (R,t) recovered. Translation is compared only by
// DIRECTION (unit vector) -- the Essential matrix fixes translation only
// up to an unknown scale.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include "camera/camera.hpp"
#include "tracking/essential_matrix.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

int main() {
  PinholeCamera cam{520.9, 521.0, 325.1, 249.7};

  // Known ground-truth relative pose, this repo's convention: x2 = R_gt*x1 + t_gt.
  Eigen::Matrix3d R_gt =
      Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.1, 1.0, 0.2).normalized()).toRotationMatrix();
  Eigen::Vector3d t_gt(0.3, -0.05, 0.1);

  mt19937 rng(42);
  uniform_real_distribution<double> xy(-1.0, 1.0), depth(3.0, 8.0);
  vector<Point2d> pts1_cam, pts2_cam;
  for (int i = 0; i < 60; ++i) {
    Eigen::Vector3d p1(xy(rng), xy(rng), depth(rng));
    Eigen::Vector3d p2 = R_gt * p1 + t_gt;
    if (p2.z() <= 0) continue;
    pts1_cam.push_back(Point2d(p1.x() / p1.z(), p1.y() / p1.z()));
    pts2_cam.push_back(Point2d(p2.x() / p2.z(), p2.y() / p2.z()));
  }

  cout << "=== ESSENTIAL MATRIX TEST ===\n" << endl;
  cout << "synthetic correspondences: " << pts1_cam.size() << endl;

  Mat E = EstimateEssentialMatrix(pts1_cam, pts2_cam);
  cout << "\nE =\n" << E << endl;

  Mat R, t;
  if (!RecoverPoseFromEssential(E, pts1_cam, pts2_cam, R, t)) {
    cout << "\nFATAL: no cheirality-consistent pose found." << endl;
    return 1;
  }

  Mat R_gt_cv = EigenRotToCv(R_gt);
  double rot_err = RotationAngleDeg(R_gt_cv, R);
  cout << "\nGround truth R:\n" << R_gt_cv << endl;
  cout << "Recovered R:\n" << R << endl;
  cout << "Rotation error: " << rot_err << " deg" << endl;

  Eigen::Vector3d t_gt_dir = t_gt.normalized();
  Eigen::Vector3d t_est_dir(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  t_est_dir.normalize();
  double cos_angle = max(-1.0, min(1.0, t_gt_dir.dot(t_est_dir)));
  double t_dir_err_deg = acos(cos_angle) * 180.0 / CV_PI;

  cout << "\nGround truth t direction: " << t_gt_dir.transpose() << endl;
  cout << "Recovered t direction:    " << t_est_dir.transpose() << endl;
  cout << "Translation direction error: " << t_dir_err_deg << " deg "
       << "(magnitude NOT compared -- E fixes t only up to scale)" << endl;

  cout << "\nRotation error < 1 deg: " << (rot_err < 1.0 ? "PASS" : "FAIL") << endl;
  cout << "Translation direction error < 1 deg: " << (t_dir_err_deg < 1.0 ? "PASS" : "FAIL") << endl;

  return 0;
}
