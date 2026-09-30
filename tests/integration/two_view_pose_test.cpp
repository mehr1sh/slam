// Adapted from pose_estimation_2d2d.cpp's main(): find matches, estimate
// 2D-2D pose, verify E = t^R and the epipolar constraint per match.

#include <filesystem>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "geometry/geometry.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

// Not book code: "reasonable residual" pass/fail threshold for the
// normalized-camera-coordinate epipolar constraint y2^T [t]_x R y1 ~= 0.
static constexpr double kEpipolarResidualThreshold = 1e-2;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: two_view_pose_test img1 img2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "matches: " << matches.size() << endl;

  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  // Book code starts
  //-- 验证E=t^R*scale
  Mat t_x =
    (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);
  cout << "t^R=" << endl << t_x * R << endl;

  //-- 验证对极约束
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<double> residuals;
  for (DMatch m: matches) {
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
    residuals.push_back(std::abs(d.at<double>(0, 0)));  // Not book code
  }
  // Book code ends

  // Not book code: quantitative reporting for the presentation.
  Stats resid_stats = ComputeStats(residuals);
  double t_norm = cv::norm(t);
  double angle_deg = RotationAngleDeg(Mat::eye(3, 3, R.type()), R);
  size_t n_pass = 0;
  for (double r : residuals) if (r <= kEpipolarResidualThreshold) ++n_pass;
  double pass_pct = residuals.empty() ? 0.0 : 100.0 * n_pass / residuals.size();

  // Simple epipolar-line visualization: draw the epipolar line in image 2
  // corresponding to each point in image 1 (and vice versa), using F derived
  // from K and (R,t): F = K^-T [t]_x R K^-1.
  Mat K_inv = K.inv();
  Mat F = K_inv.t() * t_x * R * K_inv;
  vector<Point2f> pts1, pts2;
  for (DMatch m : matches) {
    pts1.push_back(keypoints_1[m.queryIdx].pt);
    pts2.push_back(keypoints_2[m.trainIdx].pt);
  }
  vector<Vec3f> lines_in_2, lines_in_1;
  cv::computeCorrespondEpilines(pts1, 1, F, lines_in_2);
  cv::computeCorrespondEpilines(pts2, 2, F, lines_in_1);
  Mat img1_epi = img_1.clone();
  Mat img2_epi = img_2.clone();
  RNG rng(12345);
  for (size_t i = 0; i < pts1.size(); ++i) {
    Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    auto draw_line = [](Mat &img, const Vec3f &l, const Scalar &color) {
      float a = l[0], b = l[1], c = l[2];
      Point2f p0(0, -c / b), p1(img.cols - 1.f, -(c + a * (img.cols - 1)) / b);
      cv::line(img, p0, p1, color, 1);
    };
    draw_line(img2_epi, lines_in_2[i], color);
    cv::circle(img2_epi, pts2[i], 3, color, -1);
    draw_line(img1_epi, lines_in_1[i], color);
    cv::circle(img1_epi, pts1[i], 3, color, -1);
  }
  std::filesystem::path output_dir = "../output/two_view_pose";
  std::filesystem::create_directories(output_dir);
  imwrite((output_dir / "epipolar_lines_img1.png").string(), img1_epi);
  imwrite((output_dir / "epipolar_lines_img2.png").string(), img2_epi);
  cout << "wrote " << (output_dir / "epipolar_lines_img1.png").string() << endl;
  cout << "wrote " << (output_dir / "epipolar_lines_img2.png").string() << endl;

  cout << "\n=== TWO-VIEW GEOMETRY SUMMARY ===" << endl;
  cout << "correspondences used: " << matches.size() << endl;
  cout << "translation norm (unit scale): " << t_norm << endl;
  cout << "rotation angle (deg): " << angle_deg << endl;
  cout << "epipolar residual |d| mean/median/rms/max: " << resid_stats.mean << " / "
       << resid_stats.median << " / " << resid_stats.rms << " / " << resid_stats.max << endl;
  cout << "residual <= " << kEpipolarResidualThreshold << ": " << n_pass << "/"
       << residuals.size() << " (" << pass_pct << "%)" << endl;

  return 0;
}
