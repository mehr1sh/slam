// Adapted from triangulation.cpp's main(): find matches, estimate pose,
// triangulate, and depth-color-visualize the reprojection in both images.

#include <filesystem>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR

#include "camera/camera.hpp"
#include "features/features.hpp"
#include "geometry/geometry.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

// Book code starts
/// 作图用
inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}
// Book code ends

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: triangulation_test img1 img2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "matches: " << matches.size() << endl;

  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  // Book code starts
  //-- 验证三角化点与特征点的重投影关系
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  Mat img1_plot = img_1.clone();
  Mat img2_plot = img_2.clone();
  vector<double> depth1_all, depth2_all;  // Not book code
  for (int i = 0; i < matches.size(); i++) {
    float depth1 = points[i].z;
    cout << "depth: " << depth1 << endl;
    Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
    cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

    Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
    cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    depth1_all.push_back(depth1);  // Not book code
    depth2_all.push_back(depth2);  // Not book code
  }
  // Book code ends
  // (book displays img1_plot/img2_plot with imshow()/waitKey() here; saved
  // to disk instead so the test runs headless -- visualization unchanged.)
  std::filesystem::path output_dir = "../output/triangulation";
  std::filesystem::create_directories(output_dir);
  imwrite((output_dir / "img1_reprojection.png").string(), img1_plot);
  imwrite((output_dir / "img2_reprojection.png").string(), img2_plot);
  cout << "wrote " << (output_dir / "img1_reprojection.png").string() << endl;
  cout << "wrote " << (output_dir / "img2_reprojection.png").string() << endl;

  // Not book code: quantitative reporting for the presentation.
  vector<double> valid_depths;
  vector<Point3d> valid_points;
  for (size_t i = 0; i < depth1_all.size(); ++i) {
    if (depth1_all[i] > 0 && depth2_all[i] > 0) {
      valid_depths.push_back(depth1_all[i]);
      valid_points.push_back(points[i]);
    }
  }
  Stats depth_stats = ComputeStats(valid_depths);
  double cheirality_pct = points.empty() ? 0.0 : 100.0 * valid_points.size() / points.size();

  // Camera centers in the frame-1 coordinate system: camera 1 at origin,
  // camera 2 at -R^T t (since P2 = K[R|t] maps frame-1 points into camera 2).
  vector<Point3d> ply_points = valid_points;
  vector<Rgb> ply_colors(valid_points.size(), Rgb{200, 200, 200});
  Mat cam2_center = -R.t() * t;
  ply_points.push_back(Point3d(0, 0, 0));
  ply_colors.push_back(Rgb{255, 0, 0});
  ply_points.push_back(Point3d(cam2_center.at<double>(0), cam2_center.at<double>(1),
                                cam2_center.at<double>(2)));
  ply_colors.push_back(Rgb{0, 255, 0});
  WritePLY((output_dir / "points_and_cameras.ply").string(), ply_points, ply_colors);
  cout << "wrote " << (output_dir / "points_and_cameras.ply").string() << endl;

  cout << "\n=== TRIANGULATION SUMMARY ===" << endl;
  cout << "triangulated points: " << points.size() << endl;
  cout << "positive depth in both cameras: " << valid_points.size() << " (" << cheirality_pct
       << "%)" << endl;
  cout << "depth (valid points) min/max/mean/median: " << depth_stats.min << " / "
       << depth_stats.max << " / " << depth_stats.mean << " / " << depth_stats.median << endl;
  cout << "NOTE: monocular two-view reconstruction has arbitrary scale -- "
       << "t from the essential matrix is unit-norm, so depths/points above are in that "
       << "same unit-norm-t scale, not metres." << endl;

  return 0;
}
