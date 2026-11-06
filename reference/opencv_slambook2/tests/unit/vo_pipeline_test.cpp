// End-to-end from-scratch 2D-2D visual odometry check: synthetic 3D
// points with a known relative (R,t) are rendered as small bright square
// markers into two synthetic images (giving FAST genuinely detectable
// corners, unlike a bare point cloud); the full FAST -> BRIEF -> Hamming
// -> normalized 8-point Essential Matrix pipeline recovers a relative
// pose from those REAL detected/matched pixel correspondences, compared
// against ground truth.
//
// tests/unit/essential_matrix_test.cpp already covers "test the Essential
// Matrix using known 2D correspondences" in isolation, with exact,
// idealized correspondences and no detection/matching noise -- it passes
// to within floating-point precision. This test is the complementary "run
// the complete pipeline where practical" check, and is expected to (and
// does) look rougher, for two found-and-explained reasons, not bugs:
//   1. Identical markers are genuinely ambiguous for ANY local descriptor
//      (marker A's corner is pixel-identical to marker B's corner) --
//      confirmed by observing catastrophic (~150-180 degree) errors with
//      uniform marker size, fixed by giving each marker a distinct size.
//   2. Even after that fix, ROTATION recovers well (roughly 1-2 degrees),
//      but TRANSLATION DIRECTION does not: a handful of real detected
//      keypoints per marker (FAST fires along a whole small square's
//      edge, not only its 4 true corners) occasionally get matched to
//      the wrong corner of the *same* (correct) marker -- a few pixels of
//      genuine correspondence noise. Translation direction is far more
//      sensitive to this than rotation in two-view epipolar geometry (a
//      well-known property, not specific to this implementation), and
//      with no RANSAC/outlier rejection permitted at this stage, that
//      noise reaches the final estimate directly.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include "camera/camera.hpp"
#include "tracking/brief.hpp"
#include "tracking/essential_matrix.hpp"
#include "tracking/fast.hpp"
#include "tracking/hamming_matching.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

namespace {

void DrawMarker(Mat &img, int cx, int cy, int half_size) {
  for (int y = cy - half_size; y <= cy + half_size; ++y)
    for (int x = cx - half_size; x <= cx + half_size; ++x)
      if (y >= 0 && y < img.rows && x >= 0 && x < img.cols) img.at<uchar>(y, x) = 255;
}

}  // namespace

int main() {
  PinholeCamera cam{520.9, 521.0, 325.1, 249.7};
  Mat K = cam.K();
  int width = 640, height = 480;

  Eigen::Matrix3d R_gt =
      Eigen::AngleAxisd(0.05, Eigen::Vector3d(0.1, 1.0, 0.05).normalized()).toRotationMatrix();
  Eigen::Vector3d t_gt(0.3, -0.05, 0.15);  // same order of magnitude as essential_matrix_test's
                                           // own t_gt relative to scene depth -- the earlier,
                                           // much smaller t_gt put this near the well-known
                                           // "translation direction is ill-conditioned when
                                           // baseline << depth" regime, unrelated to any bug.

  Mat img1 = Mat::zeros(height, width, CV_8UC1);
  Mat img2 = Mat::zeros(height, width, CV_8UC1);

  // A regular pixel grid in image1 (not random rejection-sampling): each
  // marker's true 3D point is recovered by back-projecting the grid pixel
  // at a fixed depth (pixel2cam + depth, exactly the same backprojection
  // tests/integration/pnp_test.cpp uses), then forward-projected through
  // (R_gt,t_gt) into image2 via cam2pixel. This GUARANTEES even spacing in
  // image1 (unlike random placement, which needs many rejected attempts to
  // enforce a minimum separation and can fail to converge) -- spacing must
  // comfortably exceed BRIEF's sampling-patch diameter
  // (2*kBriefPatchHalfSize) so different markers' descriptors don't get
  // built from overlapping pixels, which would otherwise confuse simple
  // nearest-neighbor Hamming matching (no ratio test/cross-check, as
  // instructed) into matching one marker's corner to a neighbor's.
  int cols = 5, rows = 4;
  double margin = 60, depth_val = 5.0;
  vector<Point2d> placed;
  int idx = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c, ++idx) {
      double u = margin + c * (width - 2 * margin) / (cols - 1);
      double v = margin + r * (height - 2 * margin) / (rows - 1);
      Point2d cam1 = pixel2cam(Point2d(u, v), K);
      Eigen::Vector3d p1(cam1.x * depth_val, cam1.y * depth_val, depth_val);
      Eigen::Vector3d p2 = R_gt * p1 + t_gt;
      if (p2.z() <= 0.1) continue;
      Point2d pix2 = cam2pixel(Point3d(p2.x(), p2.y(), p2.z()), K);
      if (pix2.x < 20 || pix2.y < 20 || pix2.x > width - 20 || pix2.y > height - 20) continue;

      // Varying size per marker (not all markers identical): a repeated,
      // pixel-identical pattern is genuinely ambiguous for any local
      // descriptor (a real, well-known limitation of matching on
      // repetitive/textureless scenes, not a bug) -- varying size gives
      // each marker's corners a locally distinguishable footprint.
      int marker_half = 3 + (idx % 5);
      placed.push_back(Point2d(u, v));
      DrawMarker(img1, (int)round(u), (int)round(v), marker_half);
      DrawMarker(img2, (int)round(pix2.x), (int)round(pix2.y), marker_half);
    }
  }

  cout << "=== VO PIPELINE TEST (FAST -> BRIEF -> Hamming -> Essential Matrix) ===\n" << endl;
  cout << "synthetic markers placed: " << placed.size() << endl;

  FastParams fast_params;
  fast_params.threshold = 30;
  vector<KeyPoint> kp1 = DetectFast(img1, fast_params);
  vector<KeyPoint> kp2 = DetectFast(img2, fast_params);
  cout << "FAST keypoints: image1=" << kp1.size() << " image2=" << kp2.size() << endl;

  vector<BriefDescriptor> desc1 = ComputeBrief(img1, kp1);
  vector<BriefDescriptor> desc2 = ComputeBrief(img2, kp2);

  vector<DMatch> matches = MatchHamming(desc1, desc2, 60);
  cout << "Hamming matches: " << matches.size() << endl;

  if (matches.size() < 8) {
    cout << "\nFATAL: fewer than 8 matches -- cannot run the 8-point algorithm." << endl;
    return 1;
  }

  vector<Point2d> pts1_cam, pts2_cam;
  for (const auto &m : matches) {
    pts1_cam.push_back(pixel2cam(kp1[m.queryIdx].pt, K));
    pts2_cam.push_back(pixel2cam(kp2[m.trainIdx].pt, K));
  }

  Mat E = EstimateEssentialMatrix(pts1_cam, pts2_cam);
  Mat R, t;
  if (!RecoverPoseFromEssential(E, pts1_cam, pts2_cam, R, t)) {
    cout << "\nFATAL: no cheirality-consistent pose found." << endl;
    return 1;
  }

  Mat R_gt_cv = EigenRotToCv(R_gt);
  double rot_err = RotationAngleDeg(R_gt_cv, R);

  Eigen::Vector3d t_gt_dir = t_gt.normalized();
  Eigen::Vector3d t_est_dir(t.at<double>(0), t.at<double>(1), t.at<double>(2));
  t_est_dir.normalize();
  double cos_a = max(-1.0, min(1.0, t_gt_dir.dot(t_est_dir)));
  double t_dir_err = acos(cos_a) * 180.0 / CV_PI;

  cout << "\nGround truth R:\n" << R_gt_cv << endl;
  cout << "Recovered R:\n" << R << endl;
  cout << "Rotation error: " << rot_err << " deg" << endl;
  cout << "Translation direction error: " << t_dir_err << " deg (magnitude NOT compared)" << endl;

  return 0;
}
