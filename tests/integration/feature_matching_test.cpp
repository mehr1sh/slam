// Adapted from orb_cv.cpp's main(): same detect/compute/match/filter/draw
// sequence, routed through the shared features::find_feature_matches().

#include <filesystem>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>  // CV_LOAD_IMAGE_COLOR

#include "features/features.hpp"
#include "../report_utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_matching_test img1 img2" << endl;
    return 1;
  }
  Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches, all_matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches, &all_matches);
  cout << "all matches: " << all_matches.size() << endl;
  cout << "matches: " << matches.size() << endl;

  // Not book code: quantitative reporting for the presentation.
  cout << "keypoints in image 1: " << keypoints_1.size() << endl;
  cout << "keypoints in image 2: " << keypoints_2.size() << endl;
  vector<double> good_dist;
  for (const auto &m : matches) good_dist.push_back(m.distance);
  Stats dist_stats = ComputeStats(good_dist);
  PrintStats("good match Hamming distance", dist_stats);
  double retained_pct = all_matches.empty() ? 0.0 : 100.0 * matches.size() / all_matches.size();

  // Book code starts
  // orb_cv.cpp's main() draws all three of these (keypoints, all matches,
  // good matches) via imshow(); saved to disk instead so the test runs
  // headless. Drawing calls/arguments otherwise unchanged.
  Mat outimg1;
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  Mat img_match;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, all_matches, img_match);

  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch);
  // Book code ends

  std::filesystem::path output_dir = "../output/feature_matching";
  std::filesystem::create_directories(output_dir);
  imwrite((output_dir / "keypoints.png").string(), outimg1);
  imwrite((output_dir / "all_matches.png").string(), img_match);
  imwrite((output_dir / "good_matches.png").string(), img_goodmatch);
  cout << "wrote " << (output_dir / "keypoints.png").string() << endl;
  cout << "wrote " << (output_dir / "all_matches.png").string() << endl;
  cout << "wrote " << (output_dir / "good_matches.png").string() << endl;

  // Not book code: presentation-friendly summary.
  cout << "\n=== FEATURE MATCHING SUMMARY ===" << endl;
  cout << "keypoints image 1: " << keypoints_1.size() << endl;
  cout << "keypoints image 2: " << keypoints_2.size() << endl;
  cout << "candidate matches: " << all_matches.size() << endl;
  cout << "good matches: " << matches.size() << endl;
  cout << "retained: " << retained_pct << "%" << endl;
  cout << "Hamming distance (good matches) min/max/mean/median: "
       << dist_stats.min << " / " << dist_stats.max << " / " << dist_stats.mean
       << " / " << dist_stats.median << endl;

  return 0;
}
