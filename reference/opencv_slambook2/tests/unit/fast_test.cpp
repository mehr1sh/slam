// From-scratch FAST corner detector, tested on a synthetic image with
// known corner locations: a filled white square on a black background has
// exactly 4 geometric corners, which FAST should fire on; flat interior
// and ba9ckground regions should not.

#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>

#include "tracking/fast.hpp"

using namespace std;
using namespace cv;

int main() {
  Mat img = Mat::zeros(100, 100, CV_8UC1);
  int x0 = 30, y0 = 30, size = 20;
  for (int y = y0; y < y0 + size; ++y)
    for (int x = x0; x < x0 + size; ++x)
      img.at<uchar>(y, x) = 255;

  vector<Point2i> known_corners = {
      {x0, y0}, {x0 + size - 1, y0}, {x0, y0 + size - 1}, {x0 + size - 1, y0 + size - 1}};

  FastParams params;
  params.threshold = 30;
  params.contiguous = 9;
  vector<KeyPoint> keypoints = DetectFast(img, params);

  cout << "=== FAST TEST ===\n" << endl;
  cout << "detected keypoints: " << keypoints.size() << endl;

  int found = 0;
  for (const auto &corner : known_corners) {
    bool near = false;
    for (const auto &kp : keypoints) {
      if (abs(kp.pt.x - corner.x) <= 2 && abs(kp.pt.y - corner.y) <= 2) {
        near = true;
        break;
      }
    }
    cout << "corner (" << corner.x << "," << corner.y << "): " << (near ? "DETECTED" : "MISSED") << endl;
    if (near) found++;
  }

  bool flat_flagged = false;
  for (const auto &kp : keypoints) {
    if (abs(kp.pt.x - 40) <= 1 && abs(kp.pt.y - 40) <= 1) flat_flagged = true;  // square interior, flat
  }
  cout << "flat interior point (40,40) incorrectly flagged: "
       << (flat_flagged ? "YES (FAIL)" : "no (PASS)") << endl;

  cout << "\ncorners detected: " << found << "/4" << endl;
  bool pass = (found == 4 && !flat_flagged);
  cout << (pass ? "PASS" : "FAIL") << endl;

  return pass ? 0 : 1;
}
