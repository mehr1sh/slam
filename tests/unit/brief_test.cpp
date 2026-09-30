// From-scratch BRIEF descriptor, tested on a synthetic deterministic
// image: verifies descriptors are reproducible across repeated runs,
// that different-content keypoints get meaningfully different
// descriptors, and that border keypoints are handled safely (empty
// descriptor rather than an out-of-bounds read).

#include <iostream>
#include <opencv2/core/core.hpp>

#include "tracking/brief.hpp"

using namespace std;
using namespace cv;

int main() {
  Mat img(100, 100, CV_8UC1);
  for (int y = 0; y < 100; ++y)
    for (int x = 0; x < 100; ++x)
      img.at<uchar>(y, x) = (uchar)((x * 7 + y * 13) % 256);

  vector<KeyPoint> keypoints = {KeyPoint(Point2f(50, 50), 7), KeyPoint(Point2f(20, 80), 7)};

  vector<BriefDescriptor> desc_a = ComputeBrief(img, keypoints);
  vector<BriefDescriptor> desc_b = ComputeBrief(img, keypoints);

  cout << "=== BRIEF TEST ===\n" << endl;

  bool deterministic = (desc_a == desc_b);
  cout << "determinism (identical descriptors across repeated runs): "
       << (deterministic ? "PASS" : "FAIL") << endl;

  cout << "descriptor size (uint32 words): " << desc_a[0].size() << " (expect 8, i.e. 256 bits)" << endl;

  int hamming = 0;
  for (size_t i = 0; i < desc_a[0].size(); ++i) hamming += __builtin_popcount(desc_a[0][i] ^ desc_a[1][i]);
  cout << "Hamming distance between two different keypoints: " << hamming << " / 256" << endl;
  bool discriminative = (hamming > 20 && hamming < 236);
  cout << "descriptors differ meaningfully: " << (discriminative ? "PASS" : "FAIL (degenerate)") << endl;

  vector<KeyPoint> edge_kp = {KeyPoint(Point2f(1, 1), 7)};
  vector<BriefDescriptor> edge_desc = ComputeBrief(img, edge_kp);
  bool boundary_ok = edge_desc[0].empty();
  cout << "edge keypoint descriptor empty (boundary handled safely): "
       << (boundary_ok ? "PASS" : "FAIL") << endl;

  bool pass = deterministic && discriminative && boundary_ok;
  cout << "\n" << (pass ? "PASS" : "FAIL") << endl;

  return pass ? 0 : 1;
}
