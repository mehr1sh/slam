#include "tracking/brief.hpp"

#include <random>

using namespace std;
using namespace cv;

namespace {

struct PointPair { int dx1, dy1, dx2, dy2; };

const vector<PointPair> &BriefPattern() {
  static const vector<PointPair> pattern = [] {
    vector<PointPair> p;
    p.reserve(256);
    mt19937 rng(12345);
    uniform_int_distribution<int> offset(-kBriefPatchHalfSize, kBriefPatchHalfSize);
    for (int i = 0; i < 256; ++i) {
      p.push_back({offset(rng), offset(rng), offset(rng), offset(rng)});
    }
    return p;
  }();
  return pattern;
}

}  // namespace

vector<BriefDescriptor> ComputeBrief(const Mat &gray, const vector<KeyPoint> &keypoints) {
  const auto &pattern = BriefPattern();
  vector<BriefDescriptor> descriptors;
  descriptors.reserve(keypoints.size());

  for (const auto &kp : keypoints) {
    int x = (int)kp.pt.x, y = (int)kp.pt.y;
    if (x < kBriefPatchHalfSize || y < kBriefPatchHalfSize ||
        x >= gray.cols - kBriefPatchHalfSize || y >= gray.rows - kBriefPatchHalfSize) {
      descriptors.push_back({});
      continue;
    }

    BriefDescriptor desc(8, 0);
    for (int i = 0; i < 256; ++i) {
      const PointPair &pp = pattern[i];
      uchar a = gray.at<uchar>(y + pp.dy1, x + pp.dx1);
      uchar b = gray.at<uchar>(y + pp.dy2, x + pp.dx2);
      if (a < b) desc[i / 32] |= (1u << (i % 32));
    }
    descriptors.push_back(desc);
  }
  return descriptors;
}
