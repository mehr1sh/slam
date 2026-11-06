#include "tracking/fast.hpp"

using namespace std;
using namespace cv;

namespace {

constexpr int kCircleSize = 16; //look at 16 pixels
constexpr int kBorder = 3;  // radius of the Bresenham circle

const int kCircleDx[kCircleSize] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
const int kCircleDy[kCircleSize] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

bool IsCorner(const Mat &gray, int x, int y, const FastParams &params) {
  int center = gray.at<uchar>(y, x);
  int signs[kCircleSize];
  for (int i = 0; i < kCircleSize; ++i) {
    int v = gray.at<uchar>(y + kCircleDy[i], x + kCircleDx[i]);
    if (v > center + params.threshold) signs[i] = 1;
    else if (v < center - params.threshold) signs[i] = -1;
    else signs[i] = 0;
  }

  for (int start = 0; start < kCircleSize; ++start) {
    if (signs[start] == 0) continue;
    int run = 1;
    for (int k = 1; k < params.contiguous; ++k) {
      if (signs[(start + k) % kCircleSize] == signs[start]) run++;
      else break;
    }
    if (run >= params.contiguous) return true;
  }
  return false;
}

}  // namespace

vector<KeyPoint> DetectFast(const Mat &gray, const FastParams &params) {
  vector<KeyPoint> keypoints;
  for (int y = kBorder; y < gray.rows - kBorder; ++y) {
    for (int x = kBorder; x < gray.cols - kBorder; ++x) {
      if (IsCorner(gray, x, y, params)) {
        keypoints.emplace_back(Point2f((float)x, (float)y), 7.0f);
      }
    }
  }
  return keypoints;
}
