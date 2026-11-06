#include "tracking/icp.hpp"
#include "tracking/rgbd_icp.hpp"
#include <stdexcept>

// Compatibility API: fixed correspondences, pts1 ~= R*pts2+t.
// Dense RGB-D ICP with changing correspondences is registerFrames() in
// rgbd_icp.
void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1,
                          const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                          cv::Mat &t) {
  std::vector<rgbd_icp::V3> source, target;
  for (auto p : pts1)
    target.emplace_back(p.x, p.y, p.z);
  for (auto p : pts2)
    source.emplace_back(p.x, p.y, p.z);
  auto pose = rgbd_icp::rigidSVD(source, target);
  R = cv::Mat(3, 3, CV_64F);
  t = cv::Mat(3, 1, CV_64F);
  for (int i = 0; i < 3; ++i) {
    t.at<double>(i) = pose.t[i];
    for (int j = 0; j < 3; ++j)
      R.at<double>(i, j) = pose.R(i, j);
  }
}

// Historical name retained for existing callers. The fixed, unweighted 3D-3D
// objective already has an exact SVD solution; rerunning g2o from identity is
// redundant. Real point-to-plane refinement is provided by planeICP().
void bundleAdjustment(const std::vector<cv::Point3f> &pts1,
                      const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                      cv::Mat &t) {
  pose_estimation_3d3d(pts1, pts2, R, t);
}
