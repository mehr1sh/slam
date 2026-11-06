#include "camera/camera.hpp"

// Book code starts
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}
// Book code ends

cv::Point2d cam2pixel(const cv::Point3d &P, const cv::Mat &K) {
  return cv::Point2d(
    K.at<double>(0, 0) * P.x / P.z + K.at<double>(0, 2),
    K.at<double>(1, 1) * P.y / P.z + K.at<double>(1, 2)
  );
}
