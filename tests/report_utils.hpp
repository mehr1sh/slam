#pragma once

// Shared reporting helpers for the test executables. NOT book code --
// basic descriptive statistics, pose-comparison helpers, and a minimal
// ASCII PLY writer, reused across tests to avoid duplicating this
// bookkeeping in every file. Contains no SLAM/book algorithm logic.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

struct Stats {
  size_t n = 0;
  double min = 0, max = 0, mean = 0, median = 0, rms = 0;
};

inline Stats ComputeStats(std::vector<double> v) {
  Stats s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  double sum = 0, sq_sum = 0;
  for (double x : v) {
    sum += x;
    sq_sum += x * x;
  }
  s.mean = sum / v.size();
  s.rms = std::sqrt(sq_sum / v.size());
  size_t mid = v.size() / 2;
  s.median = (v.size() % 2 == 0) ? 0.5 * (v[mid - 1] + v[mid]) : v[mid];
  return s;
}

inline void PrintStats(const std::string &label, const Stats &s) {
  std::cout << label << ": n=" << s.n << " min=" << s.min << " max=" << s.max
             << " mean=" << s.mean << " median=" << s.median << " rms=" << s.rms
             << std::endl;
}

// Angle (degrees) between two rotation matrices.
inline double RotationAngleDeg(const cv::Mat &R1, const cv::Mat &R2) {
  cv::Mat R = R1.t() * R2;
  double tr = cv::trace(R)[0];
  double c = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
  return std::acos(c) * 180.0 / CV_PI;
}

inline double TranslationDiff(const cv::Mat &t1, const cv::Mat &t2) {
  return cv::norm(t1 - t2);
}

inline cv::Mat EigenRotToCv(const Eigen::Matrix3d &R) {
  cv::Mat m(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m.at<double>(i, j) = R(i, j);
  return m;
}

inline cv::Mat EigenVecToCv(const Eigen::Vector3d &v) {
  return (cv::Mat_<double>(3, 1) << v(0), v(1), v(2));
}

struct Rgb {
  unsigned char r, g, b;
};

// Minimal ASCII PLY writer (points + optional per-point color).
inline void WritePLY(const std::string &path, const std::vector<cv::Point3d> &points,
                      const std::vector<Rgb> &colors = {}) {
  bool has_color = colors.size() == points.size();
  std::ofstream f(path);
  f << "ply\nformat ascii 1.0\n";
  f << "element vertex " << points.size() << "\n";
  f << "property float x\nproperty float y\nproperty float z\n";
  if (has_color) f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  f << "end_header\n";
  for (size_t i = 0; i < points.size(); ++i) {
    f << points[i].x << " " << points[i].y << " " << points[i].z;
    if (has_color) f << " " << (int)colors[i].r << " " << (int)colors[i].g << " " << (int)colors[i].b;
    f << "\n";
  }
}
