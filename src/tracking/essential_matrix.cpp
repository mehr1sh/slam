#include "tracking/essential_matrix.hpp"

#include "camera/camera.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>

using namespace std;
using namespace cv;

namespace {

Eigen::Vector3d ToHomogeneous(const Point2d &p) { return Eigen::Vector3d(p.x, p.y, 1.0); }

Mat EigenToCv(const Eigen::Matrix3d &M) {
  Mat m(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m.at<double>(i, j) = M(i, j);
  return m;
}

Eigen::Matrix3d CvToEigen(const Mat &M) {
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) m(i, j) = M.at<double>(i, j);
  return m;
}

}  // namespace

Mat EstimateEssentialMatrix(const vector<Point2d> &pts1_cam, const vector<Point2d> &pts2_cam) {
  int n = (int)pts1_cam.size();
  Eigen::MatrixXd A(n, 9);
  for (int i = 0; i < n; ++i) {
    double x1 = pts1_cam[i].x, y1 = pts1_cam[i].y;
    double x2 = pts2_cam[i].x, y2 = pts2_cam[i].y;
    A.row(i) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd_a(A, Eigen::ComputeFullV);
  Eigen::VectorXd e = svd_a.matrixV().col(8);
  Eigen::Matrix3d e_raw;
  e_raw << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

  Eigen::JacobiSVD<Eigen::Matrix3d> svd_e(e_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  double s = (svd_e.singularValues()(0) + svd_e.singularValues()(1)) / 2.0;
  Eigen::Matrix3d sigma = Eigen::Vector3d(s, s, 0.0).asDiagonal();
  Eigen::Matrix3d E = svd_e.matrixU() * sigma * svd_e.matrixV().transpose();

  return EigenToCv(E);
}

bool RecoverPoseFromEssential(const Mat &E_cv, const vector<Point2d> &pts1_cam,
                              const vector<Point2d> &pts2_cam, Mat &R, Mat &t) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(CvToEigen(E_cv), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  if (U.determinant() < 0) U.col(2) *= -1;
  if (V.determinant() < 0) V.col(2) *= -1;

  Eigen::Matrix3d W;
  W << 0, -1, 0,
       1, 0, 0,
       0, 0, 1;

  Eigen::Matrix3d r_a = U * W * V.transpose();
  Eigen::Matrix3d r_b = U * W.transpose() * V.transpose();
  Eigen::Vector3d t_pos = U.col(2);
  Eigen::Vector3d t_neg = -t_pos;

  Eigen::Matrix3d candidate_r[4] = {r_a, r_a, r_b, r_b};
  Eigen::Vector3d candidate_t[4] = {t_pos, t_neg, t_pos, t_neg};

  int best = -1, best_count = 0;
  for (int c = 0; c < 4; ++c) {
    const Eigen::Matrix3d &rc = candidate_r[c];
    const Eigen::Vector3d &tc = candidate_t[c];
    int count = 0;
    for (size_t i = 0; i < pts1_cam.size(); ++i) {
      Eigen::Vector3d x1 = ToHomogeneous(pts1_cam[i]);
      Eigen::Vector3d x2 = ToHomogeneous(pts2_cam[i]);

      // depth2*x2 - R*depth1*x1 = t, solved via least squares for (depth1, depth2).
      Eigen::Matrix<double, 3, 2> a;
      a.col(0) = -(rc * x1);
      a.col(1) = x2;
      Eigen::Vector2d depths = (a.transpose() * a).ldlt().solve(a.transpose() * tc);
      if (depths(0) > 0 && depths(1) > 0) count++;
    }
    if (count > best_count) {
      best_count = count;
      best = c;
    }
  }

  if (best < 0) return false;

  R = EigenToCv(candidate_r[best]);
  const Eigen::Vector3d &t_best = candidate_t[best];
  t = (Mat_<double>(3, 1) << t_best(0), t_best(1), t_best(2));
  return true;
}

bool EstimateRelativePoseEssentialMatrix(
    const vector<KeyPoint> &keypoints_1, const vector<KeyPoint> &keypoints_2,
    const vector<DMatch> &matches, const Mat &K, Mat &R, Mat &t) {
  if (matches.size() < 8) return false;

  vector<Point2d> pts1_cam, pts2_cam;
  for (const DMatch &m : matches) {
    pts1_cam.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
    pts2_cam.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
  }

  Mat E = EstimateEssentialMatrix(pts1_cam, pts2_cam);
  return RecoverPoseFromEssential(E, pts1_cam, pts2_cam, R, t);
}
