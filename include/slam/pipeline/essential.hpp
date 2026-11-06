#pragma once

// Essential matrix from scratch (Eigen only): normalized 8-point estimate,
// rank-2 / equal-singular-value enforcement, decomposition into the four
// (R, t) candidates, and the cheirality test that picks the physical one.
//
// Coordinate convention (the same as the scratch PnP and the trajectory code):
//   x1, x2 are NORMALIZED image coordinates, x = K^-1 [u v 1]^T, of the same
//   3D point in camera 1 and camera 2 (for a frame pair i -> i+1: camera 1 =
//   frame i, camera 2 = frame i+1). The recovered pose satisfies
//       X_2 = R * X_1 + t        (i.e. T_{i+1<-i})
//   with the epipolar constraint x2^T E x1 = 0 and E = [t]_x R.
//   A two-view essential matrix fixes t only up to scale: |t| = 1 is returned.

#include <array>
#include <vector>

#include <Eigen/Core>

#include "projection.hpp"

namespace scratch {

using Points2 = std::vector<Eigen::Vector2d>;

struct PoseRt {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// [t]_x, the matrix with [t]_x v = t x v.
Eigen::Matrix3d Skew(const Eigen::Vector3d &t);

// Normalized 8-point algorithm (needs >= 8 correspondences, returns false
// otherwise). Steps: Hartley-normalize both point sets -> one row per
// correspondence of A e = 0 -> e = right singular vector of the smallest
// singular value -> undo the normalization -> enforce singular values
// (s, s, 0). The result has unit Frobenius norm.
bool EstimateEssential8Point(const Points2 &x1, const Points2 &x2, Eigen::Matrix3d &E);

// Projects any 3x3 matrix onto the essential manifold: SVD M = U S V^T,
// E = U diag(s, s, 0) V^T with s = (s1 + s2) / 2.
Eigen::Matrix3d EnforceEssentialConstraints(const Eigen::Matrix3d &M);

// The four (R, t) pairs consistent with E (det R = +1, |t| = 1):
//   (U W V^T, +u3), (U W V^T, -u3), (U W^T V^T, +u3), (U W^T V^T, -u3).
std::array<PoseRt, 4> DecomposeEssential(const Eigen::Matrix3d &E);

// Linear two-view triangulation of one correspondence for a candidate pose:
// solves  d2 * x2 - d1 * R x1 = t  for the depths (d1, d2) in least squares.
// Returns false if the rays are (nearly) parallel.
bool TriangulateDepths(const PoseRt &P, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2, double &d1, double &d2);

// Number of correspondences with positive depth in BOTH cameras.
int CountInFront(const PoseRt &P, const Points2 &x1, const Points2 &x2);

struct RecoveredPose {
  PoseRt pose;
  int in_front = 0;                        // of the chosen candidate
  std::array<int, 4> candidate_in_front{};  // for all four candidates
  int chosen = -1;
};

// Decomposes E and keeps the candidate with the most points in front of
// both cameras (cheirality). Returns false if no candidate has any.
bool RecoverPoseFromEssential(const Eigen::Matrix3d &E, const Points2 &x1, const Points2 &x2, RecoveredPose &out);

// First-order geometric (Sampson) distance of one correspondence to E, in
// normalized-coordinate units squared (multiply by f^2 for pixels^2).
double SampsonDistance(const Eigen::Matrix3d &E, const Eigen::Vector2d &x1, const Eigen::Vector2d &x2);

// --- epipolar geometry in PIXEL coordinates (visualization and checks) ---
// The constraint above holds for normalized coordinates. Keypoints are in
// pixels, p = K x, so for pixels it reads p2^T F p1 = 0 with
//   F = K^-T E K^-1.
Eigen::Matrix3d FundamentalFromEssential(const Eigen::Matrix3d &E, const Intrinsics &K);

// Epipolar line in image 2 of pixel p1 of image 1: l = F [p1; 1], scaled so
// that a^2 + b^2 = 1; then l . [p2; 1] is the signed distance of pixel p2 to
// the line, in pixels. (The line in image 1 of a pixel of image 2 is F^T p2.)
Eigen::Vector3d EpipolarLine(const Eigen::Matrix3d &F, const Eigen::Vector2d &p1);

// Distance in pixels of p2 to the epipolar line of p1 (one-sided, in image 2).
double EpipolarDistance(const Eigen::Matrix3d &F, const Eigen::Vector2d &p1, const Eigen::Vector2d &p2);

}  // namespace scratch
