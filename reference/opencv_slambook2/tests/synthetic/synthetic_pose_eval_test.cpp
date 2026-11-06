// Reads a synthetic dataset's groundtruth.txt (TUM format, T_world_camera
// per frame) and prints the ground-truth relative pose between two frames in
// both conventions this repo's executables use:
//   - x_j = R*x_i + t  (two_view_pose_test, pnp_test)
//   - x_i = R*x_j + t  (icp_test -- the inverse of the above)
// so its rotation-angle/translation-norm can be compared by eye against
// those executables' own printed R,t for the same frame pair.

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

#include "../report_utils.hpp"

using namespace std;

namespace {

vector<Sophus::SE3d> ReadGroundtruth(const string &path) {
  vector<Sophus::SE3d> poses;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    double ts, tx, ty, tz, qx, qy, qz, qw;
    iss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    poses.emplace_back(q, Eigen::Vector3d(tx, ty, tz));
  }
  return poses;
}

void PrintPose(const string &label, const Sophus::SE3d &T) {
  // Reuse the same RotationAngleDeg/EigenRotToCv helpers pnp_test/icp_test
  // already use for their own method-vs-method comparisons.
  cv::Mat R = EigenRotToCv(T.rotationMatrix());
  cv::Mat t = EigenVecToCv(T.translation());
  double angle_deg = RotationAngleDeg(cv::Mat::eye(3, 3, R.type()), R);
  cout << label << ":" << endl;
  cout << "  R =\n" << R << endl;
  cout << "  t = " << t.t() << endl;
  cout << "  rotation angle (deg): " << angle_deg << endl;
  cout << "  translation norm (m): " << cv::norm(t) << endl;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "usage: synthetic_pose_eval_test <groundtruth.txt> <frame_i> <frame_j>" << endl;
    return 1;
  }
  vector<Sophus::SE3d> T_wc = ReadGroundtruth(argv[1]);
  int i = atoi(argv[2]), j = atoi(argv[3]);
  if (i < 0 || j < 0 || i >= (int)T_wc.size() || j >= (int)T_wc.size()) {
    cerr << "frame index out of range (dataset has " << T_wc.size() << " frames)" << endl;
    return 1;
  }

  // PnP / two_view_pose_test convention: x_j = R*x_i + t.
  Sophus::SE3d T_ji = T_wc[j].inverse() * T_wc[i];
  // icp_test convention: x_i = R*x_j + t -- the inverse of the above.
  Sophus::SE3d T_ij = T_ji.inverse();

  cout << "=== GROUND TRUTH RELATIVE POSE: frame " << i << " -> frame " << j << " ===" << endl;
  PrintPose("T_ji (compare against two_view_pose_test / pnp_test R,t)", T_ji);
  cout << endl;
  PrintPose("T_ij (compare against icp_test R,t)", T_ij);

  return 0;
}
