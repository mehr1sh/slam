// Self-checking test of the TUM RGB-D adapter (src/dataset/tum.cpp): file
// lists, timestamp association, ground-truth interpolation, pose convention,
// depth conversion, intrinsics, missing data and deterministic ordering.
// Uses small hand-made sequence files in a temporary directory.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "dataset/tum.hpp"

namespace fs = std::filesystem;

namespace {

int failures = 0;
void Check(bool ok, const std::string &text) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", text.c_str());
  failures += !ok;
}
void Write(const fs::path &p, const std::string &s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << s;
}
double AngleDeg(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) {
  return std::acos(std::min(1.0, std::max(-1.0, ((A.transpose() * B).trace() - 1) / 2))) * 180 / M_PI;
}

}  // namespace

int main() {
  std::printf("=== TUM RGB-D adapter ===\n");
  const fs::path dir = fs::temp_directory_path() / "tum_dataset_test";
  fs::remove_all(dir);

  // --- file lists: comments skipped, sorted by timestamp ---
  Write(dir / "list.txt", "# color images\n# timestamp filename\n2.0 rgb/2.png\n1.0 rgb/1.png\n\n3.0 rgb/3.png\n");
  std::vector<tum::Stamped> l;
  Check(tum::ReadFileList((dir / "list.txt").string(), l) && l.size() == 3 && l[0].t == 1.0 && l[2].file == "rgb/3.png",
        "rgb.txt-style list: comments and blank lines skipped, entries sorted by timestamp");
  Write(dir / "bad.txt", "1.0\n");
  std::string err, err2;
  const bool bad = !tum::ReadFileList((dir / "bad.txt").string(), l, &err);
  Check(bad && !err.empty(), "malformed line reported: " + err);
  const bool none = !tum::ReadFileList((dir / "none.txt").string(), l, &err2);
  Check(none && err2.find("cannot open") != std::string::npos, "missing file reported: " + err2);

  // --- association ---
  {
    const std::vector<tum::Stamped> rgb = {{1.000, "r1"}, {1.033, "r2"}, {1.066, "r3"}, {2.000, "r4"}, {2.010, "r5"}};
    const std::vector<tum::Stamped> dep = {{0.995, "d1"}, {1.030, "d2"}, {1.100, "d3"}, {2.006, "d4"}};
    const auto a = tum::Associate(rgb, dep, 0.02);
    Check(a.size() == 3 && a[0].rgb == "r1" && a[0].depth == "d1" && a[1].rgb == "r2" && a[1].depth == "d2",
          "pairs within 0.02 s associated (r1-d1 5 ms, r2-d2 3 ms)");
    Check(a.size() == 3 && a[2].rgb == "r5" && a[2].depth == "d4",
          "a depth image nearest to two RGB images goes to the closer one (r5, 4 ms; not r4, 6 ms), used once");
    bool r3 = false;
    for (const auto &x : a) r3 |= x.rgb == "r3";
    Check(!r3, "r3 has no depth within 0.02 s (nearest 34 ms): not associated");
    Check(tum::Associate(rgb, dep, 0.05).size() == 4, "a larger max_dt (0.05 s) also associates r3-d3 (34 ms)");
    // deterministic ordering: shuffled input gives the identical result
    std::vector<tum::Stamped> rs = {rgb[3], rgb[0], rgb[4], rgb[2], rgb[1]}, ds = {dep[2], dep[0], dep[3], dep[1]};
    const auto b = tum::Associate(rs, ds, 0.02);
    bool same = a.size() == b.size();
    for (size_t k = 0; same && k < a.size(); ++k) same = a[k].rgb == b[k].rgb && a[k].depth == b[k].depth;
    Check(same, "shuffled input lists: identical associations in RGB-timestamp order");
  }

  // --- quaternion -> rotation, pose convention ---
  {
    Check(AngleDeg(tum::QuaternionToR(0, 0, 0, 1), Eigen::Matrix3d::Identity()) < 1e-12, "q = (0,0,0,1): identity");
    const double s = std::sqrt(0.5);
    Eigen::Matrix3d Rz;
    Rz << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    Check((tum::QuaternionToR(0, 0, s, s) - Rz).norm() < 1e-12, "q = (0,0,sin 45,cos 45): +90 deg about z");
    Check((tum::QuaternionToR(0, 0, 2 * s, 2 * s) - Rz).norm() < 1e-12, "a non-unit quaternion is normalised");
    // T_wc: X_world = R X_cam + t. Camera at (1,2,3) rotated +90 deg about world z:
    // its centre is t, and its optical axis (camera +z) points along R e_z.
    tum::Pose T;
    T.R = Rz;
    T.t = Eigen::Vector3d(1, 2, 3);
    const Eigen::Vector3d centre = T.R * Eigen::Vector3d::Zero() + T.t, ahead = T.R * Eigen::Vector3d(0, 0, 1) + T.t;
    Check(centre.isApprox(Eigen::Vector3d(1, 2, 3)) && ahead.isApprox(Eigen::Vector3d(1, 2, 4)),
          "T_wc maps the camera origin to its centre t and the camera's +z (forward) axis to R e_z");
    // relative pose T_{j<-i} = T_wc[j]^-1 T_wc[i] maps camera-i points to camera-j points
    tum::Pose Ti, Tj;
    Ti.R = tum::QuaternionToR(0.1, -0.2, 0.05, 0.97);
    Ti.t = Eigen::Vector3d(0.3, -0.1, 1.2);
    Tj.R = tum::QuaternionToR(-0.05, 0.1, 0.2, 0.97);
    Tj.t = Eigen::Vector3d(0.25, 0.0, 1.1);
    const Eigen::Matrix3d R = Tj.R.transpose() * Ti.R;
    const Eigen::Vector3d tr = Tj.R.transpose() * (Ti.t - Tj.t), Xw(0.4, 0.7, 2.5);
    const Eigen::Vector3d Xi = Ti.R.transpose() * (Xw - Ti.t), Xj = Tj.R.transpose() * (Xw - Tj.t);
    Check((R * Xi + tr - Xj).norm() < 1e-12, "T_{j<-i} = T_wc[j]^-1 T_wc[i] maps a world point seen from camera i into camera j");
  }

  // --- ground-truth interpolation ---
  {
    std::vector<tum::GtSample> gt(3);
    gt[0].t = 0.0;
    gt[1].t = 1.0, gt[1].p = Eigen::Vector3d(1, 2, 0);
    gt[1].q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    gt[2].t = 1.5, gt[2].p = Eigen::Vector3d(1, 2, 0), gt[2].q = gt[1].q;
    tum::Pose p;
    std::string why;
    Check(tum::InterpolateGroundtruth(gt, 0.5, 1.0, p) && p.t.isApprox(Eigen::Vector3d(0.5, 1, 0)) &&
              std::abs(AngleDeg(p.R, Eigen::Matrix3d::Identity()) - 45.0) < 1e-9,
          "midpoint: translation linear (0.5, 1, 0), rotation slerp 45 deg");
    Check(tum::InterpolateGroundtruth(gt, 1.0, 1.0, p) && p.t.isApprox(gt[1].p), "exact sample returned as is");
    const bool after = !tum::InterpolateGroundtruth(gt, 2.0, 1.0, p, &why);
    Check(after && why.find("outside") != std::string::npos, "after the ground-truth span: missing (" + why + ")");
    const bool gap = !tum::InterpolateGroundtruth(gt, 0.5, 0.5, p, &why);
    Check(gap && why.find("apart") != std::string::npos, "bracketing samples 1 s apart, max 0.5 s: missing (" + why + ")");
    std::vector<tum::GtSample> g2 = gt;
    g2[1].q.coeffs() *= -1;  // the same rotation with the opposite sign
    Check(tum::InterpolateGroundtruth(g2, 0.5, 1.0, p) && std::abs(AngleDeg(p.R, Eigen::Matrix3d::Identity()) - 45.0) < 1e-9,
          "q and -q interpolate along the shortest arc (45 deg, not 135)");
  }

  // --- depth conversion ---
  Check(tum::DepthMetres(5000, 5000.0) == 1.0 && tum::DepthMetres(2500, 5000.0) == 0.5 &&
            std::abs(tum::DepthMetres(65535, 5000.0) - 13.107) < 1e-12,
        "depth: raw / 5000 = metres (5000 -> 1 m, 2500 -> 0.5 m, 65535 -> 13.107 m)");
  Check(!tum::ValidDepth(0) && tum::DepthMetres(0, 5000.0) == 0.0 && tum::ValidDepth(1), "raw 0 = no measurement: invalid");

  // --- intrinsics ---
  {
    tum::CameraIntrinsics k;
    Check(tum::OfficialIntrinsics("rgbd_dataset_freiburg1_desk", k) && k.sensor == "freiburg1" && k.fx == 517.3 &&
              k.fy == 516.5 && k.cx == 318.6 && k.cy == 255.3 && k.d[0] == 0.2624 && k.depth_factor == 5000.0,
          "freiburg1_desk: fx 517.3, fy 516.5, cx 318.6, cy 255.3 (k1 0.2624), depth factor 5000");
    Check(!(k.fx == 520.9 && k.fy == 521.0), "not the synthetic Bunny intrinsics (520.9, 521.0)");
    Check(tum::OfficialIntrinsics("rgbd_dataset_freiburg3_long_office_household", k) && k.fx == 535.4 && k.d[0] == 0,
          "freiburg3: fx 535.4, no distortion");
    Check(!tum::OfficialIntrinsics("my_sequence", k), "unknown sensor: not guessed");
  }

  // --- frame selection with missing data ---
  {
    const fs::path seq = dir / "rgbd_dataset_freiburg1_test";
    Write(seq / "rgb.txt", "# color\n1.00 rgb/1.00.png\n1.10 rgb/1.10.png\n1.20 rgb/1.20.png\n1.30 rgb/1.30.png\n"
                           "9.00 rgb/9.00.png\n");
    Write(seq / "depth.txt", "# depth\n1.005 depth/1.005.png\n1.105 depth/1.105.png\n1.205 depth/1.205.png\n"
                             "1.305 depth/1.305.png\n9.005 depth/9.005.png\n");
    Write(seq / "groundtruth.txt", "# gt\n0.9 0 0 0 0 0 0 1\n1.5 1 0 0 0 0 0 1\n");
    for (const char *f : {"rgb/1.00.png", "rgb/1.10.png", "rgb/1.30.png", "rgb/9.00.png", "depth/1.005.png",
                          "depth/1.105.png", "depth/1.205.png", "depth/1.305.png", "depth/9.005.png"})
      Write(seq / f, "");  // rgb/1.20.png is missing
    std::vector<tum::Frame> fr;
    tum::SelectionReport rep;
    Check(tum::SelectFrames(seq.string(), 0.02, 1.0, 1, 0, fr, rep, &err) && rep.associated == 5 && fr.size() == 3,
          "5 associated, 3 usable frames");
    Check(rep.missing_image == 1 && rep.skipped.size() == 2 && rep.skipped[0].find("rgb/1.20.png") != std::string::npos,
          "missing image reported: " + (rep.skipped.empty() ? std::string() : rep.skipped[0]));
    Check(rep.missing_groundtruth == 1 && rep.skipped[1].find("9.000000") == 0,
          "frame without ground truth reported: " + (rep.skipped.size() > 1 ? rep.skipped[1] : std::string()));
    Check(fr.size() == 3 && fr[0].t_rgb == 1.00 && fr[1].t_rgb == 1.10 && fr[2].t_rgb == 1.30 && fr[0].index == 0 &&
              fr[2].index == 2 && std::abs(fr[1].T_wc.t.x() - 1.0 / 3.0) < 1e-12,
          "frames in timestamp order, indexed 0..2, ground truth interpolated at t_rgb (x = 1/3 at 1.10 s)");
    Check(tum::SelectFrames(seq.string(), 0.02, 1.0, 2, 0, fr, rep) && fr.size() == 2 && fr[1].t_rgb == 1.30,
          "stride 2 keeps every second usable frame");
    Check(tum::SelectFrames(seq.string(), 0.02, 1.0, 1, 2, fr, rep) && fr.size() == 2, "max_frames 2 stops after 2 frames");
  }
  fs::remove_all(dir);
  std::printf("\ntum_dataset_test: %s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
