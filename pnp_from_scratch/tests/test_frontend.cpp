// Self-checking test of the scratch front end on the real frame pair 0 -> 1:
// project-owned features (FAST threshold 20, orientation, rotated BRIEF,
// Hamming) -> filtered matches -> 3D->2D correspondences. The ground-truth
// motion is used only to MEASURE how many correspondences are geometrically
// right; nothing is selected with it.

#include <cmath>

#include "check.hpp"
#include "frontend.hpp"

using namespace scratch;

int main() {
  std::printf("=== scratch front end, frames 0 -> 1 ===\n");
  const auto repo = scratch_test::Repo();
  const std::string ds = (repo / "data" / "synthetic_bunny").string();
  Intrinsics K;
  {
    std::ifstream f(ds + "/intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  RgbdFrame f0, f1;
  std::string err;
  CHECK(LoadRgbdFrame(ds, 0, f0, &err) && LoadRgbdFrame(ds, 1, f1, &err), "frames 0 and 1 load (%s)", err.c_str());
  const FrontendParams params;
  CHECK(params.fast.threshold == 20 && params.fast.max_keypoints == 500 && params.match_floor == 30,
        "configuration: FAST threshold 20, strongest 500, filter max(2 d_min, 30)");

  const FrameFeatures a = ExtractFeatures(f0.rgb, params), b = ExtractFeatures(f1.rgb, params);
  CHECK(a.keypoints.size() > 80 && a.keypoints.size() <= 500 && b.keypoints.size() > 80,
        "keypoints: %d / %d", int(a.keypoints.size()), int(b.keypoints.size()));
  const PairMatches pm = MatchFrames(a, b, f0.depth, f1.depth, K, params);
  int n3d = 0;
  for (const auto &c : pm.correspondences) n3d += c.has_3d;
  CHECK(pm.raw.size() == a.keypoints.size(), "one raw match per frame-0 keypoint (%d)", int(pm.raw.size()));
  CHECK(pm.filtered.size() == pm.correspondences.size() && pm.filtered.size() > 50,
        "filtered matches = correspondences = %d (d_min %d, d_max %d)", int(pm.filtered.size()), pm.d_min, pm.d_max);
  CHECK(n3d > 50, "%d 3D->2D correspondences (frame-0 depth available)", n3d);

  // geometric quality, measured with the ground-truth motion T_{1<-0}
  const auto Twc = ReadGroundtruthTwc(repo / "data" / "synthetic_bunny" / "groundtruth.txt");
  const Eigen::Matrix3d R = Twc[1].R.transpose() * Twc[0].R;
  const Eigen::Vector3d t = Twc[1].R.transpose() * (Twc[0].t - Twc[1].t);
  int good = 0;
  bool backproj_ok = true;
  for (const auto &c : pm.correspondences) {
    if (!c.has_3d) continue;
    const double Z = c.depth_raw_i / 5000.0;
    backproj_ok &= std::abs(c.X_i.z() - Z) < 1e-12 && std::abs(K.fx * c.X_i.x() / Z + K.cx - c.uv_i.x()) < 1e-9;
    good += (projectPoint(c.X_i, R, t, K) - c.uv_j).norm() < 3.0;
  }
  CHECK(backproj_ok, "3D points are the back-projection of (u_i, v_i, depth/5000) with K");
  CHECK(good > 0.6 * n3d, "%d / %d correspondences reproject within 3 px under the true motion", good, n3d);

  const FrameFeatures a2 = ExtractFeatures(f0.rgb, params);
  bool same = a2.keypoints.size() == a.keypoints.size();
  for (size_t i = 0; same && i < a.keypoints.size(); ++i)
    same = a2.keypoints[i].x == a.keypoints[i].x && a2.keypoints[i].y == a.keypoints[i].y &&
           a2.descriptors[i].bits == a.descriptors[i].bits;
  CHECK(same, "deterministic: a second extraction gives identical keypoints and descriptors");
  return scratch_test::Finish("test_frontend");
}
