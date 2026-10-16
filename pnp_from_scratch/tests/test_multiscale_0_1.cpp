// Self-checking test of the multiscale scratch front end (8-level pyramid,
// scale 1.2, FAST threshold 20; the default) on the real frame pair 0 -> 1:
// per-level counts, level -> original-image coordinate mapping, the depth
// lookup and 3D reconstruction at the ORIGINAL coordinates, and the whole
// pipeline with it. Ground truth is only used to measure.

#include <algorithm>
#include <cmath>

#include "check.hpp"
#include "pipeline.hpp"

using namespace scratch;

int main() {
  std::printf("=== multiscale scratch front end, frames 0 -> 1 ===\n");
  const auto repo = scratch_test::Repo();
  const std::string ds = (repo / "data" / "synthetic_bunny").string();
  Intrinsics K;
  {
    std::ifstream f(ds + "/intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  RgbdFrame f0, f1;
  CHECK(LoadRgbdFrame(ds, 0, f0) && LoadRgbdFrame(ds, 1, f1), "frames load");

  PipelineParams multi, single;
  single.frontend.pyramid.levels = 1;
  CHECK(multi.frontend.pyramid.levels == 8 && multi.frontend.fast.threshold == 20 &&
            std::abs(multi.frontend.pyramid.scale_factor - 1.2) < 1e-12,
        "configuration: default = 8 levels x 1.2, FAST threshold 20 (single scale: pyramid.levels = 1)");

  const FrameFeatures s0 = ExtractFeatures(f0.rgb, single.frontend);
  const FrameFeatures m0 = ExtractFeatures(f0.rgb, multi.frontend), m1 = ExtractFeatures(f1.rgb, multi.frontend);
  int total = 0;
  bool all_levels = m0.per_level_count.size() == 8;
  for (int c : m0.per_level_count) all_levels &= c > 0, total += c;
  CHECK(all_levels, "frame 0 keypoints per level: %d %d %d %d %d %d %d %d", m0.per_level_count[0],
        m0.per_level_count[1], m0.per_level_count[2], m0.per_level_count[3], m0.per_level_count[4],
        m0.per_level_count[5], m0.per_level_count[6], m0.per_level_count[7]);
  CHECK(total > 300 && total < 500, "total %d keypoints (reference ORB: 366; single scale: %d)", total,
        int(s0.keypoints.size()));
  // level 0 of the pyramid is exactly the single-scale result
  bool lvl0 = m0.per_level_count[0] == int(s0.keypoints.size());
  for (size_t k = 0; lvl0 && k < s0.keypoints.size(); ++k)
    lvl0 = m0.keypoints[k].x == s0.keypoints[k].x && m0.keypoints[k].y == s0.keypoints[k].y &&
           m0.descriptors[k].bits == s0.descriptors[k].bits && m0.level[k] == 0;
  CHECK(lvl0, "the level-0 keypoints and descriptors equal the single-scale ones");
  int on_bunny = 0;
  bool coords = true;
  for (size_t k = 0; k < m0.keypoints.size(); ++k) {
    const auto &kp = m0.keypoints[k];
    coords &= kp.x >= 0 && kp.y >= 0 && kp.x < 640 && kp.y < 480 &&
              std::abs(kp.x - ((m0.level_x[k] + 0.5f) * m0.scale[k] - 0.5f)) < 1e-3f;
    on_bunny += f0.depth.samples[size_t(int(kp.y)) * 640 + int(kp.x)] > 0;
  }
  CHECK(coords, "original coordinates = (level coordinate + 0.5) x scale - 0.5, inside the image");
  // Coarse-level corners on the outline map back to within ~1 level pixel
  // (several original pixels) and can fall just outside the silhouette; the
  // reference ORB keypoints of this frame show the same effect: 257 / 366 = 0.70.
  CHECK(on_bunny > 0.65 * total, "%d / %d = %.2f of the keypoints have valid depth at the mapped pixel "
        "(reference ORB: 0.70; outline corners of coarse levels land just outside the silhouette)", on_bunny, total,
        double(on_bunny) / total);

  // --- level -> original mapping in matching, depth lookup and 3D reconstruction ---
  const auto Twc = ReadGroundtruthTwc(repo / "data" / "synthetic_bunny" / "groundtruth.txt");
  const Eigen::Matrix3d R = Twc[1].R.transpose() * Twc[0].R;
  const Eigen::Vector3d t = Twc[1].R.transpose() * (Twc[0].t - Twc[1].t);
  {
    const PairMatches pm = MatchFrames(m0, m1, f0.depth, f1.depth, K, multi.frontend);
    bool uv_ok = true, depth_ok = true, backproj_ok = true;
    int coarse = 0, coarse_3d = 0, wrong_no_depth = 0;
    std::vector<double> err_right, err_wrong;  // coarse levels (>= 3): GT reprojection error into frame 1
    // per-axis scale of level l, from the level size alone: 640 / round(640 / 1.2^l), 480 / round(480 / 1.2^l)
    auto sx = [](int l) { return 640.0 / std::lround(640 / std::pow(1.2, l)); };
    auto sy = [](int l) { return 480.0 / std::lround(480 / std::pow(1.2, l)); };
    for (const Correspondence &c : pm.correspondences) {
      const int q = c.query, s = c.train;
      // pixels used downstream = level coordinates mapped to the 640x480 image
      uv_ok &= c.level_i == m0.level[q] && c.level_j == m1.level[s] &&
               std::abs(m0.scale[q] - sx(m0.level[q])) < 1e-6 &&
               std::abs(c.uv_i.x() - ((m0.level_x[q] + 0.5) * sx(c.level_i) - 0.5)) < 1e-4 &&
               std::abs(c.uv_i.y() - ((m0.level_y[q] + 0.5) * sy(c.level_i) - 0.5)) < 1e-4 &&
               std::abs(c.uv_j.x() - ((m1.level_x[s] + 0.5) * sx(c.level_j) - 0.5)) < 1e-4 &&
               std::abs(c.uv_j.y() - ((m1.level_y[s] + 0.5) * sy(c.level_j) - 0.5)) < 1e-4;
      // depth is read from the ORIGINAL depth image at the mapped pixel
      depth_ok &= c.depth_raw_i == f0.depth.samples[size_t(int(c.uv_i.y())) * 640 + int(c.uv_i.x())];
      if (c.has_3d) {
        double z;
        backproj_ok &= (projectPoint(c.X_i, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), K, &z) - c.uv_i)
                           .norm() < 1e-9 && std::abs(z - c.depth_raw_i / 5000.0) < 1e-12;
      }
      if (c.level_i < 3) continue;
      ++coarse;
      if (!c.has_3d) continue;
      ++coarse_3d;
      err_right.push_back((projectPoint(c.X_i, R, t, K) - c.uv_j).norm());
      // control: the same keypoint if its LEVEL coordinates were (wrongly) used as pixels
      const int lx = int(m0.level_x[q]), ly = int(m0.level_y[q]);
      const uint16_t dw = f0.depth.samples[size_t(ly) * 640 + lx];
      if (dw == 0) {  // the level pixel falls off the bunny in the original depth image
        ++wrong_no_depth;
        continue;
      }
      const double Z = dw / 5000.0;
      const Eigen::Vector3d Xw((lx - K.cx) / K.fx * Z, (ly - K.cy) / K.fy * Z, Z);
      err_wrong.push_back((projectPoint(Xw, R, t, K) - c.uv_j).norm());
    }
    CHECK(uv_ok, "every correspondence uses (level coordinate + 0.5) x scale - 0.5 per axis in both frames "
          "(x scale 640 / level width, y scale 480 / level height), and records the levels");
    CHECK(depth_ok, "depth is read from the 640x480 depth image at the truncated ORIGINAL pixel");
    CHECK(backproj_ok, "3D points back-project the original pixel with K and depth / 5000");
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v.empty() ? NAN : v[v.size() / 2];
    };
    // control: the level coordinates themselves land elsewhere on (or off) the object
    const bool wrong_fails = wrong_no_depth > coarse_3d / 2 || median(err_wrong) > 30.0;
    CHECK(coarse_3d > 30 && median(err_right) < 3.0 && wrong_fails,
          "%d coarse-level (>= 3) matches, %d with depth: median reprojection error under the true motion %.2f px "
          "with the mapping; using the level coordinates directly instead, %d / %d would have no depth",
          coarse, coarse_3d, median(err_right), wrong_no_depth, coarse_3d);
  }

  // the whole pipeline with multiscale features
  const PairResult r = ProcessPair(0, 1, m0, m1, f0, f1, K, multi);
  int cross = 0;
  for (const auto &c : r.correspondences) cross += c.level_i != c.level_j;
  CHECK(r.pnp_ok && r.pnp_inliers > 100, "%d filtered matches (%d between different levels), %d 3D->2D, %d RANSAC "
        "inliers", int(r.correspondences.size()), cross, int(r.pnp_index.size()), r.pnp_inliers);
  CHECK(RotErrDeg(R, r.R) < 3.0 && (t - r.t).norm() < 0.03, "pose: rotation error %.3f deg, translation error %.4f m",
        RotErrDeg(R, r.R), (t - r.t).norm());
  CHECK(r.reproj_inlier_mean < 3.0, "inlier reprojection error mean %.2f px", r.reproj_inlier_mean);
  return scratch_test::Finish("test_multiscale_0_1");
}
