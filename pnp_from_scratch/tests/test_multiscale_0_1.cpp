// Self-checking test of the multiscale scratch front end (8-level pyramid,
// scale 1.2, FAST threshold 20) on the real frame pair 0 -> 1, and of the
// whole pipeline with it. Ground truth is only used to measure.

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

  PipelineParams single, multi;
  multi.frontend.pyramid.levels = 8;
  CHECK(single.frontend.pyramid.levels == 1 && multi.frontend.fast.threshold == 20 &&
            std::abs(multi.frontend.pyramid.scale_factor - 1.2) < 1e-12,
        "configuration: default single scale; multiscale = 8 levels x 1.2, FAST threshold 20");

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

  // the whole pipeline with multiscale features
  const auto Twc = ReadGroundtruthTwc(repo / "data" / "synthetic_bunny" / "groundtruth.txt");
  const Eigen::Matrix3d R = Twc[1].R.transpose() * Twc[0].R;
  const Eigen::Vector3d t = Twc[1].R.transpose() * (Twc[0].t - Twc[1].t);
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
