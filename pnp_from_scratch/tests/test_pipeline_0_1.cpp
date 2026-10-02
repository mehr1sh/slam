// End-to-end self-checking test of the scratch pipeline on the real frame
// pair 0 -> 1: PNG decoding, project-owned features, essential-matrix stage,
// RANSAC around the scratch linear PnP. The ground-truth motion is only used
// to measure the result.

#include <cmath>

#include "check.hpp"
#include "pipeline.hpp"
#include "pnp.hpp"

using namespace scratch;

int main() {
  std::printf("=== scratch pipeline, frames 0 -> 1 (end to end) ===\n");
  const auto repo = scratch_test::Repo();
  const std::string ds = (repo / "data" / "synthetic_bunny").string();
  Intrinsics K;
  {
    std::ifstream f(ds + "/intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  const auto Twc = ReadGroundtruthTwc(repo / "data" / "synthetic_bunny" / "groundtruth.txt");
  const Eigen::Matrix3d R = Twc[1].R.transpose() * Twc[0].R;  // T_{1<-0}
  const Eigen::Vector3d t = Twc[1].R.transpose() * (Twc[0].t - Twc[1].t);

  RgbdFrame f0, f1;
  CHECK(LoadRgbdFrame(ds, 0, f0) && LoadRgbdFrame(ds, 1, f1), "frames load");
  const PipelineParams params;
  const FrameFeatures a = ExtractFeatures(f0.rgb, params.frontend), b = ExtractFeatures(f1.rgb, params.frontend);
  const PairResult r = ProcessPair(0, 1, a, b, f0, f1, K, params);

  CHECK(r.pnp_ok, "RANSAC PnP solved the pair (%s)", r.pnp_reason.c_str());
  CHECK(int(r.pnp_index.size()) > 50 && r.pnp_inliers >= 40,
        "%d 3D->2D correspondences, %d RANSAC inliers", int(r.pnp_index.size()), r.pnp_inliers);
  const double re = RotErrDeg(R, r.R), te = (t - r.t).norm();
  CHECK(re < 3.0 && te < 0.03, "pose vs ground truth: rotation error %.3f deg, translation error %.4f m "
        "(true motion 10 deg, 0.087 m)", re, te);
  CHECK(r.reproj_inlier_mean < 3.0, "inlier reprojection error mean %.2f px, median %.2f px, max %.2f px",
        r.reproj_inlier_mean, r.reproj_inlier_median, r.reproj_inlier_max);
  CHECK(std::abs(r.R.determinant() - 1) < 1e-9 && (r.R.transpose() * r.R - Eigen::Matrix3d::Identity()).norm() < 1e-9,
        "R is a proper rotation");
  int mask_count = 0;
  for (bool m : r.pnp_mask) mask_count += m;
  CHECK(mask_count == r.pnp_inliers && r.pnp_mask.size() == r.pnp_index.size(), "exported inlier mask matches the count");

  // the same correspondences without RANSAC (pure linear PnP) for contrast
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
  for (int k : r.pnp_index) X.push_back(r.correspondences[k].X_i), uv.push_back(r.correspondences[k].uv_j);
  const PnPResult plain = solvePnPDLT(X, uv, K);
  CHECK(plain.ok && RotErrDeg(R, plain.R) > re, "RANSAC improves on the plain linear PnP of the same matches "
        "(%.2f deg without RANSAC)", RotErrDeg(R, plain.R));

  // essential stage: computed and reported; it is a diagnostic (see README:
  // the orbit's rotation and translation nearly cancel in the image)
  CHECK(r.essential_mask.size() == r.correspondences.size(),
        "essential stage ran on all %d filtered matches (ok %d, %d inliers, rotation error %.2f deg)",
        int(r.correspondences.size()), int(r.essential_ok), r.essential_inliers,
        r.essential_ok ? RotErrDeg(R, r.R_essential) : -1.0);

  const PairResult r2 = ProcessPair(0, 1, a, b, f0, f1, K, params);
  CHECK(r2.R == r.R && r2.t == r.t && r2.pnp_mask == r.pnp_mask, "deterministic: identical result on a second run");
  return scratch_test::Finish("test_pipeline_0_1");
}
