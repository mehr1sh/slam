// Tests of the PnP pose-validity gate (src/validity.hpp) and of accumulation with
// rejected pairs (src/accumulate.hpp): invalid poses, duplicate rows, distinct
// keypoints / locations, coverage, reprojection, inlier ratio, a valid synthetic
// pose, held poses, determinism; and the gate inside the pipeline on Bunny 0 -> 1.

#include <algorithm>
#include <cmath>
#include <random>

#include "accumulate.hpp"
#include "check.hpp"
#include "pipeline.hpp"
#include "ransac.hpp"
#include "validity.hpp"

using namespace scratch;

namespace {

const Intrinsics K{517.3, 516.5, 318.6, 255.3};
constexpr double kThr = 8.0;  // the default RANSAC threshold

bool Has(const PoseValidity &v, const std::string &s) { return v.reason.find(s) != std::string::npos; }

// n rows on a 10 x n/10 grid spread over the image (60 px apart): distinct keypoints and locations,
// every row an inlier with the given residual.
std::vector<ValidityRow> Spread(int n, double residual = 1.0) {
  std::vector<ValidityRow> rows(n);
  for (int k = 0; k < n; ++k) {
    rows[k].src = k, rows[k].dst = 1000 + k;
    rows[k].uv_src = Eigen::Vector2d(20 + 60 * (k % 10), 20 + 60 * (k / 10));
    rows[k].uv_dst = rows[k].uv_src + Eigen::Vector2d(3, -2);
    rows[k].residual_px = residual;
    rows[k].inlier = true;
  }
  return rows;
}

// Synthetic scene: pixels spread over the image, depths 1-3 m, true motion T_{j<-i}; RANSAC PnP on it.
struct Synthetic {
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv_i, uv_j;
};
Synthetic MakeScene(int n, const Pose &T, double noise_px, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(16, 624), V(16, 464), Z(1.0, 3.0);
  std::normal_distribution<double> N(0, noise_px);
  Synthetic s;
  for (int k = 0; k < n; ++k) {
    const double u = U(rng), v = V(rng), z = Z(rng);
    const Eigen::Vector3d X((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z);
    s.X.push_back(X), s.uv_i.emplace_back(u, v);
    s.uv_j.push_back(projectPoint(X, T.R, T.t, K) + Eigen::Vector2d(N(rng), N(rng)));
  }
  return s;
}

}  // namespace

int main() {
  const ValidityParams P;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(0.01, 0, 0.02);

  std::printf("=== 1. non-finite or invalid poses ===\n");
  {
    const auto rows = Spread(60);
    Eigen::Matrix3d Rn = I;
    Rn(1, 2) = NAN;
    const PoseValidity a = EvaluatePose(true, "", Rn, t0, rows, kThr, P);
    CHECK(a.status == PoseStatus::kRejected && !a.finite && Has(a, "non-finite"), "NaN in R: rejected (%s)", a.reason.c_str());
    const PoseValidity b = EvaluatePose(true, "", I, Eigen::Vector3d(0, INFINITY, 0), rows, kThr, P);
    CHECK(b.status == PoseStatus::kRejected && Has(b, "non-finite"), "inf in t: rejected");
    const PoseValidity c = EvaluatePose(true, "", 1.01 * I, t0, rows, kThr, P);
    CHECK(c.status == PoseStatus::kRejected && !c.rotation_valid && Has(c, "SO(3)"),
          "scaled R (not orthogonal): rejected (%s)", c.reason.c_str());
    const Eigen::Matrix3d refl = Eigen::Vector3d(1, 1, -1).asDiagonal();
    const PoseValidity d = EvaluatePose(true, "", refl, t0, rows, kThr, P);
    CHECK(d.status == PoseStatus::kRejected && Has(d, "SO(3)") && d.det_error > 1.9, "reflection (det -1): rejected");
    const PoseValidity e = EvaluatePose(false, "fewer than 6 correspondences", I, t0, rows, kThr, P);
    CHECK(e.status == PoseStatus::kSolverFailed && e.reason == "fewer than 6 correspondences" && !e.accepted(),
          "solver failure is its own status, with the solver's reason");
    const PoseValidity f = EvaluatePose(true, "", I, t0, rows, kThr, P);
    CHECK(f.accepted() && f.reason.empty() && f.rotation_valid && f.orthogonality_error < 1e-12,
          "the same rows with a proper rotation: accepted");
  }

  std::printf("\n=== 2. duplicate correspondence rows are not independent support ===\n");
  {
    std::vector<ValidityRow> rows;
    for (const ValidityRow &r : Spread(10))
      for (int c = 0; c < 6; ++c) rows.push_back(r);  // 60 rows, 10 keypoints per frame
    const PoseValidity v = EvaluatePose(true, "", I, t0, rows, kThr, P);
    CHECK(v.inlier_rows == 60 && v.distinct_src == 10 && v.distinct_dst == 10 && v.distinct_loc_src == 10,
          "60 rows counted as 10 distinct keypoints / locations per frame");
    CHECK(v.status == PoseStatus::kRejected && Has(v, "distinct frame-i inlier keypoints") &&
              Has(v, "distinct frame-j inlier keypoints") && !Has(v, "inlier rows"),
          "rejected for distinct support although the row count passes (%s)", v.reason.c_str());
    // cross-level duplicates: different keypoint indices at (nearly) the same pixels
    std::vector<ValidityRow> lv;
    int id = 0;
    for (const ValidityRow &r : Spread(10))
      for (int l = 0; l < 6; ++l) {
        ValidityRow d = r;
        d.src = id, d.dst = 5000 + id++;
        d.uv_src += Eigen::Vector2d(0.3 * l, 0.2 * l), d.uv_dst += Eigen::Vector2d(0.2 * l, -0.3 * l);
        lv.push_back(d);
      }
    const PoseValidity w = EvaluatePose(true, "", I, t0, lv, kThr, P);
    CHECK(w.distinct_src == 60 && w.distinct_dst == 60 && w.distinct_loc_src == 10 && w.distinct_loc_dst == 10 &&
              w.status == PoseStatus::kRejected && Has(w, "inlier locations") && !Has(w, "inlier keypoints"),
          "60 distinct keypoint indices at 10 locations (same corner on several levels): rejected by location");
  }

  std::printf("\n=== 3. insufficient distinct source or destination keypoints ===\n");
  {
    auto src_dup = Spread(40);
    for (auto &r : src_dup) r.src = r.src % 10, r.uv_src = Spread(10)[r.src].uv_src;  // frame i: 10 keypoints
    const PoseValidity a = EvaluatePose(true, "", I, t0, src_dup, kThr, P);
    CHECK(a.distinct_src == 10 && a.distinct_dst == 40 && a.status == PoseStatus::kRejected &&
              Has(a, "frame-i inlier keypoints") && !Has(a, "frame-j inlier keypoints"),
          "10 source / 40 destination keypoints: rejected for the source only (%s)", a.reason.c_str());
    auto dst_dup = Spread(40);
    for (auto &r : dst_dup) r.dst = 1000 + r.src % 10, r.uv_dst = Spread(10)[r.src % 10].uv_dst;
    const PoseValidity b = EvaluatePose(true, "", I, t0, dst_dup, kThr, P);
    CHECK(b.distinct_src == 40 && b.distinct_dst == 10 && b.status == PoseStatus::kRejected &&
              Has(b, "frame-j inlier keypoints") && !Has(b, "frame-i inlier keypoints"),
          "40 source / 10 destination keypoints: rejected for the destination only");
    const PoseValidity c = EvaluatePose(true, "", I, t0, Spread(20), kThr, P);
    const PoseValidity d = EvaluatePose(true, "", I, t0, Spread(19), kThr, P);
    CHECK(c.accepted() && !d.accepted() && Has(d, "19 inlier rows"), "boundary: 20 accepted, 19 rejected");
  }

  std::printf("\n=== 4. poor spatial coverage ===\n");
  {
    std::vector<ValidityRow> rows(30);
    for (int k = 0; k < 30; ++k) {  // 30 locations 3 px apart inside one 40 px cell
      rows[k].src = k, rows[k].dst = 100 + k;
      rows[k].uv_src = Eigen::Vector2d(402 + 3 * (k % 6), 202 + 3 * (k / 6));
      rows[k].uv_dst = rows[k].uv_src + Eigen::Vector2d(1, 1);
      rows[k].residual_px = 1, rows[k].inlier = true;
    }
    const PoseValidity v = EvaluatePose(true, "", I, t0, rows, kThr, P);
    CHECK(v.distinct_loc_src == 30 && v.cells_src == 1 && v.status == PoseStatus::kRejected &&
              Has(v, "frame-i inliers cover 1 cells") && Has(v, "frame-j inliers cover"),
          "30 distinct locations inside one cell: rejected for coverage only (%s)", v.reason.c_str());
    CHECK(CountOccupiedCells({{0, 0}, {39.9, 39.9}, {40, 0}, {-0.1, 0}}, 40) == 3,
          "cell boundaries: [0, 40) is one cell, 40 and -0.1 start new ones");
  }

  std::printf("\n=== 5. excessive reprojection error ===\n");
  {
    const PoseValidity a = EvaluatePose(true, "", I, t0, Spread(60, 4.5), kThr, P);
    CHECK(a.status == PoseStatus::kRejected && Has(a, "median 4.500 px > 4.000 px"),
          "inlier median 4.5 px > 0.5 x 8 px: rejected (%s)", a.reason.c_str());
    auto rows = Spread(60, 1.0);
    for (int k = 0; k < 9; ++k) rows[k].residual_px = 12.0;  // 15 %: p90 above the threshold, median fine
    const PoseValidity b = EvaluatePose(true, "", I, t0, rows, kThr, P);
    CHECK(b.status == PoseStatus::kRejected && Has(b, "p90") && !Has(b, "median"),
          "p90 %.2f px > 8 px with median %.2f px: rejected for p90 only", b.reproj_p90_px, b.reproj_median_px);
    auto one_inf = Spread(60, 1.0);
    one_inf[0].residual_px = INFINITY;  // behind the camera under the final pose
    const PoseValidity oi = EvaluatePose(true, "", I, t0, one_inf, kThr, P);
    CHECK(oi.reproj_median_px == 1.0 && oi.reproj_p90_px == 1.0 && oi.accepted(),
          "one infinite residual among 60: median and p90 stay 1.0 px (not NaN), accepted");
    auto many_inf = Spread(60, 1.0);
    for (int k = 0; k < 40; ++k) many_inf[k].residual_px = (k % 2) ? INFINITY : NAN;  // NaN is treated as infinite
    const PoseValidity mi = EvaluatePose(true, "", I, t0, many_inf, kThr, P);
    CHECK(std::isinf(mi.reproj_median_px) && std::isinf(mi.reproj_p90_px) && mi.status == PoseStatus::kRejected &&
              Has(mi, "median inf px"),
          "40 of 60 residuals NaN / infinite: median and p90 infinite, rejected (%s)", mi.reason.c_str());
    CHECK(Percentile({1, 2, 3, 4, 5, 6, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY}, 50) == 6 &&
              std::isinf(Percentile({1, INFINITY}, 50)) && std::isinf(Percentile({INFINITY, INFINITY}, 50)),
          "percentile next to infinite values: the order statistic itself, never NaN");
    CHECK(Percentile({}, 50) != Percentile({}, 50) && Percentile({1, 2, 3, 4}, 50) == 2.5 &&
              std::abs(Percentile({0, 10}, 90) - 9.0) < 1e-12,
          "percentile: empty -> NaN, linear interpolation");
    const PoseValidity outliers = EvaluatePose(true, "", I, t0, [] {
      auto r = Spread(60, 1.0);
      for (int k = 0; k < 30; ++k) r[k].inlier = false, r[k].residual_px = 300;  // outliers do not count
      return r;
    }(), kThr, P);
    CHECK(outliers.accepted() && outliers.inlier_ratio == 0.5 && outliers.reproj_p90_px == 1.0,
          "residuals of non-inliers do not enter the statistics");
    ValidityParams Q = P;
    Q.min_inlier_ratio = 0.6;
    CHECK(!EvaluatePose(true, "", I, t0, [] {
             auto r = Spread(60, 1.0);
             for (int k = 0; k < 30; ++k) r[k].inlier = false;
             return r;
           }(), kThr, Q).accepted(),
          "a configured inlier-ratio minimum (0.6) rejects a 0.5 ratio; the default 0 records it only");
  }

  std::printf("\n=== 6. a valid synthetic pose is accepted ===\n");
  {
    Pose T;
    T.R = QuatToR(0.01, -0.02, 0.005, 1.0);
    T.t = Eigen::Vector3d(0.03, -0.01, 0.05);
    Synthetic s = MakeScene(400, T, 0.7, 7);
    std::mt19937 rng(11);  // 30 % gross outliers
    for (int k = 0; k < 120; ++k) s.uv_j[k] = Eigen::Vector2d(16 + rng() % 600, 16 + rng() % 440);
    const RansacPnPResult p = RansacPnP(s.X, s.uv_j, K);
    std::vector<ValidityRow> rows(s.X.size());
    for (size_t k = 0; k < rows.size(); ++k)
      rows[k] = {int(k), int(k), s.uv_i[k], s.uv_j[k], p.residual_px[k], bool(p.inlier_mask[k])};
    const PoseValidity v = EvaluatePose(p.ok, p.reason, p.R, p.t, rows, kThr, P);
    CHECK(p.ok && v.accepted(), "RANSAC PnP on 400 points (30 %% outliers, 0.7 px noise): accepted "
          "(%d inliers, %d / %d locations, %d / %d cells, median %.2f px, p90 %.2f px, ratio %.2f)",
          v.inlier_rows, v.distinct_loc_src, v.distinct_loc_dst, v.cells_src, v.cells_dst, v.reproj_median_px,
          v.reproj_p90_px, v.inlier_ratio);
    // the same scene with only 12 true correspondences (the rest outliers): a minimal-sample-sized support
    Synthetic few = MakeScene(400, T, 0.7, 8);
    for (int k = 12; k < 400; ++k) few.uv_j[k] = Eigen::Vector2d(16 + rng() % 600, 16 + rng() % 440);
    const RansacPnPResult q = RansacPnP(few.X, few.uv_j, K);
    std::vector<ValidityRow> qr(few.X.size());
    for (size_t k = 0; k < qr.size(); ++k)
      qr[k] = {int(k), int(k), few.uv_i[k], few.uv_j[k], q.ok ? q.residual_px[k] : 0.0, q.ok && q.inlier_mask[k]};
    const PoseValidity w = EvaluatePose(q.ok, q.reason, q.R, q.t, qr, kThr, P);
    CHECK(!w.accepted(), "12 true correspondences among 400: %s (%d inlier rows; %s)", PoseStatusName(w.status),
          w.inlier_rows, w.reason.c_str());
  }

  std::printf("\n=== 7. a rejected pose does not change the accumulated trajectory ===\n");
  {
    Pose anchor;
    anchor.R = QuatToR(0.3, 0.1, -0.2, 0.9);
    anchor.t = Eigen::Vector3d(1.3, 0.6, 1.6);
    Pose A, B, wild;
    A.R = QuatToR(0.01, 0.0, 0.02, 1.0), A.t = Eigen::Vector3d(0.01, 0.02, -0.03);
    B.R = QuatToR(-0.02, 0.01, 0.0, 1.0), B.t = Eigen::Vector3d(-0.02, 0.0, 0.04);
    wild.R = QuatToR(0.2, 0.0, 0.0, 1.0), wild.t = Eigen::Vector3d(0.7, -0.3, 0.2);  // a 22 deg / 0.76 m-like error
    std::vector<RelativeStep> steps = {{true, "", A}, {false, "held_previous_pose_pair_rejected", wild}, {true, "", B}};
    const AccumulatedTrajectory t = AccumulateTrajectory(anchor, steps);
    CHECK(t.Twc.size() == 4 && t.Twc[2].R == t.Twc[1].R && t.Twc[2].t == t.Twc[1].t,
          "frame after the rejected pair holds the previous pose exactly");
    const Pose expect3 = Compose(t.Twc[2], Inverse(B));
    CHECK(t.Twc[3].R == expect3.R && t.Twc[3].t == expect3.t, "the next accepted pair continues from the held pose");
    steps[1].T.R = Eigen::Matrix3d::Constant(NAN), steps[1].T.t = Eigen::Vector3d::Constant(NAN);
    const AccumulatedTrajectory u = AccumulateTrajectory(anchor, steps);
    bool same = true;
    for (size_t k = 0; k < u.Twc.size(); ++k) same &= u.Twc[k].R == t.Twc[k].R && u.Twc[k].t == t.Twc[k].t;
    CHECK(same && u.Twc[3].R.allFinite(), "the rejected pair's pose values are never read (NaN pose: same trajectory)");
    CHECK(t.source[0] == "gt_anchor" && t.source[1] == "scratch_pipeline" &&
              t.source[2] == "held_previous_pose_pair_rejected" && t.source[3] == "scratch_pipeline",
          "pose sources: gt_anchor, scratch_pipeline, held_previous_pose_pair_rejected, scratch_pipeline");
    // convention: T_wc[k]^-1 T_wc[k+1] ... T_{k+1<-k} = T_wc[k+1]^-1 T_wc[k] for accepted steps
    const Pose rel = Compose(Inverse(t.Twc[1]), t.Twc[0]);
    CHECK((rel.R - A.R).norm() < 1e-12 && (rel.t - A.t).norm() < 1e-12, "accepted step: T_wc[1]^-1 T_wc[0] = T_{1<-0}");
  }

  std::printf("\n=== 8. deterministic decisions ===\n");
  {
    auto rows = Spread(80, 1.0);
    for (int k = 0; k < 80; ++k) rows[k].residual_px = 0.5 + 0.05 * k, rows[k].inlier = k % 7 != 0;
    const PoseValidity a = EvaluatePose(true, "", I, t0, rows, kThr, P), b = EvaluatePose(true, "", I, t0, rows, kThr, P);
    std::mt19937 rng(3);
    auto shuffled = rows;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    const PoseValidity c = EvaluatePose(true, "", I, t0, shuffled, kThr, P);
    auto same = [](const PoseValidity &x, const PoseValidity &y) {
      return x.status == y.status && x.reason == y.reason && x.inlier_rows == y.inlier_rows &&
             x.distinct_loc_src == y.distinct_loc_src && x.distinct_loc_dst == y.distinct_loc_dst &&
             x.cells_src == y.cells_src && x.cells_dst == y.cells_dst && x.reproj_median_px == y.reproj_median_px &&
             x.reproj_p90_px == y.reproj_p90_px && x.inlier_ratio == y.inlier_ratio;
    };
    CHECK(same(a, b), "identical inputs: identical decision and quantities");
    CHECK(same(a, c), "row order does not matter (shuffled rows: identical decision and quantities)");
  }

  std::printf("\n=== the gate in the pipeline (Bunny 0 -> 1) ===\n");
  {
    const auto repo = scratch_test::Repo();
    const std::string ds = (repo / "data" / "synthetic_bunny").string();
    Intrinsics Kb;
    std::ifstream f(ds + "/intrinsics.txt");
    double w, h;
    f >> w >> h >> Kb.fx >> Kb.fy >> Kb.cx >> Kb.cy;
    RgbdFrame f0, f1;
    CHECK(LoadRgbdFrame(ds, 0, f0) && LoadRgbdFrame(ds, 1, f1), "frames load");
    // The Stage-2 configuration of main (FAST cell fallback, original matcher, gate on); on the
    // integration branch PipelineParams{} is the teammate's legacy front end with the gate off.
    PipelineParams on, off;
    ApplyStage2Frontend(on);
    on.validity.enabled = true;
    off = on;
    off.validity.enabled = false;
    const FrameFeatures a = ExtractFeatures(f0.rgb, on.frontend), b = ExtractFeatures(f1.rgb, on.frontend);
    const PairResult r = ProcessPair(0, 1, a, b, f0, f1, Kb, on), r2 = ProcessPair(0, 1, a, b, f0, f1, Kb, on),
                     ro = ProcessPair(0, 1, a, b, f0, f1, Kb, off);
    CHECK(r.pnp_ok && r.pose_accepted && r.validity.accepted(),
          "Bunny 0->1 accepted: %d inlier rows, %d / %d distinct, %d / %d locations, %d / %d cells, median %.2f, "
          "p90 %.2f px, ratio %.3f", r.validity.inlier_rows, r.validity.distinct_src, r.validity.distinct_dst,
          r.validity.distinct_loc_src, r.validity.distinct_loc_dst, r.validity.cells_src, r.validity.cells_dst,
          r.validity.reproj_median_px, r.validity.reproj_p90_px, r.validity.inlier_ratio);
    CHECK(r.validity.inlier_rows == 217 && r.validity.distinct_dst == 151 && r.validity.distinct_loc_dst == 84,
          "gate counts equal the Stage-1 locked measurement (217 rows, 151 distinct, 84 locations in frame j)");
    CHECK(ro.R == r.R && ro.t == r.t && ro.pnp_mask == r.pnp_mask && ro.pnp_residual_px == r.pnp_residual_px &&
              ro.pose_accepted && ro.validity.status == r.validity.status,
          "the gate changes no estimate; disabled, the pose is accepted and the evaluation is still recorded");
    CHECK(r2.validity.status == r.validity.status && r2.validity.reason == r.validity.reason &&
              r2.validity.reproj_p90_px == r.validity.reproj_p90_px, "deterministic in the pipeline");
  }
  return scratch_test::Finish("test_validity");
}
