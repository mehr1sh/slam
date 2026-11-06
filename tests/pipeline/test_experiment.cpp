// Tests of the Stage-3C experiment support (src/experiment.hpp): matcher presets, the RANSAC trace,
// offline gate diagnostics and relative support, a spatially coherent wrong consensus, degenerate 3D
// support. The production gate (validity.hpp) is only called, never changed.

#include <cmath>
#include <random>

#include "check.hpp"
#include "experiment.hpp"
#include "projection.hpp"
#include "ransac.hpp"

using namespace scratch;
using namespace scratch::experiment;

namespace {

const Intrinsics K{517.3, 516.5, 318.6, 255.3};

// A PairResult built by hand: PnP inputs (X_i, uv_i, uv_j, src, dst), a pose, an inlier mask; residuals and the
// production validity are computed exactly as ProcessPair does.
PairResult Build(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uvi,
                 const std::vector<Eigen::Vector2d> &uvj, const std::vector<int> &src, const std::vector<int> &dst,
                 const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double thr = 8.0) {
  PairResult r;
  r.pnp_ok = true, r.R = R, r.t = t;
  std::vector<ValidityRow> rows;
  for (size_t k = 0; k < X.size(); ++k) {
    Correspondence c;
    c.query = src[k], c.train = dst[k], c.uv_i = uvi[k], c.uv_j = uvj[k], c.X_i = X[k], c.has_3d = true;
    r.correspondences.push_back(c);
    r.pnp_index.push_back(int(k));
    double z;
    const double e = (projectPoint(X[k], R, t, K, &z) - uvj[k]).norm();
    r.pnp_residual_px.push_back(z > 0 ? e : INFINITY);
    r.pnp_mask.push_back(z > 0 && e < thr);
    r.pnp_inliers += r.pnp_mask.back();
    rows.push_back({src[k], dst[k], uvi[k], uvj[k], r.pnp_residual_px.back(), bool(r.pnp_mask.back())});
  }
  r.validity = EvaluatePose(true, "", R, t, rows, thr);
  r.pose_accepted = r.validity.accepted();
  return r;
}

Eigen::Vector3d Backproject(double u, double v, double z) {
  return Eigen::Vector3d((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z);
}

}  // namespace

int main() {
  const auto repo = scratch_test::Repo();
  std::string err;

  std::printf("=== matcher presets ===\n");
  {
    FrontendParams f;
    bool all = true;
    for (const std::string &id : MatcherConfigIds()) all &= ApplyMatcherConfig(id, f, &err) && !DescribeMatcherConfig(id).empty();
    CHECK(all && MatcherConfigIds().size() == 9, "nine presets: M0 M1 M2 M2n M3 M4 M5 M1L M3L");
    ApplyMatcherConfig("M0", f);
    CHECK(!f.ranked_matcher, "M0: the original matcher");
    ApplyMatcherConfig("M3", f);
    CHECK(f.ranked_matcher && f.matcher.mutual && !f.matcher.mutual_location_aware && f.matcher.ratio == 0.8 &&
              f.matcher.same_location_radius_px == 3.0f && f.matcher.max_level_diff == -1 && !f.matcher.rank,
          "M3: strict mutual + location-aware ratio 0.8");
    ApplyMatcherConfig("M2n", f);
    CHECK(f.matcher.ratio == 0.8 && f.matcher.same_location_radius_px == 0.0f && !f.matcher.mutual, "M2n: naive ratio");
    ApplyMatcherConfig("M4", f);
    const bool m4 = f.matcher.max_level_diff == 1 && f.matcher.mutual;
    ApplyMatcherConfig("M5", f);
    const bool m5 = f.matcher.rank && f.matcher.mutual && f.matcher.ratio == 0.8;
    ApplyMatcherConfig("M3L", f);
    CHECK(m4 && m5 && f.matcher.mutual_location_aware && f.matcher.ratio == 0.8, "M4 level <= 1, M5 ranked, M3L location-aware");
    CHECK(!ApplyMatcherConfig("M9", f, &err), "unknown preset rejected (%s)", err.c_str());
  }

  std::printf("\n=== integration configurations (branch integration/teammate) ===\n");
  {
    PipelineParams t0, ref, m0, l0;
    ApplyImprovedDefaults(ref);
    CHECK(ApplyIntegrationConfig("T0", t0) && t0.frontend.selection.harris == ref.frontend.selection.harris &&
              t0.frontend.selection.total_features == 1400 && t0.frontend.fast.threshold == 10 &&
              t0.frontend.mutual_ratio && t0.frontend.ratio == ref.frontend.ratio && t0.frontend.depth_edge_m == 0.012 &&
              t0.refine_params.huber_delta_px == 2.0 && t0.reselect_inliers && t0.pnp_ransac.iterations == 650 &&
              t0.pnp_ransac.threshold_px == 3.0 && !t0.validity.enabled && !t0.frontend.ranked_matcher &&
              t0.frontend.fast.cell_size == 0,
          "T0 = the teammate defaults (ApplyImprovedDefaults), gate off, no cell fallback");
    CHECK(ApplyIntegrationConfig("M0", m0) && m0.frontend.fast.threshold == 20 && m0.frontend.fast.cell_size == 30 &&
              m0.frontend.fast.min_threshold == 7 && !m0.frontend.selection.harris && !m0.frontend.mutual_ratio &&
              m0.validity.enabled && m0.pnp_ransac.iterations == 300 && m0.pnp_ransac.threshold_px == 8.0 &&
              m0.refine_params.huber_delta_px == 0 && !m0.reselect_inliers,
          "M0 = main's Stage-2 configuration (cell fallback 30/7, original matcher, 300 it / 8 px, gate on)");
    CHECK(ApplyIntegrationConfig("L0", l0) && !l0.frontend.selection.harris && l0.frontend.fast.cell_size == 0 &&
              !l0.validity.enabled && l0.pnp_ransac.iterations == 300,
          "L0 = PipelineParams{} (the teammate's --legacy)");
    int single_changes = 0;
    for (const char *id : {"T0-harris", "T0-budget", "T0-fast20", "T0-mr", "T0-reselect", "T0E", "T0H"}) {
      PipelineParams p;
      ApplyIntegrationConfig(id, p);
      const int changed = (p.frontend.selection.harris != t0.frontend.selection.harris) +
                          (p.frontend.selection.total_features != t0.frontend.selection.total_features) +
                          (p.frontend.fast.threshold != t0.frontend.fast.threshold) +
                          (p.frontend.mutual_ratio != t0.frontend.mutual_ratio) + (p.reselect_inliers != t0.reselect_inliers) +
                          (p.frontend.depth_edge_m != t0.frontend.depth_edge_m) +
                          (p.refine_params.huber_delta_px != t0.refine_params.huber_delta_px);
      single_changes += changed == 1;
    }
    CHECK(single_changes == 7, "each leave-one-out configuration changes exactly one component of T0 (%d / 7)",
          single_changes);
    bool all = true;
    for (const std::string &id : IntegrationConfigIds()) {
      PipelineParams p;
      all &= ApplyIntegrationConfig(id, p, &err);
    }
    PipelineParams bad;
    CHECK(all && !ApplyIntegrationConfig("T9", bad, &err) && !ApplyIntegrationConfig("X", bad, &err),
          "all %zu configuration ids apply; unknown ids are rejected", IntegrationConfigIds().size());
  }

  std::printf("\n=== every preset on Bunny 0 -> 1, RANSAC trace = production ===\n");
  {
    const std::string ds = (repo / "data" / "synthetic_bunny").string();
    Intrinsics Kb;
    std::ifstream fi(ds + "/intrinsics.txt");
    double w, h;
    fi >> w >> h >> Kb.fx >> Kb.fy >> Kb.cx >> Kb.cy;
    RgbdFrame a, b;
    CHECK(LoadRgbdFrame(ds, 0, a) && LoadRgbdFrame(ds, 1, b), "frames load");
    PipelineParams def;  // main's Stage-2 configuration (= integration preset M0)
    ApplyIntegrationConfig("M0", def);
    const FrameFeatures fa = ExtractFeatures(a.rgb, def.frontend), fb = ExtractFeatures(b.rgb, def.frontend);
    const PairResult r0 = ProcessPair(0, 1, fa, fb, a, b, Kb, def);
    for (const std::string &id : MatcherConfigIds()) {
      PipelineParams p;
      ApplyIntegrationConfig(id, p);
      const PairResult r = ProcessPair(0, 1, fa, fb, a, b, Kb, p), r2 = ProcessPair(0, 1, fa, fb, a, b, Kb, p);
      std::vector<Eigen::Vector3d> X;
      std::vector<Eigen::Vector2d> uv;
      for (int q : r.pnp_index) X.push_back(r.correspondences[q].X_i), uv.push_back(r.correspondences[q].uv_j);
      bool trace = true;
      for (uint32_t seed : {12345u, 1u, 7u}) {
        PipelineParams ps = p;
        ps.pnp_ransac.seed = seed;
        trace &= TraceMatches(TraceRansacPnP(X, uv, Kb, ps.pnp_ransac), ProcessPair(0, 1, fa, fb, a, b, Kb, ps));
      }
      const features::MatchCounters &c = r.match_counters;
      const bool counters = id == "M0" || c.raw == c.kept + c.removed_distance + c.removed_mutual + c.removed_ratio + c.removed_level;
      const bool same_as_default = id != "M0" || (r.R == r0.R && r.t == r0.t && r.pnp_mask == r0.pnp_mask);
      CHECK(r.pose_accepted && trace && counters && same_as_default && r2.R == r.R && r2.pnp_mask == r.pnp_mask,
            "%-3s: accepted, %d PnP inputs, %d inlier rows; deterministic; trace reproduces RansacPnP for 3 seeds%s",
            id.c_str(), int(r.pnp_index.size()), r.pnp_inliers, id == "M0" ? "; identical to the default" : "");
    }
  }

  std::printf("\n=== gate diagnostics and relative support ===\n");
  {
    // 100 inputs spread over the image; the first 40 are inliers of the identity pose, the rest 50 px off.
    // Inputs 0..9 are duplicated (rows 100..109: same keypoints and pixels).
    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector2d> ui, uj;
    std::vector<int> src, dst;
    for (int k = 0; k < 110; ++k) {
      const int b = k < 100 ? k : k - 100;
      const double u = 30 + 58 * (b % 10), v = 30 + 42 * (b / 10);
      X.push_back(Backproject(u, v, 1.5));
      ui.emplace_back(u, v);
      uj.emplace_back(u + (b < 40 ? 0.5 : 50.0), v);
      src.push_back(b), dst.push_back(1000 + b);
    }
    const PairResult r = Build(X, ui, uj, src, dst, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    const PoseValidity before = r.validity;
    const GateDiagnostics d = DiagnoseGate(r, ValidityParams(), 8.0);
    CHECK(d.v.inlier_rows == 50 && d.v.distinct_loc_dst == 40 && d.input_pairs == 100 && d.inlier_pairs == 40,
          "50 inlier rows (10 duplicated) = 40 locations; 110 input rows = 100 location pairs, 40 inlier pairs");
    CHECK(std::abs(d.ratio_rows - 50.0 / 110) < 1e-12 && std::abs(d.ratio_pairs - 0.4) < 1e-12 &&
              std::abs(d.ratio_loc_dst - 0.4) < 1e-12 && std::abs(d.rows_per_location - 1.25) < 1e-12,
          "relative support: rows %.4f (inflated by duplicates), pairs 0.40, locations 0.40; 1.25 rows per location",
          d.ratio_rows);
    const bool all_pass = d.c_finite && d.c_so3 && d.c_rows && d.c_kp_src && d.c_kp_dst && d.c_loc_src && d.c_loc_dst &&
                          d.c_ratio && d.c_cells_src && d.c_cells_dst && d.c_median && d.c_p90;
    CHECK(all_pass == d.v.accepted() && d.v.accepted(), "every individual check passes exactly when the gate accepts");
    CHECK(r.validity.status == before.status && r.validity.reason == before.reason &&
              r.validity.reproj_p90_px == before.reproj_p90_px,
          "diagnosing does not change the production evaluation");
    CHECK(d.largest_cell_share > 0 && d.largest_cell_share <= 1 && std::abs(d.depth_median - 1.5) < 1e-12 &&
              d.depth_min == 1.5 && d.depth_max == 1.5,
          "cell share in (0, 1]; depths of the inlier points");
    CHECK(d.planarity < 1e-12, "all points at one depth: planar support (planarity %.2g)", d.planarity);
    PairResult few = r;
    few.validity.inlier_rows = 10;
    const GateDiagnostics f = DiagnoseGate(few, ValidityParams(), 8.0);
    CHECK(!f.c_rows && f.c_loc_dst, "each check is evaluated separately (rows fail, locations pass)");
  }

  std::printf("\n=== 3D conditioning ===\n");
  {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<Eigen::Vector3d> cube, plane, line;
    for (int k = 0; k < 500; ++k) {
      cube.emplace_back(U(rng), U(rng), 2 + U(rng));
      plane.emplace_back(U(rng), U(rng), 2.0);
      const double s = U(rng);
      line.emplace_back(s, 0.5 * s, 2 + 0.2 * s);
    }
    const Eigen::Vector3d ec = CovarianceEigenvalues(cube), ep = CovarianceEigenvalues(plane), el = CovarianceEigenvalues(line);
    CHECK(std::sqrt(ec(2) / ec(0)) < 1.3, "uniform cube: condition %.2f (about 1)", std::sqrt(ec(2) / ec(0)));
    CHECK(ep(0) / ep(2) < 1e-12 && ep(1) > 0.2, "plane: smallest eigenvalue 0 relative to the largest (%.1e)", ep(0));
    CHECK(el(0) / el(2) < 1e-12 && el(1) / el(2) < 1e-12 && el(2) > 0.3,
          "line: two eigenvalues 0 relative to the largest (%.1e, %.1e)", el(0), el(1));
    CHECK(CovarianceEigenvalues({}).isZero() && CovarianceEigenvalues({Eigen::Vector3d(1, 2, 3)}).isZero(),
          "fewer than two points: zeros");
  }

  std::printf("\n=== spatially coherent wrong consensus (the 73 -> 74 failure mode) ===\n");
  {
    // 1500 PnP inputs. 60 of them, inside a 160 x 120 px region, agree with a WRONG pose W (a coherent set of
    // mismatches); 1100 agree with the true motion; the rest are random. Evaluated at W:
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> U(0, 1);
    Pose T, W;
    T.R = QuatToR(0.004, -0.008, 0.002, 1), T.t = Eigen::Vector3d(0.03, 0.0, 0.04);
    W.R = QuatToR(0.15, 0.05, -0.1, 1), W.t = Eigen::Vector3d(0.4, -0.3, 0.2);
    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector2d> ui, uj;
    std::vector<int> src, dst;
    for (int k = 0; k < 1500; ++k) {
      const bool coherent = k < 60;
      const double u = coherent ? 380 + 160 * U(rng) : 20 + 600 * U(rng), v = coherent ? 120 + 120 * U(rng) : 20 + 440 * U(rng);
      const Eigen::Vector3d x = Backproject(u, v, 1.0 + 1.5 * U(rng));
      Eigen::Vector2d p;
      if (coherent) p = projectPoint(x, W.R, W.t, K) + Eigen::Vector2d(U(rng) - 0.5, U(rng) - 0.5);
      else if (k < 1160) p = projectPoint(x, T.R, T.t, K) + Eigen::Vector2d(U(rng) - 0.5, U(rng) - 0.5);
      else p = Eigen::Vector2d(20 + 600 * U(rng), 20 + 440 * U(rng));
      X.push_back(x), ui.emplace_back(u, v), uj.push_back(p), src.push_back(k), dst.push_back(k);
    }
    const PairResult wrong = Build(X, ui, uj, src, dst, W.R, W.t), right = Build(X, ui, uj, src, dst, T.R, T.t);
    const GateDiagnostics dw = DiagnoseGate(wrong, ValidityParams(), 8.0), dr = DiagnoseGate(right, ValidityParams(), 8.0);
    CHECK(wrong.validity.accepted(),
          "the UNCHANGED Stage-2 gate accepts the wrong pose: %d rows, %d / %d locations, %d / %d cells, median %.2f px",
          wrong.validity.inlier_rows, wrong.validity.distinct_loc_src, wrong.validity.distinct_loc_dst,
          wrong.validity.cells_src, wrong.validity.cells_dst, wrong.validity.reproj_median_px);
    CHECK(dw.ratio_pairs < 0.06 && dr.ratio_pairs > 0.7,
          "relative support separates them: wrong %.3f vs right %.3f of the input location pairs", dw.ratio_pairs,
          dr.ratio_pairs);
    CHECK(dw.largest_cell_share > dr.largest_cell_share,
          "the wrong consensus is spatially concentrated (largest cell share %.3f vs %.3f)", dw.largest_cell_share,
          dr.largest_cell_share);
  }
  return scratch_test::Finish("test_experiment");
}
