// Tests of the Stage-1 measurement harness (src/baseline.hpp): fixture loading,
// distinct-keypoint counting, duplicate detection, diagnostic pose validity,
// determinism, ground-truth isolation, pose conventions and the baseline lock.
// The pipeline runs only on the Bunny fixture (committed data, fast); the TUM
// fixtures are exercised by the scratch_baseline_lock test.

#include <cmath>

#include <unistd.h>

#include "baseline.hpp"
#include "check.hpp"

using namespace scratch;
using namespace scratch::baseline;

namespace {

fs::path TempFile(const std::string &name, const std::string &text) {
  const fs::path p = fs::temp_directory_path() / ("test_baseline_" + std::to_string(::getpid()) + "_" + name);
  std::ofstream(p) << text;
  return p;
}

Eigen::Matrix4d H(const Pose &T) {
  Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
  M.topLeftCorner<3, 3>() = T.R, M.topRightCorner<3, 1>() = T.t;
  return M;
}

// A PairResult with the given PnP inputs (query, train) and inlier mask; nothing else is needed by Measure.
PairResult FakePair(const std::vector<std::pair<int, int>> &qt, const std::vector<bool> &mask, bool ok) {
  PairResult r;
  r.i = 3, r.j = 4;
  for (size_t k = 0; k < qt.size(); ++k) {
    Correspondence c;
    c.query = qt[k].first, c.train = qt[k].second;
    c.uv_i = Eigen::Vector2d(10.0 * c.query, 0), c.uv_j = Eigen::Vector2d(10.0 * c.train, 0);
    c.has_3d = true;
    r.correspondences.push_back(c);
    r.pnp_index.push_back(int(k));
  }
  r.filtered_matches = int(qt.size());
  r.pnp_ok = ok;
  r.pnp_mask = mask;
  for (bool b : mask) r.pnp_inliers += b;
  return r;
}

}  // namespace

int main() {
  const fs::path repo = scratch_test::Repo();
  const fs::path fixtures_csv = repo / "tests/fixtures/baseline_pairs.csv";

  std::printf("=== fixture loading ===\n");
  std::vector<Fixture> fx;
  std::vector<FixtureGroundTruth> gt;
  std::string err;
  CHECK(LoadFixtures(fixtures_csv, fx, gt, &err), "fixtures load (%s)", err.c_str());
  CHECK(fx.size() == 3 && gt.size() == 3 && fx[0].name == "bunny_0_1" && fx[1].name == "tum_xyz_stride4_46_47" &&
            fx[2].name == "tum_xyz_stride4_73_74",
        "three fixtures in order: bunny_0_1, tum_xyz_stride4_46_47, tum_xyz_stride4_73_74");
  bool names_aligned = true;
  for (size_t k = 0; k < fx.size() && k < gt.size(); ++k) names_aligned &= fx[k].name == gt[k].name;
  CHECK(names_aligned, "ground truth is returned in a separate list aligned by name");
  CHECK(fx.size() == 3 && fx[1].frame_i == 46 && fx[1].frame_j == 47 && fx[2].frame_i == 73 && fx[2].frame_j == 74 &&
            fx[1].K.fx == 517.3 && fx[1].K.cy == 255.3 && fx[0].K.fx == 520.9 && fx[0].K.cy == 249.7,
        "frame indices and intrinsics");
  {
    const auto bunny_gt = ReadGroundtruthTwc(repo / "data/synthetic_bunny/groundtruth.txt");
    CHECK(gt.size() == 3 && gt[0].Twc_i.t == bunny_gt[0].t && gt[0].Twc_j.t == bunny_gt[1].t &&
              gt[0].Twc_i.R == bunny_gt[0].R && gt[0].Twc_j.R == bunny_gt[1].R,
          "Bunny fixture ground truth is identical to groundtruth.txt poses 0 and 1");
  }
  {
    std::ifstream f(fixtures_csv);
    std::string header;
    std::getline(f, header);
    std::string row;
    std::getline(f, row);
    std::vector<Fixture> a;
    std::vector<FixtureGroundTruth> b;
    CHECK(!LoadFixtures(TempFile("dup.csv", header + "\n" + row + "\n" + row + "\n"), a, b, &err) && a.empty(),
          "duplicate fixture names are rejected (%s)", err.c_str());
    CHECK(!LoadFixtures(TempFile("nocol.csv", "name,dataset\nx,bunny\n"), a, b, &err), "missing columns are rejected");
    CHECK(!LoadFixtures(TempFile("empty.csv", header + "\n"), a, b, &err), "a file without fixtures is rejected");
    CHECK(!LoadFixtures(repo / "no/such/file.csv", a, b, &err), "a missing file is rejected");
  }

  std::printf("\n=== distinct counting ===\n");
  CHECK(CountDistinct({}) == 0 && CountDistinct({4}) == 1 && CountDistinct({3, 3, 5, 1, 5, 3}) == 3,
        "CountDistinct: {} -> 0, {4} -> 1, {3,3,5,1,5,3} -> 3");
  {
    using V = Eigen::Vector2d;
    CHECK(CountDistinctLocations({}, 2.0) == 0, "no points -> 0 locations");
    CHECK(CountDistinctLocations({V(0, 0), V(2, 0), V(10, 0)}, 2.0) == 2, "points exactly at the radius merge");
    CHECK(CountDistinctLocations({V(0, 0), V(2.01, 0), V(10, 0)}, 2.0) == 3, "points just beyond the radius do not");
    CHECK(CountDistinctLocations({V(0, 0), V(1.5, 0), V(3, 0), V(4.5, 0)}, 2.0) == 1, "single linkage chains");
  }
  CHECK(CountSharedDestinationMatches({}) == 0 && CountSharedDestinationMatches({1, 2, 3}) == 0 &&
            CountSharedDestinationMatches({7, 7, 8, 9, 9, 9}) == 5,
        "shared destinations: {1,2,3} -> 0, {7,7,8,9,9,9} -> 5");

  std::printf("\n=== duplicate detection ===\n");
  {
    FrameFeatures f;
    auto add = [&](float x, float y, int level) {
      features::Keypoint k;
      k.x = x, k.y = y;
      f.keypoints.push_back(k);
      f.level.push_back(level);
    };
    add(10, 10, 0), add(11, 10, 1);     // same corner on two levels: both duplicates
    add(50, 50, 0), add(51, 50, 0);     // same level: not cross-level duplicates
    add(100, 100, 2), add(103, 100, 3); // 3 px apart: not duplicates
    add(200, 200, 0), add(201, 201, 4), add(199.5, 200, 7);  // three levels at one corner
    CHECK(CountCrossLevelNearDuplicates(f, 2.0) == 5, "cross-level near-duplicates: %d (expected 5)",
          CountCrossLevelNearDuplicates(f, 2.0));
  }
  {
    // 5 inlier rows on 3 distinct frame-j keypoints (two pairs share a destination), one outlier
    const PairResult r = FakePair({{0, 7}, {1, 7}, {2, 8}, {3, 9}, {4, 9}, {5, 20}}, {1, 1, 1, 1, 1, 0}, true);
    const PairMeasurement m = Measure("fake", r, FrameFeatures(), FrameFeatures());
    CHECK(m.inlier_rows == 5 && m.distinct_inlier_i == 5 && m.distinct_inlier_j == 3 && m.shared_dst_inliers == 4,
          "inlier rows 5, distinct 5 / 3, rows sharing a destination 4 (got %d, %d / %d, %d)", m.inlier_rows,
          m.distinct_inlier_i, m.distinct_inlier_j, m.shared_dst_inliers);
    CHECK(m.unique_src_filtered == 6 && m.unique_dst_filtered == 4 && m.shared_dst_filtered == 4,
          "filtered: unique src 6, unique dst 4, sharing 4");
    CHECK(m.solver_ok && !m.pose_valid && !m.validity_reason.empty(),
          "solver success and pose validity are separate: solved but invalid (%s)", m.validity_reason.c_str());
    const PairResult r2 = FakePair({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}}, std::vector<bool>(7, true), true);
    const PairMeasurement m2 = Measure("fake2", r2, FrameFeatures(), FrameFeatures());
    CHECK(m2.solver_ok && m2.pose_valid && m2.distinct_inlier_j == 7, "7 distinct inliers: valid");
    const PairResult r3 = FakePair({{0, 0}, {1, 1}}, {true, true}, false);
    const PairMeasurement m3 = Measure("fake3", r3, FrameFeatures(), FrameFeatures());
    CHECK(!m3.solver_ok && !m3.pose_valid && m3.inlier_rows == 0, "failed solver: invalid, no inlier rows counted");
    PairResult r4 = r2;
    r4.R(0, 0) = NAN;
    CHECK(!Measure("fake4", r4, FrameFeatures(), FrameFeatures()).pose_valid, "non-finite rotation: invalid");
  }

  std::printf("\n=== determinism and ground-truth isolation (Bunny 0 -> 1) ===\n");
  // main's Stage-2 configuration (the locked values describe it); on the integration branch
  // PipelineParams{} is the teammate's legacy front end with the gate off.
  PipelineParams params;
  ApplyStage2Frontend(params);
  params.validity.enabled = true;
  PairMeasurement m1, m2;
  PairResult p1, p2;
  err.clear();
  CHECK(fx.size() == 3 && MeasureFixture(fx[0], repo, params, m1, &err, &p1) &&
            MeasureFixture(fx[0], repo, params, m2, &err, &p2),
        "Bunny fixture measured twice (%s)", err.c_str());
  {
    const auto a = NumericFields(m1), b = NumericFields(m2);
    bool same = a.size() == b.size();
    for (size_t k = 0; same && k < a.size(); ++k)
      same = a[k].first == b[k].first && (a[k].second == b[k].second || (std::isnan(a[k].second) && std::isnan(b[k].second)));
    CHECK(same && p1.R == p2.R && p1.t == p2.t && p1.pnp_mask == p2.pnp_mask && p1.pnp_residual_px == p2.pnp_residual_px,
          "two runs are bit-identical (every measured field, R, t, inlier mask, residuals)");
  }
  CHECK(!m1.scored && std::isnan(m1.rot_err_deg) && std::isnan(m1.gt_motion_deg),
        "the estimator's measurement carries no ground-truth values before scoring");
  {
    PairMeasurement true_scored = m1, wrong_scored = m1;
    ScoreMeasurement(true_scored, gt[0]);
    FixtureGroundTruth wrong = gt[0];
    wrong.Twc_j.t += Vector3d(0.3, -0.2, 0.1);
    wrong.Twc_j.R = wrong.Twc_j.R * QuatToR(0.1, 0.2, 0.0, 1.0);
    ScoreMeasurement(wrong_scored, wrong);
    const auto a = NumericFields(true_scored), b = NumericFields(wrong_scored), c = NumericFields(m1);
    bool estimate_same = true;
    for (size_t k = 0; k < a.size(); ++k) {
      const std::string &f = a[k].first;
      if (f == "gt_motion_deg" || f == "gt_motion_m" || f == "rot_err_deg" || f == "trans_err_m") continue;
      estimate_same &= a[k].second == b[k].second && a[k].second == c[k].second;
    }
    CHECK(estimate_same, "scoring with the true or a wrong ground truth changes no estimated field");
    CHECK(std::abs(true_scored.rot_err_deg - wrong_scored.rot_err_deg) > 1.0,
          "only the scores change (rotation error %.3f vs %.3f deg)", true_scored.rot_err_deg, wrong_scored.rot_err_deg);
  }

  std::printf("\n=== pose conventions ===\n");
  {
    const Pose G = RelativeGroundTruth(gt[0].Twc_i, gt[0].Twc_j);
    const Eigen::Matrix4d Hrel = H(gt[0].Twc_j).inverse() * H(gt[0].Twc_i);
    CHECK((H(G) - Hrel).norm() < 1e-12, "RelativeGroundTruth = T_wc[j]^-1 T_wc[i] (4x4 check, diff %.1e)",
          (H(G) - Hrel).norm());
    CHECK(std::abs(RotAngleDeg(G.R) - 10.0) < 0.01 && std::abs(G.t.norm() - 0.0872) < 0.0005,
          "Bunny 0 -> 1 true motion: %.4f deg, %.5f m (orbit: 10 deg, 0.0872 m)", RotAngleDeg(G.R), G.t.norm());
    // trajectory accumulation (scratch_pipeline): T_wc[j] = T_wc[i] * T_{j<-i}^-1 recovers frame j
    const Eigen::Matrix4d acc = H(gt[0].Twc_i) * H(G).inverse();
    CHECK((acc - H(gt[0].Twc_j)).norm() < 1e-12, "accumulation T_wc[i] T_rel^-1 reproduces T_wc[j]");
    // the estimator's (R, t) maps frame-i points into frame j: inliers reproject with the TRUE T_{j<-i}
    std::vector<Vector3d> X;
    std::vector<Vector2d> uv;
    for (size_t k = 0; k < p1.pnp_index.size(); ++k)
      if (p1.pnp_mask[k]) X.push_back(p1.correspondences[p1.pnp_index[k]].X_i), uv.push_back(p1.correspondences[p1.pnp_index[k]].uv_j);
    const double right = Reprojection(X, uv, G, fx[0].K).median;
    Pose Ginv;
    Ginv.R = G.R.transpose(), Ginv.t = -G.R.transpose() * G.t;
    const double wrong = Reprojection(X, uv, Ginv, fx[0].K).median;
    // (the orbit's rotation and translation nearly cancel in the image, so the inverted
    // convention is only a few pixels off here: compare the two rather than use a fixed bound)
    CHECK(right < 3.0 && wrong > 3.0 * right,
          "inliers under the true T_{j<-i}: median %.2f px; inverted convention: %.2f px", right, wrong);
    PairMeasurement exact = m1;
    exact.R = G.R, exact.t = G.t;
    ScoreMeasurement(exact, gt[0]);
    CHECK(exact.rot_err_deg < 1e-6 && exact.trans_err_m < 1e-12, "an estimate equal to the truth scores zero error");
    const Eigen::Vector4d q = RToQuat(gt[0].Twc_j.R);
    CHECK((QuatToR(q(0), q(1), q(2), q(3)) - gt[0].Twc_j.R).norm() < 1e-12,
          "quaternion order (qx, qy, qz, qw) round trip");
  }

  std::printf("\n=== baseline lock comparator ===\n");
  {
    PairMeasurement m = m1;
    ScoreMeasurement(m, gt[0]);
    const fs::path p = TempFile("expected.csv", ExpectedCsv({m}));
    std::vector<Expected> e;
    CHECK(LoadExpected(p, e, &err) && e.size() == 1 && e[0].name == m.name, "lock CSV round trip loads");
    CHECK(!e.empty() && Compare(m, e[0]).empty(), "the measurement matches its own lock exactly (17 digits)");
    if (!e.empty()) {
      Expected changed = e[0];
      changed.values["inlier_rows"] += 1;
      const auto d = Compare(m, changed);
      CHECK(d.size() == 1 && d[0].field == "inlier_rows", "a changed count is detected (%s)",
            d.empty() ? "none" : d[0].field.c_str());
      Expected within = e[0];
      within.values["rot_err_deg"] += 0.5 * FieldTolerance("rot_err_deg");
      CHECK(Compare(m, within).empty(), "a difference inside the documented tolerance is accepted");
      Expected beyond = e[0];
      beyond.values["trans_err_m"] += 10 * FieldTolerance("trans_err_m");
      CHECK(Compare(m, beyond).size() == 1, "a difference beyond the tolerance is detected");
      Expected flag = e[0];
      flag.values["pose_valid"] = 1 - flag.values["pose_valid"];
      CHECK(Compare(m, flag).size() == 1, "a changed validity flag is detected");
      Expected missing = e[0];
      missing.values.erase("distinct_inlier_j");
      CHECK(Compare(m, missing).size() == 1, "a field missing from the lock is reported");
    }
    CHECK(FieldTolerance("inlier_rows") == 0 && FieldTolerance("solver_ok") == 0 && FieldTolerance("refine_iterations") == 0 &&
              FieldTolerance("rot_err_deg") == 1e-6 && FieldTolerance("trans_err_m") == 1e-8 && FieldTolerance("r12") == 1e-9,
          "counts and flags are exact; documented float tolerances");
  }
  std::printf("\n=== multi-seed evaluation (Stage 3) ===\n");
  {
    std::vector<uint32_t> sd;
    CHECK(ParseSeedList("12345,1,7-9", sd) && sd == std::vector<uint32_t>({12345, 1, 7, 8, 9}),
          "seed list with a range: 12345,1,7-9");
    CHECK(!ParseSeedList("1,1", sd, &err) && !ParseSeedList("3-1", sd) && !ParseSeedList("x", sd) &&
              !ParseSeedList("", sd) && !ParseSeedList("1,,2", sd),
          "duplicates, reversed ranges, non-numbers and empty entries are rejected (%s)", err.c_str());
    CHECK(DefaultSeedSweep(20).size() == 20 && DefaultSeedSweep(20)[0] == 12345 && DefaultSeedSweep(20)[19] == 19 &&
              DefaultSeedSweep(1) == std::vector<uint32_t>({12345}) && DefaultSeedSweep(0).empty(),
          "default sweep: the pipeline seed 12345 first, then 1..N-1");
    const SweepStat st = Summarize({4, 1, NAN, 3, 2, INFINITY});
    CHECK(st.n == 4 && st.median == 2.5 && std::abs(st.p95 - 3.85) < 1e-12 && st.max == 4 &&
              std::abs(st.stddev - std::sqrt(1.25)) < 1e-12,
          "summary: non-finite ignored; median 2.5, p95 3.85 (interpolated), max 4, population std sqrt(1.25)");
    CHECK(Summarize({}).n == 0 && std::isnan(Summarize({}).median), "summary of nothing: n 0, NaN");

    std::vector<SeedRun> a, b;
    const std::vector<uint32_t> seeds = {12345, 1, 2};
    CHECK(SweepFixture(fx[0], gt[0], repo, params, seeds, a, &err) && SweepFixture(fx[0], gt[0], repo, params, seeds, b, &err),
          "Bunny fixture swept over 3 seeds, twice");
    bool inputs = a.size() == 3, repeat = a.size() == b.size();
    for (size_t k = 0; k < a.size(); ++k) {
      inputs &= a[k].inputs_identical && a[k].pnp_inputs == a[0].pnp_inputs;
      if (repeat)
        repeat &= a[k].seed == b[k].seed && a[k].gate == b[k].gate && a[k].inlier_rows == b[k].inlier_rows &&
                  a[k].rot_err_deg == b[k].rot_err_deg && a[k].trans_err_m == b[k].trans_err_m;
    }
    CHECK(inputs, "PnP inputs identical for every seed (%d each)", a.empty() ? 0 : a[0].pnp_inputs);
    CHECK(repeat, "the sweep is reproducible (every field but the wall time)");
    PairMeasurement ref = m1;
    ScoreMeasurement(ref, gt[0]);
    CHECK(!a.empty() && a[0].seed == 12345 && a[0].inlier_rows == ref.gate.inlier_rows && a[0].accepted == ref.pose_accepted &&
              a[0].rot_err_deg == ref.rot_err_deg && a[0].trans_err_m == ref.trans_err_m,
          "seed 12345 reproduces the default single run (%d rows, %.6f deg)", ref.gate.inlier_rows, ref.rot_err_deg);
    FixtureGroundTruth wrong = gt[0];
    wrong.Twc_j.t += Vector3d(0.2, 0.1, -0.1);
    std::vector<SeedRun> w;
    SweepFixture(fx[0], wrong, repo, params, seeds, w, &err);
    bool only_scores = w.size() == a.size();
    for (size_t k = 0; only_scores && k < w.size(); ++k)
      only_scores = w[k].gate == a[k].gate && w[k].accepted == a[k].accepted && w[k].inlier_rows == a[k].inlier_rows &&
                    w[k].trans_err_m != a[k].trans_err_m;
    CHECK(only_scores, "a wrong ground truth changes only the errors, never a decision");
    const SeedSummary sm = SummarizeSweep("bunny", a);
    CHECK(sm.seeds == 3 && sm.accepted + sm.rejected + sm.solver_failed == 3 && sm.inputs_changed == 0 &&
              sm.rot_returned.n == sm.solver_ok,
          "sweep summary counts add up");
  }

  return scratch_test::Finish("test_baseline");
}
