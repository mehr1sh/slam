// scratch_baseline: Stage-1 measurement and regression harness (see baseline.hpp).
// Runs the unchanged scratch pipeline stages on fixed frame pairs, measures every
// stage, scores the pose against ground truth, writes a machine-readable report
// and compares it with the locked baseline. Standard library + Eigen + scratch only.
//
// Usage: scratch_baseline [--repo DIR] [--fixtures CSV] [--expected CSV] [--out DIR]
//                         [--only NAME] [--write-expected CSV] [--ate RUN_DIR ...]
//   --fixtures CSV        fixture list (default tests/fixtures/baseline_pairs.csv)
//   --expected CSV        locked values (default tests/fixtures/baseline_expected.csv);
//                         "none" = measure and report without comparing
//   --out DIR             report directory (default results/runs/scratch_baseline/stage1; relative paths are
//                         relative to the repository root); writes baseline_pairs.json and .csv
//   --only NAME           run one fixture
//   --write-expected CSV  write the measured values as a new lock file (refuses to overwrite)
//   --expected-gate CSV   Stage-2 pose-validity gate lock (default
//                         tests/fixtures/baseline_expected_gate.csv; "none" = not compared)
//   --write-expected-gate CSV  write the gate fields as a new gate lock file (refuses to overwrite)
//   --ate RUN_DIR         raw ATE of an existing scratch_pipeline run (no re-estimation); repeatable;
//                         with --ate-expected M checks the last one against M (tolerance 1e-6 m)
//   --seeds LIST | --seed-count N   multi-seed evaluation instead of the lock comparison: every fixture is
//                         processed once per RANSAC seed (LIST e.g. "12345,1-19"; N = {12345, 1, ..., N-1}),
//                         features and PnP inputs fixed (checked); writes <seed-out>/seed_runs.csv (one row per
//                         fixture and seed), seed_summary.csv and seed_summary.json; exit 1 if PnP inputs changed
//   --seed-out DIR        default results/runs/scratch_baseline/stage3_seeds
//   --ranked-matcher, --match-mutual, --match-ratio R, --match-max-level-diff N, --match-rank,
//   --match-same-location-radius PX, --match-mutual-location-aware, --matcher-config ID
//                         matcher options as in scratch_pipeline (all off by default)
// Exit status: 0 all fixtures match; 1 a mismatch or error; 77 the comparison passed for every
// fixture whose data exists, but some fixture's images are missing (e.g. the TUM sequence is
// not downloaded) -- ctest reports that as skipped.

#include <cstdio>
#include <iostream>

#include "baseline.hpp"
#include "experiment.hpp"

using namespace scratch;
using namespace scratch::baseline;

int main(int argc, char **argv) {
  fs::path repo, fixtures_csv, expected_csv, out_arg, write_expected, expected_gate_csv, write_expected_gate;
  std::string only;
  std::vector<fs::path> ate_runs;
  double ate_expected = NAN;
  std::vector<uint32_t> seeds;
  fs::path seed_out;
  // The Stage-1/2 locks describe main's Stage-2 configuration (FAST cell fallback, original matcher, gate on);
  // on the integration branch that is no longer PipelineParams{}, so it is set explicitly.
  PipelineParams params;
  ApplyStage2Frontend(params);
  params.validity.enabled = true;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto next = [&] { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
    if (s == "--repo") repo = next();
    else if (s == "--fixtures") fixtures_csv = next();
    else if (s == "--expected") expected_csv = next();
    else if (s == "--out") out_arg = next();
    else if (s == "--only") only = next();
    else if (s == "--write-expected") write_expected = next();
    else if (s == "--expected-gate") expected_gate_csv = next();
    else if (s == "--write-expected-gate") write_expected_gate = next();
    else if (s == "--ate") ate_runs.push_back(next());
    else if (s == "--ate-expected") ate_expected = std::stod(next());
    else if (s == "--seeds") {
      std::string e;
      if (!ParseSeedList(next(), seeds, &e)) {
        std::cerr << e << "\n";
        return 2;
      }
    } else if (s == "--seed-count") {
      seeds = DefaultSeedSweep(std::stoi(next()));
    } else if (s == "--seed-out") seed_out = next();
    else if (s == "--config") {  // a complete pipeline configuration (experiment.hpp IntegrationConfigIds)
      std::string e;
      if (!experiment::ApplyIntegrationConfig(next(), params, &e)) {
        std::cerr << e << "\n";
        return 2;
      }
    }
    else if (s == "--ranked-matcher") params.frontend.ranked_matcher = true;
    else if (s == "--match-mutual") params.frontend.ranked_matcher = params.frontend.matcher.mutual = true;
    else if (s == "--match-mutual-location-aware")
      params.frontend.ranked_matcher = params.frontend.matcher.mutual = params.frontend.matcher.mutual_location_aware = true;
    else if (s == "--matcher-config") {
      std::string e;
      if (!experiment::ApplyMatcherConfig(next(), params.frontend, &e)) {
        std::cerr << e << "\n";
        return 2;
      }
    }
    else if (s == "--match-ratio") params.frontend.ranked_matcher = true, params.frontend.matcher.ratio = std::stod(next());
    else if (s == "--match-max-level-diff")
      params.frontend.ranked_matcher = true, params.frontend.matcher.max_level_diff = std::stoi(next());
    else if (s == "--match-rank") params.frontend.ranked_matcher = params.frontend.matcher.rank = true;
    else if (s == "--match-same-location-radius")
      params.frontend.ranked_matcher = true, params.frontend.matcher.same_location_radius_px = std::stof(next());
    else {
      std::cerr << "unknown option " << s << "\n";
      return 2;
    }
  }
  if (repo.empty()) repo = FindRepo(fs::current_path());
  if (repo.empty()) repo = FindRepo(fs::path(argv[0]).parent_path());
  if (repo.empty()) {
    std::cerr << "cannot find the repository root; pass --repo DIR\n";
    return 2;
  }
  auto abs = [&](const fs::path &p) { return p.is_absolute() ? p : repo / p; };

  // ---- trajectory ATE of existing runs (read-only) ----
  if (!ate_runs.empty()) {
    int bad = 0;
    for (const auto &run : ate_runs) {
      const AteResult r = RawAte(abs(run), repo);
      if (!r.ok) {
        std::cerr << run.string() << ": " << r.error << "\n";
        ++bad;
        continue;
      }
      std::printf("%-60s frames %4d  raw ATE RMSE %.9f m  final %.6f m\n", run.string().c_str(), r.frames,
                  r.ate_raw_rmse_m, r.final_error_m);
      if (&run == &ate_runs.back() && std::isfinite(ate_expected) && std::abs(r.ate_raw_rmse_m - ate_expected) > 1e-6) {
        std::printf("  MISMATCH: expected %.9f m (tolerance 1e-6 m)\n", ate_expected);
        ++bad;
      }
    }
    return bad ? 1 : 0;
  }

  if (fixtures_csv.empty()) fixtures_csv = "tests/fixtures/baseline_pairs.csv";
  if (expected_csv.empty()) expected_csv = "tests/fixtures/baseline_expected.csv";
  if (expected_gate_csv.empty()) expected_gate_csv = "tests/fixtures/baseline_expected_gate.csv";
  const fs::path out = abs(out_arg.empty() ? fs::path("results/runs/scratch_baseline/stage1") : out_arg);

  std::vector<Fixture> fixtures;
  std::vector<FixtureGroundTruth> gts;
  std::string err;
  if (!LoadFixtures(abs(fixtures_csv), fixtures, gts, &err)) {
    std::cerr << err << "\n";
    return 1;
  }
  // ---- multi-seed evaluation ----
  if (!seeds.empty()) {
    const fs::path dir = abs(seed_out.empty() ? fs::path("results/runs/scratch_baseline/stage3_seeds") : seed_out);
    fs::create_directories(dir);
    std::ofstream runs_csv(dir / "seed_runs.csv"), sum_csv(dir / "seed_summary.csv"), sum_json(dir / "seed_summary.json");
    runs_csv << "fixture,seed,inputs_identical,pnp_inputs,solver_ok,gate_status,accepted,inlier_rows,distinct_loc_dst,"
                "rot_err_deg,trans_err_m,seconds,reason\n";
    sum_csv << "fixture,seeds,solver_ok,accepted,rejected,solver_failed,inputs_changed,returned_over_2deg";
    for (const char *q : {"rot_returned", "trans_returned", "rot_accepted", "trans_accepted", "seconds"})
      for (const char *st : {"n", "median", "p95", "max", "std"}) sum_csv << "," << q << "_" << st;
    sum_csv << "\n";
    auto num = [](double v) {
      char b[40];
      std::snprintf(b, sizeof(b), "%.9g", v);
      return std::isfinite(v) ? std::string(b) : std::string();
    };
    auto jnum = [&](double v) { return std::isfinite(v) ? num(v) : std::string("null"); };
    sum_json << "{\n  \"tool\": \"scratch_baseline --seeds\",\n  \"seeds\": [";
    for (size_t k = 0; k < seeds.size(); ++k) sum_json << (k ? ", " : "") << seeds[k];
    sum_json << "],\n  \"ranked_matcher\": " << (params.frontend.ranked_matcher ? "true" : "false")
             << ",\n  \"fixtures\": [\n";
    int missing = 0, changed = 0, written = 0;
    std::printf("multi-seed evaluation: %zu seeds per fixture -> %s\n", seeds.size(), dir.string().c_str());
    std::printf("%-24s %5s %4s %4s %4s %5s | rot(returned) med   p95   max   std | trans med    p95    max | >2deg | s/run\n",
                "fixture", "seeds", "acc", "rej", "fail", "same");
    for (size_t f = 0; f < fixtures.size(); ++f) {
      if (!only.empty() && fixtures[f].name != only) continue;
      std::vector<SeedRun> runs;
      if (!SweepFixture(fixtures[f], gts[f], repo, params, seeds, runs, &err)) {
        std::printf("%-24s SKIPPED: %s\n", fixtures[f].name.c_str(), err.c_str());
        ++missing;
        continue;
      }
      for (const SeedRun &r : runs) {
        std::string reason = r.reason;
        for (char &c : reason)
          if (c == ',') c = ' ';
        runs_csv << r.name << "," << r.seed << "," << int(r.inputs_identical) << "," << r.pnp_inputs << ","
                 << int(r.solver_ok) << "," << PoseStatusName(r.gate) << "," << int(r.accepted) << "," << r.inlier_rows
                 << "," << r.distinct_loc_dst << "," << num(r.rot_err_deg) << "," << num(r.trans_err_m) << ","
                 << num(r.seconds) << "," << reason << "\n";
      }
      const SeedSummary sm = SummarizeSweep(fixtures[f].name, runs);
      changed += sm.inputs_changed;
      sum_csv << sm.name << "," << sm.seeds << "," << sm.solver_ok << "," << sm.accepted << "," << sm.rejected << ","
              << sm.solver_failed << "," << sm.inputs_changed << "," << sm.returned_over_2deg;
      sum_json << (written++ ? ",\n" : "") << "    {\"fixture\": \"" << sm.name << "\", \"seeds\": " << sm.seeds
               << ", \"solver_ok\": " << sm.solver_ok << ", \"accepted\": " << sm.accepted << ", \"rejected\": "
               << sm.rejected << ", \"solver_failed\": " << sm.solver_failed << ", \"inputs_changed\": "
               << sm.inputs_changed << ", \"returned_over_2deg\": " << sm.returned_over_2deg;
      const std::pair<const char *, const SweepStat *> stats[] = {{"rot_returned_deg", &sm.rot_returned},
                                                                  {"trans_returned_m", &sm.trans_returned},
                                                                  {"rot_accepted_deg", &sm.rot_accepted},
                                                                  {"trans_accepted_m", &sm.trans_accepted},
                                                                  {"seconds", &sm.seconds}};
      for (const auto &[key, st] : stats) {
        sum_csv << "," << st->n << "," << num(st->median) << "," << num(st->p95) << "," << num(st->max) << ","
                << num(st->stddev);
        sum_json << ", \"" << key << "\": {\"n\": " << st->n << ", \"median\": " << jnum(st->median)
                 << ", \"p95\": " << jnum(st->p95) << ", \"max\": " << jnum(st->max) << ", \"std\": "
                 << jnum(st->stddev) << "}";
      }
      sum_csv << "\n";
      sum_json << "}";
      std::printf("%-24s %5d %4d %4d %4d %5s | %13.3f %5.3f %5.3f %5.3f | %8.4f %6.4f %6.4f | %5d | %.2f\n",
                  sm.name.c_str(), sm.seeds, sm.accepted, sm.rejected, sm.solver_failed,
                  sm.inputs_changed ? "NO" : "yes", sm.rot_returned.median, sm.rot_returned.p95, sm.rot_returned.max,
                  sm.rot_returned.stddev, sm.trans_returned.median, sm.trans_returned.p95, sm.trans_returned.max,
                  sm.returned_over_2deg, sm.seconds.median);
    }
    sum_json << "\n  ]\n}\n";
    std::printf("wrote seed_runs.csv, seed_summary.csv, seed_summary.json\n");
    if (changed) {
      std::printf("ERROR: PnP inputs changed between seeds in %d run(s)\n", changed);
      return 1;
    }
    return missing ? 77 : 0;
  }

  std::vector<Expected> expected;
  const bool compare = expected_csv != "none";
  if (compare && !LoadExpected(abs(expected_csv), expected, &err)) {
    std::cerr << err << "\n";
    return 1;
  }

  std::vector<Expected> expected_gate;
  const bool compare_gate = expected_gate_csv != "none";
  if (compare_gate && !LoadExpected(abs(expected_gate_csv), expected_gate, &err)) {
    std::cerr << err << "\n";
    return 1;
  }
  std::vector<PairMeasurement> measured;
  std::vector<std::string> json_entries;
  int mismatches = 0, missing = 0;
  for (size_t f = 0; f < fixtures.size(); ++f) {
    const Fixture &fx = fixtures[f];
    if (!only.empty() && fx.name != only) continue;
    std::printf("=== %s (%s, frames %d -> %d) ===\n", fx.name.c_str(), fx.dataset.c_str(), fx.frame_i, fx.frame_j);
    PairMeasurement m;
    if (!MeasureFixture(fx, repo, params, m, &err)) {  // estimation: no ground truth involved
      std::printf("  SKIPPED: %s\n", err.c_str());
      ++missing;
      continue;
    }
    ScoreMeasurement(m, gts[f]);  // ground truth: scoring only
    std::printf("  keypoints %d / %d (cross-level near-duplicates %d / %d)\n", m.keypoints_i, m.keypoints_j,
                m.cross_level_dup_i, m.cross_level_dup_j);
    std::printf("  matches raw %d, filtered %d (unique src %d, unique dst %d, sharing a dst %d)\n", m.raw_matches,
                m.filtered_matches, m.unique_src_filtered, m.unique_dst_filtered, m.shared_dst_filtered);
    std::printf("  PnP inputs %d, inlier rows %d, distinct inlier keypoints %d / %d (locations %d / %d)\n",
                m.pnp_inputs, m.inlier_rows, m.distinct_inlier_i, m.distinct_inlier_j, m.distinct_inlier_loc_i,
                m.distinct_inlier_loc_j);
    std::printf("  solver %s; pose validity (diagnostic) %s%s\n", m.solver_ok ? "ok" : "FAILED",
                m.pose_valid ? "valid" : "INVALID", m.validity_reason.empty() ? "" : (": " + m.validity_reason).c_str());
    std::printf("  Stage-2 gate: %s%s\n", PoseStatusName(m.gate.status),
                m.gate.reason.empty() || m.gate.status == PoseStatus::kAccepted ? "" : (": " + m.gate.reason).c_str());
    std::printf("  error vs GT: rotation %.6f deg, translation %.6f m (true motion %.3f deg, %.4f m)\n",
                m.rot_err_deg, m.trans_err_m, m.gt_motion_deg, m.gt_motion_m);

    std::string verdict = "not compared";
    std::vector<Difference> diffs;
    if (compare) {
      const Expected *e = nullptr;
      for (const auto &x : expected)
        if (x.name == fx.name) e = &x;
      if (!e) {
        verdict = "no locked baseline";
        ++mismatches;
      } else {
        diffs = Compare(m, *e);
        verdict = diffs.empty() ? "matches the locked baseline" : "DIFFERS from the locked baseline";
        mismatches += !diffs.empty();
      }
      std::printf("  baseline: %s\n", verdict.c_str());
      for (const auto &d : diffs)
        std::printf("    %-24s expected %.17g, got %.17g (tolerance %g)\n", d.field.c_str(), d.expected, d.actual,
                    d.tolerance);
    }
    std::string gate_verdict = "not compared";
    std::vector<Difference> gdiffs;
    if (compare_gate) {
      const Expected *e = nullptr;
      for (const auto &x : expected_gate)
        if (x.name == fx.name) e = &x;
      if (!e) {
        gate_verdict = "no locked gate baseline";
        ++mismatches;
      } else {
        gdiffs = CompareGate(m, *e);
        gate_verdict = gdiffs.empty() ? "matches the locked gate baseline" : "DIFFERS from the locked gate baseline";
        mismatches += !gdiffs.empty();
      }
      std::printf("  gate baseline: %s\n", gate_verdict.c_str());
      for (const auto &d : gdiffs)
        std::printf("    %-24s expected %.17g, got %.17g (tolerance %g)\n", d.field.c_str(), d.expected, d.actual,
                    d.tolerance);
    }
    const ExportCheck ec = CheckAgainstExport(m, fx, repo);
    if (ec.available) {
      std::printf("  existing export %s pair %d: %s (%zu fields compared)\n", fx.export_dir.c_str(), fx.export_pair,
                  ec.problems.empty() ? "reproduced" : "NOT reproduced", ec.compared.size());
      for (const auto &p : ec.problems) std::printf("    %s\n", p.c_str());
    } else if (!fx.export_dir.empty()) {
      std::printf("  existing export %s: not available (skipped)\n", fx.export_dir.c_str());
    }

    std::ostringstream j;
    j << "    {\n      \"measurement\": " << ToJson(m, 6) << ",\n      \"baseline\": {\"compared\": "
      << (compare ? "true" : "false") << ", \"verdict\": \"" << verdict << "\", \"differences\": [";
    for (size_t d = 0; d < diffs.size(); ++d) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%s{\"field\": \"%s\", \"expected\": %.17g, \"actual\": %.17g, \"tolerance\": %g}",
                    d ? ", " : "", diffs[d].field.c_str(), diffs[d].expected, diffs[d].actual, diffs[d].tolerance);
      j << buf;
    }
    j << "]},\n      \"gate_baseline\": {\"compared\": " << (compare_gate ? "true" : "false") << ", \"verdict\": \""
      << gate_verdict << "\", \"differing_fields\": [";
    for (size_t d = 0; d < gdiffs.size(); ++d) j << (d ? ", " : "") << "\"" << gdiffs[d].field << "\"";
    j << "]},\n      \"export_check\": {\"available\": " << (ec.available ? "true" : "false")
      << ", \"export_dir\": \"" << fx.export_dir << "\", \"export_pair\": " << fx.export_pair
      << ", \"fields_compared\": " << ec.compared.size() << ", \"problems\": [";
    for (size_t p = 0; p < ec.problems.size(); ++p) j << (p ? ", " : "") << "\"" << ec.problems[p] << "\"";
    j << "]}\n    }";
    json_entries.push_back(j.str());
    mismatches += ec.available && !ec.problems.empty();
    measured.push_back(m);
  }

  // ---- report ----
  fs::create_directories(out);
  {
    std::ofstream f(out / "baseline_pairs.json");
    f << "{\n  \"tool\": \"scratch_baseline\",\n  \"fixtures\": \"" << fixtures_csv.string()
      << "\",\n  \"expected\": \"" << expected_csv.string() << "\",\n"
      << "  \"params\": {\"pyramid_levels\": " << params.frontend.pyramid.levels
      << ", \"fast_threshold\": " << params.frontend.fast.threshold << ", \"match_floor\": "
      << params.frontend.match_floor << ", \"pnp_iterations\": " << params.pnp_ransac.iterations
      << ", \"pnp_threshold_px\": " << params.pnp_ransac.threshold_px << ", \"pnp_seed\": " << params.pnp_ransac.seed
      << ", \"refine\": " << (params.refine ? "true" : "false") << "},\n"
      << "  \"definitions\": {\"duplicate_radius_px\": " << kDuplicateRadiusPx
      << ", \"min_distinct_inliers\": " << kMinDistinctInliers
      << ", \"pose_valid\": \"diagnostic only: solver ok, finite proper rotation, and >= min_distinct_inliers distinct "
         "inlier keypoints in each frame (reference evaluator AcceptPnp); not used by the pipeline\"},\n"
      << "  \"pairs\": [\n";
    for (size_t k = 0; k < json_entries.size(); ++k) f << json_entries[k] << (k + 1 < json_entries.size() ? ",\n" : "\n");
    f << "  ],\n  \"mismatches\": " << mismatches << ",\n  \"skipped\": " << missing << "\n}\n";
  }
  {
    std::ofstream f(out / "baseline_pairs.csv");
    f << ExpectedCsv(measured);
  }
  const fs::path rel = out.lexically_relative(repo);
  const bool inside = !rel.empty() && *rel.begin() != "..";
  std::printf("\nwrote %s/{baseline_pairs.json,baseline_pairs.csv}\n", (inside ? rel : out).string().c_str());
  if (!write_expected.empty()) {
    const fs::path p = abs(write_expected);
    if (fs::exists(p)) {
      std::cerr << p.string() << " exists; not overwriting\n";
      return 1;
    }
    std::ofstream(p) << ExpectedCsv(measured);
    std::printf("wrote lock file %s\n", p.string().c_str());
  }
  if (!write_expected_gate.empty()) {
    const fs::path p = abs(write_expected_gate);
    if (fs::exists(p)) {
      std::cerr << p.string() << " exists; not overwriting\n";
      return 1;
    }
    std::ofstream(p) << GateExpectedCsv(measured);
    std::printf("wrote gate lock file %s\n", p.string().c_str());
  }
  std::printf("%zu measured, %d skipped (missing data), %d mismatching\n", measured.size(), missing, mismatches);
  if (mismatches) return 1;
  return missing ? 77 : 0;
}
