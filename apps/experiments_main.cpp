// scratch_experiments: Stage-3C matcher experiments and offline gate diagnostics (see experiment.hpp).
//
// Usage (repository root):
//   scratch_experiments --out DIR [--configs M0,M1,...] [--seeds LIST | --seed-count N] [--threads T]
//                       (--fixtures CSV | --sequence-frames CSV --tum-sequence DIR --intrinsics fx,fy,cx,cy)
// Configurations: experiment.hpp IntegrationConfigIds (default M0,T0); seeds 12345 + 1..4 (5), one thread per
// configuration. DIR must not exist (prior outputs are never overwritten).
//
// Features are extracted once per frame and configuration (configurations may change extraction); per configuration the
// pipeline's ProcessPair runs once per pair and seed with only the matcher options and the RANSAC seed
// changed -- PnP, LM and the Stage-2 gate are the production ones. Each run is re-traced (TraceRansacPnP)
// and must reproduce the production RANSAC result exactly (trace_ok; the program fails otherwise).
// Outputs:
//   runs.csv       one row per configuration, pair and seed: status, every gate check, candidate relative-support
//                  metrics, 3D conditioning, RANSAC stages; columns prefixed eval_ use ground truth (evaluation only)
//   matchsets.csv  one row per configuration and pair (seed-independent): match-set structure, filter counters,
//                  strict vs location-aware mutual
//   ate.csv        (sequence mode) one row per configuration and seed: accepted / rejected / failed, raw ATE
//   command.txt
// Ground truth is read for scoring only; no decision uses it.

#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "accumulate.hpp"
#include "baseline.hpp"
#include "experiment.hpp"

using namespace scratch;
using namespace scratch::experiment;

namespace {

struct Frame {
  RgbdFrame rgbd;
  Pose Twc;  // ground truth (scoring only)
};
struct PairJob {
  std::string label;
  int pair = 0, i = 0, j = 0;  // indices into the frame list
  int frame_i = 0, frame_j = 0;
  Intrinsics K{0, 0, 0, 0};
};

std::string N(double v, const char *fmt = "%.9g") {
  if (!std::isfinite(v)) return "";
  char b[48];
  std::snprintf(b, sizeof(b), fmt, v);
  return b;
}

// Part E: strict vs location-aware mutual over the matches that passed the distance filter.
struct MutualStats {
  int passed_distance = 0, strict = 0, location = 0, rescued = 0;
  int rescued_with_depth = 0, rescued_correct = 0, strict_with_depth = 0, strict_correct = 0;
};
MutualStats MutualComparison(const PairResult &r, const Frame &a, const Frame &b, const FrameFeatures &fa,
                             const FrameFeatures &fb, const Intrinsics &K, double depth_scale, double thr) {
  MutualStats s;
  const Pose G = baseline::RelativeGroundTruth(a.Twc, b.Twc);  // evaluation only
  auto correct = [&](const features::RankedMatch &m, bool &has_depth) {
    const features::Keypoint &p = fa.keypoints[m.query], &q = fb.keypoints[m.train];
    const uint16_t d = a.rgbd.depth.samples[size_t(int(p.y)) * a.rgbd.depth.width + int(p.x)];
    has_depth = d > 0;
    if (!has_depth) return false;
    const double Z = d / depth_scale;
    const Eigen::Vector3d X((p.x - K.cx) / K.fx * Z, (p.y - K.cy) / K.fy * Z, Z);
    double z;
    const Eigen::Vector2d u = projectPoint(X, G.R, G.t, K, &z);
    return z > 0 && (u - Eigen::Vector2d(q.x, q.y)).norm() < thr;
  };
  for (const features::RankedMatch &m : r.match_details) {
    if (m.fate == features::MatchFate::kRemovedDistance) continue;
    ++s.passed_distance;
    s.strict += m.mutual, s.location += m.mutual_location;
    bool dep = false;
    if (m.mutual) {
      const bool ok = correct(m, dep);
      s.strict_with_depth += dep, s.strict_correct += ok;
    } else if (m.mutual_location) {
      ++s.rescued;
      const bool ok = correct(m, dep);
      s.rescued_with_depth += dep, s.rescued_correct += ok;
    }
  }
  return s;
}

const char *kRunHeader =
    "config,label,pair,frame_i,frame_j,seed,seconds,pnp_inputs,solver_ok,gate_status,accepted,reason,"
    "inlier_rows,distinct_src,distinct_dst,loc_src,loc_dst,cells_src,cells_dst,reproj_median_px,reproj_p90_px,"
    "inlier_ratio,c_finite,c_so3,c_rows,c_kp_src,c_kp_dst,c_loc_src,c_loc_dst,c_ratio,c_cells_src,c_cells_dst,"
    "c_median,c_p90,input_loc_src,input_loc_dst,input_pairs,inlier_pairs,input_cells_dst,ratio_pairs,ratio_loc_dst,"
    "ratio_cells_dst,rows_per_location,largest_cell_share,eig1,eig2,eig3,condition,planarity,depth_min,depth_max,"
    "depth_median,sample_inliers,refit_inliers,trace_ok,eval_rot_err_deg,eval_trans_err_m,eval_sample_rot_err_deg,"
    "eval_sample_trans_err_m,eval_refit_rot_err_deg,eval_refit_trans_err_m,eval_correct_input_share,"
    "eval_correct_inlier_share\n";
const char *kMatchHeader =
    "config,label,pair,frame_i,frame_j,keypoints_i,keypoints_j,raw,filtered,uniq_src_kp,uniq_dst_kp,uniq_src_loc,"
    "uniq_dst_loc,many_to_one,pnp_inputs,removed_distance,removed_mutual,removed_ratio,removed_level,"
    "passed_distance,strict_mutual,location_mutual,mutual_rescued,eval_correct_inputs,eval_correct_input_share,"
    "eval_rescued_with_depth,eval_rescued_correct,eval_strict_with_depth,eval_strict_correct\n";

std::string RunRow(const std::string &cfg, const PairJob &job, uint32_t seed, double sec, const PairResult &r,
                   const GateDiagnostics &d, const RansacTrace &tr, bool trace_ok, const TruthEval &t) {
  std::ostringstream o;
  std::string reason = r.validity.reason;
  for (char &c : reason)
    if (c == ',') c = ' ';
  const PoseValidity &v = r.validity;
  o << cfg << "," << job.label << "," << job.pair << "," << job.frame_i << "," << job.frame_j << "," << seed << ","
    << N(sec) << "," << r.pnp_index.size() << "," << int(r.pnp_ok) << "," << PoseStatusName(v.status) << ","
    << int(r.pose_accepted) << "," << reason << "," << v.inlier_rows << "," << v.distinct_src << "," << v.distinct_dst
    << "," << v.distinct_loc_src << "," << v.distinct_loc_dst << "," << v.cells_src << "," << v.cells_dst << ","
    << N(v.reproj_median_px) << "," << N(v.reproj_p90_px) << "," << N(v.inlier_ratio);
  for (bool c : {d.c_finite, d.c_so3, d.c_rows, d.c_kp_src, d.c_kp_dst, d.c_loc_src, d.c_loc_dst, d.c_ratio,
                 d.c_cells_src, d.c_cells_dst, d.c_median, d.c_p90})
    o << "," << int(c);
  o << "," << d.input_loc_src << "," << d.input_loc_dst << "," << d.input_pairs << "," << d.inlier_pairs << ","
    << d.input_cells_dst << "," << N(d.ratio_pairs) << "," << N(d.ratio_loc_dst) << "," << N(d.ratio_cells_dst) << ","
    << N(d.rows_per_location) << "," << N(d.largest_cell_share) << "," << N(d.eig1) << "," << N(d.eig2) << ","
    << N(d.eig3) << "," << N(d.condition) << "," << N(d.planarity) << "," << N(d.depth_min) << "," << N(d.depth_max)
    << "," << N(d.depth_median) << "," << tr.sample_inliers << "," << tr.refit_inliers << "," << int(trace_ok) << ","
    << N(t.rot_err_deg) << "," << N(t.trans_err_m) << "," << N(t.sample_rot_err_deg) << "," << N(t.sample_trans_err_m)
    << "," << N(t.refit_rot_err_deg) << "," << N(t.refit_trans_err_m) << "," << N(t.correct_input_share) << ","
    << N(t.correct_inlier_share) << "\n";
  return o.str();
}

std::string MatchRow(const std::string &cfg, const PairJob &job, const PairResult &r, const MutualStats &ms,
                     const TruthEval &t) {
  std::vector<int> src, dst;
  std::vector<Eigen::Vector2d> us, ud;
  for (const Correspondence &c : r.correspondences)
    src.push_back(c.query), dst.push_back(c.train), us.push_back(c.uv_i), ud.push_back(c.uv_j);
  std::map<int, int> uses;
  for (int k : dst) ++uses[k];
  int many = 0;
  for (int k : dst) many += uses[k] > 1;
  const features::MatchCounters &c = r.match_counters;
  const bool rk = r.ranked_matcher;
  std::ostringstream o;
  o << cfg << "," << job.label << "," << job.pair << "," << job.frame_i << "," << job.frame_j << "," << r.keypoints_i
    << "," << r.keypoints_j << "," << r.raw_matches << "," << r.filtered_matches << "," << CountDistinctIds(src) << ","
    << CountDistinctIds(dst) << "," << CountLocationGroups(us, 2.0) << "," << CountLocationGroups(ud, 2.0) << ","
    << many << "," << r.pnp_index.size() << "," << (rk ? c.removed_distance : r.raw_matches - r.filtered_matches) << ","
    << (rk ? std::to_string(c.removed_mutual) : "0") << "," << (rk ? std::to_string(c.removed_ratio) : "0") << ","
    << (rk ? std::to_string(c.removed_level) : "0") << ",";
  if (rk)
    o << ms.passed_distance << "," << ms.strict << "," << ms.location << "," << ms.rescued;
  else
    o << ",,,";
  o << "," << t.correct_inputs << "," << N(t.correct_input_share) << ",";
  if (rk)
    o << ms.rescued_with_depth << "," << ms.rescued_correct << "," << ms.strict_with_depth << "," << ms.strict_correct;
  else
    o << ",,,";
  o << "\n";
  return o.str();
}

}  // namespace

int main(int argc, char **argv) {
  fs::path repo, out, fixtures_csv, seq_frames, seq_dir;
  std::string configs_arg, intrinsics_arg;
  std::vector<uint32_t> seeds = baseline::DefaultSeedSweep(5);
  int threads = 0;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    auto next = [&] { return a + 1 < argc ? std::string(argv[++a]) : std::string(); };
    if (s == "--repo") repo = next();
    else if (s == "--out") out = next();
    else if (s == "--configs") configs_arg = next();
    else if (s == "--fixtures") fixtures_csv = next();
    else if (s == "--sequence-frames") seq_frames = next();
    else if (s == "--tum-sequence") seq_dir = next();
    else if (s == "--intrinsics") intrinsics_arg = next();
    else if (s == "--threads") threads = std::stoi(next());
    else if (s == "--seed-count") seeds = baseline::DefaultSeedSweep(std::stoi(next()));
    else if (s == "--seeds") {
      std::string e;
      if (!baseline::ParseSeedList(next(), seeds, &e)) return std::cerr << e << "\n", 2;
    } else return std::cerr << "unknown option " << s << "\n", 2;
  }
  if (repo.empty()) repo = FindRepo(fs::current_path());
  if (repo.empty() || out.empty() || (fixtures_csv.empty() == seq_frames.empty()))
    return std::cerr << "need --out DIR and exactly one of --fixtures / --sequence-frames (repository root found: "
                     << !repo.empty() << ")\n", 2;
  auto abs = [&](const fs::path &p) { return p.is_absolute() ? p : repo / p; };
  out = abs(out);
  if (fs::exists(out)) return std::cerr << out.string() << " exists; prior outputs are not overwritten\n", 2;
  std::vector<std::string> configs;
  {
    std::stringstream ss(configs_arg.empty() ? "M0,T0" : configs_arg);
    std::string c;
    PipelineParams probe;
    std::string e;
    while (std::getline(ss, c, ','))
      if (!ApplyIntegrationConfig(c, probe, &e)) return std::cerr << e << "\n", 2;
      else configs.push_back(c);
  }

  // ---- frames and pair jobs ----
  std::vector<Frame> frames;
  std::vector<PairJob> jobs;
  std::string err;
  if (!fixtures_csv.empty()) {
    std::vector<baseline::Fixture> fx;
    std::vector<baseline::FixtureGroundTruth> gt;
    if (!baseline::LoadFixtures(abs(fixtures_csv), fx, gt, &err)) return std::cerr << err << "\n", 1;
    for (size_t k = 0; k < fx.size(); ++k) {
      Frame a, b;
      if (!LoadRgbdFiles((repo / fx[k].rgb_i).string(), (repo / fx[k].depth_i).string(), a.rgbd, &err) ||
          !LoadRgbdFiles((repo / fx[k].rgb_j).string(), (repo / fx[k].depth_j).string(), b.rgbd, &err))
        return std::cerr << fx[k].name << ": " << err << "\n", 1;
      a.Twc = gt[k].Twc_i, b.Twc = gt[k].Twc_j;
      PairJob j{fx[k].name, int(k), int(frames.size()), int(frames.size()) + 1, fx[k].frame_i, fx[k].frame_j, fx[k].K};
      frames.push_back(std::move(a)), frames.push_back(std::move(b));
      jobs.push_back(j);
    }
  } else {
    Intrinsics K;
    if (std::sscanf(intrinsics_arg.c_str(), "%lf,%lf,%lf,%lf", &K.fx, &K.fy, &K.cx, &K.cy) != 4)
      return std::cerr << "--intrinsics fx,fy,cx,cy\n", 2;
    Csv fr;
    if (!fr.load(abs(seq_frames))) return std::cerr << "cannot read " << seq_frames << "\n", 1;
    for (size_t r = 0; r < fr.rows.size(); ++r) {
      Frame f;
      if (!LoadRgbdFiles((abs(seq_dir) / fr.rows[r].at(fr.col.at("rgb"))).string(),
                         (abs(seq_dir) / fr.rows[r].at(fr.col.at("depth"))).string(), f.rgbd, &err))
        return std::cerr << err << "\n", 1;
      f.Twc.R = QuatToR(fr.d(r, "gt_qx"), fr.d(r, "gt_qy"), fr.d(r, "gt_qz"), fr.d(r, "gt_qw"));
      f.Twc.t = Vector3d(fr.d(r, "gt_x"), fr.d(r, "gt_y"), fr.d(r, "gt_z"));
      frames.push_back(std::move(f));
    }
    for (int k = 0; k + 1 < int(frames.size()); ++k) jobs.push_back(PairJob{"seq", k, k, k + 1, k, k + 1, K});
  }
  std::printf("%zu frames, %zu pairs, %zu configurations, %zu seeds\n", frames.size(), jobs.size(), configs.size(),
              seeds.size());

  // ---- per configuration (one thread each, up to --threads) ----
  struct Output {
    std::string runs, matchsets, ate;
    int trace_failures = 0;
  };
  std::vector<Output> outputs(configs.size());
  auto work = [&](size_t ci) {
    const std::string &cfg = configs[ci];
    PipelineParams params;
    ApplyIntegrationConfig(cfg, params);
    Output &o = outputs[ci];
    // Features are extracted per configuration: the integration configurations change feature
    // extraction (FAST threshold / cell fallback, Harris ranking, feature budget), not only matching.
    // Within a configuration they are computed once per frame and shared by every pair and seed.
    std::vector<FrameFeatures> feats(frames.size());
    for (size_t k = 0; k < frames.size(); ++k) feats[k] = ExtractFeatures(frames[k].rgbd.rgb, params.frontend);
    std::vector<std::vector<RelativeStep>> steps(seeds.size(), std::vector<RelativeStep>(jobs.size()));
    for (const PairJob &job : jobs) {
      const Frame &a = frames[job.i], &b = frames[job.j];
      const Pose G = baseline::RelativeGroundTruth(a.Twc, b.Twc);  // evaluation only
      for (size_t si = 0; si < seeds.size(); ++si) {
        PipelineParams p = params;
        p.pnp_ransac.seed = seeds[si];
        const auto t0 = std::chrono::steady_clock::now();
        const PairResult r = ProcessPair(job.frame_i, job.frame_j, feats[job.i], feats[job.j], a.rgbd, b.rgbd, job.K, p);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<Eigen::Vector3d> X;
        std::vector<Eigen::Vector2d> uv;
        for (int q : r.pnp_index) X.push_back(r.correspondences[q].X_i), uv.push_back(r.correspondences[q].uv_j);
        const RansacTrace tr = TraceRansacPnP(X, uv, job.K, p.pnp_ransac);
        const bool trace_ok = TraceMatches(tr, r);
        o.trace_failures += !trace_ok;
        const GateDiagnostics d = DiagnoseGate(r, p.validity, p.pnp_ransac.threshold_px);
        const TruthEval t = EvaluateWithTruth(r, tr, G, job.K, p.pnp_ransac.threshold_px);
        o.runs += RunRow(cfg, job, seeds[si], sec, r, d, tr, trace_ok, t);
        if (si == 0)
          o.matchsets += MatchRow(cfg, job, r, MutualComparison(r, a, b, feats[job.i], feats[job.j], job.K,
                                                                p.frontend.depth_scale, p.pnp_ransac.threshold_px), t);
        RelativeStep &st = steps[si][job.pair];
        st.accepted = r.pose_accepted;
        st.held_source = r.pnp_ok ? "held_previous_pose_pair_rejected" : "held_previous_pose_pair_failed";
        st.T.R = r.R, st.T.t = r.t;
      }
    }
    if (seq_frames.empty()) return;
    for (size_t si = 0; si < seeds.size(); ++si) {  // trajectory per seed (sequence mode)
      const AccumulatedTrajectory tj = AccumulateTrajectory(frames[0].Twc, steps[si]);
      double sq = 0;
      int acc = 0, held_rej = 0, held_fail = 0;
      for (size_t k = 0; k < frames.size(); ++k) sq += (tj.Twc[k].t - frames[k].Twc.t).squaredNorm();
      for (size_t k = 1; k < tj.source.size(); ++k)
        acc += tj.source[k] == "scratch_pipeline", held_rej += tj.source[k] == "held_previous_pose_pair_rejected",
            held_fail += tj.source[k] == "held_previous_pose_pair_failed";
      // RPE (delta 1): E_k = (Q_k^-1 Q_k+1)^-1 (P_k^-1 P_k+1), Q = ground truth, P = estimate (held poses included)
      double rpe_t = 0, rpe_r = 0;
      for (size_t k = 0; k + 1 < frames.size(); ++k) {
        const Pose Qr = Compose(Inverse(frames[k].Twc), frames[k + 1].Twc), Pr = Compose(Inverse(tj.Twc[k]), tj.Twc[k + 1]);
        const Pose E = Compose(Inverse(Qr), Pr);
        rpe_t += E.t.squaredNorm(), rpe_r += std::pow(RotAngleDeg(E.R), 2);
      }
      const double npairs = double(frames.size() - 1);
      const Pose &last = tj.Twc.back(), &gl = frames.back().Twc;
      std::ostringstream row;
      row << cfg << "," << seeds[si] << "," << frames.size() << "," << acc << "," << held_rej << "," << held_fail << ","
          << N(std::sqrt(sq / frames.size())) << "," << N((last.t - gl.t).norm()) << "," << N(RotErrDeg(gl.R, last.R))
          << "," << N(std::sqrt(rpe_t / npairs)) << "," << N(std::sqrt(rpe_r / npairs)) << "\n";
      o.ate += row.str();
    }
  };
  const int nthreads = threads > 0 ? threads : int(configs.size());
  for (size_t start = 0; start < configs.size(); start += nthreads) {
    std::vector<std::thread> pool;
    for (size_t ci = start; ci < std::min(configs.size(), start + size_t(nthreads)); ++ci) pool.emplace_back(work, ci);
    for (auto &t : pool) t.join();
  }

  fs::create_directories(out);
  {
    std::ofstream cmd(out / "command.txt");
    for (int a = 0; a < argc; ++a) cmd << (a ? " " : "") << argv[a];
    cmd << "\n";
  }
  std::ofstream runs(out / "runs.csv"), ms(out / "matchsets.csv");
  runs << kRunHeader;
  ms << kMatchHeader;
  int trace_failures = 0;
  for (const Output &o : outputs) runs << o.runs, ms << o.matchsets, trace_failures += o.trace_failures;
  if (!seq_frames.empty()) {
    std::ofstream ate(out / "ate.csv");
    ate << "config,seed,frames,accepted,rejected,solver_failed,eval_ate_raw_rmse_m,eval_final_pos_err_m,"
           "eval_final_rot_err_deg,eval_rpe_trans_rmse_m,eval_rpe_rot_rmse_deg\n";
    for (const Output &o : outputs) ate << o.ate;
  }
  std::printf("wrote %s/{runs.csv,matchsets.csv%s,command.txt}; RANSAC traces reproducing production: %s\n",
              out.string().c_str(), seq_frames.empty() ? "" : ",ate.csv", trace_failures ? "NO" : "all");
  return trace_failures ? 1 : 0;
}
