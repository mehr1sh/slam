#include "baseline.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <numeric>

namespace scratch::baseline {

namespace {

// A fixture image path "data/rgbd_dataset_<sequence>/..." names an external TUM sequence. When the environment
// variable SLAM_TUM_ROOT is set (the directory that contains the sequence directories), the file is read from
// $SLAM_TUM_ROOT/rgbd_dataset_<sequence>/... instead of <repo>/data/...; the bytes read are the same. Every
// other path, and every path when SLAM_TUM_ROOT is unset or empty, is returned unchanged.
std::string ExternalDatasetPath(const std::string &path) {
  static const std::string kPrefix = "data/rgbd_dataset_";
  const char *root = std::getenv("SLAM_TUM_ROOT");
  if (root == nullptr || *root == '\0' || path.compare(0, kPrefix.size(), kPrefix) != 0) return path;
  return (fs::path(root) / path.substr(std::string("data/").size())).string();
}

// Lines of a CSV file without '#' comment lines (scratch_pipeline's trajectory.csv has a comment header).
bool LoadCsvSkippingComments(const fs::path &path, Csv &csv) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  bool header = true;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);
    if (line.back() == ',') cells.push_back("");
    if (header) {
      for (size_t c = 0; c < cells.size(); ++c) csv.col[cells[c]] = int(c);
      header = false;
    } else {
      csv.rows.push_back(cells);
    }
  }
  return !header;
}

Pose Inverse(const Pose &T) {
  Pose r;
  r.R = T.R.transpose();
  r.t = -r.R * T.t;
  return r;
}
Pose Compose(const Pose &A, const Pose &B) {
  Pose r;
  r.R = A.R * B.R;
  r.t = A.R * B.t + A.t;
  return r;
}

int Find(std::vector<int> &parent, int a) {
  while (parent[a] != a) a = parent[a] = parent[parent[a]];
  return a;
}

std::string Num(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}
std::string Quote(const std::string &s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o + "\"";
}

}  // namespace

// ---------------------------------------------------------------- fixtures
bool LoadFixtures(const fs::path &csv_path, std::vector<Fixture> &fixtures, std::vector<FixtureGroundTruth> &gt,
                  std::string *error) {
  fixtures.clear(), gt.clear();
  Csv csv;
  if (!csv.load(csv_path)) {
    if (error) *error = "cannot read " + csv_path.string();
    return false;
  }
  try {
    for (size_t r = 0; r < csv.rows.size(); ++r) {
      auto s = [&](const std::string &c) { return csv.rows[r].at(csv.col.at(c)); };
      Fixture f;
      f.name = s("name"), f.dataset = s("dataset");
      f.frame_i = csv.i(r, "frame_i"), f.frame_j = csv.i(r, "frame_j");
      f.rgb_i = ExternalDatasetPath(s("rgb_i")), f.depth_i = ExternalDatasetPath(s("depth_i"));
      f.rgb_j = ExternalDatasetPath(s("rgb_j")), f.depth_j = ExternalDatasetPath(s("depth_j"));
      f.K = Intrinsics{csv.d(r, "fx"), csv.d(r, "fy"), csv.d(r, "cx"), csv.d(r, "cy")};
      f.export_dir = s("export_dir");
      f.export_pair = s("export_pair").empty() ? -1 : csv.i(r, "export_pair");
      if (f.name.empty() || f.rgb_i.empty() || f.depth_i.empty() || f.rgb_j.empty() || f.depth_j.empty())
        throw std::runtime_error("empty name or file path");
      if (!(f.K.fx > 0 && f.K.fy > 0)) throw std::runtime_error("non-positive focal length");
      for (const auto &o : fixtures)
        if (o.name == f.name) throw std::runtime_error("duplicate fixture name " + f.name);
      FixtureGroundTruth g;
      g.name = f.name;
      for (const char *k : {"i", "j"}) {
        const std::string p = std::string("gt_") + k + "_";
        Pose T;
        T.R = QuatToR(csv.d(r, p + "qx"), csv.d(r, p + "qy"), csv.d(r, p + "qz"), csv.d(r, p + "qw"));
        T.t = Vector3d(csv.d(r, p + "x"), csv.d(r, p + "y"), csv.d(r, p + "z"));
        if (!T.R.allFinite() || !T.t.allFinite()) throw std::runtime_error("non-finite ground truth");
        (k[0] == 'i' ? g.Twc_i : g.Twc_j) = T;
      }
      fixtures.push_back(f);
      gt.push_back(g);
    }
  } catch (const std::exception &e) {
    if (error) *error = csv_path.string() + ": row " + std::to_string(fixtures.size() + 1) + ": " + e.what();
    fixtures.clear(), gt.clear();
    return false;
  }
  if (fixtures.empty()) {
    if (error) *error = csv_path.string() + ": no fixtures";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- counting
int CountDistinct(std::vector<int> ids) {
  std::sort(ids.begin(), ids.end());
  return int(std::unique(ids.begin(), ids.end()) - ids.begin());
}

int CountDistinctLocations(const std::vector<Eigen::Vector2d> &pts, double radius) {
  const int n = int(pts.size());
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  int groups = n;
  for (int a = 0; a < n; ++a)
    for (int b = a + 1; b < n; ++b)
      if ((pts[a] - pts[b]).norm() <= radius) {
        const int ra = Find(parent, a), rb = Find(parent, b);
        if (ra != rb) parent[ra] = rb, --groups;
      }
  return groups;
}

int CountCrossLevelNearDuplicates(const FrameFeatures &f, double radius) {
  const int n = int(f.keypoints.size());
  std::vector<char> dup(n, 0);
  const double r2 = radius * radius;
  for (int a = 0; a < n; ++a)
    for (int b = a + 1; b < n; ++b) {
      if (f.level[a] == f.level[b]) continue;
      const double dx = f.keypoints[a].x - f.keypoints[b].x, dy = f.keypoints[a].y - f.keypoints[b].y;
      if (dx * dx + dy * dy <= r2) dup[a] = dup[b] = 1;
    }
  return int(std::count(dup.begin(), dup.end(), 1));
}

int CountSharedDestinationMatches(const std::vector<int> &train) {
  std::map<int, int> uses;
  for (int t : train) ++uses[t];
  int n = 0;
  for (int t : train) n += uses[t] > 1;
  return n;
}

// ---------------------------------------------------------------- measurement
PairMeasurement Measure(const std::string &name, const PairResult &r, const FrameFeatures &fi,
                        const FrameFeatures &fj) {
  PairMeasurement m;
  m.name = name;
  m.frame_i = r.i, m.frame_j = r.j;
  m.keypoints_i = r.keypoints_i, m.keypoints_j = r.keypoints_j;
  m.cross_level_dup_i = CountCrossLevelNearDuplicates(fi, kDuplicateRadiusPx);
  m.cross_level_dup_j = CountCrossLevelNearDuplicates(fj, kDuplicateRadiusPx);
  m.raw_matches = r.raw_matches, m.filtered_matches = r.filtered_matches;

  std::vector<int> q_all, t_all;
  for (const auto &c : r.correspondences) q_all.push_back(c.query), t_all.push_back(c.train);
  m.unique_src_filtered = CountDistinct(q_all), m.unique_dst_filtered = CountDistinct(t_all);
  m.shared_dst_filtered = CountSharedDestinationMatches(t_all);

  m.pnp_inputs = int(r.pnp_index.size());
  std::vector<int> q_pnp, t_pnp, q_in, t_in;
  std::vector<Eigen::Vector2d> loc_i, loc_j;
  for (size_t k = 0; k < r.pnp_index.size(); ++k) {
    const Correspondence &c = r.correspondences[r.pnp_index[k]];
    q_pnp.push_back(c.query), t_pnp.push_back(c.train);
    if (r.pnp_ok && k < r.pnp_mask.size() && r.pnp_mask[k]) {
      q_in.push_back(c.query), t_in.push_back(c.train);
      loc_i.push_back(c.uv_i), loc_j.push_back(c.uv_j);
    }
  }
  m.unique_src_pnp = CountDistinct(q_pnp), m.unique_dst_pnp = CountDistinct(t_pnp);
  m.inlier_rows = int(q_in.size());
  m.distinct_inlier_i = CountDistinct(q_in), m.distinct_inlier_j = CountDistinct(t_in);
  m.distinct_inlier_loc_i = CountDistinctLocations(loc_i, kDuplicateRadiusPx);
  m.distinct_inlier_loc_j = CountDistinctLocations(loc_j, kDuplicateRadiusPx);
  m.shared_dst_inliers = CountSharedDestinationMatches(t_in);

  m.solver_ok = r.pnp_ok, m.solver_reason = r.pnp_reason;
  m.gate = r.validity, m.pose_accepted = r.pose_accepted;
  m.refined = r.refined, m.refine_iterations = r.refine_iterations;
  m.reproj_inlier_mean_px = r.pnp_ok ? r.reproj_inlier_mean : NAN;
  m.R = r.R, m.t = r.t;
  m.est_rotation_deg = RotAngleDeg(r.R), m.est_translation_m = r.t.norm();

  // diagnostic validity (recorded only)
  if (!m.solver_ok) {
    m.validity_reason = "solver failed: " + r.pnp_reason;
  } else if (!r.R.allFinite() || !r.t.allFinite() || std::abs(r.R.determinant() - 1) > 1e-6 ||
             (r.R.transpose() * r.R - Eigen::Matrix3d::Identity()).norm() > 1e-6) {
    m.validity_reason = "pose is not a finite proper rigid motion";
  } else if (std::min(m.distinct_inlier_i, m.distinct_inlier_j) < kMinDistinctInliers) {
    m.validity_reason = "inliers have " + std::to_string(m.distinct_inlier_i) + " distinct frame-i and " +
                        std::to_string(m.distinct_inlier_j) + " distinct frame-j keypoints (< " +
                        std::to_string(kMinDistinctInliers) + ")";
  }
  m.pose_valid = m.solver_ok && m.validity_reason.empty();
  return m;
}

bool MeasureFixture(const Fixture &fx, const fs::path &repo, const PipelineParams &params, PairMeasurement &m,
                    std::string *error, PairResult *result) {
  RgbdFrame a, b;
  if (!LoadRgbdFiles((repo / fx.rgb_i).string(), (repo / fx.depth_i).string(), a, error) ||
      !LoadRgbdFiles((repo / fx.rgb_j).string(), (repo / fx.depth_j).string(), b, error))
    return false;
  const FrameFeatures fa = ExtractFeatures(a.rgb, params.frontend), fb = ExtractFeatures(b.rgb, params.frontend);
  PairResult r = ProcessPair(fx.frame_i, fx.frame_j, fa, fb, a, b, fx.K, params);
  m = Measure(fx.name, r, fa, fb);
  if (result) *result = std::move(r);
  return true;
}

Pose RelativeGroundTruth(const Pose &Twc_i, const Pose &Twc_j) { return Compose(Inverse(Twc_j), Twc_i); }

void ScoreMeasurement(PairMeasurement &m, const FixtureGroundTruth &gt) {
  const Pose G = RelativeGroundTruth(gt.Twc_i, gt.Twc_j);
  m.scored = true;
  m.gt_motion_deg = RotAngleDeg(G.R), m.gt_motion_m = G.t.norm();
  m.rot_err_deg = m.solver_ok ? RotErrDeg(G.R, m.R) : NAN;
  m.trans_err_m = m.solver_ok ? (G.t - m.t).norm() : NAN;
}

// ---------------------------------------------------------------- lock
std::vector<std::pair<std::string, double>> NumericFields(const PairMeasurement &m) {
  std::vector<std::pair<std::string, double>> f = {
      {"frame_i", m.frame_i}, {"frame_j", m.frame_j},
      {"keypoints_i", m.keypoints_i}, {"keypoints_j", m.keypoints_j},
      {"cross_level_dup_i", m.cross_level_dup_i}, {"cross_level_dup_j", m.cross_level_dup_j},
      {"raw_matches", m.raw_matches}, {"filtered_matches", m.filtered_matches},
      {"unique_src_filtered", m.unique_src_filtered}, {"unique_dst_filtered", m.unique_dst_filtered},
      {"shared_dst_filtered", m.shared_dst_filtered},
      {"pnp_inputs", m.pnp_inputs}, {"unique_src_pnp", m.unique_src_pnp}, {"unique_dst_pnp", m.unique_dst_pnp},
      {"inlier_rows", m.inlier_rows},
      {"distinct_inlier_i", m.distinct_inlier_i}, {"distinct_inlier_j", m.distinct_inlier_j},
      {"distinct_inlier_loc_i", m.distinct_inlier_loc_i}, {"distinct_inlier_loc_j", m.distinct_inlier_loc_j},
      {"shared_dst_inliers", m.shared_dst_inliers},
      {"solver_ok", m.solver_ok}, {"pose_valid", m.pose_valid},
      {"refined", m.refined}, {"refine_iterations", m.refine_iterations},
      {"reproj_inlier_mean_px", m.reproj_inlier_mean_px},
      {"est_rotation_deg", m.est_rotation_deg}, {"est_translation_m", m.est_translation_m},
      {"t_x", m.t.x()}, {"t_y", m.t.y()}, {"t_z", m.t.z()}};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) f.push_back({"r" + std::to_string(a) + std::to_string(b), m.R(a, b)});
  f.insert(f.end(), {{"gt_motion_deg", m.gt_motion_deg}, {"gt_motion_m", m.gt_motion_m},
                     {"rot_err_deg", m.rot_err_deg}, {"trans_err_m", m.trans_err_m}});
  return f;
}

std::vector<std::pair<std::string, double>> GateFields(const PairMeasurement &m) {
  const PoseValidity &g = m.gate;
  return {{"gate_status", double(int(g.status))}, {"pose_accepted", m.pose_accepted},
          {"gate_pnp_inputs", g.pnp_inputs}, {"gate_inlier_rows", g.inlier_rows},
          {"gate_distinct_src", g.distinct_src}, {"gate_distinct_dst", g.distinct_dst},
          {"gate_distinct_loc_src", g.distinct_loc_src}, {"gate_distinct_loc_dst", g.distinct_loc_dst},
          {"gate_inlier_ratio", g.inlier_ratio}, {"gate_cells_src", g.cells_src}, {"gate_cells_dst", g.cells_dst},
          {"gate_reproj_median_px", g.reproj_median_px}, {"gate_reproj_p90_px", g.reproj_p90_px}};
}

// TOLERANCES. Counts, flags and iteration numbers are compared exactly. Floating-point
// values are written with 17 significant digits (exact round trip), and the pipeline is
// single-threaded and seeded, so on the machine that wrote the lock the comparison is
// exact; the tolerances below only absorb compiler / FMA / libm differences between
// machines. They are 4-6 orders of magnitude below anything an algorithmic change
// produces (e.g. pair 46->47 is 22.2 deg / 0.76 m off; a one-inlier change moves the
// refined pose by ~1e-4).
double FieldTolerance(const std::string &f) {
  auto ends = [&](const char *s) {
    const std::string x = s;
    return f.size() >= x.size() && f.compare(f.size() - x.size(), x.size(), x) == 0;
  };
  if (ends("_deg")) return 1e-6;                                    // degrees
  if (ends("_m") || f == "t_x" || f == "t_y" || f == "t_z") return 1e-8;  // metres
  if (ends("_px")) return 1e-6;                                     // pixels
  if (ends("_ratio")) return 1e-12;                                 // ratio of two counts
  if (f.size() == 3 && f[0] == 'r' && std::isdigit(f[1]) && std::isdigit(f[2])) return 1e-9;  // R entries
  return 0;
}

bool LoadExpected(const fs::path &csv_path, std::vector<Expected> &out, std::string *error) {
  out.clear();
  Csv csv;
  if (!csv.load(csv_path)) {
    if (error) *error = "cannot read " + csv_path.string();
    return false;
  }
  try {
    for (size_t r = 0; r < csv.rows.size(); ++r) {
      Expected e;
      e.name = csv.rows[r].at(csv.col.at("name"));
      for (const auto &[c, idx] : csv.col) {
        if (c == "name") continue;
        const std::string v = csv.rows[r].at(idx);
        e.values[c] = (v.empty() || v == "nan") ? NAN : std::stod(v);
      }
      out.push_back(e);
    }
  } catch (const std::exception &ex) {
    if (error) *error = csv_path.string() + ": " + ex.what();
    return false;
  }
  return true;
}

namespace {
using FieldFn = std::vector<std::pair<std::string, double>> (*)(const PairMeasurement &);

std::string FieldCsv(const std::vector<PairMeasurement> &ms, FieldFn fields) {
  std::ostringstream o;
  o << "name";
  for (const auto &[k, v] : fields(PairMeasurement())) o << "," << k;
  o << "\n";
  for (const auto &m : ms) {
    o << m.name;
    for (const auto &[k, v] : fields(m)) o << "," << (std::isfinite(v) ? Num(v) : "nan");
    o << "\n";
  }
  return o.str();
}

std::vector<Difference> CompareFields(const std::vector<std::pair<std::string, double>> &fields, const Expected &e) {
  std::vector<Difference> d;
  std::map<std::string, double> actual;
  for (const auto &[k, v] : fields) actual[k] = v;
  for (const auto &[k, v] : actual) {
    const double tol = FieldTolerance(k);
    const auto it = e.values.find(k);
    if (it == e.values.end()) {
      d.push_back({k, NAN, v, tol});
      continue;
    }
    const double x = it->second;
    const bool same = (std::isnan(x) && std::isnan(v)) || (std::isfinite(x) && std::isfinite(v) && std::abs(x - v) <= tol);
    if (!same) d.push_back({k, x, v, tol});
  }
  for (const auto &[k, v] : e.values)
    if (!actual.count(k)) d.push_back({k, v, NAN, 0});
  return d;
}
}  // namespace

std::string ExpectedCsv(const std::vector<PairMeasurement> &ms) { return FieldCsv(ms, NumericFields); }
std::string GateExpectedCsv(const std::vector<PairMeasurement> &ms) { return FieldCsv(ms, GateFields); }
std::vector<Difference> Compare(const PairMeasurement &m, const Expected &e) { return CompareFields(NumericFields(m), e); }
std::vector<Difference> CompareGate(const PairMeasurement &m, const Expected &e) { return CompareFields(GateFields(m), e); }

// ---------------------------------------------------------------- export cross-check
ExportCheck CheckAgainstExport(const PairMeasurement &m, const Fixture &fx, const fs::path &repo) {
  ExportCheck ec;
  if (fx.export_dir.empty() || fx.export_pair < 0) return ec;
  const fs::path dir = repo / fx.export_dir;
  Csv pairs;
  if (!pairs.load(dir / "pairs.csv")) return ec;
  ec.available = true;
  auto problem = [&](const std::string &s) { ec.problems.push_back(s); };
  size_t row = pairs.rows.size();
  for (size_t r = 0; r < pairs.rows.size(); ++r)
    if (pairs.i(r, "pair") == fx.export_pair) row = r;
  if (row == pairs.rows.size()) {
    problem("pair " + std::to_string(fx.export_pair) + " not in " + (dir / "pairs.csv").string());
    return ec;
  }
  auto exact = [&](const std::string &col, double v) {
    ec.compared.push_back(col);
    const double x = pairs.d(row, col);
    if (x != v) problem(col + ": export " + Num(x) + ", harness " + Num(v));
  };
  auto printed = [&](const std::string &col, double v, double tol) {  // export printed with fixed decimals
    ec.compared.push_back(col);
    const std::string s = pairs.rows[row].at(pairs.col.at(col));
    if (s.empty() != !std::isfinite(v) || (!s.empty() && std::abs(std::stod(s) - v) > tol))
      problem(col + ": export " + (s.empty() ? "empty" : s) + ", harness " + Num(v));
  };
  if (pairs.i(row, "frame_i") != m.frame_i || pairs.i(row, "frame_j") != m.frame_j)
    problem("export pair " + std::to_string(fx.export_pair) + " is not frames " + std::to_string(m.frame_i) + "->" +
            std::to_string(m.frame_j));
  exact("keypoints_i", m.keypoints_i), exact("keypoints_j", m.keypoints_j);
  exact("raw_matches", m.raw_matches), exact("filtered_matches", m.filtered_matches);
  exact("correspondences_3d2d", m.pnp_inputs), exact("pnp_ok", m.solver_ok);
  exact("pnp_inliers", m.inlier_rows), exact("refine_iterations", m.refine_iterations);
  printed("rot_err_deg", m.rot_err_deg, 5.01e-7), printed("trans_err_m", m.trans_err_m, 5.01e-7);
  printed("tx", m.t.x(), 5.01e-10), printed("ty", m.t.y(), 5.01e-10), printed("tz", m.t.z(), 5.01e-10);
  printed("reproj_inlier_mean_px", m.reproj_inlier_mean_px, 5.01e-5);

  // per-match export: distinct inlier keypoints, shared destinations
  Csv c;
  const fs::path cpath =
      dir / "correspondences" / ("pair_" + std::to_string(m.frame_i) + "_" + std::to_string(m.frame_j) + ".csv");
  if (c.load(cpath)) {
    std::vector<int> t_all, q_in, t_in;
    for (size_t r = 0; r < c.rows.size(); ++r) {
      t_all.push_back(c.i(r, "train"));
      if (c.i(r, "pnp_inlier")) q_in.push_back(c.i(r, "query")), t_in.push_back(c.i(r, "train"));
    }
    auto cmp = [&](const std::string &what, int x, int v) {
      ec.compared.push_back(what);
      if (x != v) problem(what + ": export " + std::to_string(x) + ", harness " + std::to_string(v));
    };
    cmp("correspondences/unique_dst_filtered", CountDistinct(t_all), m.unique_dst_filtered);
    cmp("correspondences/shared_dst_filtered", CountSharedDestinationMatches(t_all), m.shared_dst_filtered);
    cmp("correspondences/distinct_inlier_i", CountDistinct(q_in), m.distinct_inlier_i);
    cmp("correspondences/distinct_inlier_j", CountDistinct(t_in), m.distinct_inlier_j);
  } else {
    problem("missing " + cpath.string());
  }
  // keypoint export: counts and cross-level near-duplicates (positions printed with 3 decimals)
  for (const auto &[frame, count, dup] : {std::tuple<int, int, int>{m.frame_i, m.keypoints_i, m.cross_level_dup_i},
                                          std::tuple<int, int, int>{m.frame_j, m.keypoints_j, m.cross_level_dup_j}}) {
    char name[48];
    std::snprintf(name, sizeof(name), "frame_%06d.csv", frame);
    Csv k;
    if (!k.load(dir / "keypoints" / name)) {
      problem("missing keypoints/" + std::string(name));
      continue;
    }
    FrameFeatures ff;
    for (size_t r = 0; r < k.rows.size(); ++r) {
      features::Keypoint kp;
      kp.x = float(k.d(r, "x")), kp.y = float(k.d(r, "y"));
      ff.keypoints.push_back(kp);
      ff.level.push_back(k.i(r, "level"));
    }
    const std::string tag = "keypoints/" + std::string(name);
    ec.compared.push_back(tag + " count"), ec.compared.push_back(tag + " cross_level_dup");
    if (int(ff.keypoints.size()) != count)
      problem(tag + " count: export " + std::to_string(ff.keypoints.size()) + ", harness " + std::to_string(count));
    const int d = CountCrossLevelNearDuplicates(ff, kDuplicateRadiusPx);
    if (d != dup) problem(tag + " cross_level_dup: export " + std::to_string(d) + ", harness " + std::to_string(dup));
  }
  return ec;
}

std::string ToJson(const PairMeasurement &m, int indent) {
  const std::string in(indent, ' '), in2(indent + 2, ' ');
  std::ostringstream o;
  o << "{\n" << in2 << "\"name\": " << Quote(m.name) << ",\n";
  for (const auto &[k, v] : NumericFields(m)) {
    const bool flag = k == "solver_ok" || k == "pose_valid" || k == "refined";
    o << in2 << Quote(k) << ": " << (flag ? (v != 0 ? "true" : "false") : Num(v)) << ",\n";
  }
  for (const auto &[k, v] : GateFields(m))
    o << in2 << Quote(k) << ": " << (k == "pose_accepted" ? (v != 0 ? "true" : "false") : Num(v)) << ",\n";
  o << in2 << "\"gate_status_name\": " << Quote(PoseStatusName(m.gate.status)) << ",\n"
    << in2 << "\"gate_reason\": " << Quote(m.gate.reason) << ",\n";
  o << in2 << "\"solver_reason\": " << Quote(m.solver_reason) << ",\n"
    << in2 << "\"validity_reason\": " << Quote(m.validity_reason) << ",\n"
    << in2 << "\"scored\": " << (m.scored ? "true" : "false") << "\n"
    << in << "}";
  return o.str();
}

// ---------------------------------------------------------------- multi-seed evaluation
std::vector<uint32_t> DefaultSeedSweep(int n) {
  std::vector<uint32_t> s;
  if (n <= 0) return s;
  s.push_back(RansacParams().seed);
  for (int k = 1; k < n; ++k) s.push_back(uint32_t(k));
  return s;
}

bool ParseSeedList(const std::string &str, std::vector<uint32_t> &out, std::string *error) {
  out.clear();
  std::stringstream ss(str);
  std::string tok;
  try {
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) throw std::runtime_error("empty entry");
      const size_t dash = tok.find('-', 1);
      size_t used = 0;
      const unsigned long lo = std::stoul(tok.substr(0, dash), &used);
      if (used != (dash == std::string::npos ? tok.size() : dash)) throw std::runtime_error("not a number: " + tok);
      unsigned long hi = lo;
      if (dash != std::string::npos) {
        hi = std::stoul(tok.substr(dash + 1), &used);
        if (used != tok.size() - dash - 1 || hi < lo) throw std::runtime_error("bad range: " + tok);
      }
      for (unsigned long v = lo; v <= hi; ++v) {
        if (v > 0xFFFFFFFFul) throw std::runtime_error("seed out of range: " + tok);
        if (std::find(out.begin(), out.end(), uint32_t(v)) != out.end())
          throw std::runtime_error("duplicate seed " + std::to_string(v));
        out.push_back(uint32_t(v));
      }
    }
  } catch (const std::exception &e) {
    if (error) *error = std::string("seed list: ") + e.what();
    out.clear();
    return false;
  }
  if (out.empty()) {
    if (error) *error = "seed list: empty";
    return false;
  }
  return true;
}

SweepStat Summarize(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
  SweepStat s;
  s.n = int(v.size());
  if (v.empty()) return s;
  s.median = Percentile(v, 50), s.p95 = Percentile(v, 95);
  s.max = *std::max_element(v.begin(), v.end());
  double mean = 0, sq = 0;
  for (double x : v) mean += x;
  mean /= v.size();
  for (double x : v) sq += (x - mean) * (x - mean);
  s.stddev = std::sqrt(sq / v.size());
  return s;
}

bool SweepFixture(const Fixture &fx, const FixtureGroundTruth &gt, const fs::path &repo, const PipelineParams &params,
                  const std::vector<uint32_t> &seeds, std::vector<SeedRun> &runs, std::string *error) {
  runs.clear();
  RgbdFrame a, b;
  if (!LoadRgbdFiles((repo / fx.rgb_i).string(), (repo / fx.depth_i).string(), a, error) ||
      !LoadRgbdFiles((repo / fx.rgb_j).string(), (repo / fx.depth_j).string(), b, error))
    return false;
  const FrameFeatures fa = ExtractFeatures(a.rgb, params.frontend), fb = ExtractFeatures(b.rgb, params.frontend);
  const Pose G = RelativeGroundTruth(gt.Twc_i, gt.Twc_j);  // scoring only
  std::vector<Vector3d> X0;
  std::vector<Vector2d> uv0;
  for (size_t k = 0; k < seeds.size(); ++k) {
    PipelineParams p = params;
    p.pnp_ransac.seed = seeds[k];
    const auto start = std::chrono::steady_clock::now();
    const PairResult r = ProcessPair(fx.frame_i, fx.frame_j, fa, fb, a, b, fx.K, p);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<Vector3d> X;
    std::vector<Vector2d> uv;
    for (int q : r.pnp_index) X.push_back(r.correspondences[q].X_i), uv.push_back(r.correspondences[q].uv_j);
    if (k == 0) X0 = X, uv0 = uv;
    SeedRun s;
    s.name = fx.name, s.seed = seeds[k], s.seconds = sec;
    s.inputs_identical = X == X0 && uv == uv0;
    s.solver_ok = r.pnp_ok, s.accepted = r.pose_accepted, s.gate = r.validity.status, s.reason = r.validity.reason;
    s.pnp_inputs = int(X.size()), s.inlier_rows = r.validity.inlier_rows, s.distinct_loc_dst = r.validity.distinct_loc_dst;
    if (r.pnp_ok) s.rot_err_deg = RotErrDeg(G.R, r.R), s.trans_err_m = (G.t - r.t).norm();
    runs.push_back(s);
  }
  return true;
}

SeedSummary SummarizeSweep(const std::string &name, const std::vector<SeedRun> &runs) {
  SeedSummary s;
  s.name = name;
  std::vector<double> rr, tr, ra, ta, sec;
  for (const SeedRun &r : runs) {
    ++s.seeds;
    s.solver_ok += r.solver_ok, s.accepted += r.accepted, s.inputs_changed += !r.inputs_identical;
    s.rejected += r.gate == PoseStatus::kRejected, s.solver_failed += r.gate == PoseStatus::kSolverFailed;
    sec.push_back(r.seconds);
    if (!r.solver_ok) continue;
    rr.push_back(r.rot_err_deg), tr.push_back(r.trans_err_m);
    s.returned_over_2deg += r.rot_err_deg > 2.0;
    if (r.accepted) ra.push_back(r.rot_err_deg), ta.push_back(r.trans_err_m);
  }
  s.rot_returned = Summarize(rr), s.trans_returned = Summarize(tr);
  s.rot_accepted = Summarize(ra), s.trans_accepted = Summarize(ta);
  s.seconds = Summarize(sec);
  return s;
}

// ---------------------------------------------------------------- ATE
AteResult RawAte(const fs::path &run_dir, const fs::path &repo) {
  AteResult a;
  Csv traj;
  if (!LoadCsvSkippingComments(run_dir / "trajectory.csv", traj)) {
    a.error = "cannot read " + (run_dir / "trajectory.csv").string();
    return a;
  }
  std::vector<Vector3d> gt;
  Csv frames;
  if (frames.load(run_dir / "frames.csv")) {
    for (size_t r = 0; r < frames.rows.size(); ++r)
      gt.emplace_back(frames.d(r, "gt_x"), frames.d(r, "gt_y"), frames.d(r, "gt_z"));
  } else {
    for (const Pose &p : ReadGroundtruthTwc(repo / "data" / "synthetic_bunny" / "groundtruth.txt")) gt.push_back(p.t);
  }
  if (traj.rows.empty() || traj.rows.size() > gt.size()) {
    a.error = "trajectory has " + std::to_string(traj.rows.size()) + " poses, ground truth " + std::to_string(gt.size());
    return a;
  }
  double sq = 0;
  for (size_t r = 0; r < traj.rows.size(); ++r) {
    const int k = traj.i(r, "frame");
    const double e = (Vector3d(traj.d(r, "tx"), traj.d(r, "ty"), traj.d(r, "tz")) - gt.at(k)).norm();
    sq += e * e;
    a.final_error_m = e;
  }
  a.frames = int(traj.rows.size());
  a.ate_raw_rmse_m = std::sqrt(sq / a.frames);
  a.ok = true;
  return a;
}

}  // namespace scratch::baseline
