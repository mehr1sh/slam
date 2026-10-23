#include "tracking/rgbd_icp.hpp"
#include "features/matcher.hpp"
#include "features/multiscale.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <opencv2/imgcodecs.hpp>

#include <random>
#include <sstream>
#include <stdexcept>
namespace rgbd_icp {
M3 skew(const V3 &x) {
  M3 S;
  S << 0, -x.z(), x.y(), x.z(), 0, -x.x(), -x.y(), x.x(), 0;
  return S;
}
Eigen::Matrix<double, 3, 6> pointJacobian(const V3 &x) {
  Eigen::Matrix<double, 3, 6> J;
  J << -skew(x), M3::Identity();
  return J;
}
Pose leftIncrement(const Pose &T, const Eigen::Matrix<double, 6, 1> &step) {
  return Pose{expSO3(step.head<3>()), step.tail<3>()} * T;
}
M3 expSO3(const V3 &w) {
  double a = w.norm();
  if (a < 1e-15)
    return M3::Identity();
  return Eigen::AngleAxisd(a, w / a).toRotationMatrix();
}
V3 logSO3(const M3 &R) {
  Eigen::AngleAxisd a(R);
  return a.angle() * a.axis();
}
double angleDeg(const M3 &R) { return logSO3(R).norm() * 180 / std::acos(-1.); }
Pose rigidSVD(const std::vector<V3> &s, const std::vector<V3> &t) {
  if (s.size() != t.size() || s.size() < 3)
    throw std::invalid_argument("SVD needs >=3 paired points");
  V3 a = V3::Zero(), b = V3::Zero();
  for (size_t i = 0; i < s.size(); ++i) {
    if (!s[i].allFinite() || !t[i].allFinite())
      throw std::invalid_argument("Nonfinite point");
    a += s[i];
    b += t[i];
  }
  a /= s.size();
  b /= s.size();
  M3 H = M3::Zero();
  for (size_t i = 0; i < s.size(); ++i)
    H += (s[i] - a) * (t[i] - b).transpose();
  Eigen::JacobiSVD<M3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  M3 D = M3::Identity();
  D(2, 2) =
      (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0 ? -1 : 1;
  M3 R = svd.matrixV() * D * svd.matrixU().transpose();
  return {R, b - R * a};
}
namespace {
bool noncollinear(const std::vector<V3> &p) {
  return (p[1] - p[0]).cross(p[2] - p[0]).norm() > 1e-7;
}
std::vector<int> score(const Pose &T, const std::vector<V3> &s,
                       const std::vector<V3> &t, double threshold) {
  std::vector<int> ids;
  for (size_t i = 0; i < s.size(); ++i)
    if ((T.apply(s[i]) - t[i]).norm() < threshold)
      ids.push_back(i);
  return ids;
}
Pose fitSubset(const std::vector<V3> &s, const std::vector<V3> &t,
               const std::vector<int> &ids) {
  std::vector<V3> a, b;
  for (int i : ids) {
    a.push_back(s[i]);
    b.push_back(t[i]);
  }
  return rigidSVD(a, b);
}
double medianResidual(const Pose &T, const std::vector<V3> &s,
                      const std::vector<V3> &t, const std::vector<int> &ids) {
  if (ids.empty())
    return INFINITY;
  std::vector<double> r;
  for (int i : ids)
    r.push_back((T.apply(s[i]) - t[i]).norm());
  std::sort(r.begin(), r.end());
  size_t m = r.size() / 2;
  return r.size() % 2 ? r[m] : (r[m - 1] + r[m]) / 2;
}
// Exact balanced 3D KD tree. No registration library or approximate search.
class KDTree {
  const std::vector<V3> &p;
  std::vector<int> ids, left, right, axis;
  int root;
  int build(int begin, int end, int depth) {
    if (begin >= end)
      return -1;
    int mid = (begin + end) / 2, a = depth % 3;
    std::nth_element(ids.begin() + begin, ids.begin() + mid, ids.begin() + end,
                     [&](int i, int j) { return p[i][a] < p[j][a]; });
    int id = ids[mid];
    axis[id] = a;
    left[id] = build(begin, mid, depth + 1);
    right[id] = build(mid + 1, end, depth + 1);
    return id;
  }
  void query(int node, const V3 &q, int &best, double &d) const {
    if (node < 0)
      return;
    double r = (q - p[node]).squaredNorm();
    if (r < d) {
      d = r;
      best = node;
    }
    double delta = q[axis[node]] - p[node][axis[node]];
    int near = delta < 0 ? left[node] : right[node],
        far = delta < 0 ? right[node] : left[node];
    query(near, q, best, d);
    if (delta * delta < d)
      query(far, q, best, d);
  }

public:
  explicit KDTree(const std::vector<V3> &points)
      : p(points), ids(p.size()), left(p.size()), right(p.size()),
        axis(p.size()) {
    std::iota(ids.begin(), ids.end(), 0);
    root = build(0, p.size(), 0);
  }
  std::pair<int, double> nearest(const V3 &q) const {
    int id = -1;
    double d = INFINITY;
    query(root, q, id, d);
    return {id, std::sqrt(d)};
  }
};
struct Match {
  int s, t;
  double d;
};
std::vector<Match> nearestPairs(const std::vector<V3> &s,
                                const std::vector<V3> &t, const KDTree &tree,
                                const Pose &T, const Config &c) {
  std::vector<Match> v;
  for (size_t i = 0; i < s.size(); ++i) {
    auto [id, d] = tree.nearest(T.apply(s[i]));
    if (id >= 0 && std::isfinite(d) && d < c.max_distance)
      v.push_back({int(i), id, d});
  }
  std::sort(v.begin(), v.end(),
            [](const Match &a, const Match &b) { return a.d < b.d; });
  if (v.size() > 3)
    v.resize(
        std::min(v.size(), std::max(size_t(3), size_t(v.size() * c.trim))));
  std::vector<bool> used(t.size());
  std::vector<Match> out;
  for (auto m : v)
    if (!used[m.t]) {
      used[m.t] = true;
      out.push_back(m);
    }
  return out;
}
bool safeMotion(const Pose &T, const Pose &initial, const Config &c) {
  Pose d = T * initial.inverse();
  return angleDeg(d.R) <= c.max_angle_deg && d.t.norm() <= c.max_translation;
}
void finish(const std::vector<Match> &m, const Config &c, SolverInfo &info) {
  info.pairs = m.size();
  double sum = 0;
  for (auto a : m)
    sum += a.d * a.d;
  info.rmse = m.empty() ? INFINITY : std::sqrt(sum / m.size());
  info.accepted = info.iterations > 0 && info.pairs >= c.min_pairs &&
                  std::isfinite(info.rmse) && info.rmse < c.max_distance;
}
double huber(double r, double d) {
  double a = std::abs(r);
  return a <= d ? .5 * a * a : d * (a - .5 * d);
}
void recordIteration(const std::vector<Match> &pairs,
                     const std::vector<V3> &s, const std::vector<V3> &t,
                     const std::vector<V3> *normals, const Pose &before,
                     const Pose &after, const Config &c, SolverInfo &info) {
  IterationInfo row;
  row.iteration = info.iterations;
  row.pairs = int(pairs.size());
  std::vector<std::pair<int, int>> assignments;
  for (auto p : pairs) {
    V3 a = before.apply(s[p.s]) - t[p.t];
    V3 b = after.apply(s[p.s]) - t[p.t];
    row.rmse_before += a.squaredNorm();
    row.rmse_after += b.squaredNorm();
    if (normals) {
      double ra = (*normals)[p.t].dot(a), rb = (*normals)[p.t].dot(b);
      row.normal_before += ra * ra;
      row.normal_after += rb * rb;
      row.objective_before += huber(ra, c.huber);
      row.objective_after += huber(rb, c.huber);
    } else {
      row.objective_before += a.squaredNorm();
      row.objective_after += b.squaredNorm();
    }
    assignments.emplace_back(p.s, p.t);
  }
  double count = double(pairs.size());
  row.rmse_before = std::sqrt(row.rmse_before / count);
  row.rmse_after = std::sqrt(row.rmse_after / count);
  row.normal_before = std::sqrt(row.normal_before / count);
  row.normal_after = std::sqrt(row.normal_after / count);
  row.objective_before /= count;
  row.objective_after /= count;
  Pose delta = after * before.inverse();
  row.rotation_step_deg = angleDeg(delta.R);
  row.translation_step_m = delta.t.norm();
  // Sort by source ID so changes in distance ordering do not change the hash.
  std::sort(assignments.begin(), assignments.end());
  row.assignment_hash = 14695981039346656037ULL;
  for (auto [source, target] : assignments)
    for (auto id : {source, target}) {
      row.assignment_hash ^= std::uint64_t(id);
      row.assignment_hash *= 1099511628211ULL;
    }
  info.history.push_back(row);
}
V3 project(double u, double v, double z, const Camera &k) {
  return {(u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z};
}
bool stableDepth(const cv::Mat &d, int x, int y, const Config &c, double &z) {
  if (x < 2 || y < 2 || x >= d.cols - 2 || y >= d.rows - 2)
    return false;
  double lo = INFINITY, hi = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      double v = d.at<uint16_t>(y + dy, x + dx) / c.depth_scale;
      if (v <= 0)
        return false;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
  z = d.at<uint16_t>(y, x) / c.depth_scale;
  return hi - lo < c.depth_edge;
}
void cloud(const Frame &f, const Camera &k, const Config &c, std::vector<V3> &p,
           std::vector<V3> *normals) {
  for (int y = 2; y < k.height - 2; y += c.stride)
    for (int x = 2; x < k.width - 2; x += c.stride) {
      double z;
      if (!stableDepth(f.depth, x, y, c, z))
        continue;
      if (normals) {
        auto get = [&](int u, int v) {
          return project(u, v, f.depth.at<uint16_t>(v, u) / c.depth_scale, k);
        };
        V3 dx = get(x + 1, y) - get(x - 1, y),
           dy = get(x, y + 1) - get(x, y - 1), n = dx.cross(dy);
        if (n.norm() <= 1e-10)
          continue;
        normals->push_back(n.normalized());
      }
      p.push_back(project(x, y, z, k));
    }
  if (int(p.size()) > c.max_points) {
    std::vector<int> ids(p.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 rng(c.seed);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<V3> out, ns;
    for (int j = 0; j < c.max_points; ++j) {
      out.push_back(p[ids[j]]);
      if (normals)
        ns.push_back((*normals)[ids[j]]);
    }
    p = std::move(out);
    if (normals)
      *normals = std::move(ns);
  }
}
} // namespace
void matchedFeaturePoints(const Frame &a, const Frame &b, const Camera &k,
                   const Config &c, const ImageFeatures &fa,
                   const ImageFeatures &fb, std::vector<V3> &target,
                   std::vector<V3> &source, int &count) {
  if (fa.pixels.size() != fa.descriptors.size() ||
      fb.pixels.size() != fb.descriptors.size())
    throw std::invalid_argument("Feature position/descriptor count mismatch");
  auto matches = features::MatchMutualRatio(fa.descriptors, fb.descriptors, c.ratio);
  count = matches.size();
  for (auto match : matches) {
    auto uv = fa.pixels[match.query], vv = fb.pixels[match.train];
    double z1, z2;
    if (!stableDepth(a.depth, int(std::nearbyint(uv.x())),
                     int(std::nearbyint(uv.y())), c, z1) ||
        !stableDepth(b.depth, int(std::nearbyint(vv.x())),
                     int(std::nearbyint(vv.y())), c, z2))
      continue;
    target.push_back(project(uv.x(), uv.y(), z1, k));
    source.push_back(project(vv.x(), vv.y(), z2, k));
  }
}
bool ransac(const std::vector<V3> &s, const std::vector<V3> &t, const Config &c,
            Pose &T, std::vector<int> &best, unsigned seed) {
  if (s.size() != t.size())
    throw std::invalid_argument("RANSAC shape mismatch");
  best.clear();
  if (s.size() < 3)
    return false;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(0, s.size() - 1);
  double bestMedian = INFINITY;
  for (int iteration = 0; iteration < c.ransac_iterations; ++iteration) {
    std::array<int, 3> ids;
    ids[0] = d(rng);
    do {
      ids[1] = d(rng);
    } while (ids[1] == ids[0]);
    do {
      ids[2] = d(rng);
    } while (ids[2] == ids[0] || ids[2] == ids[1]);
    std::vector<V3> a, b;
    for (int id : ids) {
      a.push_back(s[id]);
      b.push_back(t[id]);
    }
    if (!noncollinear(a) || !noncollinear(b))
      continue;
    Pose candidate = rigidSVD(a, b);
    auto in = score(candidate, s, t, c.ransac_threshold);
    double med = medianResidual(candidate, s, t, in);
    if (in.size() > best.size() ||
        (in.size() == best.size() && med < bestMedian)) {
      best = in;
      bestMedian = med;
    }
  }
  if (int(best.size()) < c.min_inliers)
    return false;
  for (int i = 0; i < 4; ++i) {
    T = fitSubset(s, t, best);
    auto in = score(T, s, t, c.ransac_threshold);
    if (int(in.size()) < c.min_inliers || in == best)
      break;
    best = std::move(in);
  }
  T = fitSubset(s, t, best);
  return true;
}
Pose pointICP(const std::vector<V3> &s, const std::vector<V3> &t,
              const Pose &initial, const Config &c, SolverInfo &info) {
  info = SolverInfo{};
  Pose T = initial;
  KDTree tree(t);
  info.reason = "ITERATION_LIMIT";
  for (int i = 0; i < c.point_iterations; ++i) {
    // Refresh spatial correspondences at the CURRENT estimate, then hold them
    // fixed during linearization and backtracking.
    auto pairs = nearestPairs(s, t, tree, T, c);
    if (int(pairs.size()) < c.min_pairs) {
      info.reason = "INSUFFICIENT_SUPPORT";
      break;
    }
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    double old = 0;
    for (auto p : pairs) {
      V3 x = T.apply(s[p.s]), r = x - t[p.t];
      auto J = pointJacobian(x); // [-[x]_cross | I], rotation then translation
      H.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * r;
      old += r.squaredNorm();
    }
    // Small fixed regularization stabilizes GN; no adaptive LM schedule.
    H.diagonal().array() += 1e-5 * H.trace() / 6 + 1e-9;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
    Eigen::Matrix<double, 6, 1> step = solver.solve(b);
    if (solver.info() != Eigen::Success || !step.allFinite()) {
      info.reason = "NUMERICAL_FAILURE";
      break;
    }
    // Scale the entire direction, preserving the GN descent direction.
    double scale = 1.;
    if (step.head<3>().norm() > .1) scale = std::min(scale, .1 / step.head<3>().norm());
    if (step.tail<3>().norm() > .03) scale = std::min(scale, .03 / step.tail<3>().norm());
    step *= scale;
    bool improved = false;
    double alphaUsed = 0;
    for (double alpha : {1., .5, .25, .125, .0625}) {
      Pose candidate = leftIncrement(T, alpha * step);
      if (!safeMotion(candidate, initial, c)) continue;
      double cost = 0;
      for (auto p : pairs) cost += (candidate.apply(s[p.s]) - t[p.t]).squaredNorm();
      if (cost <= old + 1e-14 * pairs.size()) {
        Pose before = T;
        T = candidate;
        ++info.iterations;
        recordIteration(pairs, s, t, nullptr, before, T, c, info);
        improved = true;
        alphaUsed = alpha;
        break;
      }
    }
    if (!improved) { info.reason = "NO_DESCENT"; break; }
    if (alphaUsed * step.head<3>().norm() < 1e-7 &&
        alphaUsed * step.tail<3>().norm() < 1e-7) {
      info.converged = true;
      info.reason = "CONVERGED";
      break;
    }
  }
  finish(nearestPairs(s, t, tree, T, c), c, info);
  return T;
}
Pose planeICP(const std::vector<V3> &s, const std::vector<V3> &t,
              const std::vector<V3> &normals, const Pose &initial,
              const Config &c, SolverInfo &info) {
  if (t.size() != normals.size())
    throw std::invalid_argument("Normal count mismatch");
  info = SolverInfo{};
  Pose T = initial;
  KDTree tree(t);
  info.reason = "ITERATION_LIMIT";
  for (int i = 0; i < c.plane_iterations; ++i) {
    auto pairs = nearestPairs(s, t, tree, T, c);
    if (int(pairs.size()) < c.min_pairs) {
      info.reason = "INSUFFICIENT_SUPPORT";
      break;
    }
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    double old = 0;
    for (auto p : pairs) {
      V3 x = T.apply(s[p.s]), n = normals[p.t];
      double r = n.dot(x - t[p.t]),
             w = std::abs(r) <= c.huber ? 1 : c.huber / std::abs(r);
      Eigen::Matrix<double, 6, 1> J = pointJacobian(x).transpose() * n;
      H += w * J * J.transpose();
      b -= w * J * r;
      old += huber(r, c.huber);
    }
    double damp = 1e-5 * H.trace() / 6 + 1e-9;
    H.diagonal().array() += damp;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
    Eigen::Matrix<double, 6, 1> step = solver.solve(b);
    if (solver.info() != Eigen::Success || !step.allFinite()) {
      info.reason = "NUMERICAL_FAILURE";
      break;
    }
    double scale = 1.;
    if (step.head<3>().norm() > .1) scale = std::min(scale, .1 / step.head<3>().norm());
    if (step.tail<3>().norm() > .03) scale = std::min(scale, .03 / step.tail<3>().norm());
    step *= scale;
    bool improved = false;
    double alphaUsed = 0;
    for (double alpha : {1., .5, .25, .125, .0625}) {
      Pose cand = leftIncrement(T, alpha * step);
      if (!safeMotion(cand, initial, c))
        continue;
      double cost = 0;
      for (auto p : pairs)
        cost += huber(normals[p.t].dot(cand.apply(s[p.s]) - t[p.t]), c.huber);
      if (cost <= old + 1e-14 * pairs.size()) {
        Pose before = T;
        T = cand;
        improved = true;
        alphaUsed = alpha;
        ++info.iterations;
        recordIteration(pairs, s, t, &normals, before, T, c, info);
        break;
      }
    }
    if (!improved) {
      info.reason = "NO_DESCENT";
      break;
    }
    if (alphaUsed * step.head<3>().norm() < 1e-7 &&
        alphaUsed * step.tail<3>().norm() < 1e-7) {
      info.converged = true;
      info.reason = "CONVERGED";
      break;
    }
  }
  finish(nearestPairs(s, t, tree, T, c), c, info);
  return T;
}
Camera readCamera(const std::string &directory) {
  Camera k;
  std::ifstream f(directory + "/intrinsics.txt");
  if (!(f >> k.width >> k.height >> k.fx >> k.fy >> k.cx >> k.cy) ||
      k.width < 5 || k.height < 5 || k.fx <= 0 || k.fy <= 0)
    throw std::runtime_error("Invalid intrinsics");
  return k;
}
Frame readFrame(const std::string &directory, int i, const Camera &k) {
  std::ostringstream name;
  name << directory << '/' << std::setw(6) << std::setfill('0') << i;
  Frame f{cv::imread(name.str() + ".png", cv::IMREAD_COLOR),
          cv::imread(name.str() + "_depth.png", cv::IMREAD_UNCHANGED)};
  if (f.rgb.empty() || f.depth.empty() || f.depth.type() != CV_16UC1 ||
      f.depth.cols != k.width || f.depth.rows != k.height ||
      f.rgb.size() != f.depth.size())
    throw std::runtime_error("Invalid RGB-D frame " + std::to_string(i));
  return f;
}
ImageFeatures extractFeatures(const cv::Mat &image, const Config &c) {
  auto gray = features::GrayFromInterleaved(
      image.data, image.cols, image.rows, image.step, image.channels());
  features::FastParams fast;
  fast.threshold = c.fast_threshold;
  auto f = features::ExtractMultiscale(features::BuildPyramid(gray), fast, {true, c.features});
  ImageFeatures out;
  out.descriptors = std::move(f.descriptors);
  for (auto kp : f.keypoints) {
    out.pixels.emplace_back(kp.x, kp.y);
    out.levels.push_back(kp.level);
    out.angles_rad.push_back(kp.level_kp.angle);
  }
  return out;
}
PairResult registerFrames(const Frame &a, const Frame &b, const Camera &k,
                          const Config &c, unsigned seed, PairClouds *diagnostics) {
  auto fa = extractFeatures(a.rgb, c), fb = extractFeatures(b.rgb, c);
  return registerFramesWithFeatures(a, b, k, c, fa, fb, seed, diagnostics);
}
PairResult registerFramesWithFeatures(const Frame &a, const Frame &b, const Camera &k,
                          const Config &c, const ImageFeatures &fa,
                          const ImageFeatures &fb, unsigned seed, PairClouds *diagnostics) {
  if (diagnostics)
    *diagnostics = PairClouds{};
  PairResult out;
  std::vector<V3> target, source;
  matchedFeaturePoints(a, b, k, c, fa, fb, target, source, out.matches);
  out.valid3d = source.size();
  if (source.size() < 3)
    return out;
  out.baseline = rigidSVD(source, target);
  std::vector<int> in;
  if (!ransac(source, target, c, out.robust, in, seed)) {
    out.status = "RANSAC_FAILED";
    out.inliers = in.size();
    return out;
  }
  out.inliers = in.size();
  out.inlier_ratio = double(in.size()) / source.size();
  std::vector<V3> s, t, n;
  cloud(b, k, c, s, nullptr);
  cloud(a, k, c, t, &n);
  if (diagnostics)
    *diagnostics = {s, t, n};
  if (int(s.size()) < c.min_pairs || int(t.size()) < c.min_pairs) {
    out.status = "INSUFFICIENT_CLOUD";
    return out;
  }
  out.point = pointICP(s, t, out.robust, c, out.point_info);
  out.plane = planeICP(s, t, n, out.point, c, out.plane_info);
  out.accepted = out.point_info.accepted && out.plane_info.accepted;
  out.status = out.accepted ? "OK" : "ICP_REJECTED";
  return out;
}
std::vector<Edge> findLoops(const std::string &dir, const Camera &k,
                            const Config &c, const std::vector<Pose> &poses) {
  struct Candidate {
    int i, j;
    double d;
  };
  std::vector<Candidate> cand;
  for (int i = 0; i < int(poses.size()); ++i)
    for (int j = i + 25; j < int(poses.size()); ++j) {
      Pose p = poses[i].inverse() * poses[j];
      if (p.t.norm() < .12 && angleDeg(p.R) < 32)
        cand.push_back({i, j, p.t.norm()});
    }
  std::sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.d < b.d; });
  if (cand.size() > 8)
    cand.resize(8);
  std::vector<Edge> edges;
  for (auto v : cand) {
    auto r = registerFrames(readFrame(dir, v.i, k), readFrame(dir, v.j, k), k,
                            c, c.seed + v.i);
    if (!r.accepted || r.inliers < std::max(35, c.min_inliers) ||
        r.inlier_ratio < .65)
      continue;
    Pose diff = r.plane.inverse() * (poses[v.i].inverse() * poses[v.j]);
    if (angleDeg(diff.R) <= 8 && diff.t.norm() <= .045)
      edges.push_back({v.i, v.j, r.plane, true});
  }
  return edges;
}
namespace {
using V6 = Eigen::Matrix<double, 6, 1>;
V6 residual(const Edge &e, const std::vector<Pose> &poses) {
  Pose error = e.measurement.inverse() * (poses[e.i].inverse() * poses[e.j]);
  V6 r;
  r << logSO3(error.R) / (e.loop ? .015 : .025),
      error.t / (e.loop ? .003 : .005);
  return r;
}
// Component-wise soft-L1 with scale 2, matching the reference objective.
double graphCost(const std::vector<Pose> &poses,
                 const std::vector<Edge> &edges) {
  double sum = 0;
  for (auto e : edges) {
    auto r = residual(e, poses);
    for (int k = 0; k < 6; ++k)
      sum += 4 * (std::sqrt(1 + r[k] * r[k] / 4) - 1);
  }
  return sum;
}
Pose perturb(const Pose &p, int axis, double delta) {
  V3 w = V3::Zero(), v = V3::Zero();
  if (axis < 3)
    w[axis] = delta;
  else
    v[axis - 3] = delta;
  return Pose{expSO3(w), v} * p;
}
} // namespace
GraphResult optimizeGraph(const std::vector<Pose> &initial,
                          const std::vector<Edge> &edges, int limit) {
  GraphResult result;
  result.poses = initial;
  if (initial.empty())
    throw std::invalid_argument("Empty pose graph");
  for (auto e : edges)
    if (e.i < 0 || e.j < 0 || e.i >= int(initial.size()) ||
        e.j >= int(initial.size()))
      throw std::invalid_argument("Invalid edge index");
  result.initial_cost = result.final_cost = graphCost(initial, edges);
  if (edges.empty() || initial.size() < 2) {
    result.converged = true;
    return result;
  }
  // Fix one pose per disconnected component, avoiding free gauges after failed
  // odometry.
  std::vector<int> parent(initial.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int i) {
    while (parent[i] != i)
      i = parent[i];
    return i;
  };
  for (auto e : edges) {
    int a = find(e.i), b = find(e.j);
    parent[std::max(a, b)] = std::min(a, b);
  }
  std::vector<int> offsets(initial.size(), -1);
  int dim = 0;
  for (int i = 0; i < int(initial.size()); ++i)
    if (find(i) != i) {
      offsets[i] = dim;
      dim += 6;
    }
  if (!dim) {
    result.converged = true;
    return result;
  }
  double lambda = 1e-5;
  for (int it = 0; it < limit; ++it) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(dim);
    for (auto e : edges) {
      V6 r = residual(e, result.poses), weight;
      for (int q = 0; q < 6; ++q)
        weight[q] = 1 / std::sqrt(1 + r[q] * r[q] / 4);
      std::array<Eigen::Matrix<double, 6, 6>, 2> J;
      int nodes[2] = {e.i, e.j};
      for (int side = 0; side < 2; ++side) {
        int node = nodes[side];
        if (offsets[node] < 0)
          continue;
        Pose original = result.poses[node];
        for (int q = 0; q < 6; ++q) {
          result.poses[node] = perturb(original, q, 1e-6);
          V6 plus = residual(e, result.poses);
          result.poses[node] = perturb(original, q, -1e-6);
          V6 minus = residual(e, result.poses);
          J[side].col(q) = (plus - minus) / 2e-6;
        }
        result.poses[node] = original;
      }
      for (int a = 0; a < 2; ++a) {
        int oa = offsets[nodes[a]];
        if (oa < 0)
          continue;
        b.segment<6>(oa) -= J[a].transpose() * weight.asDiagonal() * r;
        for (int d = 0; d < 2; ++d) {
          int od = offsets[nodes[d]];
          if (od >= 0)
            H.block<6, 6>(oa, od) +=
                J[a].transpose() * weight.asDiagonal() * J[d];
        }
      }
    }
    if (b.lpNorm<Eigen::Infinity>() < 1e-7) {
      result.converged = true;
      break;
    }
    bool accepted = false;
    double before = result.final_cost;
    Eigen::VectorXd step;
    for (int attempt = 0; attempt < 8; ++attempt) {
      Eigen::MatrixXd damped = H;
      damped.diagonal().array() += lambda;
      step = damped.ldlt().solve(b);
      if (!step.allFinite()) {
        lambda *= 10;
        continue;
      }
      auto trial = result.poses;
      for (int i = 0; i < int(trial.size()); ++i)
        if (offsets[i] >= 0) {
          int o = offsets[i];
          trial[i] = Pose{expSO3(step.segment<3>(o)), step.segment<3>(o + 3)} *
                     trial[i];
        }
      double cost = graphCost(trial, edges);
      if (cost <= before) {
        result.poses = std::move(trial);
        result.final_cost = cost;
        lambda = std::max(1e-10, lambda / 3);
        accepted = true;
        break;
      }
      lambda *= 10;
    }
    ++result.iterations;
    if (!accepted)
      break;
    if (step.norm() < 1e-7 ||
        std::abs(before - result.final_cost) < 1e-9 * (1 + before)) {
      result.converged = true;
      break;
    }
  }
  return result;
}
} // namespace rgbd_icp
