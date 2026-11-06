#include "dataset/tum.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <tuple>

namespace tum {

namespace {

bool OpenLines(const std::string &path, std::vector<std::string> &lines, std::string *error) {
  std::ifstream f(path);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::string line;
  while (std::getline(f, line)) {
    const size_t p = line.find_first_not_of(" \t\r");
    if (p == std::string::npos || line[p] == '#') continue;
    lines.push_back(line);
  }
  return true;
}

}  // namespace

bool ReadFileList(const std::string &path, std::vector<Stamped> &out, std::string *error) {
  std::vector<std::string> lines;
  if (!OpenLines(path, lines, error)) return false;
  out.clear();
  for (const std::string &l : lines) {
    std::istringstream s(l);
    Stamped e;
    if (!(s >> e.t >> e.file)) {
      if (error) *error = path + ": malformed line '" + l + "'";
      return false;
    }
    out.push_back(e);
  }
  std::stable_sort(out.begin(), out.end(), [](const Stamped &a, const Stamped &b) { return a.t < b.t; });
  return true;
}

bool ReadGroundtruth(const std::string &path, std::vector<GtSample> &out, std::string *error) {
  std::vector<std::string> lines;
  if (!OpenLines(path, lines, error)) return false;
  out.clear();
  for (const std::string &l : lines) {
    std::istringstream s(l);
    GtSample g;
    double qx, qy, qz, qw;
    if (!(s >> g.t >> g.p.x() >> g.p.y() >> g.p.z() >> qx >> qy >> qz >> qw)) {
      if (error) *error = path + ": malformed line '" + l + "'";
      return false;
    }
    g.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    out.push_back(g);
  }
  std::stable_sort(out.begin(), out.end(), [](const GtSample &a, const GtSample &b) { return a.t < b.t; });
  return true;
}

std::vector<Association> Associate(const std::vector<Stamped> &rgb, const std::vector<Stamped> &depth, double max_dt) {
  struct Cand {
    double dt;
    size_t r, d;
  };
  std::vector<Cand> cand;
  for (size_t r = 0; r < rgb.size(); ++r)
    for (size_t d = 0; d < depth.size(); ++d) {
      const double dt = std::abs(rgb[r].t - depth[d].t);
      if (dt <= max_dt) cand.push_back({dt, r, d});
    }
  std::sort(cand.begin(), cand.end(), [&](const Cand &a, const Cand &b) {
    return std::tie(a.dt, rgb[a.r].t, depth[a.d].t) < std::tie(b.dt, rgb[b.r].t, depth[b.d].t);
  });
  std::vector<bool> used_r(rgb.size(), false), used_d(depth.size(), false);
  std::vector<Association> out;
  for (const Cand &c : cand) {
    if (used_r[c.r] || used_d[c.d]) continue;
    used_r[c.r] = used_d[c.d] = true;
    out.push_back({rgb[c.r].t, depth[c.d].t, rgb[c.r].file, depth[c.d].file});
  }
  std::sort(out.begin(), out.end(), [](const Association &a, const Association &b) { return a.t_rgb < b.t_rgb; });
  return out;
}

Eigen::Matrix3d QuaternionToR(double qx, double qy, double qz, double qw) {
  return Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
}

bool InterpolateGroundtruth(const std::vector<GtSample> &gt, double t, double max_gap, Pose &out, std::string *why) {
  if (gt.empty() || t < gt.front().t || t > gt.back().t) {
    if (why) *why = "no ground truth: timestamp outside the ground-truth time span";
    return false;
  }
  const auto hi = std::lower_bound(gt.begin(), gt.end(), t, [](const GtSample &g, double v) { return g.t < v; });
  if (hi->t == t) {  // exact sample
    out.R = hi->q.toRotationMatrix();
    out.t = hi->p;
    return true;
  }
  const GtSample &b = *hi, &a = *(hi - 1);
  if (b.t - a.t > max_gap) {
    std::ostringstream m;
    m << "no ground truth: bracketing samples " << (b.t - a.t) << " s apart (max " << max_gap << " s)";
    if (why) *why = m.str();
    return false;
  }
  const double s = (t - a.t) / (b.t - a.t);
  out.t = (1 - s) * a.p + s * b.p;
  out.R = a.q.slerp(s, b.q).normalized().toRotationMatrix();  // Eigen's slerp takes the shortest arc
  return true;
}

bool OfficialIntrinsics(const std::string &sequence_name, CameraIntrinsics &out) {
  // https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats (calibrated colour-camera intrinsics)
  struct Entry {
    const char *sensor;
    double fx, fy, cx, cy, d[5];
  };
  static const Entry table[] = {
      {"freiburg1", 517.3, 516.5, 318.6, 255.3, {0.2624, -0.9531, -0.0054, 0.0026, 1.1633}},
      {"freiburg2", 520.9, 521.0, 325.1, 249.7, {0.2312, -0.7849, -0.0033, -0.0001, 0.9172}},
      {"freiburg3", 535.4, 539.2, 320.1, 247.6, {0, 0, 0, 0, 0}},
  };
  for (const Entry &e : table)
    if (sequence_name.find(e.sensor) != std::string::npos) {
      out = CameraIntrinsics();
      out.sensor = e.sensor;
      out.fx = e.fx, out.fy = e.fy, out.cx = e.cx, out.cy = e.cy;
      std::copy(e.d, e.d + 5, out.d);
      out.depth_factor = 5000.0;
      return true;
    }
  return false;
}

bool SelectFrames(const std::string &sequence_dir, double max_dt, double max_gt_gap, int stride, int max_frames,
                  std::vector<Frame> &frames, SelectionReport &report, std::string *error) {
  namespace fs = std::filesystem;
  std::vector<Stamped> rgb, depth;
  std::vector<GtSample> gt;
  const fs::path dir(sequence_dir);
  if (!ReadFileList((dir / "rgb.txt").string(), rgb, error) || !ReadFileList((dir / "depth.txt").string(), depth, error) ||
      !ReadGroundtruth((dir / "groundtruth.txt").string(), gt, error))
    return false;
  report = SelectionReport();
  report.rgb = int(rgb.size());
  report.depth = int(depth.size());
  const std::vector<Association> assoc = Associate(rgb, depth, max_dt);
  report.associated = int(assoc.size());
  frames.clear();
  int usable = 0;
  for (const Association &a : assoc) {
    std::ostringstream ts;
    ts.precision(6);
    ts << std::fixed << a.t_rgb;
    const fs::path pr = dir / a.rgb, pd = dir / a.depth;
    if (!fs::exists(pr) || !fs::exists(pd)) {
      ++report.missing_image;
      report.skipped.push_back(ts.str() + ": missing image " + (fs::exists(pr) ? a.depth : a.rgb));
      continue;
    }
    Frame f;
    std::string why;
    if (!InterpolateGroundtruth(gt, a.t_rgb, max_gt_gap, f.T_wc, &why)) {
      ++report.missing_groundtruth;
      report.skipped.push_back(ts.str() + ": " + why);
      continue;
    }
    if (usable++ % std::max(1, stride) != 0) continue;
    f.index = int(frames.size());
    f.t_rgb = a.t_rgb;
    f.t_depth = a.t_depth;
    f.rgb = pr.string();
    f.depth = pd.string();
    frames.push_back(f);
    if (max_frames > 0 && int(frames.size()) >= max_frames) break;
  }
  return true;
}

}  // namespace tum
