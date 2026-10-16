// depth_edge_experiment: read-only experiment on the scratch pipeline (the
// production pipeline is NOT changed). It runs the unchanged front end
// (pyramid, FAST, orientation, rotated BRIEF, Hamming, filter, depth lookup)
// and then the unchanged geometric stage (RANSAC around the linear PnP + LM
// refinement) twice per pair, on two correspondence sets:
//
//   all       every correspondence with frame-i depth   (= the production pipeline)
//   interior  the same, minus the correspondences whose depth sample is at a
//             depth edge (rule below)
//
// Depth-edge rule, for the depth sample (truncated original pixel) of a frame-i
// keypoint detected on pyramid level l (scale s):
//   window   all pixels within radius r = ceil(3 s) (the FAST circle radius 3,
//            mapped from the keypoint's level to the original image)
//   edge if  any pixel of the window has no depth (silhouette against the
//            background), or two 4-adjacent pixels of the window differ in depth
//            by more than tau(Z) = (Z / fx) tan(80 deg): the step a continuous
//            surface makes between neighbouring pixels only when it is within
//            10 deg of tangent to the view ray (Z = depth of the sample).
//
// Outputs (generated): pnp_from_scratch/results/depth_edge/
//   pairs.csv                       per pair, for both sets: counts, inliers, errors, reprojection
//   trajectory_{all,interior}.csv   accumulated T_wc and its errors per frame
//   correspondences/pair_<i>_<j>.csv  every PnP input: pixels, level, depth statistics of the
//                                     window, edge flag, inlier flags in both runs
//
// Usage: depth_edge_experiment [--incidence-deg A] [--radius-factor F]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "common.hpp"
#include "pipeline.hpp"

using namespace scratch;

namespace {

struct EdgeStats {
  int radius = 0, pixels = 0, valid = 0;
  double z_min = 0, z_max = 0;  // metres, valid pixels of the window
  double max_jump = 0;          // largest 4-adjacent depth difference between valid pixels (m)
  double gradient = 0;          // central-difference |grad Z| at the sample (m / px); NaN if a neighbour is invalid
  double tau = 0;               // jump threshold at the sample's depth (m)
  bool edge = false;
};

double Depth(const PngImage &d, int x, int y, double scale) {
  if (x < 0 || y < 0 || x >= d.width || y >= d.height) return 0;
  return d.samples[size_t(y) * d.width + x] / scale;
}

EdgeStats Classify(const PngImage &depth, double u, double v, float level_scale, const Intrinsics &K,
                   double depth_scale, double incidence_deg, double radius_factor) {
  EdgeStats s;
  const int x0 = int(u), y0 = int(v);
  const double z0 = Depth(depth, x0, y0, depth_scale);
  s.radius = int(std::ceil(radius_factor * level_scale - 1e-9));
  s.tau = z0 / K.fx * std::tan(incidence_deg * M_PI / 180.0);
  s.z_min = std::numeric_limits<double>::infinity();
  s.z_max = 0;
  const int r = s.radius;
  for (int dy = -r; dy <= r; ++dy)
    for (int dx = -r; dx <= r; ++dx) {
      if (dx * dx + dy * dy > r * r) continue;  // circular window
      ++s.pixels;
      const double z = Depth(depth, x0 + dx, y0 + dy, depth_scale);
      if (z <= 0) continue;
      ++s.valid;
      s.z_min = std::min(s.z_min, z);
      s.z_max = std::max(s.z_max, z);
      for (const auto &n : {std::pair<int, int>{1, 0}, std::pair<int, int>{0, 1}}) {
        const int ex = dx + n.first, ey = dy + n.second;
        if (ex * ex + ey * ey > r * r) continue;
        const double zn = Depth(depth, x0 + ex, y0 + ey, depth_scale);
        if (zn > 0) s.max_jump = std::max(s.max_jump, std::abs(zn - z));
      }
    }
  const double zl = Depth(depth, x0 - 1, y0, depth_scale), zr = Depth(depth, x0 + 1, y0, depth_scale),
               zu = Depth(depth, x0, y0 - 1, depth_scale), zd = Depth(depth, x0, y0 + 1, depth_scale);
  s.gradient = (zl > 0 && zr > 0 && zu > 0 && zd > 0) ? std::hypot(0.5 * (zr - zl), 0.5 * (zd - zu)) : NAN;
  s.edge = s.valid < s.pixels || s.max_jump > s.tau;
  return s;
}

struct Run {
  bool ok = false;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity(), R_lin = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero(), t_lin = Eigen::Vector3d::Zero();
  int n = 0, inliers = 0;
  std::vector<bool> mask;  // per selected input
  double rep_mean = 0, rep_median = 0, rep_max = 0;
};

// The geometric stage of ProcessPair (pipeline.cpp), unchanged: RANSAC around the
// linear PnP, then LM refinement on the RANSAC inliers.
Run Geometric(const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector2d> &uv, const Intrinsics &K,
              const PipelineParams &params) {
  Run out;
  out.n = int(X.size());
  const RansacPnPResult p = RansacPnP(X, uv, K, params.pnp_ransac);
  if (!p.ok) return out;
  out.ok = true;
  out.R = out.R_lin = p.R;
  out.t = out.t_lin = p.t;
  out.inliers = p.num_inliers;
  out.mask = p.inlier_mask;
  const RefineResult f = RefinePnP(X, uv, p.inlier_mask, K, p.R, p.t, params.refine_params);
  if (f.ok) out.R = f.R, out.t = f.t;
  std::vector<double> r;
  for (size_t k = 0; k < X.size(); ++k)
    if (p.inlier_mask[k]) r.push_back((projectPoint(X[k], out.R, out.t, K) - uv[k]).norm());
  if (!r.empty()) {
    double sum = 0;
    for (double v : r) sum += v;
    out.rep_mean = sum / r.size();
    std::sort(r.begin(), r.end());
    out.rep_median = r.size() % 2 ? r[r.size() / 2] : 0.5 * (r[r.size() / 2 - 1] + r[r.size() / 2]);
    out.rep_max = r.back();
  }
  return out;
}

std::string F(double v, int p = 6) {
  if (!std::isfinite(v)) return "";
  std::ostringstream o;
  o << std::fixed << std::setprecision(p) << v;
  return o.str();
}

Pose Inv(const Pose &T) {
  Pose r;
  r.R = T.R.transpose();
  r.t = -r.R * T.t;
  return r;
}
Pose Mul(const Pose &A, const Pose &B) {
  Pose r;
  r.R = A.R * B.R;
  r.t = A.R * B.t + A.t;
  return r;
}

}  // namespace

int main(int argc, char **argv) {
  double incidence_deg = 80.0, radius_factor = 3.0;
  for (int a = 1; a + 1 < argc; ++a) {
    const std::string s = argv[a];
    if (s == "--incidence-deg") incidence_deg = std::stod(argv[++a]);
    else if (s == "--radius-factor") radius_factor = std::stod(argv[++a]);
  }
  const fs::path repo = FindRepo(fs::current_path());
  if (repo.empty()) {
    std::cerr << "run inside the repository\n";
    return 2;
  }
  const fs::path ds = repo / "data" / "synthetic_bunny";
  Intrinsics K;
  {
    std::ifstream f(ds / "intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  const std::vector<Pose> gt = ReadGroundtruthTwc(ds / "groundtruth.txt");
  const int N = int(gt.size());
  const PipelineParams params;  // the production defaults
  const fs::path out = repo / "pnp_from_scratch" / "results" / "depth_edge";
  fs::create_directories(out / "correspondences");

  std::vector<Run> all(N - 1), inner(N - 1);
  std::vector<int> n_edge(N - 1), n_edge_inl(N - 1), n_filtered(N - 1), kp_i(N - 1);
  RgbdFrame fa, fb;
  LoadRgbdFrame(ds.string(), 0, fa);
  FrameFeatures ea = ExtractFeatures(fa.rgb, params.frontend), eb;
  for (int i = 0; i + 1 < N; ++i) {
    LoadRgbdFrame(ds.string(), i + 1, fb);
    eb = ExtractFeatures(fb.rgb, params.frontend);
    const PairMatches pm = MatchFrames(ea, eb, fa.depth, fb.depth, K, params.frontend);
    n_filtered[i] = int(pm.correspondences.size());
    kp_i[i] = int(ea.keypoints.size());
    std::vector<Eigen::Vector3d> X, Xi;
    std::vector<Eigen::Vector2d> uv, uvi;
    std::vector<EdgeStats> st;
    std::vector<int> corr;
    for (size_t m = 0; m < pm.correspondences.size(); ++m) {
      const Correspondence &c = pm.correspondences[m];
      if (!c.has_3d) continue;
      const EdgeStats e = Classify(fa.depth, c.uv_i.x(), c.uv_i.y(), ea.scale[c.query], K, params.frontend.depth_scale,
                                   incidence_deg, radius_factor);
      X.push_back(c.X_i);
      uv.push_back(c.uv_j);
      st.push_back(e);
      corr.push_back(int(m));
      if (!e.edge) Xi.push_back(c.X_i), uvi.push_back(c.uv_j);
    }
    all[i] = Geometric(X, uv, K, params);
    inner[i] = Geometric(Xi, uvi, K, params);
    std::ofstream c(out / "correspondences" / ("pair_" + std::to_string(i) + "_" + std::to_string(i + 1) + ".csv"));
    c << "match,level_i,u_i,v_i,u_j,v_j,depth_raw_i,depth_raw_j,X_i,Y_i,Z_i,radius_px,window_pixels,valid_pixels,"
         "z_min_m,z_max_m,max_jump_m,gradient_m_per_px,tau_m,edge,inlier_all,inlier_interior\n";
    for (size_t k = 0, q = 0; k < st.size(); ++k) {
      const Correspondence &cc = pm.correspondences[corr[k]];
      const EdgeStats &e = st[k];
      const bool in_all = all[i].ok && all[i].mask[k];
      const bool in_int = !e.edge && inner[i].ok && inner[i].mask[q];
      if (!e.edge) ++q;
      n_edge[i] += e.edge;
      n_edge_inl[i] += e.edge && in_all;
      c << corr[k] << "," << cc.level_i << "," << F(cc.uv_i.x(), 3) << "," << F(cc.uv_i.y(), 3) << ","
        << F(cc.uv_j.x(), 3) << "," << F(cc.uv_j.y(), 3) << "," << cc.depth_raw_i << "," << cc.depth_raw_j << ","
        << F(cc.X_i.x()) << "," << F(cc.X_i.y()) << "," << F(cc.X_i.z()) << "," << e.radius << "," << e.pixels << ","
        << e.valid << "," << F(e.z_min) << "," << F(e.z_max) << "," << F(e.max_jump) << "," << F(e.gradient, 7)
        << "," << F(e.tau) << "," << int(e.edge) << "," << int(in_all) << "," << int(in_int) << "\n";
    }
    std::swap(fa, fb);
    std::swap(ea, eb);
  }

  // trajectories and errors
  auto trajectory = [&](const std::vector<Run> &runs, const std::string &name, std::vector<double> &et,
                        std::vector<double> &er) {
    std::vector<Pose> T(N);
    T[0] = gt[0];
    for (int i = 0; i + 1 < N; ++i) {
      Pose rel;
      rel.R = runs[i].R;
      rel.t = runs[i].t;
      T[i + 1] = runs[i].ok ? Mul(T[i], Inv(rel)) : T[i];
    }
    std::ofstream f(out / ("trajectory_" + name + ".csv"));
    f << "frame,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22,translation_error_m,rotation_error_deg\n";
    et.assign(N, 0);
    er.assign(N, 0);
    for (int k = 0; k < N; ++k) {
      et[k] = (T[k].t - gt[k].t).norm();
      er[k] = RotErrDeg(gt[k].R, T[k].R);
      f << k << "," << F(T[k].t.x(), 9) << "," << F(T[k].t.y(), 9) << "," << F(T[k].t.z(), 9);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) f << "," << F(T[k].R(a, b), 9);
      f << "," << F(et[k], 9) << "," << F(er[k], 6) << "\n";
    }
  };
  std::vector<double> ta, ra, ti, ri;
  trajectory(all, "all", ta, ra);
  trajectory(inner, "interior", ti, ri);

  std::ofstream f(out / "pairs.csv");
  f << "pair,keypoints_i,filtered_matches,depth_correspondences,edge_correspondences,edge_inliers_removed,"
       "all_inliers,all_rot_err_deg,all_trans_err_m,all_signed_rot_deg,all_reproj_mean,all_reproj_median,all_reproj_max,"
       "interior_correspondences,interior_ok,interior_inliers,interior_rot_err_deg,interior_trans_err_m,"
       "interior_signed_rot_deg,interior_reproj_mean,interior_reproj_median,interior_reproj_max\n";
  double sa = 0, si = 0, era = 0, eri = 0, eta = 0, eti = 0;
  int neg_a = 0, neg_i = 0, tot_d = 0, tot_e = 0, tot_ei = 0, tot_ia = 0;
  for (int i = 0; i + 1 < N; ++i) {
    const Pose G = Mul(Inv(gt[i + 1]), gt[i]);
    const double ga = RotAngleDeg(G.R);
    const Run &a = all[i], &b = inner[i];
    const double sra = RotAngleDeg(a.R) - ga, sri = b.ok ? RotAngleDeg(b.R) - ga : NAN;
    sa += sra, si += sri, neg_a += sra < 0, neg_i += sri < 0;
    era += RotErrDeg(G.R, a.R), eri += RotErrDeg(G.R, b.R);
    eta += (G.t - a.t).norm(), eti += (G.t - b.t).norm();
    tot_d += a.n, tot_e += n_edge[i], tot_ei += n_edge_inl[i], tot_ia += a.inliers;
    f << i << "," << kp_i[i] << "," << n_filtered[i] << "," << a.n << "," << n_edge[i] << "," << n_edge_inl[i] << ","
      << a.inliers << "," << F(RotErrDeg(G.R, a.R)) << "," << F((G.t - a.t).norm()) << "," << F(sra) << ","
      << F(a.rep_mean, 4) << "," << F(a.rep_median, 4) << "," << F(a.rep_max, 4) << "," << b.n << "," << int(b.ok)
      << "," << b.inliers << "," << F(RotErrDeg(G.R, b.R)) << "," << F((G.t - b.t).norm()) << "," << F(sri) << ","
      << F(b.rep_mean, 4) << "," << F(b.rep_median, 4) << "," << F(b.rep_max, 4) << "\n";
  }
  const int P = N - 1;
  double mta = 0, mti = 0, mra = 0, mri = 0;
  for (int k = 0; k < N; ++k) mta += ta[k] / N, mti += ti[k] / N, mra += ra[k] / N, mri += ri[k] / N;
  std::printf("depth-edge rule: radius ceil(%.1f x level scale), jump > (Z/fx) tan(%.0f deg), or no depth in the window\n",
              radius_factor, incidence_deg);
  std::printf("PnP inputs: %d with depth, %d edge (%.1f%%); %d of %d RANSAC inliers of the 'all' run are edge (%.1f%%)\n",
              tot_d, tot_e, 100.0 * tot_e / tot_d, tot_ei, tot_ia, 100.0 * tot_ei / tot_ia);
  std::printf("%-9s mean pair rot %.3f deg, trans %.4f m; signed rot sum %+.1f deg (%d/35 negative); "
              "trajectory mean %.4f m / %.2f deg, final %.4f m / %.2f deg\n", "all", era / P, eta / P, sa, neg_a, mta, mra,
              ta[N - 1], ra[N - 1]);
  std::printf("%-9s mean pair rot %.3f deg, trans %.4f m; signed rot sum %+.1f deg (%d/35 negative); "
              "trajectory mean %.4f m / %.2f deg, final %.4f m / %.2f deg\n", "interior", eri / P, eti / P, si, neg_i, mti,
              mri, ti[N - 1], ri[N - 1]);
  return 0;
}
