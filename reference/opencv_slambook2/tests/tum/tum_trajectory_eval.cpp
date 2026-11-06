// tum_trajectory_eval: the existing frontend, PnP and robust ICP of the
// reference pipeline on a TUM RGB-D sequence, through the dataset adapter
// (src/dataset/tum.cpp). The per-pair steps are those of
// tests/synthetic/slam_trajectory_test.cpp, unchanged:
//   find_feature_matches -> depth of frame i at the truncated keypoint pixel
//   -> solvePnPRansac(100 iterations, 8 px, 0.99) -> T_{j<-i}
//   ICP: matches with depth in both frames -> EstimateRigidRansac (1 cm) ->
//   bundleAdjustment on its inliers -> native T_{i<-j} inverted once -> T_{j<-i}
//   accumulation: T_wc[0] = T_gt[0], T_wc[i+1] = T_wc[i] * T_rel^-1 (failed pair: previous pose held)
// Only the input differs: frames, intrinsics and ground truth from the adapter.
// TUM depth images are registered to the colour camera, so the colour
// intrinsics back-project them. Lens distortion is not corrected (the pipeline
// has no undistortion step); the coefficients are reported.
//
// Usage (from build/):
//   ./tum_trajectory_eval ../data/rgbd_dataset_freiburg1_desk [--frames N] [--stride S]
//        [--max-dt SEC] [--max-gt-gap SEC] [--out-dir DIR] [--export-pairs LIST] [--pnp-min-distinct-inliers N]
// Defaults: 20 frames, stride 1, max-dt 0.02 s, max-gt-gap 0.05 s,
// out-dir ../results/runs/tum/<sequence name>. The output directory must not be
// inside the data directory: dataset, ground truth and the synthetic Bunny
// files are only read.
//
// --export-pairs LIST (opt-in, off by default): for the listed pairs i -> i+1
// (pair index = frame_i in pairs.csv; "73", "70-76", "10,73"), additionally
// write <out-dir>/diagnostics/pair_<i>_<i+1>/ with every keypoint, every
// filtered match, its depth, the PnP and ICP classification the estimators
// actually returned, residuals and the transforms (see WriteDiagnostics).
// The estimate and the four standard output files are identical with or
// without it.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>

#include "camera/camera.hpp"
#include "dataset/tum.hpp"
#include "features/features.hpp"
#include "tracking/icp.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

namespace {

struct Silence {  // the book functions print; keep the report readable
  streambuf *old;
  ostringstream sink;
  Silence() : old(cout.rdbuf(sink.rdbuf())) {}
  ~Silence() { cout.rdbuf(old); }
};

Sophus::SE3d CvToSE3(const Mat &R, const Mat &t) {
  Eigen::Matrix3d Re;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) Re(r, c) = R.at<double>(r, c);
  return Sophus::SE3d(Eigen::Quaterniond(Re).normalized(), Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
}
double RotErrDeg(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) {
  return acos(min(1.0, max(-1.0, ((A.transpose() * B).trace() - 1) / 2))) * 180 / M_PI;
}

struct PairOut {
  int matches = 0, pnp_corr = 0, pnp_inliers = 0, icp_corr = 0, icp_inliers = 0;
  bool pnp_ok = false, icp_ok = false;  // pnp_ok: pose ACCEPTED for the trajectory (see AcceptPnp)
  bool pnp_returned = false;            // solvePnPRansac returned a pose (solver status, kept separately)
  int pnp_inl_distinct_i = 0, pnp_inl_distinct_j = 0;  // distinct keypoints among the PnP inliers
  string pnp_reject;                    // why a returned pose was not accepted ("" if accepted)
  Sophus::SE3d pnp, icp;            // T_{j<-i}; pnp = the returned pose, also when rejected
  double gt_reproj_median = NAN;    // PnP inliers projected with the TRUE relative pose (convention check)
  double wrong_reproj_median = NAN; // the same with the inverted convention (T_wc read as T_cw): control
};

// Opt-in per-pair diagnostics (--export-pairs): copies of values RunPair
// computes anyway; nothing here feeds back into the estimate.
struct PairDetail {
  vector<KeyPoint> kp_i, kp_j;
  vector<DMatch> matches;            // the filtered matches, in find_feature_matches order
  vector<ushort> d_i, d_j;           // per match: raw depth at the truncated keypoint pixel
  vector<int> pnp_idx, icp_idx;      // per match: index into the PnP / ICP correspondence arrays, -1 if not used
  vector<Point3f> pts_3d, p1, p2;
  vector<Point2f> pts_2d;
  bool pnp_called = false, pnp_returned = false;
  vector<int> pnp_inliers;           // solvePnPRansac's inlier list (indices into pts_3d)
  bool icp_called = false;
  RigidRansacResult rr;
  Mat R_icp, t_icp;                  // native T_{i<-j} after bundleAdjustment on the RANSAC inliers
  // Export-only checks, not part of the estimate: cv::solvePnP on the reported
  // inliers alone, to compare with the returned pose (name, status, T_{j<-i}).
  struct Refit {
    string name, status;
    Sophus::SE3d T;
  };
  vector<Refit> refits;
};

PairOut RunPair(const tum::Frame &a, const tum::Frame &b, const Mat &K, double depth_factor, const Sophus::SE3d &G,
                const Sophus::SE3d &G_wrong, PairDetail *det = nullptr) {
  PairOut o;
  const Mat img_i = imread(a.rgb, IMREAD_COLOR), img_j = imread(b.rgb, IMREAD_COLOR);
  const Mat dep_i = imread(a.depth, IMREAD_UNCHANGED), dep_j = imread(b.depth, IMREAD_UNCHANGED);
  if (img_i.empty() || img_j.empty() || dep_i.empty() || dep_j.empty())
    throw runtime_error("cannot read the images of frames " + to_string(a.index) + "/" + to_string(b.index));
  if (dep_i.type() != CV_16UC1 || dep_j.type() != CV_16UC1) throw runtime_error("depth image is not 16-bit, 1 channel");
  vector<KeyPoint> kp_i, kp_j;
  vector<DMatch> matches;
  {
    Silence s;
    find_feature_matches(img_i, img_j, kp_i, kp_j, matches);
  }
  o.matches = int(matches.size());
  if (det) det->kp_i = kp_i, det->kp_j = kp_j, det->matches = matches;
  // PnP (as RunPnpPair)
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  vector<Point3f> p1, p2;
  vector<int> pnp_kpi, pnp_kpj;  // keypoint indices of each PnP correspondence (for the validity check)
  for (const DMatch &m : matches) {
    const Point2f &u = kp_i[m.queryIdx].pt, &v = kp_j[m.trainIdx].pt;
    const ushort d1 = dep_i.ptr<ushort>(int(u.y))[int(u.x)], d2 = dep_j.ptr<ushort>(int(v.y))[int(v.x)];
    if (det) det->d_i.push_back(d1), det->d_j.push_back(d2), det->pnp_idx.push_back(-1), det->icp_idx.push_back(-1);
    if (!tum::ValidDepth(d1)) continue;
    const float z1 = float(tum::DepthMetres(d1, depth_factor));
    const Point2d c1 = pixel2cam(u, K);
    pts_3d.push_back(Point3f(c1.x * z1, c1.y * z1, z1));
    pts_2d.push_back(v);
    pnp_kpi.push_back(m.queryIdx), pnp_kpj.push_back(m.trainIdx);
    if (det) det->pnp_idx.back() = int(pts_3d.size()) - 1;
    if (!tum::ValidDepth(d2)) continue;  // ICP needs depth in both frames
    const float z2 = float(tum::DepthMetres(d2, depth_factor));
    const Point2d c2 = pixel2cam(v, K);
    p1.push_back(Point3f(c1.x * z1, c1.y * z1, z1));
    p2.push_back(Point3f(c2.x * z2, c2.y * z2, z2));
    if (det) det->icp_idx.back() = int(p1.size()) - 1;
  }
  if (det) det->pts_3d = pts_3d, det->pts_2d = pts_2d, det->p1 = p1, det->p2 = p2;
  o.pnp_corr = int(pts_3d.size());
  if (o.pnp_corr >= 4) {
    Mat r, t, R;
    vector<int> inl;
    const bool returned = solvePnPRansac(pts_3d, pts_2d, K, Mat(), r, t, false, 100, 8.0f, 0.99, inl);
    if (det) {
      det->pnp_called = true;
      det->pnp_returned = returned;
      det->pnp_inliers = inl;
      if (returned && inl.size() >= 4) {
        vector<Point3f> s3;
        vector<Point2f> s2;
        for (int k : inl) s3.push_back(pts_3d[k]), s2.push_back(pts_2d[k]);
        Mat r_epnp, t_epnp;
        auto attempt = [&](const string &name, int flags, bool guess, Mat rv, Mat tv) {
          PairDetail::Refit f{name, "", Sophus::SE3d()};
          try {
            if (solvePnP(s3, s2, K, Mat(), rv, tv, guess, flags)) {
              Mat Rm;
              Rodrigues(rv, Rm);
              f.T = CvToSE3(Rm, tv);
              f.status = "ok";
              if (flags == SOLVEPNP_EPNP) r_epnp = rv.clone(), t_epnp = tv.clone();
            } else {
              f.status = "solvePnP returned false";
            }
          } catch (const cv::Exception &e) {
            f.status = "exception: " + e.err;
          }
          det->refits.push_back(f);
        };
        attempt("iterative_no_guess", SOLVEPNP_ITERATIVE, false, Mat(), Mat());
        attempt("epnp", SOLVEPNP_EPNP, false, Mat(), Mat());
        if (!r_epnp.empty()) attempt("iterative_from_epnp", SOLVEPNP_ITERATIVE, true, r_epnp.clone(), t_epnp.clone());
      }
    }
    if (returned) {
      Rodrigues(r, R);
      o.pnp = CvToSE3(R, t);
      o.pnp_inliers = int(inl.size());
      o.pnp_ok = o.pnp_returned = true;
      vector<int> di, dj;
      for (int k : inl) di.push_back(pnp_kpi[k]), dj.push_back(pnp_kpj[k]);
      sort(di.begin(), di.end()), sort(dj.begin(), dj.end());
      o.pnp_inl_distinct_i = int(unique(di.begin(), di.end()) - di.begin());
      o.pnp_inl_distinct_j = int(unique(dj.begin(), dj.end()) - dj.begin());
      for (int pass = 0; pass < 2; ++pass) {
        const Sophus::SE3d &T = pass == 0 ? G : G_wrong;
        vector<double> res;
        for (int k : inl) {
          const Eigen::Vector3d X = T * Eigen::Vector3d(pts_3d[k].x, pts_3d[k].y, pts_3d[k].z);
          const double u = K.at<double>(0, 0) * X.x() / X.z() + K.at<double>(0, 2);
          const double v = K.at<double>(1, 1) * X.y() / X.z() + K.at<double>(1, 2);
          res.push_back(hypot(u - pts_2d[k].x, v - pts_2d[k].y));
        }
        if (res.empty()) continue;
        nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
        (pass == 0 ? o.gt_reproj_median : o.wrong_reproj_median) = res[res.size() / 2];
      }
    }
  }
  // robust ICP (as RunIcpPair)
  o.icp_corr = int(p1.size());
  if (o.icp_corr >= 3) {
    const RigidRansacResult rr = EstimateRigidRansac(p1, p2);
    if (det) det->icp_called = true, det->rr = rr;
    if (rr.ok) {
      vector<Point3f> in1, in2;
      for (size_t k = 0; k < p1.size(); ++k)
        if (rr.inliers[k]) in1.push_back(p1[k]), in2.push_back(p2[k]);
      Mat R_icp, t_icp;
      {
        Silence s;
        bundleAdjustment(in1, in2, R_icp, t_icp);
      }
      if (det) det->R_icp = R_icp.clone(), det->t_icp = t_icp.clone();
      const Mat R_common = R_icp.t(), t_common = -R_common * t_icp;  // native T_{i<-j} -> T_{j<-i}, once
      o.icp = CvToSE3(R_common, t_common);
      o.icp_inliers = rr.num_inliers;
      o.icp_ok = true;
    }
  }
  return o;
}

// Geometric validity of a returned PnP pose (tracking acceptance; the solver status stays in pnp_returned).
// Several matches can share one keypoint (nearest-neighbour matching is many-to-one), so the inlier COUNT
// can overstate the constraints: k inlier rows on one image point fix only 2 of the 6 pose parameters.
// A pose is accepted only if its inliers contain at least min_distinct distinct keypoints in frame i AND in
// frame j. Default 6: the minimum cv::solvePnP itself needs to re-estimate a pose from the inliers alone
// (SOLVEPNP_ITERATIVE without a guess initialises with DLT, "needs at least 6 points"). 0 = accept every
// returned pose (the previous behaviour). A rejected pair is handled like a solver failure: pose held.
void AcceptPnp(PairOut &o, int min_distinct) {
  if (!o.pnp_returned || min_distinct <= 0) return;
  const int d = min(o.pnp_inl_distinct_i, o.pnp_inl_distinct_j);
  if (d < min_distinct) {
    o.pnp_ok = false;
    o.pnp_reject = "inliers have " + to_string(o.pnp_inl_distinct_i) + " distinct frame-i and " +
                   to_string(o.pnp_inl_distinct_j) + " distinct frame-j keypoints (< " + to_string(min_distinct) + ")";
  }
}

// "73", "70-76", "10,73,80-82" -> sorted unique pair indices; false on malformed input.
bool ParsePairList(const string &s, vector<int> &out) {
  out.clear();
  stringstream ss(s);
  string tok;
  while (getline(ss, tok, ',')) {
    const size_t dash = tok.find('-');
    try {
      size_t used = 0;
      const int lo = stoi(tok.substr(0, dash), &used);
      if (used != (dash == string::npos ? tok.size() : dash) || lo < 0) return false;
      int hi = lo;
      if (dash != string::npos) {
        const string rest = tok.substr(dash + 1);
        hi = stoi(rest, &used);
        if (used != rest.size() || hi < lo) return false;
      }
      for (int k = lo; k <= hi; ++k) out.push_back(k);
    } catch (const exception &) {
      return false;
    }
  }
  sort(out.begin(), out.end());
  out.erase(unique(out.begin(), out.end()), out.end());
  return !out.empty();
}

string Json(double v) {  // NaN / inf -> null
  if (!std::isfinite(v)) return "null";
  ostringstream s;
  s << setprecision(12) << v;
  return s.str();
}
string JsonStr(const string &v) {
  string r = "\"";
  for (char c : v) {
    if (c == '"' || c == '\\') r += string("\\") + c;
    else if (c == '\n') r += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) r += ' ';  // other control characters
    else r += c;
  }
  return r + "\"";
}
string JsonT(const Sophus::SE3d &T) {  // 4x4, row-major
  const Eigen::Matrix4d M = T.matrix();
  ostringstream s;
  s << "[";
  for (int r = 0; r < 4; ++r) {
    s << (r ? ", [" : "[");
    for (int c = 0; c < 4; ++c) s << (c ? ", " : "") << Json(M(r, c));
    s << "]";
  }
  return s.str() + "]";
}

// Writes dir/{pair.json, keypoints_i.csv, keypoints_j.csv, matches.csv}.
// Transforms T_{j<-i}: X_j = R X_i + t, camera optical axes (x right, y down,
// z forward), metres. Residuals are recomputed here from the exported values
// with the same formulas the estimators use (PnP: pinhole projection without
// distortion, as solvePnPRansac with empty distortion; ICP: |p_i - (R p_j + t)|
// in its native T_{i<-j} direction).
void WriteDiagnostics(const fs::path &dir, int pair, const tum::Frame &a, const tum::Frame &b, const fs::path &seq,
                      const tum::CameraIntrinsics &in, const Mat &K, const Sophus::SE3d &G, const PairOut &o,
                      const PairDetail &d) {
  fs::create_directories(dir);
  const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
  auto project = [&](const Sophus::SE3d &T, const Point3f &P, double &u, double &v, double &z) {
    const Eigen::Vector3d X = T * Eigen::Vector3d(P.x, P.y, P.z);
    z = X.z();
    u = fx * X.x() / X.z() + cx;
    v = fy * X.y() / X.z() + cy;
  };
  auto rigid_res = [](const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Point3f &p, const Point3f &q) {
    return (Eigen::Vector3d(p.x, p.y, p.z) - (R * Eigen::Vector3d(q.x, q.y, q.z) + t)).norm();
  };
  Eigen::Matrix3d R_icp_native = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_icp_native = Eigen::Vector3d::Zero();
  if (!d.R_icp.empty())
    for (int r = 0; r < 3; ++r) {
      t_icp_native(r) = d.t_icp.at<double>(r);
      for (int c = 0; c < 3; ++c) R_icp_native(r, c) = d.R_icp.at<double>(r, c);
    }
  const Sophus::SE3d G_native = G.inverse();  // T_{i<-j}
  vector<char> pnp_in(d.pts_3d.size(), 0);
  for (int k : d.pnp_inliers) pnp_in[k] = 1;
  const bool pnp_classified = d.pnp_called && d.pnp_returned;
  const bool icp_classified = d.icp_called && d.rr.ok;
  for (int side = 0; side < 2; ++side) {
    ofstream f(dir / (side == 0 ? "keypoints_i.csv" : "keypoints_j.csv"));
    f << "kp,x,y,angle_deg,response\n" << setprecision(9);
    const auto &kps = side == 0 ? d.kp_i : d.kp_j;
    for (size_t k = 0; k < kps.size(); ++k)
      f << k << "," << kps[k].pt.x << "," << kps[k].pt.y << "," << kps[k].angle << "," << kps[k].response << "\n";
  }
  {
    ofstream f(dir / "matches.csv");
    f << "match,kp_i,kp_j,hamming,u_i,v_i,u_j,v_j,depth_raw_i,depth_raw_j,depth_m_i,depth_m_j,"
         "pnp_corr,pnp_class,X_i,Y_i,Z_i,pnp_proj_u,pnp_proj_v,pnp_z_j,pnp_res_px,gt_proj_u,gt_proj_v,gt_z_j,gt_res_px,"
         "icp_corr,icp_class,Xj_x,Xj_y,Xj_z,icp_res_m,icp_ransac_res_m,icp_gt_res_m\n"
      << setprecision(9);
    for (size_t m = 0; m < d.matches.size(); ++m) {
      const DMatch &dm = d.matches[m];
      const Point2f &u = d.kp_i[dm.queryIdx].pt, &v = d.kp_j[dm.trainIdx].pt;
      f << m << "," << dm.queryIdx << "," << dm.trainIdx << "," << dm.distance << "," << u.x << "," << u.y << "," << v.x
        << "," << v.y << "," << d.d_i[m] << "," << d.d_j[m] << "," << tum::DepthMetres(d.d_i[m], in.depth_factor) << ","
        << tum::DepthMetres(d.d_j[m], in.depth_factor) << ",";
      const int pk = d.pnp_idx[m];
      if (pk < 0) {
        f << "-1,no_depth_i" << string(11, ',');  // X_i .. gt_res_px empty
      } else {
        const Point3f &P = d.pts_3d[pk];
        f << pk << "," << (!pnp_classified ? "unclassified" : pnp_in[pk] ? "inlier" : "outlier") << "," << P.x << ","
          << P.y << "," << P.z << ",";
        double pu, pv, pz, gu, gv, gz;
        if (o.pnp_returned) {
          project(o.pnp, P, pu, pv, pz);
          f << pu << "," << pv << "," << pz << "," << hypot(pu - v.x, pv - v.y) << ",";
        } else {
          f << ",,,,";
        }
        project(G, P, gu, gv, gz);
        f << gu << "," << gv << "," << gz << "," << hypot(gu - v.x, gv - v.y);
      }
      f << ",";
      const int ik = d.icp_idx[m];
      if (ik < 0) {
        f << "-1," << (pk < 0 ? "no_depth_i" : "no_depth_j") << string(6, ',');  // Xj_x .. icp_gt_res_m empty
      } else {
        const Point3f &p = d.p1[ik], &q = d.p2[ik];
        f << ik << "," << (!icp_classified ? "unclassified" : d.rr.inliers[ik] ? "inlier" : "rejected") << "," << q.x
          << "," << q.y << "," << q.z << ",";
        if (o.icp_ok) f << rigid_res(R_icp_native, t_icp_native, p, q);
        f << ",";
        if (icp_classified) f << rigid_res(d.rr.R, d.rr.t, p, q);
        f << "," << rigid_res(G_native.rotationMatrix(), G_native.translation(), p, q);
      }
      f << "\n";
    }
  }
  int n_valid_i = 0, n_valid_j = 0;
  for (size_t m = 0; m < d.matches.size(); ++m) n_valid_i += tum::ValidDepth(d.d_i[m]), n_valid_j += tum::ValidDepth(d.d_j[m]);
  ofstream f(dir / "pair.json");
  const auto rel = [&](const string &p) { return JsonStr(fs::path(p).lexically_relative(seq).string()); };
  const Sophus::SE3d Twc_i(Eigen::Quaterniond(a.T_wc.R).normalized(), a.T_wc.t),
      Twc_j(Eigen::Quaterniond(b.T_wc.R).normalized(), b.T_wc.t);
  f << "{\n  \"format\": \"tum_trajectory_eval pair diagnostics v1\",\n"
    << "  \"pair\": " << pair << ", \"frame_i\": " << a.index << ", \"frame_j\": " << b.index << ",\n"
    << "  \"t_rgb_i\": " << fixed << setprecision(6) << a.t_rgb << ", \"t_rgb_j\": " << b.t_rgb
    << ", \"t_depth_i\": " << a.t_depth << ", \"t_depth_j\": " << b.t_depth << ",\n"
    << defaultfloat << "  \"rgb_i\": " << rel(a.rgb) << ", \"rgb_j\": " << rel(b.rgb) << ", \"depth_i\": " << rel(a.depth)
    << ", \"depth_j\": " << rel(b.depth) << ",\n"
    << "  \"opencv_version\": " << JsonStr(CV_VERSION) << ",\n"
    << "  \"intrinsics\": {\"sensor\": " << JsonStr(in.sensor) << ", \"fx\": " << Json(in.fx) << ", \"fy\": " << Json(in.fy)
    << ", \"cx\": " << Json(in.cx) << ", \"cy\": " << Json(in.cy) << ", \"width\": " << in.width << ", \"height\": "
    << in.height << ", \"depth_factor\": " << Json(in.depth_factor) << ", \"distortion_applied\": false},\n"
    << "  \"conventions\": {\"T_j_from_i\": \"X_j = R X_i + t (camera optical axes x right, y down, z forward; metres)\", "
       "\"T_wc\": \"X_world = R X_cam + t\", \"icp_native\": \"T_i_from_j: p_i = R p_j + t\", "
       "\"depth_lookup\": \"raw depth at (int(x), int(y)) of the keypoint; 0 = invalid\"},\n"
    << "  \"config\": {\"matching\": \"FAST/rotated BRIEF, brute-force Hamming nearest neighbour, kept if distance <= "
       "max(2 d_min, 30)\", \"pnp\": {\"function\": \"cv::solvePnPRansac\", \"iterations\": 100, "
       "\"reprojection_threshold_px\": 8.0, \"confidence\": 0.99, \"flags\": \"SOLVEPNP_ITERATIVE\", "
       "\"use_extrinsic_guess\": false, \"min_correspondences\": 4}, \"icp\": {\"ransac\": \"EstimateRigidRansac\", "
       "\"threshold_m\": 0.01, \"iterations\": 300, \"seed\": 12345, \"min_correspondences\": 3, \"refinement\": "
       "\"bundleAdjustment (g2o) on the RANSAC inliers\"}},\n"
    << "  \"counts\": {\"keypoints_i\": " << d.kp_i.size() << ", \"keypoints_j\": " << d.kp_j.size() << ", \"matches\": "
    << d.matches.size() << ", \"depth_valid_i\": " << n_valid_i << ", \"depth_valid_j\": " << n_valid_j
    << ", \"pnp_correspondences\": " << d.pts_3d.size() << ", \"pnp_inliers\": " << d.pnp_inliers.size()
    << ", \"icp_correspondences\": " << d.p1.size() << ", \"icp_inliers\": " << d.rr.num_inliers << "},\n"
    << "  \"status\": {\"pnp_called\": " << (d.pnp_called ? "true" : "false") << ", \"pnp_solver_returned\": "
    << (d.pnp_returned ? "true" : "false") << ", \"pnp_ok\": " << (o.pnp_ok ? "true" : "false")
    << ", \"pnp_accepted\": " << (o.pnp_ok ? "true" : "false") << ", \"pnp_reject_reason\": " << JsonStr(o.pnp_reject)
    << ", \"pnp_inlier_distinct_i\": " << o.pnp_inl_distinct_i << ", \"pnp_inlier_distinct_j\": " << o.pnp_inl_distinct_j
    << ", \"icp_called\": " << (d.icp_called ? "true" : "false") << ", \"icp_ransac_ok\": " << (d.rr.ok ? "true" : "false")
    << ", \"icp_ok\": " << (o.icp_ok ? "true" : "false") << "},\n"
    << "  \"pnp_inlier_indices\": [";
  for (size_t k = 0; k < d.pnp_inliers.size(); ++k) f << (k ? ", " : "") << d.pnp_inliers[k];
  f << "],\n  \"T_wc_gt_i\": " << JsonT(Twc_i) << ",\n  \"T_wc_gt_j\": " << JsonT(Twc_j) << ",\n"
    << "  \"T_j_from_i_gt\": " << JsonT(G) << ",\n"
    << "  \"T_j_from_i_pnp\": " << (o.pnp_returned ? JsonT(o.pnp) : "null") << ",\n"
    << "  \"T_j_from_i_icp\": " << (o.icp_ok ? JsonT(o.icp) : "null") << ",\n"
    << "  \"T_i_from_j_icp_ransac_native\": "
    << (d.rr.ok ? JsonT(Sophus::SE3d(Eigen::Quaterniond(d.rr.R).normalized(), d.rr.t)) : "null") << ",\n"
    << "  \"T_i_from_j_icp_native\": "
    << (o.icp_ok ? JsonT(Sophus::SE3d(Eigen::Quaterniond(R_icp_native).normalized(), t_icp_native)) : "null") << ",\n"
    << "  \"export_only_pnp_refits_on_inliers\": {\"description\": \"not part of the estimate: cv::solvePnP on the "
       "reported PnP inliers alone (no distortion), to compare with the returned pose\", \"attempts\": [";
  for (size_t k = 0; k < d.refits.size(); ++k)
    f << (k ? ", " : "") << "{\"name\": " << JsonStr(d.refits[k].name) << ", \"status\": " << JsonStr(d.refits[k].status)
      << ", \"T_j_from_i\": " << (d.refits[k].status == "ok" ? JsonT(d.refits[k].T) : "null") << "}";
  f << "]}\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "usage: tum_trajectory_eval SEQUENCE_DIR [--frames N] [--stride S] [--max-dt SEC] [--max-gt-gap SEC] "
            "[--out-dir DIR] [--export-pairs LIST] [--pnp-min-distinct-inliers N]\n";
    return 2;
  }
  const fs::path seq = fs::weakly_canonical(fs::path(argv[1]));
  int frames_n = 20, stride = 1;
  double max_dt = 0.02, max_gap = 0.05;
  fs::path out;
  vector<int> export_pairs;  // empty: no diagnostics (default)
  int pnp_min_distinct = 6;  // AcceptPnp; 0 = accept every returned pose (previous behaviour)
  for (int a = 2; a + 1 < argc; ++a) {
    const string s = argv[a];
    if (s == "--frames") frames_n = stoi(argv[++a]);
    else if (s == "--stride") stride = stoi(argv[++a]);
    else if (s == "--max-dt") max_dt = stod(argv[++a]);
    else if (s == "--max-gt-gap") max_gap = stod(argv[++a]);
    else if (s == "--out-dir") out = argv[++a];
    else if (s == "--pnp-min-distinct-inliers") pnp_min_distinct = stoi(argv[++a]);
    else if (s == "--export-pairs") {
      if (!ParsePairList(argv[++a], export_pairs)) {
        cerr << "--export-pairs: expected pair indices such as 73, 70-76 or 10,73\n";
        return 2;
      }
    } else {
      cerr << "unknown option " << s << "\n";
      return 2;
    }
  }
  const string name = seq.filename().string();
  if (out.empty()) out = fs::path("../results/runs/tum") / name;
  out = fs::weakly_canonical(out);
  const fs::path data_dir = seq.parent_path();
  const auto rel = out.lexically_relative(data_dir);
  if (!rel.empty() && *rel.begin() != "..") {  // never write into the data directory
    cerr << "refusing to write into the data directory " << data_dir << "; choose another --out-dir\n";
    return 2;
  }
  tum::CameraIntrinsics in;
  if (!tum::OfficialIntrinsics(name, in)) {
    cerr << "no official intrinsics for '" << name << "' (expected freiburg1/2/3 in the name)\n";
    return 2;
  }
  vector<tum::Frame> frames;
  tum::SelectionReport rep;
  string err;
  if (!tum::SelectFrames(seq.string(), max_dt, max_gap, stride, frames_n, frames, rep, &err)) {
    cerr << err << "\n";
    return 1;
  }
  cout << "TUM sequence " << name << " (" << in.sensor << "): fx " << in.fx << " fy " << in.fy << " cx " << in.cx
       << " cy " << in.cy << ", depth factor " << in.depth_factor << "; distortion k1 " << in.d[0]
       << " (not corrected)\n"
       << "rgb " << rep.rgb << ", depth " << rep.depth << ", associated (|dt| <= " << max_dt << " s) " << rep.associated
       << "; skipped before the selection was complete: " << rep.missing_image << " missing image, "
       << rep.missing_groundtruth << " missing ground truth; selected " << frames.size() << " frames (stride " << stride
       << ")\n";
  for (const string &s : rep.skipped) cout << "  skipped " << s << "\n";
  if (frames.size() < 2) {
    cerr << "fewer than 2 usable frames\n";
    return 1;
  }
  const Mat K = (Mat_<double>(3, 3) << in.fx, 0, in.cx, 0, in.fy, in.cy, 0, 0, 1);
  const int N = int(frames.size());
  vector<Sophus::SE3d> T_gt(N);
  for (int k = 0; k < N; ++k) T_gt[k] = Sophus::SE3d(Eigen::Quaterniond(frames[k].T_wc.R).normalized(), frames[k].T_wc.t);

  if (!export_pairs.empty() && export_pairs.back() > N - 2) {
    cerr << "--export-pairs: pair " << export_pairs.back() << " does not exist (" << N - 1 << " pairs: 0.." << N - 2
         << ")\n";
    return 2;
  }
  vector<PairOut> pairs(N - 1);
  for (int i = 0; i + 1 < N; ++i) {  // the control reads the poses as T_cw: T_{j<-i} would be T_cw[j] T_cw[i]^-1
    const bool exp = binary_search(export_pairs.begin(), export_pairs.end(), i);
    PairDetail det;
    pairs[i] = RunPair(frames[i], frames[i + 1], K, in.depth_factor, T_gt[i + 1].inverse() * T_gt[i],
                       T_gt[i + 1] * T_gt[i].inverse(), exp ? &det : nullptr);
    AcceptPnp(pairs[i], pnp_min_distinct);
    if (exp) {
      const fs::path d = out / "diagnostics" / ("pair_" + to_string(i) + "_" + to_string(i + 1));
      WriteDiagnostics(d, i, frames[i], frames[i + 1], seq, in, K, T_gt[i + 1].inverse() * T_gt[i], pairs[i], det);
      cout << "wrote diagnostics " << d.string() << "\n";
    }
  }

  vector<Sophus::SE3d> T_pnp(N), T_icp(N);
  T_pnp[0] = T_icp[0] = T_gt[0];
  for (int i = 0; i + 1 < N; ++i) {
    T_pnp[i + 1] = pairs[i].pnp_ok ? T_pnp[i] * pairs[i].pnp.inverse() : T_pnp[i];
    T_icp[i + 1] = pairs[i].icp_ok ? T_icp[i] * pairs[i].icp.inverse() : T_icp[i];
  }

  fs::create_directories(out);
  {
    ofstream f(out / "frames.csv");
    f << "frame,t_rgb,t_depth,dt_ms,rgb,depth,gt_x,gt_y,gt_z,gt_qx,gt_qy,gt_qz,gt_qw\n" << fixed;
    for (const auto &fr : frames) {
      const Eigen::Quaterniond q(fr.T_wc.R);
      f << fr.index << "," << setprecision(6) << fr.t_rgb << "," << fr.t_depth << "," << setprecision(3)
        << 1000 * (fr.t_depth - fr.t_rgb) << "," << fs::path(fr.rgb).lexically_relative(seq).string() << ","
        << fs::path(fr.depth).lexically_relative(seq).string() << setprecision(6) << "," << fr.T_wc.t.x() << ","
        << fr.T_wc.t.y() << "," << fr.T_wc.t.z() << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
    }
  }
  vector<double> pr, pt, ir, it, gr, gw;
  int pnp_rejected = 0;
  {
    ofstream f(out / "pairs.csv");
    f << "pair,frame_i,frame_j,matches,pnp_correspondences,pnp_inliers,pnp_rot_err_deg,pnp_trans_err_m,"
         "icp_correspondences,icp_inliers,icp_rot_err_deg,icp_trans_err_m,gt_reproj_median_px,gt_motion_deg,gt_motion_m,"
         "pnp_solver_returned,pnp_accepted,pnp_inlier_distinct_i,pnp_inlier_distinct_j,pnp_returned_rot_err_deg,"
         "pnp_returned_trans_err_m,pnp_reject_reason\n";
    cout << "\npair    matches  PnP 3D2D/inl  rot err  trans err | ICP 3D3D/inl  rot err  trans err | GT motion   GT reproj\n";
    for (int i = 0; i + 1 < N; ++i) {
      const PairOut &p = pairs[i];
      const Sophus::SE3d G = T_gt[i + 1].inverse() * T_gt[i];
      const double gdeg = RotErrDeg(Eigen::Matrix3d::Identity(), G.rotationMatrix()), gm = G.translation().norm();
      const double per = p.pnp_ok ? RotErrDeg(G.rotationMatrix(), p.pnp.rotationMatrix()) : NAN;
      const double pet = p.pnp_ok ? (G.translation() - p.pnp.translation()).norm() : NAN;
      const double ier = p.icp_ok ? RotErrDeg(G.rotationMatrix(), p.icp.rotationMatrix()) : NAN;
      const double iet = p.icp_ok ? (G.translation() - p.icp.translation()).norm() : NAN;
      if (p.pnp_ok) pr.push_back(per), pt.push_back(pet);
      if (p.icp_ok) ir.push_back(ier), it.push_back(iet);
      if (!std::isnan(p.gt_reproj_median)) gr.push_back(p.gt_reproj_median), gw.push_back(p.wrong_reproj_median);
      f << i << "," << i << "," << i + 1 << "," << p.matches << "," << p.pnp_corr << "," << p.pnp_inliers << "," << per << ","
        << pet << "," << p.icp_corr << "," << p.icp_inliers << "," << ier << "," << iet << "," << p.gt_reproj_median << ","
        << gdeg << "," << gm << "," << p.pnp_returned << "," << p.pnp_ok << "," << p.pnp_inl_distinct_i << ","
        << p.pnp_inl_distinct_j << ","
        << (p.pnp_returned ? RotErrDeg(G.rotationMatrix(), p.pnp.rotationMatrix()) : NAN) << ","
        << (p.pnp_returned ? (G.translation() - p.pnp.translation()).norm() : NAN) << ",\"" << p.pnp_reject << "\"\n";
      if (p.pnp_returned && !p.pnp_ok) ++pnp_rejected;
      cout << setw(3) << i << "->" << left << setw(3) << i + 1 << right << setw(8) << p.matches << setw(8) << p.pnp_corr
           << "/" << left << setw(5) << p.pnp_inliers << right << fixed << setprecision(3) << setw(8) << per << setw(10)
           << pet << " |" << setw(8) << p.icp_corr << "/" << left << setw(5) << p.icp_inliers << right << setw(8) << ier
           << setw(10) << iet << " | " << setw(5) << setprecision(2) << gdeg << " deg " << setw(5) << 100 * gm
           << " cm  " << setw(5) << p.gt_reproj_median << " px\n";
    }
  }
  vector<double> te_p(N), re_p(N), te_i(N), re_i(N);
  {
    ofstream f(out / "trajectory.csv");
    f << "frame,timestamp,gt_x,gt_y,gt_z,pnp_x,pnp_y,pnp_z,icp_x,icp_y,icp_z,gt_qx,gt_qy,gt_qz,gt_qw,pnp_qx,pnp_qy,pnp_qz,"
         "pnp_qw,icp_qx,icp_qy,icp_qz,icp_qw,pnp_translation_error,pnp_rotation_error,icp_translation_error,"
         "icp_rotation_error\n";
    for (int k = 0; k < N; ++k) {
      te_p[k] = (T_pnp[k].translation() - T_gt[k].translation()).norm();
      re_p[k] = RotErrDeg(T_gt[k].rotationMatrix(), T_pnp[k].rotationMatrix());
      te_i[k] = (T_icp[k].translation() - T_gt[k].translation()).norm();
      re_i[k] = RotErrDeg(T_gt[k].rotationMatrix(), T_icp[k].rotationMatrix());
      f << k << "," << fixed << setprecision(6) << frames[k].t_rgb;
      for (const auto *T : {&T_gt[k], &T_pnp[k], &T_icp[k]})
        f << "," << T->translation().x() << "," << T->translation().y() << "," << T->translation().z();
      for (const auto *T : {&T_gt[k], &T_pnp[k], &T_icp[k]}) {
        const Eigen::Quaterniond q = T->unit_quaternion();
        f << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
      }
      f << "," << te_p[k] << "," << re_p[k] << "," << te_i[k] << "," << re_i[k] << "\n";
    }
  }
  auto mean = [](const vector<double> &v) -> double { double s = 0; for (double x : v) s += x; return v.empty() ? NAN : s / v.size(); };
  auto median = [](vector<double> v) -> double { if (v.empty()) return NAN; sort(v.begin(), v.end()); return v[v.size() / 2]; };
  ostringstream s;
  s << fixed << setprecision(4) << "TUM " << name << ": " << N << " frames, " << N - 1 << " pairs\n"
    << "convention check: PnP inliers projected with the GROUND-TRUTH relative pose, median of pair medians "
    << median(gr) << " px (control, poses read as T_cw instead: " << median(gw) << " px)\n"
    << "PnP acceptance: at least " << pnp_min_distinct << " distinct inlier keypoints per frame (0 = off); "
    << pnp_rejected << " returned poses rejected (pose held)\n"
    << "PnP: " << pr.size() << " pairs solved; per-pair rotation error mean " << mean(pr) << " deg, translation error mean "
    << mean(pt) << " m; trajectory final " << te_p[N - 1] << " m / " << re_p[N - 1] << " deg\n"
    << "ICP: " << ir.size() << " pairs solved; per-pair rotation error mean " << mean(ir) << " deg, translation error mean "
    << mean(it) << " m; trajectory final " << te_i[N - 1] << " m / " << re_i[N - 1] << " deg\n";
  cout << "\n" << s.str() << "wrote " << out.string() << "/{frames.csv,pairs.csv,trajectory.csv,summary.txt}\n";
  ofstream(out / "summary.txt") << s.str();
  return 0;
}
