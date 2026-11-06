// tum_orb_correspondences: OpenCV ORB feature correspondences on the frame pairs of an existing TUM run.
//
// A separate, diagnostic path (not used by any trajectory): genuine cv::ORB keypoints and descriptors
// (cv::ORB::create with OpenCV's defaults except nfeatures), brute-force Hamming matching, and an inlier
// classification by cv::solvePnPRansac with the reference evaluator's parameters (100 iterations, 8 px,
// confidence 0.99) on the matches that have frame-i depth. It does not touch the reference FAST + rotated
// BRIEF frontend (src/features) or the scratch pipeline (pnp_from_scratch).
//
// The frames, their order, RGB / depth files and ground truth are read from a frames.csv written by
// tum_trajectory_eval, so the pairs are exactly the pairs of that run (frame k -> k+1 of its selection).
//
// Usage (repository root):
//   build/tum_orb_correspondences SEQUENCE_DIR --frames-csv RUN/frames.csv --out DIR [--pairs all|73|70-76|46,73]
//                                 [--features 500] [--fx F --fy F --cx C --cy C] [--depth-factor 5000]
// Defaults: all pairs, 500 features, freiburg1 intrinsics (517.3, 516.5, 318.6, 255.3), depth = raw / 5000.
// Output (DIR must not exist inside data/): frames.csv (copy), command.txt, summary.csv (one row per pair),
//   pair_<i>_<j>/matches.csv   every match after the distance filter: query, train, hamming, u_i, v_i, u_j, v_j,
//                              depth_raw_i, pnp_used (frame-i depth > 0), pnp_inlier (solvePnPRansac inlier list)
//   pair_<i>_<j>/keypoints_i.csv, keypoints_j.csv   every ORB keypoint (x, y, size, angle, response, octave)
//   pair_<i>_<j>/pair.json     counts, solver status, T_j<-i and its error vs ground truth, configuration
// Match filter: distance <= max(2 * d_min, 30), the same rule the reference find_feature_matches applies
// (raw matches are counted too). Distortion is not corrected (as the evaluator).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
namespace fs = std::filesystem;

namespace {

struct FrameRow {
  int frame = 0;
  string t_rgb, t_depth, rgb, depth;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

bool ReadFrames(const fs::path &p, vector<FrameRow> &out, string &err) {
  ifstream f(p);
  if (!f) {
    err = "cannot read " + p.string();
    return false;
  }
  string line;
  getline(f, line);
  map<string, int> col;
  {
    stringstream hs(line);
    string h;
    for (int i = 0; getline(hs, h, ','); ++i) col[h] = i;
  }
  for (const char *c : {"frame", "t_rgb", "t_depth", "rgb", "depth", "gt_x", "gt_y", "gt_z", "gt_qx", "gt_qy", "gt_qz", "gt_qw"})
    if (!col.count(c)) {
      err = p.string() + ": missing column " + c;
      return false;
    }
  while (getline(f, line)) {
    if (line.empty()) continue;
    vector<string> c;
    stringstream ss(line);
    string cell;
    while (getline(ss, cell, ',')) c.push_back(cell);
    FrameRow r;
    r.frame = stoi(c.at(col["frame"]));
    r.t_rgb = c.at(col["t_rgb"]), r.t_depth = c.at(col["t_depth"]), r.rgb = c.at(col["rgb"]), r.depth = c.at(col["depth"]);
    const Eigen::Quaterniond q(stod(c.at(col["gt_qw"])), stod(c.at(col["gt_qx"])), stod(c.at(col["gt_qy"])),
                               stod(c.at(col["gt_qz"])));
    r.R = q.normalized().toRotationMatrix();
    r.t = Eigen::Vector3d(stod(c.at(col["gt_x"])), stod(c.at(col["gt_y"])), stod(c.at(col["gt_z"])));
    if (r.frame != int(out.size())) {
      err = p.string() + ": frame column is not 0, 1, 2, ...";
      return false;
    }
    out.push_back(r);
  }
  return true;
}

bool ParsePairs(const string &s, int n_pairs, vector<int> &out) {
  out.clear();
  if (s == "all") {
    for (int k = 0; k < n_pairs; ++k) out.push_back(k);
    return true;
  }
  stringstream ss(s);
  string tok;
  while (getline(ss, tok, ',')) {
    const size_t d = tok.find('-');
    try {
      const int lo = stoi(tok.substr(0, d)), hi = d == string::npos ? lo : stoi(tok.substr(d + 1));
      if (lo < 0 || hi < lo || hi >= n_pairs) return false;
      for (int k = lo; k <= hi; ++k) out.push_back(k);
    } catch (const exception &) {
      return false;
    }
  }
  sort(out.begin(), out.end());
  out.erase(unique(out.begin(), out.end()), out.end());
  return !out.empty();
}

double RotErrDeg(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) {
  return acos(min(1.0, max(-1.0, ((A.transpose() * B).trace() - 1) / 2))) * 180 / M_PI;
}

string JsonNum(double v) {
  if (!isfinite(v)) return "null";
  ostringstream o;
  o << setprecision(10) << v;
  return o.str();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "usage: tum_orb_correspondences SEQUENCE_DIR --frames-csv CSV --out DIR [--pairs LIST] [--features N]\n";
    return 2;
  }
  const fs::path seq = fs::weakly_canonical(argv[1]);
  fs::path frames_csv, out;
  string pairs_arg = "all";
  int nfeatures = 500;
  double fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3, depth_factor = 5000.0;
  for (int a = 2; a + 1 < argc; ++a) {
    const string s = argv[a];
    if (s == "--frames-csv") frames_csv = argv[++a];
    else if (s == "--out") out = argv[++a];
    else if (s == "--pairs") pairs_arg = argv[++a];
    else if (s == "--features") nfeatures = stoi(argv[++a]);
    else if (s == "--fx") fx = stod(argv[++a]);
    else if (s == "--fy") fy = stod(argv[++a]);
    else if (s == "--cx") cx = stod(argv[++a]);
    else if (s == "--cy") cy = stod(argv[++a]);
    else if (s == "--depth-factor") depth_factor = stod(argv[++a]);
    else {
      cerr << "unknown option " << s << "\n";
      return 2;
    }
  }
  if (frames_csv.empty() || out.empty()) {
    cerr << "--frames-csv and --out are required\n";
    return 2;
  }
  out = fs::weakly_canonical(out);
  const auto rel = out.lexically_relative(seq.parent_path());
  if (!rel.empty() && *rel.begin() != "..") {
    cerr << "refusing to write into the data directory\n";
    return 2;
  }
  vector<FrameRow> fr;
  string err;
  if (!ReadFrames(frames_csv, fr, err)) {
    cerr << err << "\n";
    return 2;
  }
  vector<int> pairs;
  if (fr.size() < 2 || !ParsePairs(pairs_arg, int(fr.size()) - 1, pairs)) {
    cerr << "--pairs: expected all, K, A-B or a comma list within 0.." << int(fr.size()) - 2 << "\n";
    return 2;
  }
  fs::create_directories(out);
  fs::copy_file(frames_csv, out / "frames.csv", fs::copy_options::overwrite_existing);
  {
    ofstream c(out / "command.txt");
    for (int a = 0; a < argc; ++a) c << (a ? " " : "") << argv[a];
    c << "\n";
  }
  const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures);  // OpenCV defaults: 1.2 scale, 8 levels, FAST 20, Harris score
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  ofstream summary(out / "summary.csv");
  summary << "pair,frame_i,frame_j,t_rgb_i,t_rgb_j,keypoints_i,keypoints_j,raw_matches,matches,pnp_correspondences,"
             "pnp_solver_returned,pnp_inliers,pnp_outliers,pnp_rot_err_deg,pnp_trans_err_m,gt_motion_deg,gt_motion_m\n";
  for (int k : pairs) {
    const FrameRow &a = fr[k], &b = fr[k + 1];
    const cv::Mat img_i = cv::imread((seq / a.rgb).string(), cv::IMREAD_GRAYSCALE),
                  img_j = cv::imread((seq / b.rgb).string(), cv::IMREAD_GRAYSCALE);
    const cv::Mat dep_i = cv::imread((seq / a.depth).string(), cv::IMREAD_UNCHANGED);
    if (img_i.empty() || img_j.empty() || dep_i.empty() || dep_i.type() != CV_16UC1) {
      cerr << "pair " << k << ": cannot read the RGB / 16-bit depth images\n";
      return 1;
    }
    vector<cv::KeyPoint> kp_i, kp_j;
    cv::Mat d_i, d_j;
    orb->detectAndCompute(img_i, cv::noArray(), kp_i, d_i);
    orb->detectAndCompute(img_j, cv::noArray(), kp_j, d_j);
    vector<cv::DMatch> raw;
    if (!d_i.empty() && !d_j.empty()) matcher.match(d_i, d_j, raw);
    double dmin = 1e9;
    for (const auto &m : raw) dmin = min(dmin, double(m.distance));
    vector<cv::DMatch> matches;
    for (const auto &m : raw)
      if (m.distance <= max(2 * dmin, 30.0)) matches.push_back(m);
    vector<cv::Point3f> P;
    vector<cv::Point2f> p;
    vector<int> pnp_idx(matches.size(), -1);
    vector<unsigned short> raw_depth(matches.size(), 0);
    for (size_t m = 0; m < matches.size(); ++m) {
      const cv::Point2f u = kp_i[matches[m].queryIdx].pt, v = kp_j[matches[m].trainIdx].pt;
      const unsigned short d = dep_i.at<unsigned short>(int(u.y), int(u.x));  // truncated pixel, as the evaluator
      raw_depth[m] = d;
      if (d == 0) continue;
      const double z = d / depth_factor;
      pnp_idx[m] = int(P.size());
      P.emplace_back(float((u.x - cx) / fx * z), float((u.y - cy) / fy * z), float(z));
      p.push_back(v);
    }
    bool returned = false;
    vector<int> inl;
    cv::Mat rvec, tvec;
    if (P.size() >= 4) returned = cv::solvePnPRansac(P, p, K, cv::Mat(), rvec, tvec, false, 100, 8.0f, 0.99, inl);
    vector<char> is_inl(P.size(), 0);
    for (int q : inl) is_inl[q] = 1;
    const Eigen::Matrix3d Rg = b.R.transpose() * a.R;  // T_j<-i = T_wc[j]^-1 T_wc[i]
    const Eigen::Vector3d tg = b.R.transpose() * (a.t - b.t);
    double rot_err = NAN, trans_err = NAN;
    Eigen::Matrix3d Re = Eigen::Matrix3d::Identity();
    Eigen::Vector3d te = Eigen::Vector3d::Zero();
    if (returned) {
      cv::Mat Rm;
      cv::Rodrigues(rvec, Rm);
      for (int r = 0; r < 3; ++r) {
        te(r) = tvec.at<double>(r);
        for (int c = 0; c < 3; ++c) Re(r, c) = Rm.at<double>(r, c);
      }
      rot_err = RotErrDeg(Rg, Re);
      trans_err = (tg - te).norm();
    }
    const fs::path d = out / ("pair_" + to_string(k) + "_" + to_string(k + 1));
    fs::create_directories(d);
    {
      ofstream f(d / "matches.csv");
      f << "match,query,train,hamming,u_i,v_i,u_j,v_j,depth_raw_i,pnp_used,pnp_inlier\n" << setprecision(9);
      for (size_t m = 0; m < matches.size(); ++m) {
        const cv::Point2f u = kp_i[matches[m].queryIdx].pt, v = kp_j[matches[m].trainIdx].pt;
        f << m << "," << matches[m].queryIdx << "," << matches[m].trainIdx << "," << matches[m].distance << "," << u.x
          << "," << u.y << "," << v.x << "," << v.y << "," << raw_depth[m] << "," << int(pnp_idx[m] >= 0) << ","
          << int(pnp_idx[m] >= 0 && returned && is_inl[pnp_idx[m]]) << "\n";
      }
    }
    for (int side = 0; side < 2; ++side) {
      ofstream f(d / (side ? "keypoints_j.csv" : "keypoints_i.csv"));
      f << "kp,x,y,size,angle_deg,response,octave\n" << setprecision(9);
      const auto &kps = side ? kp_j : kp_i;
      for (size_t q = 0; q < kps.size(); ++q)
        f << q << "," << kps[q].pt.x << "," << kps[q].pt.y << "," << kps[q].size << "," << kps[q].angle << ","
          << kps[q].response << "," << kps[q].octave << "\n";
    }
    const int n_in = returned ? int(inl.size()) : 0;
    {
      ofstream f(d / "pair.json");
      f << "{\"format\": \"tum_orb_correspondences v1\", \"pair\": " << k << ", \"frame_i\": " << k
        << ", \"frame_j\": " << k + 1 << ", \"t_rgb_i\": " << a.t_rgb << ", \"t_rgb_j\": " << b.t_rgb
        << ", \"rgb_i\": \"" << a.rgb << "\", \"rgb_j\": \"" << b.rgb << "\", \"depth_i\": \"" << a.depth << "\""
        << ", \"opencv_version\": \"" << CV_VERSION << "\""
        << ", \"config\": {\"detector\": \"cv::ORB::create(nfeatures=" << nfeatures
        << ") with OpenCV defaults (scaleFactor 1.2, nlevels 8, edgeThreshold 31, fastThreshold 20, HARRIS_SCORE)\""
        << ", \"matcher\": \"cv::BFMatcher(NORM_HAMMING), nearest neighbour, kept if distance <= max(2 d_min, 30)\""
        << ", \"inliers\": \"cv::solvePnPRansac(100 it, 8 px, 0.99) on matches with frame-i depth (raw / "
        << depth_factor << "); diagnostic only, not a trajectory\"}"
        << ", \"counts\": {\"keypoints_i\": " << kp_i.size() << ", \"keypoints_j\": " << kp_j.size()
        << ", \"raw_matches\": " << raw.size() << ", \"matches\": " << matches.size() << ", \"pnp_correspondences\": "
        << P.size() << ", \"pnp_inliers\": " << n_in << ", \"pnp_outliers\": " << (returned ? int(P.size()) - n_in : 0)
        << "}, \"pnp_solver_returned\": " << (returned ? "true" : "false") << ", \"pnp_rot_err_deg\": "
        << JsonNum(rot_err) << ", \"pnp_trans_err_m\": " << JsonNum(trans_err) << ", \"gt_motion_deg\": "
        << JsonNum(RotErrDeg(Eigen::Matrix3d::Identity(), Rg)) << ", \"gt_motion_m\": " << JsonNum(tg.norm()) << "}\n";
    }
    summary << k << "," << k << "," << k + 1 << "," << a.t_rgb << "," << b.t_rgb << "," << kp_i.size() << ","
            << kp_j.size() << "," << raw.size() << "," << matches.size() << "," << P.size() << "," << int(returned)
            << "," << n_in << "," << (returned ? int(P.size()) - n_in : 0) << "," << JsonNum(rot_err) << ","
            << JsonNum(trans_err) << "," << JsonNum(RotErrDeg(Eigen::Matrix3d::Identity(), Rg)) << ","
            << JsonNum(tg.norm()) << "\n";
    cout << "pair " << k << "->" << k + 1 << ": keypoints " << kp_i.size() << "/" << kp_j.size() << ", matches "
         << matches.size() << " (raw " << raw.size() << "), PnP " << n_in << "/" << P.size() << " inliers"
         << (returned ? ", error " + JsonNum(rot_err) + " deg / " + JsonNum(trans_err) + " m" : ", solver failed") << "\n";
  }
  cout << "wrote " << out.string() << "\n";
  return 0;
}
