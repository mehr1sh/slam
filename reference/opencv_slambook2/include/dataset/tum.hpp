#pragma once

// TUM RGB-D dataset adapter (standard library + Eigen only).
// https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
//
//   rgb.txt, depth.txt   "timestamp filename" per line, '#' comments
//   groundtruth.txt      "timestamp tx ty tz qx qy qz qw" per line, '#' comments:
//                        position and orientation of the colour camera's optical
//                        centre in the world (motion-capture) frame, i.e. T_wc
//                        (camera -> world) with the optical axes x right, y down,
//                        z forward -- the project's convention (pose files store
//                        T_wc; X_world = R * X_cam + t). No axis change is needed.
//   depth PNG            16-bit, depth in metres = value / 5000; 0 = no measurement
//
// Association (as the dataset's associate.py): every RGB / depth pair with
// |t_rgb - t_depth| <= max_dt is a candidate; candidates are taken greedily in
// order of increasing |dt| (ties: earlier RGB, then earlier depth), each image
// at most once; the result is sorted by RGB timestamp.
//
// Ground truth at a frame: at the frame's RGB timestamp, from the two
// bracketing ground-truth samples: translation linearly interpolated, rotation
// spherically (slerp, shortest arc). Missing if the timestamp is outside the
// ground-truth time span or the two samples are more than max_gap apart.

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tum {

struct Stamped {
  double t = 0;
  std::string file;  // relative to the sequence directory
};

struct GtSample {
  double t = 0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct Pose {  // T_wc: X_world = R * X_cam + t
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// rgb.txt / depth.txt. Sorted by timestamp (stable). false + *error if unreadable or malformed.
bool ReadFileList(const std::string &path, std::vector<Stamped> &out, std::string *error = nullptr);
// groundtruth.txt. Sorted by timestamp (stable); quaternions normalised.
bool ReadGroundtruth(const std::string &path, std::vector<GtSample> &out, std::string *error = nullptr);

struct Association {
  double t_rgb = 0, t_depth = 0;
  std::string rgb, depth;
};
std::vector<Association> Associate(const std::vector<Stamped> &rgb, const std::vector<Stamped> &depth, double max_dt);

// TUM quaternion (qx, qy, qz, qw) -> rotation matrix (normalised first).
Eigen::Matrix3d QuaternionToR(double qx, double qy, double qz, double qw);

// Ground-truth pose at time t (see the header comment). false + *why if missing.
bool InterpolateGroundtruth(const std::vector<GtSample> &gt, double t, double max_gap, Pose &out,
                            std::string *why = nullptr);

// Raw 16-bit depth -> metres; 0 (no measurement) -> 0, which callers treat as invalid.
inline double DepthMetres(uint16_t raw, double depth_factor) { return raw == 0 ? 0.0 : raw / depth_factor; }
inline bool ValidDepth(uint16_t raw) { return raw != 0; }

struct CameraIntrinsics {
  std::string sensor;  // "freiburg1", "freiburg2", "freiburg3"
  int width = 640, height = 480;
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double d[5] = {0, 0, 0, 0, 0};  // radial-tangential distortion k1 k2 p1 p2 k3 (not applied by the pipeline)
  double depth_factor = 5000.0;
};
// The calibrated intrinsics published for the sensor named in the sequence name
// (rgbd_dataset_freiburg<N>_...), from
// https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
// false if the name contains no known sensor.
bool OfficialIntrinsics(const std::string &sequence_name, CameraIntrinsics &out);

struct Frame {
  int index = 0;           // position in the selected sequence (0, 1, ...)
  double t_rgb = 0, t_depth = 0;
  std::string rgb, depth;  // absolute paths
  Pose T_wc;               // ground truth at t_rgb
};

struct SelectionReport {
  int rgb = 0, depth = 0, associated = 0;
  int missing_image = 0, missing_groundtruth = 0;
  std::vector<std::string> skipped;  // one line per excluded frame: timestamp and reason
};

// Associated frames, in RGB-timestamp order, that have both image files and
// ground truth; every excluded frame is listed with its reason. stride > 1
// keeps every stride-th usable frame; max_frames > 0 stops after that many.
bool SelectFrames(const std::string &sequence_dir, double max_dt, double max_gt_gap, int stride, int max_frames,
                  std::vector<Frame> &frames, SelectionReport &report, std::string *error = nullptr);

}  // namespace tum
