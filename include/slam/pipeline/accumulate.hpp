#pragma once

// Trajectory accumulation of the scratch pipeline:
//   T_wc[0] = anchor,  T_wc[k+1] = T_wc[k] * T_{k+1<-k}^-1   for an ACCEPTED pair,
//   T_wc[k+1] = T_wc[k]                                       otherwise (previous pose held).
// A pair that is not accepted contributes nothing -- its pose values are never read.

#include <string>
#include <vector>

#include "common.hpp"

namespace scratch {

inline Pose Inverse(const Pose &T) {
  Pose r;
  r.R = T.R.transpose();
  r.t = -r.R * T.t;
  return r;
}
inline Pose Compose(const Pose &A, const Pose &B) {
  Pose r;
  r.R = A.R * B.R;
  r.t = A.R * B.t + A.t;
  return r;
}

struct RelativeStep {
  bool accepted = false;
  std::string held_source;  // pose_source of the next frame when the step is not accepted
  Pose T;                   // T_{k+1<-k}; ignored when not accepted
};

struct AccumulatedTrajectory {
  std::vector<Pose> Twc;
  std::vector<std::string> source;
};

inline AccumulatedTrajectory AccumulateTrajectory(const Pose &anchor, const std::vector<RelativeStep> &steps,
                                                  const std::string &estimated_source = "scratch_pipeline") {
  AccumulatedTrajectory a;
  a.Twc.resize(steps.size() + 1);
  a.source.assign(steps.size() + 1, estimated_source);
  a.Twc[0] = anchor;
  a.source[0] = "gt_anchor";
  for (size_t k = 0; k < steps.size(); ++k) {
    if (steps[k].accepted) {
      a.Twc[k + 1] = Compose(a.Twc[k], Inverse(steps[k].T));
    } else {
      a.Twc[k + 1] = a.Twc[k];
      a.source[k + 1] = steps[k].held_source;
    }
  }
  return a;
}

}  // namespace scratch
