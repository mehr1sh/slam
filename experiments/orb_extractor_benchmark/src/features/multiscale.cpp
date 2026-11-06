#include "features/multiscale.hpp"

#include "features/brief.hpp"
#include "features/orientation.hpp"

namespace features {

MultiscaleFeatures ExtractMultiscale(const ImagePyramid &pyramid, const FastParams &fast) {
  MultiscaleFeatures out;
  for (size_t l = 0; l < pyramid.levels.size(); ++l) {
    const GrayImage &img = pyramid.levels[l];
    std::vector<Keypoint> kps = DetectFast(img, fast);
    AssignOrientations(img, kps);
    const std::vector<BriefDescriptor> desc = ComputeBrief(GaussianSmooth(img), kps);
    out.per_level_count.push_back(int(kps.size()));
    for (size_t k = 0; k < kps.size(); ++k) {
      MultiscaleKeypoint m;
      m.level_kp = kps[k];
      m.level = int(l);
      m.scale = float(pyramid.scale_x[l]);
      m.x = LevelToBaseX(pyramid, int(l), kps[k].x);
      m.y = LevelToBaseY(pyramid, int(l), kps[k].y);
      out.keypoints.push_back(m);
      out.descriptors.push_back(desc[k]);
    }
  }
  return out;
}

}  // namespace features
