// Self-checking test of features/pyramid (resampling, level sizes, coordinate
// mapping) and features/multiscale (per-level detection and description).

#include <algorithm>
#include <cmath>

#include "features/brief.hpp"
#include "features/multiscale.hpp"
#include "features/orientation.hpp"
#include "features/pyramid.hpp"
#include "feature_test_utils.hpp"

using namespace features;

int main() {
  std::printf("=== features/pyramid + features/multiscale ===\n");

  // --- level count and sizes for the Bunny frames (640 x 480, 8 levels, 1.2) ---
  const GrayImage tex = ftest::Texture(640, 480, 9);
  const ImagePyramid p = BuildPyramid(tex);
  const int expect[8][2] = {{640, 480}, {533, 400}, {444, 333}, {370, 278},
                            {309, 231}, {257, 193}, {214, 161}, {179, 134}};
  bool sizes = p.levels.size() == 8;
  for (int l = 0; sizes && l < 8; ++l) sizes = p.levels[l].width == expect[l][0] && p.levels[l].height == expect[l][1];
  FCHECK(sizes, "8 levels: 640x480, 533x400, 444x333, 370x278, 309x231, 257x193, 214x161, 179x134");
  bool sc = true;
  for (int l = 0; l < 8; ++l)
    sc &= std::abs(p.scale_x[l] - 640.0 / expect[l][0]) < 1e-12 && std::abs(p.scale_y[l] - 480.0 / expect[l][1]) < 1e-12 &&
          std::abs(p.scale_x[l] - std::pow(1.2, l)) < 0.01 * std::pow(1.2, l);
  FCHECK(sc, "scale = original size / level size per axis, within 1%% of 1.2^level (level 7: %.4f x %.4f)",
         p.scale_x[7], p.scale_y[7]);
  FCHECK(p.levels[0].pixels == tex.pixels, "level 0 is the input image");
  FCHECK(BuildPyramid(tex).levels[7].pixels == p.levels[7].pixels, "deterministic: identical levels on a second build");
  {
    PyramidParams one;
    one.levels = 1;
    const ImagePyramid q = BuildPyramid(tex, one);
    FCHECK(q.levels.size() == 1 && q.levels[0].pixels == tex.pixels, "levels = 1 gives the input image only");
  }

  // --- resampling preserves constants and reproduces linear ramps exactly ---
  {
    GrayImage flat(640, 480, 77);
    const ImagePyramid f = BuildPyramid(flat);
    bool same = true;
    for (const auto &lv : f.levels) same &= std::all_of(lv.pixels.begin(), lv.pixels.end(), [](uint8_t v) { return v == 77; });
    FCHECK(same, "a constant image stays constant on every level");
  }
  {
    // I(x, y) = 0.3 x + 0.1 y. Bilinear interpolation is exact for a linear
    // function, so level-l pixel (x, y) must hold I(x0, y0) with
    // (x0, y0) = LevelToBase(x, y), up to the per-level rounding to integers.
    GrayImage ramp(640, 480);
    for (int y = 0; y < 480; ++y)
      for (int x = 0; x < 640; ++x) ramp.at(x, y) = uint8_t(std::lround(0.3 * x + 0.1 * y));
    const ImagePyramid r = BuildPyramid(ramp);
    double worst = 0;
    for (int l = 1; l < 8; ++l) {
      const GrayImage &lv = r.levels[l];
      for (int y = 4; y < lv.height - 4; y += 3)
        for (int x = 4; x < lv.width - 4; x += 3) {
          const double v = 0.3 * LevelToBaseX(r, l, x) + 0.1 * LevelToBaseY(r, l, y);
          worst = std::max(worst, std::abs(lv.at(x, y) - v));
        }
    }
    FCHECK(worst < 2.0, "linear ramp: every level equals the ramp at the mapped coordinates (max |diff| %.2f grey "
           "levels; one rounding per level)", worst);
  }
  {
    // A bright 7x7 square centred at a known original position: its brightest
    // pixel on each level maps back to within ~1 level pixel of the centre.
    GrayImage dot(640, 480, 0);
    const int cx = 301, cy = 217;
    for (int y = cy - 3; y <= cy + 3; ++y)
      for (int x = cx - 3; x <= cx + 3; ++x) dot.at(x, y) = 255;
    const ImagePyramid d = BuildPyramid(dot);
    double worst = 0;
    for (int l = 0; l < 8; ++l) {
      const GrayImage &lv = d.levels[l];
      double sw = 0, sx = 0, sy = 0;  // intensity-weighted centroid on the level
      for (int y = 0; y < lv.height; ++y)
        for (int x = 0; x < lv.width; ++x) {
          sw += lv.at(x, y);
          sx += lv.at(x, y) * double(x);
          sy += lv.at(x, y) * double(y);
        }
      const double bx = LevelToBaseX(d, l, float(sx / sw)), by = LevelToBaseY(d, l, float(sy / sw));
      worst = std::max(worst, std::hypot(bx - cx, by - cy) / d.scale_x[l]);
    }
    FCHECK(worst < 0.5, "blob centroid on every level maps back to (301, 217) within %.3f level pixels", worst);
  }

  // --- multiscale extraction ---
  {
    FastParams fp;
    fp.threshold = 20;
    const MultiscaleFeatures m = ExtractMultiscale(p, fp);
    bool lv_ok = m.per_level_count.size() == 8, in_img = true, map_ok = true;
    int total = 0;
    for (int c : m.per_level_count) lv_ok &= c > 0, total += c;
    for (const auto &k : m.keypoints) {
      in_img &= k.x >= 0 && k.y >= 0 && k.x < 640 && k.y < 480;
      map_ok &= std::abs(k.x - LevelToBaseX(p, k.level, k.level_kp.x)) < 1e-4 &&
                std::abs(k.scale - float(p.scale_x[k.level])) < 1e-6;
    }
    FCHECK(lv_ok && total == int(m.keypoints.size()) && m.descriptors.size() == m.keypoints.size(),
           "keypoints on all 8 levels (%d %d %d %d %d %d %d %d, total %d), one descriptor each", m.per_level_count[0],
           m.per_level_count[1], m.per_level_count[2], m.per_level_count[3], m.per_level_count[4], m.per_level_count[5],
           m.per_level_count[6], m.per_level_count[7], total);
    FCHECK(in_img && map_ok, "level, level coordinates, scale and original coordinates are consistent");
    // descriptors come from the keypoint's OWN level (not the 640x480 image)
    bool own_level = true;
    int checked = 0;
    for (size_t i = 0; i < m.keypoints.size(); i += 17) {
      const auto &k = m.keypoints[i];
      BriefDescriptor d;
      ComputeBriefDescriptor(GaussianSmooth(p.levels[k.level]), k.level_kp, d);
      own_level &= d.bits == m.descriptors[i].bits;
      own_level &= std::abs(k.level_kp.angle - IntensityCentroidAngle(p.levels[k.level], int(k.level_kp.x),
                                                                      int(k.level_kp.y))) < 1e-6;
      ++checked;
    }
    FCHECK(own_level, "orientation and descriptor are computed on the keypoint's own level (%d checked)", checked);
    // a single-level pyramid reproduces the single-scale path exactly
    PyramidParams one;
    one.levels = 1;
    const MultiscaleFeatures s = ExtractMultiscale(BuildPyramid(tex, one), fp);
    std::vector<Keypoint> direct = DetectFast(tex, fp);
    AssignOrientations(tex, direct);
    const auto dd = ComputeBrief(GaussianSmooth(tex), direct);
    bool same = s.keypoints.size() == direct.size();
    for (size_t i = 0; same && i < direct.size(); ++i)
      same = s.keypoints[i].x == direct[i].x && s.keypoints[i].y == direct[i].y &&
             s.keypoints[i].level_kp.angle == direct[i].angle && s.descriptors[i].bits == dd[i].bits;
    FCHECK(same, "1 level == the single-scale DetectFast/orientation/BRIEF path, bit for bit (%d keypoints)",
           int(direct.size()));
  }
  return ftest::Finish("features_pyramid_test");
}
