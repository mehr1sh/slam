// The ranked matcher inside the scratch front end and pipeline (FrontendParams::ranked_matcher),
// on the Bunny pair 0 -> 1: with every matcher option off it reproduces the default front end and
// pose exactly; with options on, the correspondences carry consistent instrumentation.

#include <algorithm>

#include "check.hpp"
#include "pipeline.hpp"

using namespace scratch;

namespace {

bool SameCorrespondences(const std::vector<Correspondence> &a, const std::vector<Correspondence> &b) {
  if (a.size() != b.size()) return false;
  for (size_t k = 0; k < a.size(); ++k)
    if (a[k].query != b[k].query || a[k].train != b[k].train || a[k].hamming != b[k].hamming ||
        a[k].uv_i != b[k].uv_i || a[k].uv_j != b[k].uv_j || a[k].depth_raw_i != b[k].depth_raw_i ||
        a[k].has_3d != b[k].has_3d || a[k].X_i != b[k].X_i ||
        a[k].level_i != b[k].level_i || a[k].level_j != b[k].level_j)
      return false;
  return true;
}

}  // namespace

int main() {
  std::printf("=== ranked matcher in the front end (Bunny 0 -> 1) ===\n");
  const auto repo = scratch_test::Repo();
  const std::string ds = (repo / "data" / "synthetic_bunny").string();
  Intrinsics K;
  {
    std::ifstream f(ds + "/intrinsics.txt");
    double w, h;
    f >> w >> h >> K.fx >> K.fy >> K.cx >> K.cy;
  }
  RgbdFrame f0, f1;
  CHECK(LoadRgbdFrame(ds, 0, f0) && LoadRgbdFrame(ds, 1, f1), "frames load");
  PipelineParams def, rk;
  rk.frontend.ranked_matcher = true;
  const FrameFeatures a = ExtractFeatures(f0.rgb, def.frontend), b = ExtractFeatures(f1.rgb, def.frontend);
  CHECK(a.descriptor_valid.size() == a.descriptors.size() &&
            std::count(a.descriptor_valid.begin(), a.descriptor_valid.end(), false) == 0,
        "every descriptor valid (%zu): the 16 px border keeps all BRIEF tests inside the image", a.descriptors.size());

  const PairResult d = ProcessPair(0, 1, a, b, f0, f1, K, def), r = ProcessPair(0, 1, a, b, f0, f1, K, rk);
  CHECK(!d.ranked_matcher && d.match_details.empty(), "default: original matcher, no ranked details");
  CHECK(r.ranked_matcher && r.raw_matches == d.raw_matches && r.filtered_matches == d.filtered_matches &&
            r.d_min == d.d_min && r.d_max == d.d_max && SameCorrespondences(d.correspondences, r.correspondences),
        "ranked, all options off: identical raw (%d) / filtered (%d) counts, d_min / d_max and correspondences",
        r.raw_matches, r.filtered_matches);
  CHECK(r.R == d.R && r.t == d.t && r.pnp_mask == d.pnp_mask && r.pose_accepted == d.pose_accepted &&
            r.validity.reason == d.validity.reason && r.essential_mask == d.essential_mask,
        "identical pose, inlier mask, gate decision and essential-matrix stage");
  const features::MatchCounters &c = r.match_counters;
  CHECK(c.raw == r.raw_matches && c.kept == r.filtered_matches && c.removed_distance == c.raw - c.kept &&
            c.removed_mutual + c.removed_ratio + c.removed_level == 0 && c.invalid_query + c.invalid_train == 0,
        "counters: raw %d = kept %d + distance %d", c.raw, c.kept, c.removed_distance);

  PipelineParams on = rk;
  on.frontend.matcher.mutual = true, on.frontend.matcher.ratio = 0.8, on.frontend.matcher.rank = true;
  const PairResult m = ProcessPair(0, 1, a, b, f0, f1, K, on);
  bool consistent = true, ordered = true;
  std::vector<int> trains;
  for (size_t k = 0; k < m.correspondences.size(); ++k) {
    const Correspondence &cc = m.correspondences[k];
    consistent &= cc.mutual && cc.rank == int(k) && (!cc.has_second || cc.hamming < 0.8 * cc.second_distance);
    if (k) ordered &= m.correspondences[k - 1].ratio <= cc.ratio;
    trains.push_back(cc.train);
  }
  std::sort(trains.begin(), trains.end());
  const bool one_to_one = std::adjacent_find(trains.begin(), trains.end()) == trains.end();
  const features::MatchCounters &mc = m.match_counters;
  CHECK(consistent && ordered && one_to_one && mc.kept == int(m.correspondences.size()) &&
            mc.raw == mc.kept + mc.removed_distance + mc.removed_mutual + mc.removed_ratio + mc.removed_level,
        "mutual + ratio 0.8 + rank: %d kept (removed: distance %d, mutual %d, ratio %d); all mutual, one-to-one, "
        "ratio < 0.8, in quality order with ranks 0..n-1", mc.kept, mc.removed_distance, mc.removed_mutual,
        mc.removed_ratio);
  CHECK(m.match_details.size() == size_t(mc.raw), "every raw match has a details row");
  const PairResult m2 = ProcessPair(0, 1, a, b, f0, f1, K, on);
  CHECK(m2.R == m.R && m2.t == m.t && SameCorrespondences(m.correspondences, m2.correspondences),
        "deterministic with the options on");
  return scratch_test::Finish("test_ranked_frontend");
}
