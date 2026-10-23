#include "features/matcher.hpp"
#include "features/multiscale.hpp"
#include "geometry/fundamental.hpp"
#include "refine.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <stdexcept>
void check(bool b, const char *message) {
  if (!b)
    throw std::runtime_error(message);
}
int main() {
  try {
    auto quota = features::LevelQuotas(1400, 8, 1.2);
    int sum = 0;
    for (int q : quota)
      sum += q;
    check(sum == 1400, "Feature budget conservation");
    check(quota.front() > quota.back(), "Geometric level allocation");
    features::GrayImage corner(41, 41), edge(41, 41);
    for (int y = 0; y < 41; ++y)
      for (int x = 0; x < 41; ++x) {
        corner.at(x, y) = x >= 20 && y >= 20 ? 255 : 0;
        edge.at(x, y) = x >= 20 ? 255 : 0;
      }
    check(features::HarrisResponse(corner, 20, 20) >
              features::HarrisResponse(edge, 20, 20),
          "Harris corner versus edge");
    std::vector<features::BriefDescriptor> a(3);
    a[1].bits[0] = 15;
    a[2].bits[0] = 255;
    auto matches = features::MatchMutualRatio(a, a, .8);
    check(matches.size() == 3, "Mutual exact matching");
    auto duplicate = a;
    duplicate.push_back(a[0]);
    check(features::MatchMutualRatio(a, duplicate, .8).size() == 2,
          "Ambiguous descriptor rejection");
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<Eigen::Vector2d> p, q;
    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector2d> uv;
    Eigen::Matrix3d R =
        Eigen::AngleAxisd(.12, Eigen::Vector3d(.2, 1, .1).normalized())
            .toRotationMatrix();
    Eigen::Vector3d t(.2, .03, .05);
    Intrinsics K;
    K.fx = 520;
    K.fy = 521;
    K.cx = 320;
    K.cy = 240;
    for (int i = 0; i < 100; ++i) {
      Eigen::Vector3d x(d(rng), d(rng), 3 + d(rng)), y = R * x + t;
      X.push_back(x);
      p.emplace_back(K.fx * x.x() / x.z() + K.cx, K.fy * x.y() / x.z() + K.cy);
      q.emplace_back(K.fx * y.x() / y.z() + K.cx, K.fy * y.y() / y.z() + K.cy);
      uv.push_back(q.back());
    }
    Eigen::Matrix3d F;
    check(geometry_scratch::EstimateFundamental8Point(p, q, F),
          "Fundamental recovery");
    check(std::abs(F.determinant()) < 1e-10, "Fundamental rank two");
    for (size_t i = 0; i < p.size(); ++i) {
      Eigen::Vector3d a(p[i].x(), p[i].y(), 1), b(q[i].x(), q[i].y(), 1);
      check(std::abs(b.dot(F * a)) < 1e-7, "Epipolar consistency");
    }
    std::vector<Eigen::Vector2d> degenerate(10, Eigen::Vector2d(2, 2));
    check(
        !geometry_scratch::EstimateFundamental8Point(degenerate, degenerate, F),
        "Degenerate fundamental rejection");
    for (int i = 0; i < 10; ++i)
      uv[i] += Eigen::Vector2d(80, -60);
    std::vector<bool> use(X.size(), true);
    scratch::RefineParams ls, huber;
    huber.huber_delta_px = 2.;
    auto plain = scratch::RefinePnP(X, uv, use, K, R, t, ls),
         robust = scratch::RefinePnP(X, uv, use, K, R, t, huber);
    check(plain.ok && robust.ok, "Refinement validity");
    check((robust.t - t).norm() < (plain.t - t).norm(),
          "Huber resistance to contaminated inliers");
    std::cout << "PASS: Harris, budgets, mutual matching, fundamental "
                 "geometry/degeneracy and robust PnP\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
