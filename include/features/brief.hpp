#pragma once

// Rotated BRIEF descriptor (Calonder et al. 2010; steered as in oriented
// FAST / rotated BRIEF):
//   1. the image is pre-smoothed with a Gaussian (sigma 2, 9x9 kernel)
//   2. 256 binary tests compare two smoothed pixels around the keypoint
//   3. the test locations come from a fixed, project-owned pattern drawn once
//      from an isotropic Gaussian (sigma = 31/5, the BRIEF paper's "G II"
//      setting) with a fixed seed, restricted to a disc of radius 15
//   4. the pattern is rotated by the keypoint angle before sampling
// Bit i = 1 iff I_smooth(p_i) < I_smooth(q_i).

#include <cstdint>
#include <vector>

#include "features/types.hpp"

namespace features {

constexpr int kBriefTests = 256;
constexpr int kBriefPatchRadius = 15;      // all pattern points lie within this disc
constexpr uint32_t kBriefSeed = 0x5B21EFu; // fixed seed of the sampling pattern
constexpr double kBriefSmoothingSigma = 2.0;
constexpr int kBriefSmoothingRadius = 4;   // 9x9 kernel

struct BriefTest {
  int x1, y1, x2, y2;  // offsets of the two compared pixels (unrotated)
};

// The fixed pattern (generated deterministically on first use).
const std::vector<BriefTest> &BriefPattern();

// Separable Gaussian blur, border mirrored (…cba|abc…, edge pixel not repeated),
// rounded to the nearest integer.
GrayImage GaussianSmooth(const GrayImage &img, double sigma = kBriefSmoothingSigma,
                         int radius = kBriefSmoothingRadius);

// Descriptor of one keypoint on an ALREADY SMOOTHED image. Returns false
// (and leaves `out` zero) if a rotated test location leaves the image.
bool ComputeBriefDescriptor(const GrayImage &smoothed, const Keypoint &kp, BriefDescriptor &out);

// Descriptors for all keypoints on the smoothed image; `valid` (optional)
// receives one flag per keypoint.
std::vector<BriefDescriptor> ComputeBrief(const GrayImage &smoothed, const std::vector<Keypoint> &keypoints,
                                          std::vector<bool> *valid = nullptr);

int HammingDistance(const BriefDescriptor &a, const BriefDescriptor &b);

}  // namespace features
