// Minimal image I/O and command-line adapter for the original, unchanged
// src/features and include/features implementation from slam-main.zip.
// No OpenCV, Eigen, g2o or SLAM dependencies are used in this executable.
#include "features/brief.hpp"
#include "features/fast.hpp"
#include "features/multiscale.hpp"
#include "features/orientation.hpp"
#include "features/pyramid.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string Token(std::istream& in) {
  std::string value;
  char c;
  while (in.get(c)) {
    if (c == '#') { std::string discard; std::getline(in, discard); continue; }
    if (!std::isspace(static_cast<unsigned char>(c))) { value += c; break; }
  }
  while (in.get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) break;
    value += c;
  }
  if (value.empty()) throw std::runtime_error("Unexpected end of PGM header");
  return value;
}
features::GrayImage ReadPGM(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open PGM: " + filename);
  if (Token(in) != "P5") throw std::runtime_error("Only binary 8-bit PGM (P5) is supported");
  const int width = std::stoi(Token(in)), height = std::stoi(Token(in));
  const int maxval = std::stoi(Token(in));
  if (width <= 0 || height <= 0 || maxval != 255)
    throw std::runtime_error("Invalid PGM dimensions or max value (expected 255)");
  features::GrayImage image(width, height);
  in.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image.pixels.size()))
    throw std::runtime_error("Truncated PGM image: " + filename);
  return image;
}
void WriteRow(std::ostream &o, float x, float y, float angle, float scale,
              int score, int level, const features::BriefDescriptor &d) {
  o << std::fixed << std::setprecision(7)
    << x << ',' << y << ',' << angle << ',' << scale << ',' << score << ',' << level << ',';
  const char hex[] = "0123456789abcdef";
  for (int byte = 0; byte < 32; ++byte) {
    const unsigned b = static_cast<unsigned>((d.bits[byte / 8] >> (8 * (byte % 8))) & 0xffU);
    o << hex[b >> 4] << hex[b & 15];
  }
  o << '\n';
}
}
int main(int argc, char **argv) {
  try {
    std::string input, output;
    features::PyramidParams pyramid;
    features::FastParams fast;
    auto integer = [&](int &i) -> int {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + argv[i]);
      return std::stoi(argv[++i]);
    };
    auto number = [&](int &i) -> double {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + argv[i]);
      return std::stod(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--input" && i + 1 < argc) input = argv[++i];
      else if (arg == "--output" && i + 1 < argc) output = argv[++i];
      else if (arg == "--levels") pyramid.levels = integer(i);
      else if (arg == "--scale-factor") pyramid.scale_factor = number(i);
      else if (arg == "--fast-threshold") fast.threshold = integer(i);
      else if (arg == "--features-per-level") fast.max_keypoints = integer(i);
      else if (arg == "--cell-size") fast.cell_size = integer(i);
      else if (arg == "--min-threshold") fast.min_threshold = integer(i);
      else if (arg == "--help" || arg == "-h") {
        std::cout << "orb_extract --input image.pgm --output features.csv [--levels 8] "
                     "[--scale-factor 1.2] [--fast-threshold 30] "
                     "[--features-per-level 500] [--cell-size 0] [--min-threshold 0]\n";
        return 0;
      } else throw std::runtime_error("Unknown/incomplete argument: " + arg);
    }
    if (input.empty() || output.empty()) throw std::runtime_error("--input and --output are required");
    if (pyramid.levels < 1 || pyramid.levels > 30 || pyramid.scale_factor <= 1.0 ||
        fast.threshold < 1 || fast.threshold > 254 || fast.max_keypoints < 1)
      throw std::runtime_error("Invalid extractor parameter");
    const auto gray = ReadPGM(input);
    auto start = std::chrono::steady_clock::now();
    const auto pyr = features::BuildPyramid(gray, pyramid);
    const auto output_features = features::ExtractMultiscale(pyr, fast);
    auto stop = std::chrono::steady_clock::now();
    std::ofstream out(output);
    if (!out) throw std::runtime_error("Cannot write output: " + output);
    out << "x,y,angle_rad,scale,score,level,descriptor_hex\n";
    for (size_t i = 0; i < output_features.keypoints.size(); ++i) {
      const auto &kp = output_features.keypoints[i];
      WriteRow(out, kp.x, kp.y, kp.level_kp.angle, kp.scale,
               kp.level_kp.score, kp.level, output_features.descriptors[i]);
    }
    if (!out) throw std::runtime_error("Failed writing features CSV");
    const auto ms = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << std::fixed << std::setprecision(4)
              << "{\"keypoints\":" << output_features.keypoints.size()
              << ",\"extract_ms\":" << ms << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "orb_extract: " << e.what() << '\n';
    return 1;
  }
}
