// Self-checking test of the scratch PNG decoder (DEFLATE, zlib, CRC, filters).
// Reference values come from independent implementations: the zlib streams
// were produced by Python's zlib, the image fingerprints by Pillow, and the
// depth values by the reference pipeline's export (frozen baseline).

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.hpp"
#include "png.hpp"

using namespace scratch;

namespace {

std::vector<uint8_t> Hex(const std::string &h) {
  std::vector<uint8_t> b;
  for (size_t i = 0; i + 1 < h.size(); i += 2) b.push_back(uint8_t(std::stoi(h.substr(i, 2), nullptr, 16)));
  return b;
}

uint64_t Fingerprint(const PngImage &img) {  // FNV-1a over (sample + 1), row-major interleaved
  uint64_t h = 1469598103934665603ull;
  for (uint16_t v : img.samples) h = (h ^ uint64_t(v + 1)) * 1099511628211ull;
  return h;
}

}  // namespace

int main() {
  std::printf("=== scratch PNG decoder ===\n");
  const auto s = [](const char *t) { return std::vector<uint8_t>(t, t + std::string(t).size()); };

  CHECK(Crc32(s("123456789").data(), 9) == 0xCBF43926u, "CRC-32 check value of \"123456789\"");
  CHECK(Adler32(s("Wikipedia").data(), 9) == 0x11E60398u, "Adler-32 of \"Wikipedia\"");

  std::vector<uint8_t> out;
  std::string err;
  // stored block: header 78 01, BFINAL=1 BTYPE=00, LEN=5, NLEN, "hello", Adler-32
  std::vector<uint8_t> stored = {0x78, 0x01, 0x01, 0x05, 0x00, 0xFA, 0xFF, 'h', 'e', 'l', 'l', 'o'};
  const uint32_t ad = Adler32(s("hello").data(), 5);
  for (int k = 3; k >= 0; --k) stored.push_back(uint8_t(ad >> (8 * k)));
  CHECK(ZlibDecompress(stored, out, &err) && std::string(out.begin(), out.end()) == "hello", "stored block");
  CHECK(ZlibDecompress(Hex("78dacb48cdc9c90700062c0215"), out, &err) && std::string(out.begin(), out.end()) == "hello",
        "fixed-Huffman block (Python zlib)");
  {
    std::string text;
    for (int i = 0; i < 3; ++i) text += "The quick brown fox jumps over the lazy dog. ";
    for (int i = 0; i < 256; ++i) text += char(i);
    for (int i = 0; i < 20; ++i) text += "abracadabra";
    const auto z = Hex(
        "78da0bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb2984d04c31032313330b2b"
        "1b3b072717370f2f1fbf80a090b088a898b884a494b48cac9cbc82a292b28aaa9aba86a696b68eae9ebe81a191b189a999b985a595b58dad9"
        "dbd83a393b38bab9bbb87a797b78faf9f7f4060507048685878446454744c6c5c7c426252724a6a5a7a466656764e6e5e7e4161517149695"
        "979456555754d6d5d7d436353734b6b5b7b476757774f6f5fff848993264f993a6dfa8c99b366cf993b6ffe82858b162f59ba6cf98a95ab56"
        "af59bb6efd868d9b366fd9ba6dfb8e9dbb76efd9bb6fff8183870e1f397aecf88993a74e9f397beefc858b972e5fb97aedfa8d9bb76edfb97"
        "beffe83878f1e3f79faecf98b97af5ebf79fbeefd878f9f3e7ff9faedfb8f9fbf7efff9fbef7f6252516272620a881a764c00bd600695");
    CHECK(ZlibDecompress(z, out, &err) && std::string(out.begin(), out.end()) == text,
          "611 bytes with length/distance back-references (Python zlib)");
    auto bad = z;
    bad[40] ^= 0x10;
    CHECK(!ZlibDecompress(bad, out, &err), "corrupted stream is rejected (%s)", err.c_str());
  }

  // --- real dataset images ---
  const auto repo = scratch_test::Repo();
  const std::string ds = (repo / "data" / "synthetic_bunny").string();
  PngImage rgb0, dep0, dep1;
  err.clear();
  CHECK(LoadPng(ds + "/000000.png", rgb0, &err), "frame 0 RGB decodes (%s)", err.c_str());
  CHECK(rgb0.width == 640 && rgb0.height == 480 && rgb0.channels == 3 && rgb0.bit_depth == 8, "frame 0: 640x480 RGB, 8-bit");
  CHECK(Fingerprint(rgb0) == 0x183173ae11485121ull, "frame 0 RGB samples equal an independent decoder's (fingerprint)");
  err.clear();
  CHECK(LoadPng(ds + "/000000_depth.png", dep0, &err) && LoadPng(ds + "/000001_depth.png", dep1, &err),
        "depth frames decode");
  CHECK(dep0.channels == 1 && dep0.bit_depth == 16, "depth: 1 channel, 16-bit");
  CHECK(Fingerprint(dep0) == 0xb23f5e7f8c9e1f41ull, "frame 0 depth samples equal an independent decoder's (fingerprint)");
  {
    // the dataset PNGs use dynamic-Huffman blocks (BTYPE 2): read the first block header
    std::ifstream f(ds + "/000000.png", std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t p = 8;
    while (p + 8 < b.size() && std::string(b.begin() + p + 4, b.begin() + p + 8) != "IDAT")
      p += 12 + ((b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]);
    CHECK(((b[p + 10] >> 1) & 3) == 2, "the image's first DEFLATE block is dynamic Huffman (exercises that path)");
  }
  {
    // exact 16-bit values: every filtered match of the frozen baseline export
    Csv c;
    c.load(repo / "docs" / "migration" / "baseline" / "correspondences" / "pair_0_1_correspondences.csv");
    int ok = 0;
    for (size_t r = 0; r < c.rows.size(); ++r) {
      const int ui = int(c.d(r, "u_i")), vi = int(c.d(r, "v_i")), uj = int(c.d(r, "u_j")), vj = int(c.d(r, "v_j"));
      ok += dep0.samples[size_t(vi) * 640 + ui] == c.i(r, "depth_raw_i") &&
            dep1.samples[size_t(vj) * 640 + uj] == c.i(r, "depth_raw_j");
    }
    CHECK(ok == int(c.rows.size()) && ok > 0, "%d / %d depth values equal the baseline export's", ok, int(c.rows.size()));
  }
  return scratch_test::Finish("test_png");
}
