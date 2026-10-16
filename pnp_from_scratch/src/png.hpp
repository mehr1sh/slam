#pragma once

// Minimal PNG decoder written for the scratch pipeline (standard library only).
//
// Supported (everything the synthetic Bunny dataset uses, and more):
//   colour type 0 (grey), 2 (RGB), 4 (grey + alpha), 6 (RGBA); bit depth 8 or 16;
//   no interlacing. Palette images and Adam7 interlacing are rejected with a
//   clear error.
// Steps (PNG spec, RFC 1950/1951):
//   signature -> chunks (CRC-32 checked) -> IHDR -> concatenated IDAT
//   -> zlib stream (header + DEFLATE + Adler-32 checked)
//   -> per-scanline filter reversal (None/Sub/Up/Average/Paeth) -> samples.

#include <cstdint>
#include <string>
#include <vector>

namespace scratch {

struct PngImage {
  int width = 0, height = 0;
  int channels = 0;    // samples per pixel (1, 2, 3 or 4)
  int bit_depth = 0;   // 8 or 16
  // Samples in row-major, interleaved order. For bit_depth 16 each value is
  // the big-endian 16-bit sample; for bit_depth 8 the 8-bit sample.
  std::vector<uint16_t> samples;
};

// Decodes a PNG held in memory. Returns false and sets *error on failure.
bool DecodePng(const std::vector<uint8_t> &bytes, PngImage &out, std::string *error = nullptr);

// Reads and decodes a PNG file.
bool LoadPng(const std::string &path, PngImage &out, std::string *error = nullptr);

// DEFLATE stream (RFC 1951) -> bytes. Exposed for tests.
bool Inflate(const uint8_t *data, size_t size, std::vector<uint8_t> &out, std::string *error = nullptr);

// zlib stream (RFC 1950: 2-byte header, DEFLATE data, Adler-32) -> bytes.
bool ZlibDecompress(const std::vector<uint8_t> &in, std::vector<uint8_t> &out, std::string *error = nullptr);

uint32_t Crc32(const uint8_t *data, size_t size, uint32_t crc = 0);
uint32_t Adler32(const uint8_t *data, size_t size);

}  // namespace scratch
