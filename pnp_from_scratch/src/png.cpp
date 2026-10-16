// PNG decoding for the scratch pipeline: DEFLATE, zlib wrapper, chunk
// parsing and scanline filters, written from the specifications
// (RFC 1950, RFC 1951, PNG ISO/IEC 15948). Standard library only.

#include "png.hpp"

#include <cstdlib>
#include <fstream>
#include <iterator>

namespace scratch {

namespace {

bool Fail(std::string *error, const std::string &msg) {
  if (error) *error = msg;
  return false;
}

// ----------------------------------------------------------------- DEFLATE --

// LSB-first bit reader over a byte buffer (RFC 1951 section 3.1.1).
struct BitReader {
  const uint8_t *data;
  size_t size, pos = 0;
  uint32_t bitbuf = 0;
  int bitcnt = 0;
  bool overrun = false;

  int bits(int n) {  // next n bits, first bit read = least significant
    uint32_t v = bitbuf;
    while (bitcnt < n) {
      if (pos >= size) {
        overrun = true;
        return 0;
      }
      v |= uint32_t(data[pos++]) << bitcnt;
      bitcnt += 8;
    }
    bitbuf = v >> n;
    bitcnt -= n;
    return int(v & ((1u << n) - 1));
  }
  void align() {  // drop the remaining bits of the current byte
    bitbuf = 0;
    bitcnt = 0;
  }
};

// Canonical Huffman code (RFC 1951 section 3.2.2), represented by the number
// of codes of each length and the symbols sorted by (length, value). Decoding
// reads one bit at a time and checks, per length, whether the code read so
// far falls into that length's range.
struct Huffman {
  int count[16] = {};
  std::vector<int> symbol;

  // Returns false for an over-subscribed set of lengths.
  bool build(const int *lengths, int n) {
    for (int &c : count) c = 0;
    for (int i = 0; i < n; ++i) ++count[lengths[i]];
    count[0] = 0;
    int left = 1;
    for (int len = 1; len < 16; ++len) {
      left <<= 1;
      left -= count[len];
      if (left < 0) return false;
    }
    int offs[16];
    offs[1] = 0;
    for (int len = 1; len < 15; ++len) offs[len + 1] = offs[len] + count[len];
    symbol.assign(n, 0);
    for (int i = 0; i < n; ++i)
      if (lengths[i]) symbol[offs[lengths[i]]++] = i;
    return true;
  }

  int decode(BitReader &br) const {  // -1 on an invalid code
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
      code |= br.bits(1);
      const int c = count[len];
      if (code - c < first) return symbol[index + (code - first)];
      index += c;
      first += c;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }
};

const int kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                          35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const int kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const int kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                           257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

// Decodes literal/length + distance codes until end-of-block (symbol 256).
bool InflateCodes(BitReader &br, const Huffman &lit, const Huffman &dist, std::vector<uint8_t> &out,
                  std::string *error) {
  for (;;) {
    int sym = lit.decode(br);
    if (sym < 0 || br.overrun) return Fail(error, "inflate: invalid literal/length code");
    if (sym < 256) {
      out.push_back(uint8_t(sym));
    } else if (sym == 256) {
      return true;
    } else {
      sym -= 257;
      if (sym >= 29) return Fail(error, "inflate: invalid length symbol");
      const int len = kLenBase[sym] + br.bits(kLenExtra[sym]);
      const int ds = dist.decode(br);
      if (ds < 0 || ds >= 30) return Fail(error, "inflate: invalid distance symbol");
      const size_t d = size_t(kDistBase[ds] + br.bits(kDistExtra[ds]));
      if (br.overrun) return Fail(error, "inflate: truncated stream");
      if (d > out.size()) return Fail(error, "inflate: distance before start of output");
      const size_t from = out.size() - d;
      for (int k = 0; k < len; ++k) out.push_back(out[from + k]);  // overlapping copy is intended
    }
  }
}

bool InflateFixed(BitReader &br, std::vector<uint8_t> &out, std::string *error) {
  static Huffman lit, dist;
  static bool built = false;
  if (!built) {  // fixed codes of RFC 1951 section 3.2.6
    int l[288];
    for (int i = 0; i < 144; ++i) l[i] = 8;
    for (int i = 144; i < 256; ++i) l[i] = 9;
    for (int i = 256; i < 280; ++i) l[i] = 7;
    for (int i = 280; i < 288; ++i) l[i] = 8;
    lit.build(l, 288);
    int d[30];
    for (int &x : d) x = 5;
    dist.build(d, 30);
    built = true;
  }
  return InflateCodes(br, lit, dist, out, error);
}

bool InflateDynamic(BitReader &br, std::vector<uint8_t> &out, std::string *error) {
  static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  const int nlen = br.bits(5) + 257, ndist = br.bits(5) + 1, ncode = br.bits(4) + 4;
  if (nlen > 286 || ndist > 30) return Fail(error, "inflate: bad dynamic block counts");
  int lengths[320] = {};
  for (int i = 0; i < ncode; ++i) lengths[order[i]] = br.bits(3);
  Huffman cl;
  if (!cl.build(lengths, 19)) return Fail(error, "inflate: bad code-length code");
  int idx = 0;
  while (idx < nlen + ndist) {
    int sym = cl.decode(br);
    if (sym < 0 || br.overrun) return Fail(error, "inflate: bad code-length symbol");
    if (sym < 16) {
      lengths[idx++] = sym;
      continue;
    }
    int repeat_len = 0, times;
    if (sym == 16) {
      if (idx == 0) return Fail(error, "inflate: repeat with no previous length");
      repeat_len = lengths[idx - 1];
      times = 3 + br.bits(2);
    } else if (sym == 17) {
      times = 3 + br.bits(3);
    } else {
      times = 11 + br.bits(7);
    }
    if (idx + times > nlen + ndist) return Fail(error, "inflate: too many code lengths");
    while (times--) lengths[idx++] = repeat_len;
  }
  if (lengths[256] == 0) return Fail(error, "inflate: no end-of-block code");
  Huffman lit, dist;
  if (!lit.build(lengths, nlen) || !dist.build(lengths + nlen, ndist))
    return Fail(error, "inflate: bad literal/distance lengths");
  return InflateCodes(br, lit, dist, out, error);
}

// --------------------------------------------------------------------- PNG --

uint32_t BE32(const uint8_t *p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }

int Paeth(int a, int b, int c) {  // left, up, upper-left
  const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

}  // namespace

uint32_t Crc32(const uint8_t *data, size_t size, uint32_t crc) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[n] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

uint32_t Adler32(const uint8_t *data, size_t size) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < size; ++i) {
    a = (a + data[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

bool Inflate(const uint8_t *data, size_t size, std::vector<uint8_t> &out, std::string *error) {
  BitReader br{data, size};
  for (;;) {
    const int last = br.bits(1), type = br.bits(2);
    if (br.overrun) return Fail(error, "inflate: truncated block header");
    if (type == 0) {  // stored block
      br.align();
      if (br.pos + 4 > size) return Fail(error, "inflate: truncated stored block");
      const unsigned len = data[br.pos] | (data[br.pos + 1] << 8), nlen = data[br.pos + 2] | (data[br.pos + 3] << 8);
      br.pos += 4;
      if ((len ^ 0xFFFFu) != nlen) return Fail(error, "inflate: stored block length check failed");
      if (br.pos + len > size) return Fail(error, "inflate: truncated stored data");
      out.insert(out.end(), data + br.pos, data + br.pos + len);
      br.pos += len;
    } else if (type == 1) {
      if (!InflateFixed(br, out, error)) return false;
    } else if (type == 2) {
      if (!InflateDynamic(br, out, error)) return false;
    } else {
      return Fail(error, "inflate: invalid block type 3");
    }
    if (last) return true;
  }
}

bool ZlibDecompress(const std::vector<uint8_t> &in, std::vector<uint8_t> &out, std::string *error) {
  if (in.size() < 6) return Fail(error, "zlib: stream too short");
  const int cmf = in[0], flg = in[1];
  if ((cmf & 0x0F) != 8) return Fail(error, "zlib: compression method is not DEFLATE");
  if ((cmf * 256 + flg) % 31 != 0) return Fail(error, "zlib: header check failed");
  if (flg & 0x20) return Fail(error, "zlib: preset dictionary not supported");
  out.clear();
  if (!Inflate(in.data() + 2, in.size() - 6, out, error)) return false;
  if (Adler32(out.data(), out.size()) != BE32(in.data() + in.size() - 4))
    return Fail(error, "zlib: Adler-32 mismatch (corrupt data or decoder error)");
  return true;
}

bool DecodePng(const std::vector<uint8_t> &b, PngImage &img, std::string *error) {
  static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  if (b.size() < 8 || !std::equal(sig, sig + 8, b.begin())) return Fail(error, "png: bad signature");
  size_t p = 8;
  int color_type = -1;
  std::vector<uint8_t> idat;
  bool seen_end = false;
  while (p + 12 <= b.size() && !seen_end) {
    const uint32_t len = BE32(&b[p]);
    if (p + 12 + size_t(len) > b.size()) return Fail(error, "png: truncated chunk");
    const std::string type(reinterpret_cast<const char *>(&b[p + 4]), 4);
    const uint8_t *data = &b[p + 8];
    if (Crc32(&b[p + 4], len + 4) != BE32(&b[p + 8 + len])) return Fail(error, "png: CRC mismatch in " + type);
    if (type == "IHDR") {
      if (len != 13) return Fail(error, "png: bad IHDR");
      img.width = int(BE32(data));
      img.height = int(BE32(data + 4));
      img.bit_depth = data[8];
      color_type = data[9];
      if (data[10] != 0 || data[11] != 0) return Fail(error, "png: unknown compression/filter method");
      if (data[12] != 0) return Fail(error, "png: interlaced images are not supported");
    } else if (type == "IDAT") {
      idat.insert(idat.end(), data, data + len);
    } else if (type == "IEND") {
      seen_end = true;
    } else if (type == "PLTE") {
      return Fail(error, "png: palette images are not supported");
    }
    p += 12 + len;
  }
  switch (color_type) {
    case 0: img.channels = 1; break;
    case 2: img.channels = 3; break;
    case 4: img.channels = 2; break;
    case 6: img.channels = 4; break;
    default: return Fail(error, "png: unsupported colour type " + std::to_string(color_type));
  }
  if (img.bit_depth != 8 && img.bit_depth != 16) return Fail(error, "png: only bit depths 8 and 16 are supported");
  if (img.width <= 0 || img.height <= 0) return Fail(error, "png: bad dimensions");

  std::vector<uint8_t> raw;
  if (!ZlibDecompress(idat, raw, error)) return false;
  const size_t bpp = size_t(img.channels) * (img.bit_depth / 8);  // bytes per pixel
  const size_t stride = bpp * img.width;
  if (raw.size() != (stride + 1) * img.height) return Fail(error, "png: decompressed size does not match IHDR");

  // Reverse the scanline filters in place (PNG spec section 9).
  std::vector<uint8_t> prev(stride, 0), cur(stride);
  img.samples.assign(size_t(img.width) * img.height * img.channels, 0);
  for (int y = 0; y < img.height; ++y) {
    const uint8_t *row = &raw[y * (stride + 1)];
    const int filter = row[0];
    for (size_t x = 0; x < stride; ++x) {
      const int a = x >= bpp ? cur[x - bpp] : 0, up = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
      int pred;
      switch (filter) {
        case 0: pred = 0; break;
        case 1: pred = a; break;
        case 2: pred = up; break;
        case 3: pred = (a + up) / 2; break;
        case 4: pred = Paeth(a, up, c); break;
        default: return Fail(error, "png: invalid filter type " + std::to_string(filter));
      }
      cur[x] = uint8_t(row[1 + x] + pred);
    }
    uint16_t *dst = &img.samples[size_t(y) * img.width * img.channels];
    if (img.bit_depth == 8)
      for (size_t x = 0; x < stride; ++x) dst[x] = cur[x];
    else
      for (size_t x = 0; x < stride / 2; ++x) dst[x] = uint16_t((cur[2 * x] << 8) | cur[2 * x + 1]);
    std::swap(prev, cur);
  }
  return true;
}

bool LoadPng(const std::string &path, PngImage &out, std::string *error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Fail(error, "cannot open " + path);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return DecodePng(bytes, out, error);
}

}  // namespace scratch
