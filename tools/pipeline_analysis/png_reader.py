"""Minimal PNG reader for the scratch figure tools: Python standard library
(struct, zlib) + NumPy. It mirrors src/png.cpp (the C++ decoder): chunks with
CRC-32 check, IHDR, concatenated IDAT, zlib decompression, reversal of the five
scanline filters. Supports 8/16-bit grey, grey+alpha, RGB, RGBA; no palette,
no interlacing -- everything the Bunny dataset uses. No imaging library.
"""

import struct
import zlib

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _unfilter(raw, height, stride, bpp):
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos + 1).astype(np.int32)
        pos += stride + 1
        if ftype == 0:
            cur = line
        elif ftype == 2:  # Up
            cur = (line + prev) & 0xFF
        else:  # Sub, Average, Paeth depend on already reconstructed bytes of this row
            cur = np.empty(stride, dtype=np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                if ftype == 1:
                    p = a
                elif ftype == 3:
                    p = (a + b) >> 1
                elif ftype == 4:
                    c = prev[x - bpp] if x >= bpp else 0
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                else:
                    raise ValueError(f"invalid PNG filter type {ftype}")
                cur[x] = (line[x] + p) & 0xFF
        out[y] = cur
        prev = cur
    return out


def read_png(path):
    """Returns a NumPy array (H, W) or (H, W, C); uint8 or uint16."""
    data = open(path, "rb").read()
    if data[:8] != _SIG:
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, hdr = 8, [], None
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"{path}: CRC mismatch in {ctype!r}")
        if ctype == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat.append(body)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    width, height, depth, ctype_, _, _, interlace = hdr
    if interlace or ctype_ not in _CHANNELS or depth not in (8, 16):
        raise ValueError(f"{path}: unsupported PNG (colour type {ctype_}, depth {depth}, interlace {interlace})")
    ch = _CHANNELS[ctype_]
    bpp = ch * depth // 8
    raw = zlib.decompress(b"".join(idat))
    rows = _unfilter(raw, height, width * bpp, bpp)
    if depth == 16:
        img = rows.reshape(height, width * ch, 2).astype(np.uint16)
        img = (img[..., 0] << 8) | img[..., 1]
    else:
        img = rows
    img = img.reshape(height, width, ch)
    return img[..., 0] if ch == 1 else img
