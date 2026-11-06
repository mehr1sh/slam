"""PLY reader shared by build_tum_scene.py (inside Blender) and the tests (outside Blender); NumPy only."""

import numpy as np


def read_ply(path):
    """Binary little-endian PLY written by export_tum_scene.write_ply -> xyz (N,3) float32, rgb (N,3) uint8."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        n = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[2])
        rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                             ("b", "u1")], count=n)
    return np.c_[rec["x"], rec["y"], rec["z"]], np.c_[rec["r"], rec["g"], rec["b"]]
