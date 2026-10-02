#!/usr/bin/env python3
"""One-off export (reference side only, uses the frozen baseline's library):
the Hamming-distance histogram of ALL frame-0 x frame-1 ORB descriptor pairs,
for the descriptor-distance figure. The frozen baseline stores only the
best-match distances; this recomputes the descriptors with ORB::create()
defaults and asserts that the best-match distances equal the frozen
docs/migration/baseline/reference_features/pair_0_1_raw_matches.csv.

Run once in the default (library) environment, from the repository root:
    pixi run python docs/migration/05_figures_depth_edges/export_reference_orb_distances.py
Writes docs/migration/05_figures_depth_edges/reference_orb_distances_0_1.csv. The scratch figure
tools only read that CSV; they never import the library.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
im = [cv2.imread(str(ROOT / f"data/synthetic_bunny/00000{k}.png"), cv2.IMREAD_COLOR) for k in (0, 1)]
orb = cv2.ORB_create()
kd = [orb.compute(i, orb.detect(i, None)) for i in im]
(k0, d0), (k1, d1) = kd
B0, B1 = np.unpackbits(d0, axis=1), np.unpackbits(d1, axis=1)
dist = (B0[:, None, :] != B1[None, :, :]).sum(2)
frozen = [int(r["hamming_distance"]) for r in
          csv.DictReader(open(ROOT / "docs/migration/baseline/reference_features/pair_0_1_raw_matches.csv"))]
assert (len(k0), len(k1)) == (366, 367) and dist.min(1).tolist() == frozen, "does not reproduce the frozen baseline"
hist = np.bincount(dist.ravel(), minlength=257)
out = Path(__file__).with_name("reference_orb_distances_0_1.csv")
with open(out, "w") as f:
    f.write(f"# ORB::create() defaults, OpenCV {cv2.__version__}; all {dist.size} frame-0 x frame-1 descriptor pairs; "
            "best-match distances verified equal to the frozen baseline\n")
    f.write("distance,count\n")
    for d, c in enumerate(hist):
        f.write(f"{d},{c}\n")
print("wrote", out.relative_to(ROOT), "pairs", dist.size, "median", np.median(dist))
