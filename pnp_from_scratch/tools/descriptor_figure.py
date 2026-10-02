#!/usr/bin/env python3
"""Hamming-distance figure: why the same match filter keeps 346 scratch
matches but 202 reference matches on pair 0 -> 1 (NumPy + Matplotlib only).

  scratch    descriptors from pnp_from_scratch/results/pipeline/keypoints/frame_00000{0,1}.csv
  reference  all-pairs histogram: docs/migration/05_figures_depth_edges/reference_orb_distances_0_1.csv
             (exported once by export_reference_orb_distances.py next to it);
             best matches: docs/migration/baseline/reference_features/ (frozen)
  "true match": the best match reprojects within 8 px of its frame-1 keypoint when
  the frame-0 depth point is moved with the ground-truth motion (depth required).

Output: --out (default pnp_from_scratch/results/figures/)/descriptor_distances_0_1.png|pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import make_figures as mf  # noqa: E402
from png_reader import read_png  # noqa: E402

ROOT = mf.ROOT
FX, FY, CX, CY = 520.9, 521.0, 325.1, 249.7
THRESHOLD = 30


def gt_rel():
    gt = [list(map(float, l.split())) for l in open(ROOT / "data/synthetic_bunny/groundtruth.txt") if not l.startswith("#")]

    def R_(x, y, z, w):
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    Ri, Rj = R_(*gt[0][4:8]), R_(*gt[1][4:8])
    return Rj.T @ Ri, Rj.T @ (np.array(gt[0][1:4]) - np.array(gt[1][1:4]))


def true_match(ui, uj, depth, R, t):
    out = np.zeros(len(ui), int)  # 1 true, 0 false, -1 no depth
    for k, ((u, v), (a, b)) in enumerate(zip(ui, uj)):
        z = depth[int(v), int(u)] / 5000
        if z == 0:
            out[k] = -1
            continue
        X = R @ np.array([(u - CX) / FX * z, (v - CY) / FY * z, z]) + t
        out[k] = np.hypot(FX * X[0] / X[2] + CX - a, FY * X[1] / X[2] + CY - b) < 8
    return out


def scratch():
    def load(k):
        r = mf.rows(ROOT / f"pnp_from_scratch/results/pipeline/keypoints/frame_{k:06d}.csv")
        bits = np.array([[(int(x["descriptor_hex"][w * 16:(w + 1) * 16], 16) >> b) & 1 for w in range(4)
                          for b in range(64)] for x in r], np.uint8)
        return np.array([[float(x["x"]), float(x["y"])] for x in r]), bits
    (p0, b0), (p1, b1) = load(0), load(1)
    dist = (b0[:, None, :] != b1[None, :, :]).sum(2)
    best = dist.argmin(1)
    return np.bincount(dist.ravel(), minlength=257), dist.min(1), p0, p1[best], b0


def reference():
    h = np.loadtxt(ROOT / "docs/migration/05_figures_depth_edges/reference_orb_distances_0_1.csv", delimiter=",",
                   skiprows=2, dtype=int)
    hist = np.zeros(257, int)
    hist[h[:, 0]] = h[:, 1]
    kp = mf.rows(mf.BASE / "reference_features" / "pair_0_1_keypoints.csv")
    ki = [k for k in kp if k["frame"] == "i"]
    kj = [k for k in kp if k["frame"] == "j"]
    rm = mf.rows(mf.BASE / "reference_features" / "pair_0_1_raw_matches.csv")
    d = np.array([int(r["hamming_distance"]) for r in rm])
    ui = np.array([[float(ki[int(r["query_index"])]["u"]), float(ki[int(r["query_index"])]["v"])] for r in rm])
    uj = np.array([[float(kj[int(r["train_index"])]["u"]), float(kj[int(r["train_index"])]["v"])] for r in rm])
    return hist, d, ui, uj


def median_of_hist(h):
    c = np.cumsum(h)
    return int(np.searchsorted(c, c[-1] / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(mf.OUT.relative_to(ROOT)))
    mf.OUT = (ROOT / ap.parse_args().out).resolve()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    R, t = gt_rel()
    depth = read_png(ROOT / "data/synthetic_bunny/000000_depth.png")
    hs, ds, si, sj, bits = scratch()
    hr, dr, ri, rj = reference()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    bins = np.arange(0, 161, 4)
    for ax, name, h, d, ui, uj, colr in ((axs[0], "Scratch (random Gaussian pattern)", hs, ds, si, sj, mf.SCRATCH_REF),
                                         (axs[1], "Reference ORB (learned pattern)", hr, dr, ri, rj, mf.REF_PNP)):
        x = np.arange(257)
        ax.fill_between(x, h / h.sum(), step="mid", color="0.75", label=f"unrelated pairs (all {h.sum()}): median "
                        f"{median_of_hist(h)}, {100 * h[:THRESHOLD + 1].sum() / h.sum():.2f} % ≤ {THRESHOLD}")
        tm = true_match(ui, uj, depth, R, t)
        w = 1.0 / (4 * len(d))
        ax.hist([d[tm == 1], d[tm == 0], d[tm == -1]], bins=bins, stacked=True,
                weights=[np.full((tm == k).sum(), w) for k in (1, 0, -1)],
                color=[colr, (0.85, 0.15, 0.45), (0.55, 0.55, 0.55)],
                label=[f"best match, true ({(tm == 1).sum()})", f"best match, wrong ({(tm == 0).sum()})",
                       f"best match, no depth ({(tm == -1).sum()})"])
        ax.axvline(THRESHOLD + 0.5, color="black", ls="--", lw=1.2)
        ax.text(THRESHOLD + 2, ax.get_ylim()[1] * 0.97, f"filter: ≤ max(2·d_min, 30) = 30\nkept {(d <= THRESHOLD).sum()} "
                f"of {len(d)}", va="top", fontsize=8)
        ax.set_title(f"{name}: best-match median {int(np.median(d))}", fontsize=10)
        ax.set_xlabel("Hamming distance (256 bits)")
        ax.set_ylabel("fraction per distance")
        ax.set_xlim(0, 160)
        ax.legend(frameon=False, fontsize=7.5, loc="center right")
        ax.spines[["top", "right"]].set_visible(False)
    p = np.sort(bits.mean(0))
    axs[2].plot(np.arange(256), p, color=mf.SCRATCH_REF, lw=1.6)
    axs[2].axhspan(0.2, 0.8, color="0.92")
    axs[2].axhline(0.5, color="0.5", lw=0.8)
    axs[2].set_title(f"Scratch: fraction of 1s per bit, sorted\n{int(((p < 0.2) | (p > 0.8)).sum())} of 256 bits outside "
                     "0.2–0.8 (ORB: 29)", fontsize=10)
    axs[2].set_xlabel("bit (sorted)")
    axs[2].set_ylabel("P(bit = 1), frame-0 keypoints")
    axs[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pair 0 → 1: the same filter keeps more scratch matches because scratch distances are smaller overall",
                 fontsize=11)
    fig.tight_layout()
    mf.save(fig, "descriptor_distances_0_1")


if __name__ == "__main__":
    main()
