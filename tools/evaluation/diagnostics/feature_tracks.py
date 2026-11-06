#!/usr/bin/env python3
"""Feature tracking across consecutive frames, drawn on the real RGB images (from --export-pairs exports).

Usage (repository root):
  python3 tools/evaluation/diagnostics/feature_tracks.py --run <evaluator dir> --frames 71-75 \
          --dataset data/rgbd_dataset_freiburg1_xyz [--output DIR] [--max-tracks 120]
Needs the exports of pairs 71..74 for frames 71-75. Output (default <run>/../tracking/frames_<a>_<b>/):
tracking.png, tracks.csv (every track, all links), tracking.json (per-frame counts, sampling, checks).

Tracks. The evaluator matches each pair independently; a track chains the exported matches: the match
(keypoint a of frame k -> keypoint b of frame k+1) continues with the match of pair k+1 whose frame-i
keypoint is b. This needs frame k+1's keypoints to be identical in both exports (same image, same
detector): checked, and no link is made where it fails. Several matches can end on one keypoint; each
link belongs to the first track that reaches it (later tracks end there). Each link keeps the PnP class
solvePnPRansac returned for that match; nothing is re-matched.

Drawing, per frame (native resolution): every detected keypoint (yellow ring); for the drawn tracks, the
trail of the last positions (up to 3 frames) in that frame's pixel coordinates, each segment coloured by
the class of its match; a track born in the frame = open circle, a track ending there = cross. Drawn
tracks are a deterministic sample (most inlier links first, then longest, then first frame and match row;
outlier segments drawn thinner); the counts printed
under each frame are over ALL tracks.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GAP, HEAD, FOOT = 16, 74, 96
TRAIL = 3


def parse_ranges(s):
    out = []
    for tok in s.split(","):
        lo, _, hi = tok.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def build_tracks(pairs):
    """pairs: {k: loaded export}. Returns (tracks, keypoints_consistent, merges).

    A track is a list of (pair k, match row) links; each row's kp_j continues as kp_i of pair k+1.
    Several matches can end on the same frame-j keypoint, so chains can merge; each link is given to
    the first track (in pair, then row order) that reaches it, and the later track ends at the merge."""
    ks = sorted(pairs)
    consistent = {}
    for k in ks:
        if k + 1 in pairs:
            a, b = pairs[k]["kp_j_xy"], pairs[k + 1]["kp_i_xy"]
            consistent[k] = a.shape == b.shape and bool(np.array_equal(a, b))
    nxt = {k: {int(q): r for r, q in enumerate(pairs[k]["kp_i"])} for k in ks}  # one match per frame-i keypoint
    continued = {k: set() for k in ks}
    for k in ks:
        if not consistent.get(k):
            continue
        for b in pairs[k]["kp_j"]:
            if int(b) in nxt[k + 1]:
                continued[k + 1].add(nxt[k + 1][int(b)])
    tracks, seen, merges = [], set(), 0
    for k in ks:
        for r in range(pairs[k]["n"]):
            if r in continued[k]:
                continue  # not a track start
            links, kk, rr = [(k, r)], k, r
            seen.add((k, r))
            while consistent.get(kk) and int(pairs[kk]["kp_j"][rr]) in nxt.get(kk + 1, {}):
                rr = nxt[kk + 1][int(pairs[kk]["kp_j"][rr])]
                kk += 1
                if (kk, rr) in seen:
                    merges += 1
                    break
                seen.add((kk, rr))
                links.append((kk, rr))
            tracks.append(links)
    return tracks, consistent, merges


def positions(pairs, track):
    """[(frame, u, v, class of the match leaving this position or None)] along a track."""
    pos = []
    for k, r in track:
        p = pairs[k]
        pos.append((k, p["u_i"][r], p["v_i"][r], p["pnp_class"][r]))
    k, r = track[-1]
    pos.append((k + 1, pairs[k]["u_j"][r], pairs[k]["v_j"][r], None))
    return pos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--frames", required=True, help="frame window a-b (2 to 6 frames), e.g. 71-75")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--max-tracks", type=int, default=120)
    a = ap.parse_args()
    run = Path(a.run)
    frames = parse_ranges(a.frames)
    if not 2 <= len(frames) <= 6 or frames != list(range(frames[0], frames[-1] + 1)):
        print("error: --frames must be a range of 2 to 6 consecutive frames", file=sys.stderr)
        return 2
    out = Path(a.output) if a.output else run.parent / "tracking" / f"frames_{frames[0]}_{frames[-1]}"
    rows = C.pairs_rows(run)
    try:
        pairs = {k: C.load_pair(run / "diagnostics" / f"pair_{k}_{k + 1}") for k in frames[:-1]}
    except C.ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    problems = {k: C.validate_pair(p, rows.get(k)) for k, p in pairs.items()}
    tracks, consistent, merges = build_tracks(pairs)
    meta = {k: pairs[min(k, frames[-2])]["meta"] for k in frames}
    side = {k: "i" if k < frames[-1] else "j" for k in frames}
    t_of = {k: meta[k][f"t_rgb_{side[k]}"] for k in frames}
    rgb_of = {k: meta[k][f"rgb_{side[k]}"] for k in frames}
    kp_of = {k: (pairs[k]["kp_i_xy"] if k < frames[-1] else pairs[frames[-2]]["kp_j_xy"]) for k in frames}
    imgs = {k: C.read_rgb(a.dataset, rgb_of[k]) for k in frames}
    H, W = imgs[frames[0]].shape[:2]
    pos = [positions(pairs, t) for t in tracks]
    # per-frame counts over all tracks
    stat = {k: dict(alive=0, born=0, ended=0) for k in frames}
    for P in pos:
        fs = [q[0] for q in P]
        for f in fs:
            stat[f]["alive"] += 1
        stat[fs[0]]["born"] += fs[0] > frames[0]
        stat[fs[-1]]["ended"] += fs[-1] < frames[-1]
    n_inl = [sum(q[3] == "inlier" for q in P) for P in pos]
    order = sorted(range(len(pos)), key=lambda n: (-n_inl[n], -len(pos[n]), pos[n][0][0], tracks[n][0][1]))
    drawn = [n for n in order if len(pos[n]) >= 2][: a.max_tracks]
    total_w = len(frames) * W + (len(frames) - 1) * GAP
    fig, ax = C.pixel_figure(total_w, HEAD + H + FOOT)
    xo = {k: n * (W + GAP) for n, k in enumerate(frames)}
    for k in frames:
        ax.imshow(imgs[k], extent=(xo[k], xo[k] + W, HEAD + H, HEAD), alpha=0.8)
        xy = kp_of[k]
        ax.scatter(xy[:, 0] + xo[k], xy[:, 1] + HEAD, s=8, facecolors="none", edgecolors="#ffd400", lw=0.5, alpha=0.8)
        s = stat[k]
        ax.text(xo[k] + 4, HEAD + H + 8, f"frame {k}   t = {t_of[k]:.6f} s", fontsize=11, weight="bold", va="top")
        ax.text(xo[k] + 4, HEAD + H + 30, f"{len(xy)} keypoints   {rgb_of[k]}", fontsize=9.5, va="top", color="#444")
        ax.text(xo[k] + 4, HEAD + H + 50, f"tracks here {s['alive']}   born {s['born']}   ending {s['ended']}",
                fontsize=10, va="top")
    for k in frames[:-1]:
        c = pairs[k]["meta"]["counts"]
        ratio = c["pnp_inliers"] / c["pnp_correspondences"] if c["pnp_correspondences"] else float("nan")
        bad = c["pnp_inliers"] < 20
        ax.text(xo[k] + W + GAP / 2, HEAD - 8,
                f"{k}->{k + 1}: {c['matches']} matches, PnP inliers {c['pnp_inliers']}/{c['pnp_correspondences']} "
                f"({100 * ratio:.0f} %)", fontsize=10, ha="center", va="bottom", color="#CC3311" if bad else "black",
                weight="bold" if bad else "normal", bbox=dict(fc="white", ec="#CC3311" if bad else "#999", pad=2))
    for n in drawn:
        P = pos[n]
        for idx, (f, u, v, _) in enumerate(P):
            lo = max(0, idx - TRAIL + 1)
            seg = P[lo:idx + 1]
            for (f0, u0, v0, cls), (f1, u1, v1, _) in zip(seg[:-1], seg[1:]):
                st = C.CLASS_STYLE[cls]
                ax.plot([u0 + xo[f], u1 + xo[f]], [v0 + HEAD, v1 + HEAD], color=st["color"],
                        lw=1.8 if cls == "inlier" else 0.8, alpha=0.95 if cls == "inlier" else 0.55, ls=st["ls"],
                        solid_capstyle="round")
            cls_in = P[idx - 1][3] if idx > 0 else P[0][3]
            col = C.CLASS_STYLE[cls_in]["color"]
            if idx == 0 and f > frames[0]:
                ax.scatter([u + xo[f]], [v + HEAD], s=40, facecolors="none", edgecolors=col, lw=1.4, zorder=5)
            elif idx == len(P) - 1 and f < frames[-1]:
                ax.scatter([u + xo[f]], [v + HEAD], s=40, marker="x", color=col, lw=1.6, zorder=5)
            else:
                ax.scatter([u + xo[f]], [v + HEAD], s=12, color=col, zorder=5)
    ax.text(4, 6, f"Feature tracks over frames {frames[0]}-{frames[-1]} ({run.parent.name}): trails of the last "
                  f"{TRAIL} positions in each frame's own pixel coordinates; {len(drawn)} of "
                  f"{sum(1 for P in pos if len(P) >= 2)} tracks drawn (most inlier links first)", fontsize=12, va="top", weight="bold")
    x = 4
    for cls, txt in (("inlier", "match was a PnP inlier"), ("outlier", "PnP outlier"),
                     ("no_depth_i", "no depth at the earlier keypoint (not given to PnP)")):
        st = C.CLASS_STYLE[cls]
        ax.plot([x, x + 26], [HEAD - 44] * 2, color=st["color"], lw=2.5, ls=st["ls"])
        ax.text(x + 32, HEAD - 44, txt, fontsize=10, va="center")
        x += 50 + 7 * len(txt)
    ax.scatter([x], [HEAD - 44], s=40, facecolors="none", edgecolors="black", lw=1.2)
    ax.text(x + 12, HEAD - 44, "track born", fontsize=10, va="center")
    ax.scatter([x + 100], [HEAD - 44], s=40, marker="x", color="black", lw=1.4)
    ax.text(x + 112, HEAD - 44, "track ends (no match onward)", fontsize=10, va="center")
    ax.scatter([x + 330], [HEAD - 44], s=40, facecolors="none", edgecolors="#ffd400", lw=1.2)
    ax.text(x + 342, HEAD - 44, "detected keypoint", fontsize=10, va="center")
    made = []
    C.save_png(fig, out / "tracking.png", made)
    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "drawn", "position", "frame", "u", "v", "pnp_class_of_next_link", "pair", "match_row"])
        dset = set(drawn)
        for n, P in enumerate(pos):
            for idx, (fr, u, v, cls) in enumerate(P):
                link = tracks[n][idx] if idx < len(tracks[n]) else ("", "")
                w.writerow([n, int(n in dset), idx, fr, u, v, cls or "", link[0], link[1]])
    C.write_json(out / "tracking.json", {
        "run": str(run), "frames": frames, "timestamps": {str(k): t_of[k] for k in frames},
        "rgb": {str(k): rgb_of[k] for k in frames}, "keypoints_identical_across_exports":
            {str(k): v for k, v in consistent.items()}, "merges_ended": merges, "tracks": len(tracks),
        "tracks_with_2_or_more_positions": sum(1 for P in pos if len(P) >= 2), "drawn_tracks": len(drawn),
        "sampling": f"most inlier links, then longest, then first frame, then match row; at most {a.max_tracks}",
        "per_frame": {str(k): stat[k] for k in frames},
        "per_pair": {str(k): pairs[k]["meta"]["counts"] for k in frames[:-1]},
        "export_problems": {str(k): v for k, v in problems.items() if v}, "figures": made})
    print("\n".join(made))
    return 1 if any(problems.values()) or not all(consistent.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
