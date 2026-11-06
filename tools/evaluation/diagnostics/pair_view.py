#!/usr/bin/env python3
"""Feature correspondences of one frame pair on the real RGB images (from a --export-pairs export).

Usage (repository root):
  python3 tools/evaluation/diagnostics/pair_view.py --run <evaluator dir> --pair 73 \
          --dataset data/rgbd_dataset_freiburg1_xyz [--output DIR] [--max-lines 150]
Output (default <run>/../pairs/pair_<i>_<j>/): matches.png and matches_drawn.json.

matches.png, at the images' native resolution:
  top     frame i | frame j; every detected keypoint (small yellow ring), each filtered match as a line
          coloured by the class solvePnPRansac returned: inlier (blue), outlier (red), no depth at the
          frame-i keypoint, never given to PnP (grey, dotted); a frame-j keypoint that carries at least
          max(3, 20 %) of the inliers is ringed.
  bottom  left: frame i with the start of EVERY match, by class (not sampled: shows where the inliers are);
          right: frame j with the displacement u_i -> u_j of the drawn matches in frame j's pixel coordinates.
Lines are a deterministic sample (every k-th match of each class, in match order, at most --max-lines in
total, inliers first); matches_drawn.json lists exactly which match rows were drawn, and the export keeps all.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GAP, HEAD = 24, 132


def select(p, max_lines):
    """Rows to draw per class: inliers get up to 60 % of the budget, the rest shared by the others."""
    out = {}
    budget = max_lines
    inl = np.nonzero(p["pnp_class"] == "inlier")[0]
    out["inlier"] = C.sample_rows(inl, int(0.6 * max_lines))
    budget -= len(out["inlier"])
    others = [c for c in ("outlier", "unclassified", "no_depth_i") if (p["pnp_class"] == c).any()]
    for n, c in enumerate(others):
        rows = np.nonzero(p["pnp_class"] == c)[0]
        out[c] = C.sample_rows(rows, max(1, budget // (len(others) - n)))
        budget -= len(out[c])
    return out


def draw(p, imgs, out_png, max_lines, made, title_extra=""):
    m = p["meta"]
    H, W = imgs["i"].shape[:2]
    total_w, total_h = 2 * W + GAP, HEAD + H + 40 + H + 30
    fig, ax = C.pixel_figure(total_w, total_h)
    ax.imshow(imgs["i"], extent=(0, W, HEAD + H, HEAD))
    ax.imshow(imgs["j"], extent=(W + GAP, 2 * W + GAP, HEAD + H, HEAD))
    y2 = HEAD + H + 40
    ax.imshow(imgs["i"], extent=(0, W, y2 + H, y2), alpha=0.6)
    ax.imshow(imgs["j"], extent=(W + GAP, 2 * W + GAP, y2 + H, y2), alpha=0.7)
    for c in ("no_depth_i", "unclassified", "outlier", "inlier"):
        k = p["pnp_class"] == c
        ax.scatter(p["u_i"][k], p["v_i"][k] + y2, s=34 if c == "inlier" else 16, color=C.CLASS_STYLE[c]["color"],
                   edgecolors="white" if c == "inlier" else "none", lw=0.6, zorder=4 + (c == "inlier"))
    for side, ox in (("i", 0), ("j", W + GAP)):
        xy = p[f"kp_{side}_xy"]
        ax.scatter(xy[:, 0] + ox, xy[:, 1] + HEAD, s=9, facecolors="none", edgecolors="#ffd400", lw=0.6, alpha=0.9)
    sel = select(p, max_lines)
    order = ["no_depth_i", "unclassified", "outlier", "inlier"]
    for c in order:
        st = C.CLASS_STYLE[c]
        for r in sel.get(c, []):
            ax.plot([p["u_i"][r], p["u_j"][r] + W + GAP], [p["v_i"][r] + HEAD, p["v_j"][r] + HEAD], color=st["color"],
                    lw=st["lw"] * 1.2, alpha=min(1, st["alpha"] + 0.15), ls=st["ls"])
            ax.scatter([p["u_i"][r], p["u_j"][r] + W + GAP], [p["v_i"][r] + HEAD, p["v_j"][r] + HEAD], s=14,
                       color=st["color"], zorder=4)
        rows = sel.get(c, [])
        if len(rows):
            ox = W + GAP
            ax.quiver(p["u_i"][rows] + ox, p["v_i"][rows] + y2, (p["u_j"] - p["u_i"])[rows], (p["v_j"] - p["v_i"])[rows],
                      angles="xy", scale_units="xy", scale=1, color=st["color"], width=0.0018 if c != "inlier" else 0.0028,
                      alpha=0.95, zorder=5)
    # many-to-one: ring a frame-j keypoint only if it carries a large share of the inliers
    # (two-to-one inlier matches are common in normal pairs; they are counted in the header instead)
    kj, cnt = np.unique(p["kp_j"][p["pnp_class"] == "inlier"], return_counts=True)
    shared_inl = int(cnt[cnt > 1].sum()) if len(cnt) else 0
    for k, n in zip(kj, cnt):
        if n >= max(3, 0.2 * cnt.sum()):
            x, y = p["kp_j_xy"][int(k)]
            ax.scatter([x + W + GAP], [y + HEAD], s=420, facecolors="none", edgecolors="#0072B2", lw=2.2, zorder=6)
            right = x > W * 0.55
            ax.text(x + W + GAP + (-18 if right else 18), y + HEAD - 18,
                    f"{n} inlier matches end on this one keypoint (#{int(k)})", color="white", fontsize=10,
                    weight="bold", zorder=7, ha="right" if right else "left",
                    bbox=dict(fc="#0072B2", ec="none", alpha=0.9, pad=2))
    c = m["counts"]
    n_cls = {k: int((p["pnp_class"] == k).sum()) for k in C.PNP_CLASSES}
    ratio = c["pnp_inliers"] / c["pnp_correspondences"] if c["pnp_correspondences"] else float("nan")
    ax.text(8, 8, f"pair {m['pair']}: frame {m['frame_i']} -> {m['frame_j']}{title_extra}", fontsize=15, weight="bold",
            va="top")
    for side, ox in (("i", 8), ("j", W + GAP)):
        ax.text(ox, 38, f"frame {m['frame_' + side]}   t = {m['t_rgb_' + side]:.6f} s   {c['keypoints_' + side]} keypoints",
                fontsize=11, va="top", weight="bold")
        ax.text(ox, 58, m["rgb_" + side], fontsize=9.5, va="top", color="#444")
    ax.text(8, 82, f"filtered matches {c['matches']}  |  PnP correspondences (depth in frame i) {c['pnp_correspondences']}  |  "
                   f"PnP inliers {c['pnp_inliers']} ({100 * ratio:.1f} %)  |  outliers {n_cls['outlier']}  |  no depth {n_cls['no_depth_i']}"
                   + (f"  |  unclassified {n_cls['unclassified']}" if n_cls['unclassified'] else "")
, fontsize=10.5, va="top")
    ax.text(8, 104, f"PnP inliers sharing a frame-j keypoint with another inlier: {shared_inl}; distinct frame-j keypoints "
                    f"among the inliers: {len(kj)}", fontsize=10.5, va="top")
    legend = [("inlier", f"PnP inlier: {len(sel.get('inlier', []))} of {n_cls['inlier']} drawn"),
              ("outlier", f"PnP outlier: {len(sel.get('outlier', []))} of {n_cls['outlier']} drawn"),
              ("no_depth_i", f"no depth in frame i (not given to PnP): {len(sel.get('no_depth_i', []))} of {n_cls['no_depth_i']} drawn")]
    if n_cls["unclassified"]:
        legend.append(("unclassified", f"unclassified (solver failed): {len(sel.get('unclassified', []))} of "
                                       f"{n_cls['unclassified']} drawn"))
    x = 8
    for cls, txt in legend:
        st = C.CLASS_STYLE[cls]
        ax.plot([x, x + 30], [HEAD + H + 20] * 2, color=st["color"], lw=2.5, ls=st["ls"])
        ax.text(x + 36, HEAD + H + 20, txt, fontsize=10, va="center")
        x += 50 + 7.2 * len(txt)
    ax.scatter([x + 6], [HEAD + H + 20], s=40, facecolors="none", edgecolors="#ffd400", lw=1.2)
    ax.text(x + 18, HEAD + H + 20, "detected keypoint", fontsize=10, va="center")
    ax.text(8, y2 + H + 8, f"frame {m['frame_i']}: start of every match, by class (all {c['matches']}, not sampled)",
            fontsize=10, va="top")
    ax.text(W + GAP, y2 + H + 8, f"frame {m['frame_j']}: displacement u_i -> u_j of the drawn matches (1:1 px)",
            fontsize=10, va="top")
    C.save_png(fig, out_png, made)
    return {k: [int(r) for r in v] for k, v in sel.items()}


def run(run_dir, pair, dataset, out=None, max_lines=150, title_extra=""):
    run_dir = Path(run_dir)
    p = C.load_pair(run_dir / "diagnostics" / f"pair_{pair}_{pair + 1}")
    problems = C.validate_pair(p, C.pairs_rows(run_dir).get(pair))
    out = Path(out) if out else run_dir.parent / "pairs" / f"pair_{pair}_{pair + 1}"
    imgs = {s: C.read_rgb(dataset, p["meta"][f"rgb_{s}"]) for s in "ij"}
    made = []
    sel = draw(p, imgs, out / "matches.png", max_lines, made, title_extra)
    C.write_json(out / "matches_drawn.json", {
        "source": str(p["dir"]), "pair": pair, "frame_i": p["meta"]["frame_i"], "frame_j": p["meta"]["frame_j"],
        "t_rgb_i": p["meta"]["t_rgb_i"], "t_rgb_j": p["meta"]["t_rgb_j"], "max_lines": max_lines,
        "sampling": "per class, every k-th match row in match order; inliers up to 60 % of the budget",
        "drawn_match_rows": sel, "class_counts": {k: int((p["pnp_class"] == k).sum()) for k in C.PNP_CLASSES},
        "export_problems": problems})
    return made, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--max-lines", type=int, default=150)
    a = ap.parse_args()
    try:
        made, problems = run(a.run, a.pair, a.dataset, a.output, a.max_lines)
    except (C.ExportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(made))
    if problems:
        print("EXPORT INCONSISTENT:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
