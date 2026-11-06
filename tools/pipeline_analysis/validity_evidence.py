"""Evidence for the PnP pose-validity thresholds (Stage 2), from existing scratch_pipeline exports.

Read-only. For every pair of every run directory given, recomputes from correspondences/pair_<i>_<j>.csv
the quantities the validity gate (src/validity.hpp) uses -- inlier rows, distinct inlier keypoints in
each frame (by index and by location), inlier ratio, image coverage of the inliers, reprojection
statistics of the inliers under the final pose -- and joins the ground-truth error of the returned pose
from pairs.csv. Ground truth is used here ONLY to judge how well candidate thresholds separate good
from bad poses offline; the gate itself never sees it.

Usage: python3 tools/pipeline_analysis/validity_evidence.py RUN_DIR [RUN_DIR ...] [--csv OUT.csv]
"""

import csv
import math
import sys
from pathlib import Path

CELL_PX = 80          # coverage grid: 8 x 6 cells over 640 x 480 (must match ValidityParams::coverage_cell_px)
LOCATION_RADIUS = 2.0  # px, distinct locations (single linkage), as in the Stage-1 harness


def distinct_locations(pts, r=LOCATION_RADIUS):
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    groups = len(pts)
    order = sorted(range(len(pts)), key=lambda k: pts[k][0])
    for ia, a in enumerate(order):
        for b in order[ia + 1:]:
            if pts[b][0] - pts[a][0] > r:
                break
            if math.dist(pts[a], pts[b]) <= r:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
                    groups -= 1
    return groups


def cells(pts):
    return len({(int(x // CELL_PX), int(y // CELL_PX)) for x, y in pts})


def percentile(v, q):  # linear interpolation between order statistics (as ValidityPercentile)
    s = sorted(v)
    if not s:
        return float("nan")
    pos = q / 100 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    if frac == 0 or s[hi] == s[lo]:
        return s[lo]
    if math.isinf(s[hi]):
        return s[hi]
    return s[lo] + frac * (s[hi] - s[lo])


def pair_rows(run):
    pairs = list(csv.DictReader(open(run / "pairs.csv")))
    for p in pairs:
        i, j = int(p["frame_i"]), int(p["frame_j"])
        rows = list(csv.DictReader(open(run / "correspondences" / f"pair_{i}_{j}.csv")))
        used = [r for r in rows if r["pnp_used"] == "1"]
        inl = [r for r in used if r["pnp_inlier"] == "1"]
        pi = [(float(r["u_i"]), float(r["v_i"])) for r in inl]
        pj = [(float(r["u_j"]), float(r["v_j"])) for r in inl]
        res = [float(r["pnp_residual_px"]) for r in inl if r["pnp_residual_px"]]
        yield {
            "run": run.name, "pair": int(p["pair"]), "frame_i": i, "frame_j": j,
            "pnp_inputs": len(used), "solver_ok": int(p["pnp_ok"]), "inlier_rows": len(inl),
            "distinct_src": len({r["query"] for r in inl}), "distinct_dst": len({r["train"] for r in inl}),
            "loc_src": distinct_locations(pi), "loc_dst": distinct_locations(pj),
            "inlier_ratio": len(inl) / len(used) if used else 0.0,
            "cells_src": cells(pi), "cells_dst": cells(pj),
            "reproj_median_px": percentile(res, 50), "reproj_p90_px": percentile(res, 90),
            "rot_err_deg": float(p["rot_err_deg"]) if p["rot_err_deg"] else float("nan"),
            "trans_err_m": float(p["trans_err_m"]) if p["trans_err_m"] else float("nan"),
        }


def main(argv):
    out = None
    if "--csv" in argv:
        k = argv.index("--csv")
        out = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    rows = [r for d in argv for r in pair_rows(Path(d))]
    if out:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keys = ["inlier_rows", "distinct_src", "distinct_dst", "loc_src", "loc_dst", "inlier_ratio", "cells_src",
            "cells_dst", "reproj_median_px", "reproj_p90_px"]
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    for run, rs in by_run.items():
        print(f"\n== {run}: {len(rs)} pairs")
        print(f"{'metric':18s} {'min':>9s} {'p1':>9s} {'p5':>9s} {'median':>9s} {'max':>9s}")
        for k in keys:
            v = [r[k] for r in rs if r["solver_ok"]]
            print(f"{k:18s} {min(v):9.3f} {percentile(v, 1):9.3f} {percentile(v, 5):9.3f} "
                  f"{percentile(v, 50):9.3f} {max(v):9.3f}")
        worst = sorted(rs, key=lambda r: -r["rot_err_deg"] if r["rot_err_deg"] == r["rot_err_deg"] else 0)[:3]
        for r in worst:
            print("  worst by GT: pair %d  rows %d  src/dst %d/%d  loc %d/%d  ratio %.3f  cells %d/%d  "
                  "med %.2f  p90 %.2f  -> %.2f deg %.3f m" % (
                      r["pair"], r["inlier_rows"], r["distinct_src"], r["distinct_dst"], r["loc_src"], r["loc_dst"],
                      r["inlier_ratio"], r["cells_src"], r["cells_dst"], r["reproj_median_px"], r["reproj_p90_px"],
                      r["rot_err_deg"], r["trans_err_m"]))


if __name__ == "__main__":
    main(sys.argv[1:])
