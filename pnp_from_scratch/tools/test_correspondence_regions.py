#!/usr/bin/env python3
"""Self-checking test of the correspondence-figure data (correspondence_regions.py: neighbourhood colours)
and of the exported match indexing it relies on. NumPy only.

Usage (repository root, after pnp_from_scratch/build/scratch_pipeline):
    python3 pnp_from_scratch/tools/test_correspondence_regions.py
Exit status 1 if a check fails.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correspondence_regions as cr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pnp_from_scratch" / "results" / "pipeline"
failures = 0


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


print("=== correspondence neighbourhoods ===")
# --- synthetic scene: three separated groups of 20 points, all moved by (+5, +2) px ---
rng = np.random.default_rng(3)
centres = np.array([[100.0, 100], [300, 120], [200, 300]])
pi = np.concatenate([c + rng.normal(0, 8, (20, 2)) for c in centres])
pj = pi + [5.0, 2.0]
n = len(pi)
used, inlier = np.ones(n, bool), np.ones(n, bool)
pj[5] = pi[45] + [5, 2]           # match 5 (group 0) lands inside group 2: a wrong match ...
inlier[5] = True                  # ... that geometric verification (here: by fiat) accepted
inlier[7] = False                 # match 7 is a RANSAC outlier that lands where its group went
used[9], inlier[9] = False, False  # match 9 has no depth
m = dict(ui=pi[:, 0], vi=pi[:, 1], uj=pj[:, 0], vj=pj[:, 1], used=used, inlier=inlier)
v = cr.classify(m, k=3)
groups_ok = all(len(set(v["label_i"][g * 20:(g + 1) * 20])) == 1 for g in range(3)) and len(set(v["label_i"])) == 3
check(groups_ok, "k-means neighbourhoods: each spatial group gets one label, the groups get different labels")
v2 = cr.classify(m, k=3)
check(np.array_equal(v["label_i"], v2["label_i"]) and np.array_equal(v["colour"], v2["colour"]),
      "deterministic: identical labels and colours on a second run")
check(np.array_equal(v["colour"][0], v["colour"][1]) and not np.array_equal(v["colour"][0], v["colour"][25]),
      "nearby keypoints share a colour, separate neighbourhoods do not")
check(v["neighbourhood_changed"][5] and v["label_j"][5] == v["label_i"][45],
      "a match landing in another neighbourhood is flagged (its frame-1 end votes for that neighbourhood)")
check(v["status"][5] == "inlier", "the flag does not change the geometric status (still an inlier)")
check(not v["neighbourhood_changed"][7] and v["status"][7] == "outlier",
      "a RANSAC outlier that lands with its neighbourhood stays an outlier and is not flagged")
check(int(v["neighbourhood_changed"].sum()) == 1, f"only the misplaced match is flagged ({int(v['neighbourhood_changed'].sum())})")
check(list(v["status"][[0, 7, 9]]) == ["inlier", "outlier", "nodepth"], "status = inlier / outlier / no depth from the flags")
try:
    cr.classify(dict(m, used=np.zeros(n, bool)))
    check(False, "an inlier without a PnP input is rejected")
except ValueError:
    check(True, "an inlier without a PnP input is rejected")

# --- on the exported pair 0 -> 1 ---
corr = PIPE / "correspondences" / "pair_0_1.csv"
if not corr.exists():
    check(False, f"missing {corr.relative_to(ROOT)}: run pnp_from_scratch/build/scratch_pipeline")
else:
    c = rows(corr)
    k0 = rows(PIPE / "keypoints" / "frame_000000.csv")
    k1 = rows(PIPE / "keypoints" / "frame_000001.csv")
    pair = rows(PIPE / "pairs.csv")[0]
    idx_ok = all(abs(float(x["u_i"]) - float(k0[int(x["query"])]["x"])) < 1e-3 and
                 abs(float(x["v_i"]) - float(k0[int(x["query"])]["y"])) < 1e-3 and
                 abs(float(x["u_j"]) - float(k1[int(x["train"])]["x"])) < 1e-3 and
                 abs(float(x["v_j"]) - float(k1[int(x["train"])]["y"])) < 1e-3 and
                 int(x["level_i"]) == int(k0[int(x["query"])]["level"]) for x in c)
    check(idx_ok, f"all {len(c)} exported matches index the right keypoints of frames 0 and 1 (pixels, level)")
    m = dict(ui=np.array([float(x["u_i"]) for x in c]), vi=np.array([float(x["v_i"]) for x in c]),
             uj=np.array([float(x["u_j"]) for x in c]), vj=np.array([float(x["v_j"]) for x in c]),
             used=np.array([x["pnp_used"] == "1" for x in c]), inlier=np.array([x["pnp_inlier"] == "1" for x in c]))
    v = cr.classify(m)
    n_in, n_out, n_nd = (int((v["status"] == s).sum()) for s in ("inlier", "outlier", "nodepth"))
    check(n_in == int(pair["pnp_inliers"]) and n_in + n_out == int(pair["correspondences_3d2d"]) and
          n_in + n_out + n_nd == int(pair["filtered_matches"]),
          f"figure counts = pipeline counts: {n_in} inliers, {n_out} outliers, {n_nd} without depth")
    res = np.array([float(x["pnp_residual_px"]) if x["pnp_residual_px"] else np.nan for x in c])
    ri = res[v["status"] == "inlier"]
    # the inlier set is RANSAC's (8 px under the RANSAC pose); refinement keeps that set, and the exported
    # residual is under the refined pose, so a few inliers can end slightly above 8 px
    print(f"  (info) inlier residuals under the refined pose: median {np.median(ri):.2f} px, max {ri.max():.2f} px, "
          f"{int((ri >= 8).sum())} at or above 8 px")
    print(f"  (info) neighbourhood warnings: {int((v['neighbourhood_changed'] & (v['status'] == 'inlier')).sum())} "
          f"inliers, {int((v['neighbourhood_changed'] & (v['status'] == 'outlier')).sum())} RANSAC outliers")

print(f"\ntest_correspondence_regions: {'FAIL' if failures else 'PASS'} ({failures} failures)")
sys.exit(1 if failures else 0)
