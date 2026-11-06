#!/usr/bin/env python3
"""Self-checking tests of plot_tum_results.py on synthetic runs (no dataset needed; Matplotlib, headless).

Usage (repository root): python3 tools/evaluation/test_plot_tum_results.py      exit 1 on failure
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthetic_runs import Rz, write_run  # noqa: E402

failures = 0


def check(ok, text):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    failures += not ok


def plot(run_dir, *extra):
    return subprocess.run([sys.executable, str(HERE / "plot_tum_results.py"), "--input", str(run_dir), *extra],
                          capture_output=True, text=True)


def failed_run(d, n, fail):
    """Ground truth turning 2 deg per frame; the estimate is exact except that pair `fail` failed (pose held)."""
    gt = [(Rz(2 * k), np.array([0.01 * k, 0.002 * k * k, 1.0])) for k in range(n)]
    est = list(gt[:fail + 1]) + [gt[fail]]
    C = gt[fail][0] @ gt[fail + 1][0].T  # later poses follow the true motion from the held pose
    est += [(C @ R, C @ (p - gt[fail + 1][1]) + gt[fail][1]) for R, p in gt[fail + 2:]]
    write_run(d, gt, est, est, failed=(fail,))


print("=== plot_tum_results ===")
tmp = Path(tempfile.mkdtemp(prefix="plot_tum_test_"))
FIGS = ("trajectory_3d", "trajectory_errors", "relative_pose_errors", "run_metrics")

for name, n in (("short", 10), ("long", 80)):  # bars (<= 60 pairs) and log-scale lines (> 60 pairs)
    failed_run(tmp / name, n, fail=3)
    p = plot(tmp / name)
    check(p.returncode == 0, f"{name} ({n} frames, one failed pair): exit 0 ({p.stderr.strip()[-200:]})")
    out = tmp / name / "figures"
    check(all((out / f"{f}.{e}").is_file() for f in FIGS for e in ("png", "pdf")), f"{name}: PNG and PDF of {FIGS}")
    check(not (out / "rgb_depth_examples.png").exists(), f"{name}: no RGB-D figure without a sequence directory")
    check(not (out / "feature_matches.png").exists(), f"{name}: no feature_matches figure (no match export)")
    res = json.loads((out / "metrics.json").read_text())
    check(res["plotted_values_match_csv"] and res["sanity_problems"] == [], f"{name}: plotted values match the CSVs")
    m = res["metrics"]["pnp"]
    check(m["failed_pairs"] == 1 and m["successful_pairs"] == n - 2 and m["held_poses"] == 1,
          f"{name}: the failed pair is counted ({m['successful_pairs']}/{n - 1}) and its pose flagged as held")
    nots = " ".join(res["not_generated"])
    check("feature_matches" in nots and "rgb_depth_examples" in nots, f"{name}: blocked figures listed with reasons")
    rep = (out / "report.md").read_text()
    check("| failed pairs | 1 | 1 |" in rep, f"{name}: report.md table shows the failure")

d = tmp / "bad"
failed_run(d, 10, fail=3)
(d / "pairs.csv").write_text((d / "pairs.csv").read_text().replace("pnp_inliers", "pnp_inl"))
p = plot(d)
check(p.returncode == 2 and "pnp_inliers" in p.stderr and not (d / "figures").exists(),
      "missing column: exit 2, named in the error, no figures written")

d = tmp / "nodata"
failed_run(d, 10, fail=3)
p = plot(d, "--output", str(tmp / "elsewhere"))
check(p.returncode == 0 and (tmp / "elsewhere" / "run_metrics.png").is_file(), "--output directory honoured")

shutil.rmtree(tmp)
print(f"\ntest_plot_tum_results: {'PASS' if failures == 0 else 'FAIL'} ({failures} failures)")
sys.exit(1 if failures else 0)
