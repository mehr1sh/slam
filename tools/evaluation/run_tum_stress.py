#!/usr/bin/env python3
"""Reproducible stress runs of tum_trajectory_eval on one TUM RGB-D sequence, with sanity checks,
figures and an aggregate table.

Usage (repository root; the sequence must already be downloaded, nothing is fetched):
  python3 tools/evaluation/run_tum_stress.py --dataset data/rgbd_dataset_freiburg1_xyz \
          --output results/runs/stress/xyz_session1 [--binary build/tum_trajectory_eval] [--force] [--cases a,b]

Cases (each in its own directory <output>/<case>/):
  full          --frames 0 (every usable frame)        n20 / n100 / n300   --frames N
  n100_stride2  --frames 100 --stride 2                 n100_stride4        --frames 100 --stride 4
  repeat_a / repeat_b   --frames 20, two fresh directories, compared file by file
n300 is skipped (status "skipped") when the sequence has fewer than 300 usable frames; the count is
taken from the full run.

Per case: command.txt, config.json, stdout.log, stderr.log, status.json (sanity checks and metrics), the
evaluator CSVs, and with --figures also figures/ (plot_tum_results.py: generic charts, metrics.json,
report.md; off by default). Aggregate: <output>/stress_summary.json and stress_summary.csv.
Image-based diagnostics are a separate, explicit step: tools/evaluation/diagnostics/.

Status per case:
  ok                    evaluator exit 0, all sanity checks passed. Tracking failures (NaN pairs)
                        are results, counted in failed_pairs, not errors.
  sanity_failed         evaluator exit 0 but a check failed (listed in status.json)
  infrastructure_error  evaluator exit != 0 (usage, data, I/O, too few frames), or its CSVs could not be read
  skipped               case not applicable (reason recorded)
Missing metrics are null with a reason. Exit status: 0 if every case is ok or skipped, else 1;
2 for invalid arguments or an existing output without --force.
"""

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tum_metrics as tm  # noqa: E402

MAX_DT, MAX_GT_GAP = 0.02, 0.05  # evaluator defaults, passed explicitly so config.json is complete
CASES = [("full", 0, 1), ("n20", 20, 1), ("n100", 100, 1), ("n300", 300, 1),
         ("n100_stride2", 100, 2), ("n100_stride4", 100, 4), ("repeat_a", 20, 1), ("repeat_b", 20, 1)]
COMPARED = ("frames.csv", "pairs.csv", "trajectory.csv", "summary.txt")  # logs are excluded


def git_state():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return {"commit": rev or None, "working_tree_dirty": bool(dirty.strip())}
    except OSError:
        return {"commit": None, "working_tree_dirty": None}


def run_case(name, frames, stride, a, binary, case_dir):
    if case_dir.exists():
        shutil.rmtree(case_dir)  # only reached with --force (checked in main)
    case_dir.mkdir(parents=True)
    cmd = [str(binary), str(a.dataset), "--frames", str(frames), "--stride", str(stride),
           "--max-dt", str(MAX_DT), "--max-gt-gap", str(MAX_GT_GAP), "--out-dir", str(case_dir.resolve())]
    (case_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    cfg = {"case": name, "dataset": str(a.dataset), "dataset_resolved": str(Path(a.dataset).resolve()),
           "binary": str(binary), "frames": frames, "stride": stride, "max_dt": MAX_DT, "max_gt_gap": MAX_GT_GAP,
           "cwd": str(ROOT), "git": git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (case_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    t0 = time.monotonic()
    with open(case_dir / "stdout.log", "w") as fo, open(case_dir / "stderr.log", "w") as fe:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fo, stderr=fe).returncode
    st = {"case": name, "frames_requested": frames, "stride": stride, "exit_code": rc,
          "wall_time_s": round(time.monotonic() - t0, 2), "problems": [], "metrics": None, "metrics_null_reason": None}
    if rc != 0:
        tail = (case_dir / "stderr.log").read_text().strip().splitlines()[-3:]
        st.update(status="infrastructure_error", metrics_null_reason=f"evaluator exited with {rc}: {' | '.join(tail)}")
        return st
    try:
        run = tm.load_run(case_dir)
    except tm.SchemaError as e:
        st.update(status="infrastructure_error", metrics_null_reason=f"evaluator output unreadable: {e}")
        return st
    p = tm.check_consistency(run) + tm.ground_truth_checks(run, a.dataset, MAX_DT, MAX_GT_GAP)
    if frames and run["n"] != frames:
        p.append(f"requested {frames} frames, evaluator produced {run['n']}")
    dt = np.diff(run["t"])
    st["median_frame_interval_s"] = float(np.median(dt))
    st["frames_produced"] = int(run["n"])
    st["metrics"] = tm.to_json_safe(tm.compute_metrics(run))
    if a.figures:
        plot = subprocess.run([sys.executable, str(HERE / "plot_tum_results.py"), "--input", str(case_dir)],
                              cwd=ROOT, capture_output=True, text=True)
        (case_dir / "plot.log").write_text(plot.stdout + plot.stderr)
        st["plot_exit_code"] = plot.returncode
        if plot.returncode == 2:
            p.append("plot_tum_results.py rejected the run (see plot.log)")
        elif plot.returncode == 1:
            p.append("plot_tum_results.py reported sanity problems (see figures/metrics.json)")
    st["problems"] = p
    st["status"] = "sanity_failed" if p else "ok"
    return st


def compare_runs(da, db):
    out = {"files": {}, "identical": True}
    for f in COMPARED:
        ba, bb = (da / f).read_bytes(), (db / f).read_bytes()
        if ba == bb:
            out["files"][f] = "byte-identical"
            continue
        out["identical"] = False
        if f.endswith(".csv"):
            ra = list(csv.reader(ba.decode().splitlines()))
            rb = list(csv.reader(bb.decode().splitlines()))
            if len(ra) != len(rb) or ra[0] != rb[0]:
                out["files"][f] = "different shape or header"
                continue
            md = 0.0
            for x, y in zip(ra[1:], rb[1:]):
                for u, v in zip(x, y):
                    try:
                        fu, fv = float(u), float(v)
                        if not (np.isnan(fu) and np.isnan(fv)):
                            md = max(md, abs(fu - fv))
                    except ValueError:
                        md = md if u == v else float("inf")
            out["files"][f] = f"differs, max abs numeric difference {md:.3g}"
        else:
            out["files"][f] = "differs"
    return out


FLAT = ["successful_pairs", "failed_pairs", "success_rate", "ate_raw_rmse_m", "ate_aligned_rmse_m",
        "final_position_error_m", "final_orientation_error_deg", "correspondences_mean", "inliers_mean_successful",
        "inliers_min_successful"]


def flatten(st):
    row = {k: st.get(k) for k in ("case", "status", "exit_code", "frames_requested", "stride", "frames_produced",
                                  "median_frame_interval_s", "wall_time_s")}
    met = st.get("metrics")
    row["time_span_s"] = met and met["time_span_s"]
    row["matches_mean"] = met and met["matches_mean"]
    for m in tm.METHODS:
        for k in FLAT:
            row[f"{m}_{k}"] = met and met[m][k]
        for rk in ("rpe_1frame", "rpe_1s"):
            r = met and met[m].get(rk)
            row[f"{m}_{rk}_trans_rmse_m"] = r and r["trans_rmse_m"]
            row[f"{m}_{rk}_rot_rmse_deg"] = r and r["rot_rmse_deg"]
    reasons = [st.get("metrics_null_reason")] + (met["notes"] if met else [])
    row["null_reasons"] = "; ".join(r for r in reasons if r)
    row["problems"] = "; ".join(st.get("problems") or [])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="downloaded TUM sequence directory")
    ap.add_argument("--output", required=True, help="session directory, e.g. results/runs/stress/xyz_session1")
    ap.add_argument("--binary", default=str(ROOT / "build" / "tum_trajectory_eval"))
    ap.add_argument("--force", action="store_true", help="replace existing case directories")
    ap.add_argument("--figures", action="store_true", help="also draw the generic per-case charts (plot_tum_results.py)")
    ap.add_argument("--cases", help="comma-separated subset of: " + ",".join(c[0] for c in CASES))
    a = ap.parse_args()
    ds, out, binary = Path(a.dataset), Path(a.output), Path(a.binary)
    if not (ds / "rgb.txt").is_file() or not (ds / "groundtruth.txt").is_file():
        print(f"error: {ds} is not a downloaded TUM sequence (rgb.txt / groundtruth.txt missing)", file=sys.stderr)
        return 2
    if not binary.is_file():
        print(f"error: evaluator {binary} not found; build it first (cmake --build build)", file=sys.stderr)
        return 2
    if (ROOT / "data").resolve() in [out.resolve(), *out.resolve().parents]:
        print("error: --output must not be inside data/", file=sys.stderr)
        return 2
    cases = CASES
    if a.cases:
        want = a.cases.split(",")
        bad = set(want) - {c[0] for c in CASES}
        if bad:
            print(f"error: unknown cases {sorted(bad)}", file=sys.stderr)
            return 2
        cases = [c for c in CASES if c[0] in want]
    existing = [c[0] for c in cases if (out / c[0]).exists()]
    if existing and not a.force:
        print(f"error: {out} already holds {existing}; use a new --output or --force", file=sys.stderr)
        return 2
    results, usable = [], None
    for name, frames, stride in cases:
        if name == "n300" and usable is not None and usable < 300:
            st = {"case": name, "frames_requested": 300, "stride": 1, "status": "skipped", "exit_code": None,
                  "metrics": None, "metrics_null_reason": f"sequence has only {usable} usable frames (< 300)"}
            print(f"[{name}] skipped: {st['metrics_null_reason']}")
            results.append(st)
            continue
        print(f"[{name}] --frames {frames} --stride {stride} ...", flush=True)
        st = run_case(name, frames, stride, a, binary, out / name)
        if name == "full" and st.get("frames_produced"):
            usable = st["frames_produced"]
        print(f"[{name}] {st['status']} (exit {st['exit_code']}, {st['wall_time_s']} s)"
              + (f": {st['problems'] or st['metrics_null_reason']}" if st["status"] != "ok" else ""), flush=True)
        (out / name / "status.json").write_text(json.dumps(st, indent=2) + "\n")
        results.append(st)
    rep = None
    by = {r["case"]: r for r in results}
    if by.get("repeat_a", {}).get("exit_code") == 0 and by.get("repeat_b", {}).get("exit_code") == 0:
        rep = compare_runs(out / "repeat_a", out / "repeat_b")
        print(f"[repeatability] identical: {rep['identical']} {rep['files']}")
    summary = {"dataset": str(ds), "binary": str(binary), "git": git_state(), "max_dt": MAX_DT,
               "max_gt_gap": MAX_GT_GAP, "usable_frames_full_sequence": usable, "repeatability": rep, "cases": results}
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [flatten(r) for r in results]
    with open(out / "stress_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"wrote {out / 'stress_summary.json'} and {out / 'stress_summary.csv'}")
    bad = [r["case"] for r in results if r["status"] not in ("ok", "skipped")]
    if rep and not rep["identical"]:
        bad.append("repeatability")
    if bad:
        print(f"NOT OK: {bad}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
