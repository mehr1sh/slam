#!/usr/bin/env python3
"""Reproduce the visual diagnostics of one evaluator configuration (one command, explicit operation).

Usage (repository root; the sequence must already be downloaded):
  python3 tools/evaluation/diagnostics/make_diagnostics.py --dataset data/rgbd_dataset_freiburg1_xyz \
      --frames 100 --stride 4 --export-pairs 66-80 --failure 73 [--reference 68] \
      [--windows 66-70,71-75] [--baseline results/runs/stress/xyz_session1/n100_stride4] [--output DIR] [--force]

Output: results/runs/diagnostics/<sequence>/n<frames>_stride<stride>/ (or --output), refused if it exists unless --force:
  evaluator/   tum_trajectory_eval outputs with --export-pairs, config.json (command, baseline comparison), logs
  pairs/       pair_<i>_<j>/index.html for the failure (with the written comparison) and the reference pair
  icp/         registration before / after of both pairs
  tracking/    frames_<a>_<b>/tracking.png for each --windows entry
  scene/       scene.png (static render of the reconstruction and the trajectories)
  divergence/  divergence.png (connectors GT -> estimate, RGB of the selected frames)
  viewer/      index.html: the interactive viewer (pair selector, correspondences, 3D scene, registration)
  manifest.json  every command with its exit code
and results/runs/diagnostics/index.html lists every <sequence>/<run> with a viewer.
--reference defaults to pair_report.choose_reference (nearest pair with typical errors and motion).
Steps: evaluator; if --baseline: frames.csv, pairs.csv, trajectory.csv, summary.txt must be byte-identical;
then the views. Exit: 0 all ok, 1 a step failed or the baseline differs, 2 invalid arguments.
"""

import argparse
import filecmp
import html
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STANDARD = ("frames.csv", "pairs.csv", "trajectory.csv", "summary.txt")


def write_index(diag_root):
    """results/runs/diagnostics/index.html: one entry per <sequence>/<run> that has a viewer."""
    rows = []
    for v in sorted(diag_root.glob("*/*/viewer/index.html")):
        run = v.parents[1]
        cfg = run / "evaluator" / "config.json"
        cmd = json.loads(cfg.read_text()).get("command", "") if cfg.is_file() else ""
        pages = sorted(run.glob("pairs/pair_*/index.html"))
        links = " ".join(f'<a href="{p.relative_to(diag_root)}">{p.parent.name.replace("pair_", "").replace("_", "&rarr;")}</a>'
                         for p in pages)
        rows.append(f"<tr><td>{html.escape(run.parent.name)}</td><td><a href='{v.relative_to(diag_root)}'>"
                    f"{html.escape(run.name)}</a></td><td>{links}</td><td><code>{html.escape(cmd)}</code></td></tr>")
    (diag_root / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SLAM diagnostics</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;color:#111;background:#fff}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}
code{{font-size:12px}}</style></head><body><h1>SLAM diagnostics</h1>
<table><tr><th>sequence</th><th>run (opens the viewer)</th><th>pair pages</th><th>evaluator command</th></tr>
{''.join(rows)}</table></body></html>
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--export-pairs", required=True, help="e.g. 66-80 (must contain the failure, reference and windows)")
    ap.add_argument("--failure", type=int, required=True)
    ap.add_argument("--reference", type=int)
    ap.add_argument("--windows", default="", help="frame windows for the tracking strips, e.g. 66-70,71-75")
    ap.add_argument("--baseline", help="an earlier run of the same configuration to compare byte-for-byte")
    ap.add_argument("--output")
    ap.add_argument("--binary", default=str(ROOT / "build" / "tum_trajectory_eval"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seq = Path(a.dataset).name
    diag_root = ROOT / "results" / "runs" / "diagnostics"
    out = Path(a.output) if a.output else diag_root / seq / f"n{a.frames}_stride{a.stride}"
    if (ROOT / "data").resolve() in [out.resolve(), *out.resolve().parents]:
        print("error: --output must not be inside data/", file=sys.stderr)
        return 2
    if out.exists() and not a.force:
        print(f"error: {out} exists; choose another --output or pass --force", file=sys.stderr)
        return 2
    if out.exists():
        shutil.rmtree(out)
    ev = out / "evaluator"
    logs = out / "logs"
    ev.mkdir(parents=True)
    logs.mkdir()
    py, D = sys.executable, str(HERE)
    steps, ok = [], True

    def step(name, cmd):
        nonlocal ok
        t0 = time.monotonic()
        with open(logs / f"{name}.stdout.log", "w") as fo, open(logs / f"{name}.stderr.log", "w") as fe:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fo, stderr=fe).returncode
        steps.append({"step": name, "command": shlex.join(cmd), "exit_code": rc, "seconds": round(time.monotonic() - t0, 2)})
        print(f"[{name}] exit {rc}", flush=True)
        ok &= rc == 0
        return rc

    rel = lambda q: str(Path(q).resolve().relative_to(ROOT)) if ROOT in Path(q).resolve().parents else str(q)
    ev_cmd = [rel(a.binary), a.dataset, "--frames", str(a.frames), "--stride", str(a.stride), "--max-dt", "0.02",
              "--max-gt-gap", "0.05", "--out-dir", rel(ev), "--export-pairs", a.export_pairs]  # run from ROOT
    if step("evaluator", ev_cmd) != 0:
        (out / "manifest.json").write_text(json.dumps({"steps": steps, "ok": False}, indent=2) + "\n")
        return 1
    cfg = {"command": shlex.join(ev_cmd), "dataset": a.dataset, "frames": a.frames, "stride": a.stride,
           "export_pairs": a.export_pairs, "max_dt": 0.02, "max_gt_gap": 0.05}
    if a.baseline:
        same = {f: filecmp.cmp(ev / f, Path(a.baseline) / f, shallow=False) for f in STANDARD}
        cfg["baseline_compare"] = {"baseline": a.baseline, "byte_identical": same}
        print(f"[baseline_compare] {same}", flush=True)
        ok &= all(same.values())
    (ev / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    sys.path.insert(0, D)
    import common as C
    import pair_report as PR
    ref = a.reference if a.reference is not None else PR.choose_reference(C.pairs_rows(ev), a.failure)
    lo, _, hi = a.export_pairs.partition("-")
    ctx = f"{lo}-{hi or lo}" if "," not in a.export_pairs else f"{a.failure}-{a.failure}"
    step("pair_page_failure", [py, f"{D}/pair_page.py", "--run", str(ev), "--pair", str(a.failure), "--dataset",
                               a.dataset, "--compare", str(ref), "--context", ctx])
    step("pair_page_reference", [py, f"{D}/pair_page.py", "--run", str(ev), "--pair", str(ref), "--dataset", a.dataset])
    for w in [x for x in a.windows.split(",") if x]:
        step(f"tracking_{w}", [py, f"{D}/feature_tracks.py", "--run", str(ev), "--frames", w, "--dataset", a.dataset])
    sel = sorted({0, a.frames - 1 if a.frames else 0, a.failure, a.failure + 1, max(0, a.failure - 3)})
    step("scene", [py, f"{D}/scene_3d.py", "--run", str(ev), "--dataset", a.dataset, "--frames",
                   f"{a.failure},{a.failure + 1}"])
    step("divergence", [py, f"{D}/trajectory_divergence.py", "--run", str(ev), "--dataset", a.dataset, "--select",
                        ",".join(map(str, sel))])
    step("viewer", [py, f"{D}/build_viewer.py", "--run", str(ev), "--dataset", a.dataset, "--pair", str(a.failure),
                    "--frames", f"{a.failure},{a.failure + 1}"])
    (out / "manifest.json").write_text(json.dumps({"dataset": a.dataset, "output": str(out), "reference_pair": ref,
                                                   "steps": steps, "ok": ok}, indent=2) + "\n")
    write_index(diag_root)
    print(f"viewer: {out / 'viewer' / 'index.html'}\nindex:  {diag_root / 'index.html'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
