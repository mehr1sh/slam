"""Offline analysis of the integration experiments (scratch_experiments outputs; branch integration/teammate).

Ground truth enters only through the eval_ columns (evaluation). Read-only.

Usage: python3 tools/pipeline_analysis/integration_analysis.py OUT_DIR fixtures=DIR[,DIR...] [stride4=DIR[,DIR...]]
       [bunny=DIR[,DIR...]] [full=DIR[,DIR...]]
Several directories per input are concatenated (e.g. a later run of an extra configuration).
Writes OUT_DIR/{fixture_pairs.csv, sequences.csv, attribution.csv} and prints the tables (markdown).
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

BAD_DEG, BAD_M, GROSS_DEG = 2.0, 0.05, 5.0


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def load_all(dirs, name):
    rows = []
    for d in dirs.split(","):
        rows += load(Path(d) / name)
    return rows


def st(v):
    v = np.array([x for x in v if x == x], float)
    if not len(v):
        return dict(n=0, median=float("nan"), p95=float("nan"), max=float("nan"), std=float("nan"), mean=float("nan"))
    return dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()),
                std=float(v.std()), mean=float(v.mean()))


def med(rows, k):
    return st([f(r[k]) for r in rows])["median"]


def bad(r):
    return f(r["eval_rot_err_deg"]) > BAD_DEG or f(r["eval_trans_err_m"]) > BAD_M


def order(cfgs):
    pref = ["M0", "M3", "L0", "T0", "T0G", "T0R", "T0GR", "T0M", "T0Q", "T0E", "T0H", "T0-harris", "T0-budget",
            "T0-fast20", "T0-mr", "T0-reselect", "T0-ransac", "T0C", "T0CG"]
    return sorted(cfgs, key=lambda c: pref.index(c) if c in pref else 99)


def pair_table(runs, ms):
    out = []
    keys = sorted({(r["config"], r["label"]) for r in runs})
    for cfg in order({k[0] for k in keys}):
        for label in sorted({k[1] for k in keys if k[0] == cfg}):
            rs = [r for r in runs if r["config"] == cfg and r["label"] == label]
            m = [x for x in ms if x["config"] == cfg and x["label"] == label]
            acc = [r for r in rs if r["accepted"] == "1"]
            rot, tr = st([f(r["eval_rot_err_deg"]) for r in acc]), st([f(r["eval_trans_err_m"]) for r in acc])
            out.append(dict(
                config=cfg, pair=label, seeds=len(rs), accepted=len(acc),
                rejected=sum(r["gate_status"] == "rejected" and r["accepted"] == "0" for r in rs),
                solver_failed=sum(r["solver_ok"] == "0" for r in rs),
                bad=sum(bad(r) for r in acc), gross=sum(f(r["eval_rot_err_deg"]) > GROSS_DEG for r in acc),
                rot_median=rot["median"], rot_p95=rot["p95"], rot_max=rot["max"], rot_std=rot["std"],
                trans_median=tr["median"], trans_max=tr["max"],
                keypoints=f(m[0]["keypoints_i"]) if m else float("nan"),
                raw=f(m[0]["raw"]) if m else float("nan"), filtered=f(m[0]["filtered"]) if m else float("nan"),
                uniq_dst_loc=f(m[0]["uniq_dst_loc"]) if m else float("nan"),
                many_to_one=f(m[0]["many_to_one"]) if m else float("nan"),
                pnp_inputs=f(m[0]["pnp_inputs"]) if m else float("nan"),
                correct_share=f(m[0]["eval_correct_input_share"]) if m else float("nan"),
                inliers=med(acc, "inlier_rows") if acc else float("nan"),
                loc_dst=med(acc, "loc_dst") if acc else float("nan"),
                cells_dst=med(acc, "cells_dst") if acc else float("nan"),
                reproj_median=med(acc, "reproj_median_px") if acc else float("nan"),
                reproj_p90=med(acc, "reproj_p90_px") if acc else float("nan"),
                seconds=med(rs, "seconds")))
    return out


def seq_table(name, runs, ms, ate):
    out = []
    for cfg in order({r["config"] for r in runs}):
        rs = [r for r in runs if r["config"] == cfg]
        m = [x for x in ms if x["config"] == cfg]
        a = [x for x in ate if x["config"] == cfg]
        acc = [r for r in rs if r["accepted"] == "1"]
        rot = st([f(r["eval_rot_err_deg"]) for r in acc])
        av = st([f(x["eval_ate_raw_rmse_m"]) for x in a])
        out.append(dict(
            sequence=name, config=cfg, pairs=len({r["pair"] for r in rs}), seeds=len(a), runs=len(rs),
            accepted=len(acc), rejected=sum(r["gate_status"] == "rejected" and r["accepted"] == "0" for r in rs),
            solver_failed=sum(r["solver_ok"] == "0" for r in rs),
            bad=sum(bad(r) for r in acc), gross=sum(f(r["eval_rot_err_deg"]) > GROSS_DEG for r in acc),
            rot_median=rot["median"], rot_p95=rot["p95"], rot_max=rot["max"],
            ate_median=av["median"], ate_min=min([f(x["eval_ate_raw_rmse_m"]) for x in a], default=float("nan")),
            ate_max=av["max"],
            rpe_t_median=med(a, "eval_rpe_trans_rmse_m"), rpe_r_median=med(a, "eval_rpe_rot_rmse_deg"),
            final_pos_median=med(a, "eval_final_pos_err_m"), final_rot_median=med(a, "eval_final_rot_err_deg"),
            keypoints=np.mean([f(x["keypoints_i"]) for x in m]), filtered=np.mean([f(x["filtered"]) for x in m]),
            uniq_dst_loc=np.mean([f(x["uniq_dst_loc"]) for x in m]), pnp_inputs=np.mean([f(x["pnp_inputs"]) for x in m]),
            correct_share=np.mean([f(x["eval_correct_input_share"]) for x in m]),
            inliers=med(acc, "inlier_rows"), loc_dst=med(acc, "loc_dst"), cells_dst=med(acc, "cells_dst"),
            reproj_median=med(acc, "reproj_median_px"), reproj_p90=med(acc, "reproj_p90_px"),
            seconds=med(rs, "seconds")))
    return out


def write(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def md(rows, cols, fmt):
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if isinstance(r[c], (int, float)) and r[c] == r[c]
                                 else ("-" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")


def main(argv):
    out = Path(argv[1])
    args = dict(a.split("=", 1) for a in argv[2:])
    out.mkdir(parents=True, exist_ok=False)
    fr, fm = load_all(args["fixtures"], "runs.csv"), load_all(args["fixtures"], "matchsets.csv")
    assert all(r["trace_ok"] == "1" for r in fr), "a RANSAC trace did not reproduce production"
    pt = pair_table(fr, fm)
    write(out / "fixture_pairs.csv", pt)
    fmt = {k: "{:.3f}" for k in ["rot_median", "rot_p95", "rot_max", "rot_std", "correct_share", "reproj_median",
                                  "reproj_p90", "seconds"]}
    fmt.update({k: "{:.4f}" for k in ["trans_median", "trans_max"]})
    fmt.update({k: "{:.0f}" for k in ["keypoints", "filtered", "uniq_dst_loc", "pnp_inputs", "inliers", "loc_dst",
                                      "cells_dst", "many_to_one", "raw"]})
    for label in sorted({r["pair"] for r in pt}):
        print(f"\n### {label} ({pt[0]['seeds']} seeds)")
        md([r for r in pt if r["pair"] == label],
           ["config", "accepted", "rejected", "solver_failed", "bad", "gross", "rot_median", "rot_p95", "rot_max",
            "trans_median", "keypoints", "filtered", "uniq_dst_loc", "pnp_inputs", "correct_share", "inliers",
            "loc_dst", "cells_dst", "reproj_median", "reproj_p90", "seconds"], fmt)
    seqs = []
    for name in ["bunny", "stride4", "full"]:
        if name in args:
            d = args[name]
            r, m, a = load_all(d, "runs.csv"), load_all(d, "matchsets.csv"), load_all(d, "ate.csv")
            assert all(x["trace_ok"] == "1" for x in r), f"{name}: a RANSAC trace did not reproduce production"
            seqs += seq_table(name, r, m, a)
    write(out / "sequences.csv", seqs)
    fmt.update({k: "{:.4f}" for k in ["ate_median", "ate_min", "ate_max", "rpe_t_median", "final_pos_median"]})
    fmt.update({k: "{:.3f}" for k in ["rpe_r_median", "final_rot_median"]})
    for name in ["bunny", "stride4", "full"]:
        rows = [r for r in seqs if r["sequence"] == name]
        if not rows:
            continue
        print(f"\n### sequence {name}: {rows[0]['pairs']} pairs x {rows[0]['seeds']} seeds")
        md(rows, ["config", "accepted", "rejected", "solver_failed", "bad", "gross", "rot_median", "rot_p95",
                  "rot_max", "ate_median", "ate_min", "ate_max", "rpe_t_median", "rpe_r_median", "final_pos_median",
                  "final_rot_median", "keypoints", "filtered", "uniq_dst_loc", "pnp_inputs", "correct_share",
                  "inliers", "loc_dst", "cells_dst", "reproj_median", "reproj_p90", "seconds"], fmt)
    # attribution: leave-one-out from T0 (difference in median ATE / per-pair rotation error)
    att = []
    for name in ["bunny", "stride4", "full"]:
        rows = {r["config"]: r for r in seqs if r["sequence"] == name}
        if "T0" not in rows:
            continue
        base = rows["T0"]
        for cfg, what in [("T0-harris", "Harris ranking"), ("T0-budget", "1400-feature budget"),
                          ("T0-fast20", "FAST threshold 10"), ("T0-mr", "mutual + ratio 0.87 matcher"),
                          ("T0E", "12 mm depth-edge rejection"), ("T0H", "Huber delta 2 px"),
                          ("T0-reselect", "inlier re-selection"), ("T0-ransac", "RANSAC 650 it / 3 px")]:
            if cfg in rows:
                r = rows[cfg]
                att.append(dict(sequence=name, removed=what, config=cfg,
                                ate_without=r["ate_median"], ate_with=base["ate_median"],
                                ate_change_when_removed=r["ate_median"] - base["ate_median"],
                                rot_median_without=r["rot_median"], rot_median_with=base["rot_median"],
                                bad_without=r["bad"], bad_with=base["bad"]))
    write(out / "attribution.csv", att)
    if att:
        print("\n### Leave-one-out from the teammate defaults (T0): what removing each change does")
        md(att, ["sequence", "removed", "ate_with", "ate_without", "ate_change_when_removed", "rot_median_with",
                 "rot_median_without", "bad_with", "bad_without"],
           {"ate_with": "{:.4f}", "ate_without": "{:.4f}", "ate_change_when_removed": "{:+.4f}",
            "rot_median_with": "{:.3f}", "rot_median_without": "{:.3f}"})


if __name__ == "__main__":
    main(sys.argv)
