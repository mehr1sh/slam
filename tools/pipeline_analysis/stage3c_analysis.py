"""Offline analysis of the Stage-3C experiments (scratch_experiments outputs). Read-only.

Ground truth enters ONLY as evaluation labels (columns prefixed eval_ in runs.csv); every candidate gate
metric evaluated here is computed by scratch_experiments from correspondences and poses alone.

Usage: python3 tools/pipeline_analysis/stage3c_analysis.py FIXTURE_DIR SEQUENCE_DIR OUT_DIR
Writes OUT_DIR/{config_fixtures.csv, config_sequence.csv, gate_73_74.csv, candidate_gate.csv,
metric_correlation.csv, mutual_comparison.csv} and prints the tables.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

CONFIGS = ["M0", "M1", "M2", "M2n", "M3", "M4", "M5", "M1L", "M3L"]
BAD_DEG, BAD_M = 2.0, 0.05         # evaluation label: an accepted pose worse than this is a false accept
VERY_BAD_DEG = 5.0


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    return list(csv.DictReader(open(path)))


def stat(v):
    v = np.array([x for x in v if x == x], float)
    if len(v) == 0:
        return dict(n=0, median=float("nan"), p95=float("nan"), max=float("nan"), std=float("nan"))
    return dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()),
                std=float(v.std()))


def is_bad(r, gross=False):
    if gross:
        return f(r["eval_rot_err_deg"]) > VERY_BAD_DEG
    return f(r["eval_rot_err_deg"]) > BAD_DEG or f(r["eval_trans_err_m"]) > BAD_M


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def config_table(runs, matchsets, by_pair):
    out = []
    keys = sorted({(r["config"], r["label"] if by_pair else "all") for r in runs},
                  key=lambda k: (CONFIGS.index(k[0]), k[1]))
    for cfg, label in keys:
        rs = [r for r in runs if r["config"] == cfg and (not by_pair or r["label"] == label)]
        ms = [m for m in matchsets if m["config"] == cfg and (not by_pair or m["label"] == label)]
        acc = [r for r in rs if r["accepted"] == "1"]
        rot = stat([f(r["eval_rot_err_deg"]) for r in acc])
        tr = stat([f(r["eval_trans_err_m"]) for r in acc])
        out.append(dict(
            config=cfg, pair=label, runs=len(rs), accepted=len(acc),
            rejected=sum(r["gate_status"] == "rejected" for r in rs),
            solver_failed=sum(r["gate_status"] == "solver_failed" for r in rs),
            accepted_bad=sum(is_bad(r) for r in acc),
            accepted_over5deg=sum(f(r["eval_rot_err_deg"]) > VERY_BAD_DEG for r in acc),
            rot_median=rot["median"], rot_p95=rot["p95"], rot_max=rot["max"], rot_std=rot["std"],
            trans_median=tr["median"], trans_max=tr["max"],
            seconds_median=stat([f(r["seconds"]) for r in rs])["median"],
            keypoints_i=np.mean([f(m["keypoints_i"]) for m in ms]), raw=np.mean([f(m["raw"]) for m in ms]),
            filtered=np.mean([f(m["filtered"]) for m in ms]),
            uniq_src_kp=np.mean([f(m["uniq_src_kp"]) for m in ms]), uniq_dst_kp=np.mean([f(m["uniq_dst_kp"]) for m in ms]),
            uniq_src_loc=np.mean([f(m["uniq_src_loc"]) for m in ms]), uniq_dst_loc=np.mean([f(m["uniq_dst_loc"]) for m in ms]),
            many_to_one=np.mean([f(m["many_to_one"]) for m in ms]), pnp_inputs=np.mean([f(m["pnp_inputs"]) for m in ms]),
            eval_correct_input_share=np.mean([f(m["eval_correct_input_share"]) for m in ms]),
            inlier_rows=np.median([f(r["inlier_rows"]) for r in acc]) if acc else float("nan"),
            loc_dst=np.median([f(r["loc_dst"]) for r in acc]) if acc else float("nan"),
            cells_dst=np.median([f(r["cells_dst"]) for r in acc]) if acc else float("nan"),
            reproj_median=np.median([f(r["reproj_median_px"]) for r in acc]) if acc else float("nan"),
            reproj_p90=np.median([f(r["reproj_p90_px"]) for r in acc]) if acc else float("nan")))
    return out


def candidate_table(rows, label, gross=False):
    """False accepts / false rejects of candidate thresholds on the poses the Stage-2 gate ACCEPTED.
    Metrics ending in '<=' reject when the value is ABOVE the threshold."""
    acc = [r for r in rows if r["accepted"] == "1"]
    bad = [is_bad(r, gross) for r in acc]
    nb, ng = sum(bad), len(bad) - sum(bad)
    grid = {
        "inlier_ratio": [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        "ratio_pairs": [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        "ratio_loc_dst": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        "ratio_cells_dst": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
        "inlier_rows": [20, 50, 100, 150, 200],
        "loc_dst": [20, 40, 60, 80, 100],
        "largest_cell_share<=": [0.3, 0.25, 0.2, 0.15, 0.12],
        "condition<=": [100, 50, 20, 10, 8],
    }
    out = []
    for metric, ths in grid.items():
        upper = metric.endswith("<=")
        col = metric.rstrip("<=")
        for th in ths:
            keep = lambda r: f(r[col]) <= th if upper else f(r[col]) >= th
            fa = sum(1 for r, b in zip(acc, bad) if b and keep(r))
            fr = sum(1 for r, b in zip(acc, bad) if not b and not keep(r))
            out.append(dict(data=label, label="gross>5deg" if gross else "bad>2deg|5cm", metric=metric, threshold=th,
                            gate_accepted=len(acc), bad=nb, good=ng, false_accept=fa, false_reject=fr))
    return out


def main(argv):
    fx_dir, seq_dir, out_dir = map(Path, argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=False)
    fr, fm = load(fx_dir / "runs.csv"), load(fx_dir / "matchsets.csv")
    sr, sm, ate = load(seq_dir / "runs.csv"), load(seq_dir / "matchsets.csv"), load(seq_dir / "ate.csv")
    assert all(r["trace_ok"] == "1" for r in fr + sr), "a RANSAC trace did not reproduce production"

    # ---- 1. per configuration ----
    cf = config_table(fr, fm, by_pair=True)
    write(out_dir / "config_fixtures.csv", cf)
    print("\n### Fixtures, 20 seeds: accepted poses")
    print(f"{'cfg':4} {'pair':22} acc rej fail bad >5deg | rot med   p95    max    std | inputs corr | s")
    for c in cf:
        print(f"{c['config']:4} {c['pair'][:22]:22} {c['accepted']:3} {c['rejected']:3} {c['solver_failed']:4} "
              f"{c['accepted_bad']:3} {c['accepted_over5deg']:5} | {c['rot_median']:6.3f} {c['rot_p95']:6.3f} "
              f"{c['rot_max']:6.3f} {c['rot_std']:6.3f} | {c['pnp_inputs']:6.0f} {c['eval_correct_input_share']:.3f} | "
              f"{c['seconds_median']:.3f}")
    cs = config_table(sr, sm, by_pair=False)
    for c in cs:
        a = [x for x in ate if x["config"] == c["config"]]
        v = [f(x["eval_ate_raw_rmse_m"]) for x in a]
        c.update(ate_median=float(np.median(v)), ate_min=min(v), ate_max=max(v),
                 final_pos_median=float(np.median([f(x["eval_final_pos_err_m"]) for x in a])),
                 final_rot_median=float(np.median([f(x["eval_final_rot_err_deg"]) for x in a])))
    write(out_dir / "config_sequence.csv", cs)
    print("\n### TUM stride 4: 99 pairs x 5 seeds")
    print(f"{'cfg':4} acc rej fail bad >5deg | rot med  p95   max | ATE med   min   max | final pos rot | inputs corr m2o | s")
    for c in cs:
        print(f"{c['config']:4} {c['accepted']:3} {c['rejected']:3} {c['solver_failed']:4} {c['accepted_bad']:3} "
              f"{c['accepted_over5deg']:5} | {c['rot_median']:.3f} {c['rot_p95']:.3f} {c['rot_max']:6.2f} | "
              f"{c['ate_median']:.4f} {c['ate_min']:.4f} {c['ate_max']:.4f} | {c['final_pos_median']:.4f} "
              f"{c['final_rot_median']:.2f} | {c['pnp_inputs']:6.0f} {c['eval_correct_input_share']:.3f} "
              f"{c['many_to_one']:6.0f} | {c['seconds_median']:.3f}")

    # ---- 2. 73 -> 74, M0, 20 seeds ----
    cols = ["seed", "gate_status", "accepted", "inlier_rows", "distinct_src", "distinct_dst", "loc_src", "loc_dst",
            "inlier_ratio", "ratio_pairs", "ratio_loc_dst", "cells_src", "cells_dst", "ratio_cells_dst",
            "largest_cell_share", "rows_per_location", "reproj_median_px", "reproj_p90_px", "eig1", "eig2", "eig3",
            "condition", "planarity", "depth_min", "depth_max", "sample_inliers", "refit_inliers",
            "c_rows", "c_kp_src", "c_kp_dst", "c_loc_src", "c_loc_dst", "c_cells_src", "c_cells_dst", "c_median",
            "c_p90", "c_so3", "eval_sample_rot_err_deg", "eval_refit_rot_err_deg", "eval_rot_err_deg",
            "eval_trans_err_m", "eval_correct_inlier_share", "reason"]
    g = sorted([r for r in fr if r["config"] == "M0" and r["label"] == "tum_xyz_stride4_73_74"], key=lambda r: f(r["eval_rot_err_deg"]))
    write(out_dir / "gate_73_74.csv", [{k: r[k] for k in cols} for r in g])
    print("\n### TUM 73->74, M0, 20 seeds (sorted by rotation error; eval_ = evaluation only)")
    print("seed  gate      rows  locD cells ratio_r ratio_p maxcell rows/loc  med   p90  cond  sampleErr refitErr finalErr corrInl")
    for r in g:
        print(f"{r['seed']:>5} {r['gate_status']:9} {r['inlier_rows']:>5} {r['loc_dst']:>5} {r['cells_dst']:>5} "
              f"{f(r['inlier_ratio']):7.3f} {f(r['ratio_pairs']):7.3f} {f(r['largest_cell_share']):7.3f} "
              f"{f(r['rows_per_location']):8.2f} {f(r['reproj_median_px']):5.2f} {f(r['reproj_p90_px']):5.2f} "
              f"{f(r['condition']):5.1f} {f(r['eval_sample_rot_err_deg']):9.2f} {f(r['eval_refit_rot_err_deg']):8.2f} "
              f"{f(r['eval_rot_err_deg']):8.2f} {f(r['eval_correct_inlier_share']):7.3f}")

    # ---- 3. candidate relative-support thresholds ----
    cand = []
    for label, rows in [("stride4_all_configs", sr), ("stride4_M0", [r for r in sr if r["config"] == "M0"]),
                        ("stride4_excluding_pair73", [r for r in sr if r["frame_i"] != "73"]),
                        ("fixtures_all_configs", fr), ("fixtures_M0", [r for r in fr if r["config"] == "M0"])]:
        cand += candidate_table(rows, label) + candidate_table(rows, label, gross=True)
    write(out_dir / "candidate_gate.csv", cand)
    print(f"\n### Candidate thresholds on Stage-2-ACCEPTED poses (bad = rot > {BAD_DEG} deg or trans > {BAD_M} m)")
    for label in ["stride4_all_configs", "stride4_M0", "stride4_excluding_pair73", "fixtures_all_configs", "fixtures_M0"]:
      for lab in ["bad>2deg|5cm", "gross>5deg"]:
        rows = [c for c in cand if c["data"] == label and c["label"] == lab]
        print(f"-- {label} [{lab}]: {rows[0]['gate_accepted']} accepted, {rows[0]['bad']} bad, {rows[0]['good']} good")
        for c in rows:
            print(f"   {c['metric']:16} >= {c['threshold']:<5}  false accepts {c['false_accept']:3}   false rejects {c['false_reject']:4}")

    # ---- 4. independence of the gate metrics (accepted M0 + M3 runs, stride 4) ----
    met = ["inlier_rows", "distinct_dst", "loc_dst", "cells_dst", "inlier_ratio", "ratio_pairs", "reproj_median_px",
           "reproj_p90_px", "condition", "largest_cell_share"]
    acc = [r for r in sr if r["accepted"] == "1" and r["config"] == "M0"]
    M = np.array([[f(r[m]) for m in met] for r in acc])
    ranks = np.argsort(np.argsort(M, axis=0), axis=0).astype(float)
    C = np.corrcoef(ranks.T)
    write(out_dir / "metric_correlation.csv", [dict(metric=a, **{b: round(C[i, j], 3) for j, b in enumerate(met)})
                                                for i, a in enumerate(met)])
    print(f"\n### Spearman correlation of gate metrics (stride 4, M0, {len(acc)} accepted poses)")
    print(" " * 18 + " ".join(f"{m[:7]:>7}" for m in met))
    for i, a in enumerate(met):
        print(f"{a:18}" + " ".join(f"{C[i, j]:7.2f}" for j in range(len(met))))

    # ---- 5. strict vs location-aware mutual ----
    mc = []
    for label, ms in [("fixtures", fm), ("stride4", sm)]:
        rows = [m for m in ms if m["config"] == "M1"]
        tot = lambda k: sum(int(m[k]) for m in rows)
        mc.append(dict(data=label, pairs=len(rows), passed_distance=tot("passed_distance"), strict=tot("strict_mutual"),
                       location=tot("location_mutual"), rescued=tot("mutual_rescued"),
                       strict_with_depth=tot("eval_strict_with_depth"), strict_correct=tot("eval_strict_correct"),
                       rescued_with_depth=tot("eval_rescued_with_depth"), rescued_correct=tot("eval_rescued_correct")))
    write(out_dir / "mutual_comparison.csv", mc)
    print("\n### Strict vs location-aware mutual (M1 match sets; correctness = evaluation only)")
    for m in mc:
        print(f"{m['data']}: {m['pairs']} pairs; strict mutual {m['strict']}, location-aware {m['location']} "
              f"(+{m['rescued']} rescued); correct share with depth: strict "
              f"{m['strict_correct'] / max(1, m['strict_with_depth']):.3f}, rescued "
              f"{m['rescued_correct'] / max(1, m['rescued_with_depth']):.3f}")


if __name__ == "__main__":
    main(sys.argv)
