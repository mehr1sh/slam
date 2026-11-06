#!/usr/bin/env python3
"""Per-pair failure timeline of a tum_trajectory_eval run (pairs.csv + trajectory.csv; no export needed).

Usage (repository root):
  python3 tools/evaluation/diagnostics/failure_timeline.py --run results/runs/diagnostics/xyz_n100_stride4 \
          [--output DIR] [--mark 73]
Default output: <run>/figures/failure_timeline/: timeline.{png,pdf}, timeline.csv, timeline.json, index.html
(index.html lists the flagged pairs and links each to its detailed figures when they exist).

Fields per pair (all from the run's files):
  solver status   pnp_solver_ok / icp_solver_ok: the evaluator's own definition (a pose was returned);
                  unchanged here.
  warnings        diagnostic, computed WITHOUT ground truth, never change the status:
                    low_pnp_inliers        PnP inliers < 20
                    low_pnp_inlier_ratio   PnP inliers / correspondences < 0.25
                    low_icp_inliers        ICP inliers < 10
                    pnp_icp_disagree       relative poses differ by > 5 deg or > 0.05 m
  gt flags        evaluation only (need ground truth): pnp_gt_error / icp_gt_error, > 5 deg or > 0.05 m.
PnP-ICP disagreement uses the relative poses T_j<-i recovered from consecutive trajectory.csv poses,
T_j<-i = T_wc[j]^-1 T_wc[i] (exact for solved pairs; a failed pair has no estimate and gets null).
"""

import argparse
import csv
import html
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TH = {"low_pnp_inliers": 20, "low_pnp_inlier_ratio": 0.25, "low_icp_inliers": 10, "disagree_deg": 5.0,
      "disagree_m": 0.05, "gt_deg": 5.0, "gt_m": 0.05}
WARNINGS = ("low_pnp_inliers", "low_pnp_inlier_ratio", "low_icp_inliers", "pnp_icp_disagree")
GT_FLAGS = ("pnp_gt_error", "icp_gt_error")


def relative(run, m, k):
    """T_j<-i of method m for pair k from the accumulated poses (None for a failed pair)."""
    if not run[f"{m}_ok"][k]:
        return None
    Ti = C.se3(run[f"{m}_R"][k], run[f"{m}_p"][k])
    Tj = C.se3(run[f"{m}_R"][k + 1], run[f"{m}_p"][k + 1])
    return np.linalg.inv(Tj) @ Ti


def timeline(run_dir):
    run = C.tm.load_run(run_dir)
    rows = C.pairs_rows(run_dir)
    out = []
    for k in range(run["n"] - 1):
        r = rows[k]
        f = lambda key: float(r[key])
        d = {"pair": k, "frame_i": k, "frame_j": k + 1, "t_i": float(run["t"][k]),
             "matches": int(f("matches")), "pnp_correspondences": int(f("pnp_correspondences")),
             "pnp_inliers": int(f("pnp_inliers")), "icp_correspondences": int(f("icp_correspondences")),
             "icp_inliers": int(f("icp_inliers")),
             "pnp_solver_ok": bool(run["pnp_solver_returned"][k]) if "pnp_solver_returned" in run else bool(run["pnp_ok"][k]),
             "pnp_accepted": bool(run["pnp_ok"][k]), "icp_solver_ok": bool(run["icp_ok"][k]),
             "pnp_rejected": bool("pnp_solver_returned" in run and run["pnp_solver_returned"][k] and not run["pnp_ok"][k]),
             "pnp_returned_rot_err_deg": (float(run["pnp_returned_rot_err_deg"][k]) if "pnp_returned_rot_err_deg" in run
                                          else f("pnp_rot_err_deg")),
             "pnp_returned_trans_err_m": (float(run["pnp_returned_trans_err_m"][k]) if "pnp_returned_trans_err_m" in run
                                          else f("pnp_trans_err_m")),
             "pnp_rot_err_deg": f("pnp_rot_err_deg"), "pnp_trans_err_m": f("pnp_trans_err_m"),
             "icp_rot_err_deg": f("icp_rot_err_deg"), "icp_trans_err_m": f("icp_trans_err_m"),
             "gt_motion_deg": f("gt_motion_deg"), "gt_motion_m": f("gt_motion_m")}
        d["pnp_inlier_ratio"] = d["pnp_inliers"] / d["pnp_correspondences"] if d["pnp_correspondences"] else None
        d["icp_inlier_ratio"] = d["icp_inliers"] / d["icp_correspondences"] if d["icp_correspondences"] else None
        Tp, Ti = relative(run, "pnp", k), relative(run, "icp", k)
        if Tp is not None and Ti is not None:
            d["pnp_icp_rot_deg"] = C.rot_deg(Tp[:3, :3].T @ Ti[:3, :3])
            d["pnp_icp_trans_m"] = float(np.linalg.norm(Tp[:3, 3] - Ti[:3, 3]))
        else:
            d["pnp_icp_rot_deg"] = d["pnp_icp_trans_m"] = None
        w = []
        if d["pnp_inliers"] < TH["low_pnp_inliers"]:
            w.append("low_pnp_inliers")
        if d["pnp_inlier_ratio"] is not None and d["pnp_inlier_ratio"] < TH["low_pnp_inlier_ratio"]:
            w.append("low_pnp_inlier_ratio")
        if d["icp_inliers"] < TH["low_icp_inliers"]:
            w.append("low_icp_inliers")
        if d["pnp_icp_rot_deg"] is not None and (d["pnp_icp_rot_deg"] > TH["disagree_deg"] or
                                                 d["pnp_icp_trans_m"] > TH["disagree_m"]):
            w.append("pnp_icp_disagree")
        g = [f"{m}_gt_error" for m in ("pnp", "icp") if np.isfinite(d[f"{m}_rot_err_deg"]) and
             (d[f"{m}_rot_err_deg"] > TH["gt_deg"] or d[f"{m}_trans_err_m"] > TH["gt_m"])]
        d["warnings"], d["gt_flags"] = w, g
        out.append(d)
    return run, out


def figure(run_dir, tl, out, mark, exported, made):
    k = np.array([d["pair"] for d in tl])
    g = lambda key: np.array([np.nan if d[key] is None else d[key] for d in tl], float)
    fig, axs = plt.subplots(5, 1, figsize=(15, 13), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1.2, 1.2, 0.9], hspace=0.08))
    ax = axs[0]
    ax.plot(k, g("matches"), color="0.35", lw=1, label="filtered matches")
    ax.plot(k, g("pnp_correspondences"), color=C.METHOD_COLOR["pnp"], lw=1, ls="--", label="PnP correspondences (depth in i)")
    ax.plot(k, g("icp_correspondences"), color=C.METHOD_COLOR["icp"], lw=1, ls="--", label="ICP correspondences (depth in i and j)")
    ax.set_ylabel("count")
    ax = axs[1]
    ax.plot(k, g("pnp_inliers"), color=C.METHOD_COLOR["pnp"], lw=1.2, marker=".", label="PnP inliers")
    ax.plot(k, g("icp_inliers"), color=C.METHOD_COLOR["icp"], lw=1.2, marker=".", label="ICP inliers")
    ax.axhline(TH["low_pnp_inliers"], color=C.METHOD_COLOR["pnp"], ls=":", lw=1,
               label=f"warning: PnP inliers < {TH['low_pnp_inliers']}")
    ax.axhline(TH["low_icp_inliers"], color=C.METHOD_COLOR["icp"], ls=":", lw=1,
               label=f"warning: ICP inliers < {TH['low_icp_inliers']}")
    ax.set_yscale("log")
    ax.set_ylabel("inliers")
    ax2 = ax.twinx()
    ax2.plot(k, g("pnp_inlier_ratio"), color=C.METHOD_COLOR["pnp"], lw=0.8, alpha=0.5, ls="-.")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("PnP inlier ratio (dash-dot)", fontsize=8)
    for ax, unit, keys, th in ((axs[2], "deg", ("pnp_rot_err_deg", "icp_rot_err_deg", "pnp_icp_rot_deg", "gt_motion_deg"),
                                TH["gt_deg"]),
                               (axs[3], "m", ("pnp_trans_err_m", "icp_trans_err_m", "pnp_icp_trans_m", "gt_motion_m"),
                                TH["gt_m"])):
        ax.plot(k, g(keys[0]), color=C.METHOD_COLOR["pnp"], lw=1.2, label="PnP error vs ground truth")
        ax.plot(k, g(keys[1]), color=C.METHOD_COLOR["icp"], lw=1.2, label="ICP error vs ground truth")
        ax.plot(k, g(keys[2]), color="black", lw=1, ls="--", label="PnP vs ICP disagreement (no GT)")
        ax.plot(k, g(keys[3]), color=C.METHOD_COLOR["gt"], lw=0.8, alpha=0.7, label="true motion of the pair")
        ax.axhline(th, color="0.3", ls=":", lw=1, label=f"flag threshold {th:g} {unit}")
        ax.set_yscale("log")
        ax.set_ylabel(f"relative pose ({unit})")
    # status strip
    ax = axs[4]
    rows = [("PnP solver returned", [not d["pnp_solver_ok"] for d in tl], 2),
            ("ICP solver returned", [not d["icp_solver_ok"] for d in tl], 2)] + \
           [(w.replace("_", " "), [w in d["warnings"] for d in tl], 1) for w in WARNINGS] + \
           [(f.replace("_", " ") + " (GT)", [f in d["gt_flags"] for d in tl], 3) for f in GT_FLAGS]
    M = np.array([[lvl if v else 0 for v in vals] for _, vals, lvl in rows])
    ax.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#f2f2f2", "#e69f00", "#000000", "#CC3311"]),
              vmin=0, vmax=3, extent=(k[0] - 0.5, k[-1] + 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("pair index i (frame i -> i+1 of the run)")
    for a in axs[:4]:
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5, loc="upper left", ncol=3, framealpha=0.85)
    for a in axs:
        for e in exported:
            a.axvline(e, color="#0072B2", lw=0.6, alpha=0.25)
        if mark is not None:
            a.axvline(mark, color="#CC3311", lw=1.2, alpha=0.7)
    if mark is not None:
        d = tl[mark]
        axs[0].annotate(f"pair {mark} -> {mark + 1}: PnP {d['pnp_inliers']} inliers, error {d['pnp_rot_err_deg']:.1f} deg "
                        f"/ {d['pnp_trans_err_m']:.2f} m; solver status: returned", (mark, axs[0].get_ylim()[0]),
                        xytext=(-8, 8), textcoords="offset points", color="#CC3311", fontsize=9, ha="right",
                        bbox=dict(fc="white", ec="none", alpha=0.85))
    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.suptitle(f"Failure timeline of {Path(run_dir).name} ({len(tl)} pairs). Thin blue lines: pairs with a detailed "
                 "export. Status rows: black = solver failed (no pose), orange = diagnostic warning (no ground truth),\n"
                 "red = ground-truth error flag (evaluation only); none of them changes the evaluator's success "
                 "definition", fontsize=10.5, y=0.995)
    C.save(fig, out, "timeline", made)


def write_index(run_dir, tl, out):
    figs = Path(run_dir) / "figures"
    flagged = [d for d in tl if d["warnings"] or d["gt_flags"] or not d["pnp_solver_ok"] or not d["icp_solver_ok"]]
    rows = []
    for d in flagged:
        pd = figs / f"pair_{d['pair']}_{d['pair'] + 1}"
        link = (f'<a href="../pair_{d["pair"]}_{d["pair"] + 1}/pair_{d["pair"]}_{d["pair"] + 1}_report.md">report</a> · '
                f'<a href="../pair_{d["pair"]}_{d["pair"] + 1}/correspondences.png">figures</a>') if pd.is_dir() else \
            f"no detailed export (rerun with <code>--export-pairs {d['pair']}</code>)"
        rows.append(f"<tr><td>{d['pair']}&rarr;{d['pair'] + 1}</td><td>{d['pnp_inliers']}/{d['pnp_correspondences']}</td>"
                    f"<td>{d['icp_inliers']}/{d['icp_correspondences']}</td>"
                    f"<td>{'yes' if d['pnp_solver_ok'] else 'NO'} / {'yes' if d['icp_solver_ok'] else 'NO'}</td>"
                    f"<td>{html.escape(', '.join(d['warnings']))}</td><td>{html.escape(', '.join(d['gt_flags']))}</td>"
                    f"<td>{d['pnp_rot_err_deg']:.2f}&deg; / {d['pnp_trans_err_m']:.3f} m</td><td>{link}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Failure timeline</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}}img{{max-width:100%}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:3px 6px;text-align:left}}
th{{background:#f2f2f2}}</style></head><body>
<h1>Failure timeline: {html.escape(Path(run_dir).name)}</h1>
<p>Solver status is the evaluator's (a pose was returned). Warnings use no ground truth; GT flags are evaluation only.
Thresholds: {html.escape(str(TH))}. Data: <a href="timeline.csv">timeline.csv</a>, <a href="timeline.json">timeline.json</a>.</p>
<img src="timeline.png" alt="failure timeline">
<h2>Flagged pairs ({len(flagged)} of {len(tl)})</h2>
<table><tr><th>pair</th><th>PnP inl./corr.</th><th>ICP inl./corr.</th><th>solver PnP / ICP</th><th>warnings</th>
<th>GT flags</th><th>PnP error</th><th>details</th></tr>
{''.join(rows)}</table></body></html>
"""
    (Path(out) / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--output")
    ap.add_argument("--mark", type=int, help="pair to highlight")
    a = ap.parse_args()
    out = Path(a.output) if a.output else Path(a.run) / "figures" / "failure_timeline"
    try:
        run, tl = timeline(a.run)
    except C.tm.SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.mark is not None and not 0 <= a.mark < len(tl):
        print(f"error: --mark {a.mark}: the run has pairs 0..{len(tl) - 1}", file=sys.stderr)
        return 2
    diag = Path(a.run) / "diagnostics"
    exported = sorted(int(p.name.split("_")[1]) for p in diag.glob("pair_*_*")) if diag.is_dir() else []
    made = []
    figure(a.run, tl, out, a.mark, exported, made)
    keys = [k for k in tl[0] if k not in ("warnings", "gt_flags")]
    with open(out / "timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["warnings", "gt_flags"])
        for d in tl:
            w.writerow(["" if d[k] is None else d[k] for k in keys] + [";".join(d["warnings"]), ";".join(d["gt_flags"])])
    C.write_json(out / "timeline.json", {"run": str(a.run), "thresholds": TH, "exported_pairs": exported, "pairs": tl})
    write_index(a.run, tl, out)
    print("\n".join(made + [str(out / "index.html")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
