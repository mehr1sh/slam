#!/usr/bin/env python3
"""Focused diagnostic page of one frame pair (HTML + images), from a --export-pairs export.

Usage (repository root):
  python3 tools/evaluation/diagnostics/pair_page.py --run <evaluator dir> --pair 73 \
          --dataset data/rgbd_dataset_freiburg1_xyz [--compare 68 --context 66-80] [--output DIR]
Output (default <run>/../pairs/pair_<i>_<j>/): index.html with
  1 matches.png        RGB correspondences by PnP class (pair_view.py)
  2 depth.png          RGB, depth in metres (no measurement in magenta), valid-depth mask, matched keypoints
  3 (in matches.png)   where the PnP inliers and outliers are in frame i (every match, not sampled)
  4 pnp_residuals.png  frame j: observed keypoint -> projection of its 3D point (X_i from frame i's depth) under
                       the RETURNED PnP pose and under the ground-truth pose; every PnP correspondence
  5 registration.png   ICP before / after with its actual correspondences (icp_registration.py; under ../../icp/)
  6 orientation.png    camera j placed in camera i's frame by GT, PnP and ICP (T_i<-j), with camera axes,
                       over frame i's back-projected depth
  7 pair_<i>_<j>_report.md (pair_report.py), shown on the page: measurements, facts, hypotheses
"""

import argparse
import html
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import icp_registration as IR  # noqa: E402
import pair_report as PR  # noqa: E402
import pair_view as PV  # noqa: E402
import render3d as R3  # noqa: E402

AXIS_COLORS = ("#d62728", "#17becf", "#1f77b4")
MAGENTA = np.array([0.84, 0.15, 0.56])


def depth_image(p, imgs, deps, out, made):
    m, K = p["meta"], C.intrinsics(p)
    W, H, G, HEAD = 640, 480, 14, 56
    total = (3 * W + 2 * G, HEAD + 2 * (H + 46) + 10)
    fig, ax = C.pixel_figure(*total)
    zmax = max(float(deps[s][deps[s] > 0].max()) for s in "ij") / K["depth_factor"]
    cmap = matplotlib.colormaps["viridis"]
    for r, s in enumerate("ij"):
        y0 = HEAD + r * (H + 46)
        raw = deps[s]
        z = raw / K["depth_factor"]
        rgbz = cmap(np.clip(z / zmax, 0, 1))[..., :3]
        rgbz[raw == 0] = MAGENTA
        ax.imshow(imgs[s], extent=(0, W, y0 + H, y0))
        ax.imshow(rgbz, extent=(W + G, 2 * W + G, y0 + H, y0))
        ax.imshow(raw > 0, cmap="gray", vmin=0, vmax=1, extent=(2 * (W + G), 3 * W + 2 * G, y0 + H, y0))
        valid = p[f"depth_raw_{s}"] > 0
        u, v = p[f"u_{s}"], p[f"v_{s}"]
        for ox in (W + G, 2 * (W + G)):
            ax.scatter(u[valid] + ox, v[valid] + y0, s=16, c="white", edgecolors="black", lw=0.6, zorder=4)
            ax.scatter(u[~valid] + ox, v[~valid] + y0, s=30, marker="x", c="red", lw=1.5, zorder=5)
        ax.text(4, y0 + H + 6, f"frame {m['frame_' + s]}: RGB  t = {m['t_rgb_' + s]:.6f} s", fontsize=11, weight="bold",
                va="top")
        ax.text(W + G + 4, y0 + H + 4, f"depth t = {m['t_depth_' + s]:.6f} s (raw / {K['depth_factor']:g} m)\n"
                                       f"matched keypoints: {int(valid.sum())} with depth (white), {int((~valid).sum())} "
                                       "without (red x)", fontsize=10, va="top")
        ax.text(2 * (W + G) + 4, y0 + H + 6, f"valid depth (white): {100 * (raw > 0).mean():.1f} % of pixels", fontsize=10,
                va="top")
    ax.text(4, 6, f"Depth of pair {m['frame_i']} -> {m['frame_j']}: PnP needs depth at the frame-{m['frame_i']} keypoint, "
                  f"ICP at both (lookup at the truncated keypoint pixel)", fontsize=14, weight="bold", va="top")
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(W + G, W + G + 300, 48, 36), cmap=cmap, aspect="auto")
    ax.text(W + G - 6, 42, "0 m", fontsize=9.5, ha="right", va="center")
    ax.text(W + G + 306, 42, f"{zmax:.2f} m", fontsize=9.5, va="center")
    ax.add_patch(matplotlib.patches.Rectangle((W + G + 380, 36), 14, 12, color=MAGENTA))
    ax.text(W + G + 400, 42, "no measurement (raw 0)", fontsize=9.5, va="center")
    C.save_png(fig, out / "depth.png", made)


def residual_image(p, imgs, out, made):
    m, K = p["meta"], C.intrinsics(p)
    W, H, G, HEAD = 640, 480, 20, 96
    total = (2 * W + G, HEAD + H + 70)
    fig, ax = C.pixel_figure(*total)
    s = p["pnp_corr"] >= 0
    stats = {}
    for n, (key, title) in enumerate((("pnp", "returned PnP pose"), ("gt", "ground-truth pose"))):
        x0 = n * (W + G)
        pax = fig.add_axes([x0 / total[0], 1 - (HEAD + H) / total[1], W / total[0], H / total[1]])  # clips arrows
        pax.imshow(imgs["j"], extent=(0, W, H, 0), alpha=0.55)
        pax.set_xlim(0, W), pax.set_ylim(H, 0)
        pax.set_axis_off()
        if key == "pnp" and p["T_j_from_i_pnp"] is None:
            pax.text(W / 2, H / 2, "PnP returned no pose", ha="center", fontsize=14)
            continue
        pu, pv, res = p[f"{key}_proj_u"], p[f"{key}_proj_v"], p[f"{key}_res_px"]
        for c in ("outlier", "inlier", "unclassified"):
            k = s & (p["pnp_class"] == c)
            if not k.any():
                continue
            pax.quiver(p["u_j"][k], p["v_j"][k], pu[k] - p["u_j"][k], pv[k] - p["v_j"][k], angles="xy",
                       scale_units="xy", scale=1, color=C.CLASS_STYLE[c]["color"],
                       width=0.003 if c == "inlier" else 0.0018, alpha=1 if c == "inlier" else 0.7, zorder=4 + (c == "inlier"))
            pax.scatter(p["u_j"][k], p["v_j"][k], s=22 if c == "inlier" else 8, color=C.CLASS_STYLE[c]["color"],
                        edgecolors="white" if c == "inlier" else "none", lw=0.6, zorder=6)
        r = res[s]
        inl = (p["pnp_class"] == "inlier")[s]
        outside = int(((pu[s] < 0) | (pu[s] >= W) | (pv[s] < 0) | (pv[s] >= H)).sum())
        stats[key] = {"within_8px": int((r < 8).sum()), "inliers_median_px": float(np.median(r[inl])) if inl.any() else None,
                      "projections_outside_image": outside}
        ax.text(x0 + 4, HEAD + H + 8, f"{stats[key]['within_8px']} of {int(s.sum())} within 8 px; PnP inliers: median "
                                      f"{stats[key]['inliers_median_px']:.1f} px\n{outside} projections outside the image "
                                      "(arrows clipped)", fontsize=10.5, va="top")
        ax.text(x0 + 4, HEAD - 6, f"({'a' if n == 0 else 'b'}) under the {title}", fontsize=12, weight="bold",
                va="bottom")
    ax.text(4, 6, f"PnP reprojection residuals, pair {m['frame_i']} -> {m['frame_j']}, drawn on frame {m['frame_j']} (1:1 px)",
            fontsize=14, weight="bold", va="top")
    ax.text(4, 30, f"Arrow: observed keypoint u_j -> projection of its 3D point X_i; all {int(s.sum())} PnP correspondences; "
                   "blue = solver inlier, red = outlier.\nX_i = frame-i keypoint back-projected with its depth; pinhole "
                   "projection without distortion, as solvePnPRansac; residuals checked against the export.",
            fontsize=10, va="top")
    C.save_png(fig, out / "pnp_residuals.png", made)
    return stats


def orientation_image(p, deps, imgs, out, made):
    m, K = p["meta"], C.intrinsics(p)
    xyz, px = C.backproject(deps["i"], K, step=4)
    col = imgs["i"][px[:, 1], px[:, 0]] / 255.0
    poses = {"gt": p["T_j_from_i_gt"], "pnp": p["T_j_from_i_pnp"], "icp": p["T_j_from_i_icp"]}
    centres = [np.linalg.inv(T)[:3, 3] for T in poses.values() if T is not None]
    PW, PH, HEAD = 900, 680, 60
    total = (2 * PW + 12, HEAD + PH + 150)
    fig, ax = C.pixel_figure(*total)
    c_scene = np.median(xyz, axis=0)
    ext = np.r_[np.percentile(xyz, [5, 95], axis=0), np.zeros((1, 3)), *[c[None] for c in centres]]
    for n, (title, wide) in enumerate((("all camera poses", True), ("close-up around camera i", False))):
        if wide:
            tgt = (ext.max(0) + ext.min(0)) / 2
            rad = np.linalg.norm(ext.max(0) - ext.min(0)) / 2
            view = R3.View(tgt + np.array([-0.55, -0.6, -1.0]) / np.linalg.norm([-0.55, -0.6, -1.0]) * rad * 2.3, tgt,
                           up=(0, -1, 0), fov_deg=50, size=(PW, PH))
        else:
            tgt = 0.25 * c_scene
            view = R3.View(np.array([-0.35, -0.35, -0.45]), tgt, up=(0, -1, 0), fov_deg=50, size=(PW, PH))
        x0 = n * (PW + 12)
        pax = fig.add_axes([x0 / total[0], 1 - (HEAD + PH) / total[1], PW / total[0], PH / total[1]])
        img, _ = view.splat(xyz, col * 0.6 + 0.4, radius=1)
        pax.imshow(img, extent=(0, PW, PH, 0))
        pax.set_xlim(0, PW), pax.set_ylim(PH, 0)
        pax.set_axis_off()
        s = 0.06 if not wide else 0.22
        cams = [("black", np.eye(4), f"camera i (frame {m['frame_i']})")] + \
               [(C.METHOD_COLOR[k], np.linalg.inv(T), f"camera j: {C.METHOD_LABEL[k]}") for k, T in poses.items() if T is not None]
        for colr, T, lab in cams:
            R3.draw_segments(pax, view, C.frustum_lines(T, K, s), color=colr, lw=2.2)
            for a, ac in enumerate(AXIS_COLORS):
                R3.draw_segments(pax, view, [[T[:3, 3], T[:3, 3] + s * 0.9 * T[:3, a]]], color=ac, lw=2)
        pax.text(8, 8, title, fontsize=12, weight="bold", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax.text(4, 6, f"Camera orientations, pair {m['frame_i']} -> {m['frame_j']}: camera j placed in camera {m['frame_i']}'s frame "
                  "by T_i<-j = (T_j<-i)^-1 of each source, over frame i's depth (x right, y down, z forward; metres)",
            fontsize=13, weight="bold", va="top")
    y = HEAD + PH + 16
    for k, (colr, lab) in enumerate([("black", f"camera i = frame {m['frame_i']} (origin)")] +
                                    [(C.METHOD_COLOR[q], f"camera j by {C.METHOD_LABEL[q]}") for q in poses if poses[q] is not None]):
        ax.plot([10, 40], [y + 22 * k] * 2, color=colr, lw=3)
        ax.text(46, y + 22 * k, lab, fontsize=10.5, va="center")
    for k, (ac, lab) in enumerate(zip(AXIS_COLORS, ("camera x (right)", "camera y (down)", "camera z (optical axis)"))):
        ax.plot([340, 370], [y + 22 * k] * 2, color=ac, lw=3)
        ax.text(376, y + 22 * k, lab, fontsize=10.5, va="center")
    lines = []
    G = poses["gt"]
    for k in ("gt", "pnp", "icp"):
        T = poses[k]
        if T is None:
            lines.append(f"{C.METHOD_LABEL[k]}: no pose")
            continue
        e = "" if k == "gt" else (f"; error vs GT {C.rot_deg(G[:3, :3].T @ T[:3, :3]):.2f} deg, "
                                  f"{np.linalg.norm(G[:3, 3] - T[:3, 3]):.4f} m")
        lines.append(f"{C.METHOD_LABEL[k]}: rotation {C.rot_deg(T[:3, :3]):.2f} deg, |t| = {np.linalg.norm(T[:3, 3]):.4f} m{e}")
    ax.text(700, y - 8, "T_j<-i (X_j = R X_i + t):\n" + "\n".join(lines), fontsize=10.5, va="top", family="monospace")
    C.save_png(fig, out / "orientation.png", made)


def md_to_html(md):
    """Minimal Markdown (headings, tables, numbered and bulleted lists, paragraphs, `code`) to HTML."""
    out, para, table = [], [], []
    inline = lambda s: re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(s))

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if table:
            rows = [r.strip().strip("|").split("|") for r in table if not re.match(r"^\|[-| ]+\|$", r.strip())]
            out.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{inline(c.strip())}"
                                                            f"</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>"
                                           for i, r in enumerate(rows)) + "</table>")
            table.clear()
    for line in md.splitlines():
        if line.startswith("#"):
            flush()
            lvl = len(line) - len(line.lstrip("#"))
            out.append(f"<h{lvl + 1}>{inline(line.lstrip('#').strip())}</h{lvl + 1}>")
        elif line.startswith("|"):
            if para:
                flush()
            table.append(line)
        elif re.match(r"^(\d+\.|-) ", line):
            flush()
            out.append("<p class='item'>" + inline(line) + "</p>")
        elif not line.strip():
            flush()
        else:
            if table:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def page(out, title, sections, report_md, viewer_link):
    body = []
    for n, (h, img, text) in enumerate(sections, 1):
        body.append(f"<h2>{n}. {html.escape(h)}</h2><p class='muted'>{html.escape(text)}</p>")
        if img:
            body.append(f"<a href='{img}'><img src='{img}' alt='{html.escape(h)}'></a>")
    body.append(f"<h2>{len(sections) + 1}. What the evidence supports</h2>" + md_to_html(report_md))
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;max-width:1500px;color:#111;background:#fff}}
img{{max-width:100%;border:1px solid #ddd}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}
h3{{font-size:15px}}.muted{{color:#555}}table{{border-collapse:collapse;font-size:13px;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}
code{{font-size:12px;background:#f4f4f4;padding:0 3px}}p.item{{margin:4px 0 4px 12px}}</style></head><body>
<h1>{html.escape(title)}</h1><p><a href="{viewer_link}">interactive viewer of this run</a></p>
{''.join(body)}</body></html>
""")


def run(run_dir, pair, dataset, out=None, compare=None, context=None):
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir.parent / "pairs" / f"pair_{pair}_{pair + 1}"
    p = C.load_pair(run_dir / "diagnostics" / f"pair_{pair}_{pair + 1}")
    m = p["meta"]
    imgs = {s: C.read_rgb(dataset, m[f"rgb_{s}"]) for s in "ij"}
    deps = {s: C.read_depth_raw(dataset, m[f"depth_{s}"]) for s in "ij"}
    made, problems = PV.run(run_dir, pair, dataset, out)
    depth_image(p, imgs, deps, out, made)
    res = residual_image(p, imgs, out, made)
    icp_dir = run_dir.parent / "icp" / f"pair_{pair}_{pair + 1}"
    made += IR.run(run_dir, pair, dataset, icp_dir)[0]
    orientation_image(p, deps, imgs, out, made)
    ind, pr, _ = PR.run(run_dir, pair, dataset, out)
    ind["pnp_residual_views"] = res
    C.write_json(out / "pair_summary.json", ind)
    if compare is not None:
        ref, rp, _ = PR.run(run_dir, compare, dataset, run_dir.parent / "pairs" / f"pair_{compare}_{compare + 1}")
        lo, hi = (int(x) for x in context.split("-")) if context else (pair, pair)
        PR.comparison_report(ind, ref, PR.context_rows(run_dir, dataset, range(lo, hi + 1)), out, made, run_dir,
                             PR.choose_reference(C.pairs_rows(run_dir), pair) == compare)
        pr = pr + [f"reference: {x}" for x in rp]
    report_md = (out / f"pair_{pair}_{pair + 1}_report.md").read_text()
    c = m["counts"]
    rel_icp = f"../../icp/pair_{pair}_{pair + 1}/registration.png"
    sections = [
        ("RGB correspondences", "matches.png",
         f"frame {m['frame_i']} (t = {m['t_rgb_i']:.6f} s) and frame {m['frame_j']} (t = {m['t_rgb_j']:.6f} s): "
         f"{c['keypoints_i']} / {c['keypoints_j']} keypoints, {c['matches']} filtered matches, coloured by the class "
         "solvePnPRansac returned; lines are a deterministic sample (matches_drawn.json lists them)."),
        ("Depth maps and valid-depth masks", "depth.png",
         f"{c['depth_valid_i']} of {c['matches']} matched keypoints have depth in frame {m['frame_i']}, "
         f"{c['depth_valid_j']} in frame {m['frame_j']}."),
        ("PnP inlier / outlier distribution", None,
         "See the lower-left panel of section 1: the start of every match in frame i by class (not sampled). "
         f"PnP inliers: {c['pnp_inliers']} of {c['pnp_correspondences']} correspondences."),
        ("PnP reprojection residuals", "pnp_residuals.png",
         f"Under the returned pose {res.get('pnp', {}).get('within_8px', 'n/a')} correspondences lie within 8 px; under "
         f"the ground-truth pose {res['gt']['within_8px']}."),
        ("ICP correspondence residuals (registration before / after)", rel_icp,
         f"{c['icp_correspondences']} ICP correspondences (depth in both frames), {c['icp_inliers']} inliers by "
         "EstimateRigidRansac; vectors coloured by residual; bottom row: dense depth check (not used by ICP)."),
        ("Camera orientations: ground truth, PnP, ICP", "orientation.png",
         "Camera j drawn in camera i's frame by each source's T_i<-j, with its axes."),
    ]
    page(out, f"Pair {m['frame_i']} -> {m['frame_j']}: diagnostic page ({run_dir.parent.name})", sections, report_md,
         "../../viewer/index.html#pair=%d&mode=registration&state=icp" % pair)
    made.append(str(out / "index.html"))
    return made, problems + pr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output")
    ap.add_argument("--compare", type=int, help="reference pair for the written comparison")
    ap.add_argument("--context", help="exported pair range listed in the report, e.g. 66-80")
    a = ap.parse_args()
    try:
        made, problems = run(a.run, a.pair, a.dataset, a.output, a.compare, a.context)
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
