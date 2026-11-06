"""Data layer of the results dashboard: discovers evaluator runs under results/runs/ and serves their contents.

Everything is read from the run directories at request time (cached by file modification time); no metric
is stored in the frontend. Sources per run directory (any directory with frames.csv, pairs.csv and
trajectory.csv written by tum_trajectory_eval):
  metrics       tum_metrics.compute_metrics on the CSVs (the same function the stress runner uses)
  timeline      diagnostics/failure_timeline.timeline (pairs.csv + trajectory.csv)
  config        config.json (stress runner) or evaluator config.json (make_diagnostics); else command.txt
                / summary.txt; unknown fields stay null
  status        status.json (stress runner), if present
  match export  <run>/diagnostics/pair_<i>_<j>/ (tum_trajectory_eval --export-pairs); if a run has none, an
                export of an EQUIVALENT run is used: same dataset, frames and stride, and byte-identical
                pairs.csv and trajectory.csv (stated in the response)
Images come from the dataset directory of the run (RGB as stored; depth colourised here).
"""

import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parents[1]
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "diagnostics"))
import tum_metrics as tm  # noqa: E402
import common as C  # noqa: E402
import failure_timeline as FT  # noqa: E402
import pair_report as PR  # noqa: E402

OUTPUT = ROOT / "results" / "runs"


def dataset_dir(dataset):
    """Directory of a run's dataset, recorded by the run as "data/<sequence>": <repo>/<dataset> if it exists,
    otherwise $SLAM_TUM_ROOT/<sequence> when SLAM_TUM_ROOT is set and that exists (docs/DATA_AND_ARCHIVE.md);
    otherwise <repo>/<dataset> (missing; callers check is_dir())."""
    local = ROOT / dataset
    if local.is_dir() or not os.environ.get("SLAM_TUM_ROOT"):
        return local
    external = Path(os.environ["SLAM_TUM_ROOT"]) / Path(dataset).name
    return external if external.is_dir() else local

SKIP_PARTS = {"archive"}            # archived figures are not runs
ALIGN_CAUTION_DEG = 10.0            # aligned ATE flagged when the best rigid fit rotates the estimate more than this
CODE = {"inlier": "I", "outlier": "O", "no_depth_i": "D", "unclassified": "U"}
_lock = threading.Lock()
_cache = {}


def _cached(key, deps, fn):
    """Memoise fn() on key while the modification times of `deps` are unchanged."""
    stamp = tuple((str(d), d.stat().st_mtime_ns) for d in deps if d.exists())
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (stamp, val)
    return val


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _fingerprint(run_dir):
    h = hashlib.sha256()
    for f in ("pairs.csv", "trajectory.csv"):
        h.update((run_dir / f).read_bytes())
    return h.hexdigest()[:16]


def _describe(run_dir):
    """Static facts about one run directory (configuration, provenance, exports)."""
    rel = run_dir.relative_to(OUTPUT).as_posix()
    cfg = _json(run_dir / "config.json") or {}
    status = _json(run_dir / "status.json")
    command = (run_dir / "command.txt").read_text().strip() if (run_dir / "command.txt").is_file() else cfg.get("command")
    man = _json(run_dir / "manifest.json")  # make_diagnostics.py (first version): the evaluator step's command
    if command is None and man:
        command = next((s_["command"] for s_ in man.get("steps", []) if s_.get("step") == "evaluator"), None)
    dataset = cfg.get("dataset")
    frames_req, stride = cfg.get("frames"), cfg.get("stride")
    if command and (frames_req is None or stride is None):
        m = re.search(r"--frames (\d+)", command)
        s = re.search(r"--stride (\d+)", command)
        frames_req = int(m.group(1)) if m and frames_req is None else frames_req
        stride = int(s.group(1)) if s and stride is None else stride
        if dataset is None:
            parts = command.split()
            dataset = parts[1] if len(parts) > 1 else None
    if dataset is None and (run_dir / "summary.txt").is_file():
        m = re.match(r"TUM (\S+):", (run_dir / "summary.txt").read_text())
        if m and dataset_dir(f"data/{m.group(1)}").is_dir():
            dataset = f"data/{m.group(1)}"
    parts = Path(rel).parts
    kind = "stress" if parts[0] == "stress" else "diagnostics" if parts[0] == "diagnostics" else parts[0]
    exports = sorted(int(d.name.split("_")[1]) for d in (run_dir / "diagnostics").glob("pair_*_*")
                     if (d / "pair.json").is_file()) if (run_dir / "diagnostics").is_dir() else []
    git = cfg.get("git") or {}
    return {"id": rel, "kind": kind, "session": "/".join(parts[:-1]) or parts[0], "name": parts[-1],
            "dataset": dataset, "sequence": Path(dataset).name if dataset else None,
            "dataset_available": bool(dataset and dataset_dir(dataset).is_dir()),
            "frames_requested": frames_req, "stride": stride, "command": command,
            "started": cfg.get("started"), "git_commit": git.get("commit"), "git_dirty": git.get("working_tree_dirty"),
            "status": status.get("status") if status else None,
            "exit_code": status.get("exit_code") if status else None,
            "wall_time_s": status.get("wall_time_s") if status else None,
            "runner_problems": status.get("problems") if status else None,
            "exports": exports, "fingerprint": _fingerprint(run_dir),
            "config_source": "config.json" if cfg else ("command.txt" if command else "summary.txt only")}


def discover():
    """All evaluator run directories under results/runs/ (and incomplete ones, with the reason)."""
    runs, incomplete = [], []
    for f in sorted(OUTPUT.rglob("frames.csv")):
        d = f.parent
        if SKIP_PARTS & set(d.relative_to(OUTPUT).parts):
            continue
        cmd = d / "command.txt"
        if cmd.is_file() and any(t in cmd.read_text() for t in ("scratch_pipeline", "tum_orb_correspondences")):
            continue  # a scratch-pipeline or ORB-correspondence output (methods.py), not a tum_trajectory_eval run
        missing = [x for x in ("pairs.csv", "trajectory.csv") if not (d / x).is_file()]
        if missing:
            incomplete.append({"id": d.relative_to(OUTPUT).as_posix(), "missing": missing})
            continue
        runs.append(_cached(("describe", str(d)), [d / "pairs.csv", d / "trajectory.csv", d / "config.json",
                                                   d / "status.json", d / "diagnostics"], lambda d=d: _describe(d)))
    # equivalent export source for runs without match-level exports
    for r in runs:
        r["export_source"] = r["id"] if r["exports"] else None
        if not r["exports"]:
            for q in runs:
                if q["exports"] and q["fingerprint"] == r["fingerprint"] and q["dataset"] == r["dataset"]:
                    r["export_source"] = q["id"]
                    break
    return runs, incomplete


def run_dir(run_id):
    d = (OUTPUT / run_id).resolve()
    if OUTPUT.resolve() not in d.parents or not (d / "frames.csv").is_file():
        raise KeyError(f"unknown run {run_id!r}")
    return d


def _run_meta(run_id):
    runs, _ = discover()
    for r in runs:
        if r["id"] == run_id:
            return r
    raise KeyError(f"unknown run {run_id!r}")


def _safe(x):
    return tm.to_json_safe(x)


def _metrics(d):
    run = tm.load_run(d)
    met = tm.compute_metrics(run)
    problems = tm.check_consistency(run)
    for m in tm.METHODS:
        a = met[m].get("alignment_rotation_deg")
        met[m]["aligned_ate_caution"] = (
            f"the best single rigid fit rotates the estimate by {a:.1f} deg (> {ALIGN_CAUTION_DEG:g}): the aligned "
            "ATE is poorly constrained (short or nearly straight path) or absorbs a gross failure; interpret with care"
            if a is not None and a > ALIGN_CAUTION_DEG else None)
    gaps = []
    dt = np.diff(run["t"])
    med = float(np.median(dt)) if len(dt) else None
    for k, x in enumerate(dt):
        if med and x > 1.5 * med:
            gaps.append({"pair": k, "dt_s": float(x), "median_dt_s": med})
    return {"metrics": met, "consistency_problems": problems, "median_frame_interval_s": med, "frame_gaps": gaps,
            "held": {m: [bool(v) for v in run[f"{m}_held"]] for m in tm.METHODS}}


def run_summary(run_id):
    d = run_dir(run_id)
    deps = [d / "pairs.csv", d / "trajectory.csv", d / "frames.csv"]
    out = _cached(("metrics", run_id), deps, lambda: _metrics(d))
    tl = _cached(("timeline", run_id), deps, lambda: FT.timeline(d)[1])
    return _safe({"meta": _run_meta(run_id), **out,
                  "warnings": sum(1 for p in tl if p["warnings"]), "gt_flags": sum(1 for p in tl if p["gt_flags"])})


def runs_overview():
    runs, incomplete = discover()
    rows, errors = [], []
    for r in runs:
        try:
            rows.append(run_summary(r["id"]))
        except Exception as e:  # a malformed run is listed with its error, never silently dropped
            errors.append({"id": r["id"], "error": f"{type(e).__name__}: {e}"})
    return {"runs": rows, "incomplete": incomplete, "errors": errors, "thresholds": FT.TH,
            "align_caution_deg": ALIGN_CAUTION_DEG}


def timeline(run_id):
    d = run_dir(run_id)
    deps = [d / "pairs.csv", d / "trajectory.csv"]
    tl = _cached(("timeline", run_id), deps, lambda: FT.timeline(d)[1])
    frames = _frames(d)
    meta = _run_meta(run_id)
    src = meta["export_source"]
    exported = set(_run_meta(src)["exports"]) if src else set()
    for p in tl:
        p["dt_s"] = frames[p["frame_j"]]["t_rgb"] - frames[p["frame_i"]]["t_rgb"]
        p["exported"] = p["pair"] in exported
    return _safe({"run": run_id, "pairs": tl, "thresholds": FT.TH, "export_source": src})


def _frames(d):
    import csv
    with open(d / "frames.csv", newline="") as f:
        return [{"k": int(r["frame"]), "t_rgb": float(r["t_rgb"]), "t_depth": float(r["t_depth"]),
                 "rgb": r["rgb"], "depth": r["depth"]} for r in csv.DictReader(f)]


def frame_path(run_id, k, kind):
    d = run_dir(run_id)
    meta = _run_meta(run_id)
    if not meta["dataset_available"]:
        raise FileNotFoundError(f"dataset {meta['dataset']!r} of run {run_id} is not available")
    fr = _frames(d)[k]
    return dataset_dir(meta["dataset"]) / fr["rgb" if kind == "rgb" else "depth"]


def depth_png(run_id, k, mode):
    """Depth of frame k as PNG: 'color' (viridis 0..max, no measurement magenta) or 'mask' (valid white)."""
    import matplotlib
    from PIL import Image
    p = frame_path(run_id, k, "depth")
    raw = C.read_depth_raw(p.parent.parent, p.relative_to(p.parent.parent))
    if mode == "mask":
        img = np.where(raw > 0, 255, 0).astype(np.uint8)
    else:
        z = raw / 5000.0
        zmax = float(z[raw > 0].max()) if (raw > 0).any() else 1.0
        rgb = matplotlib.colormaps["viridis"](np.clip(z / zmax, 0, 1))[..., :3]
        rgb[raw == 0] = (0.84, 0.15, 0.56)
        img = (rgb * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue(), {"valid_fraction": float((raw > 0).mean()),
                            "max_depth_m": float(raw.max() / 5000.0) if raw.size else None}


def r4(a, d=4):
    return [round(float(x), d) for x in np.asarray(a).reshape(-1)]


def pair_payload(p, row):
    """Arrays of one export for the frontend; classes and transforms exactly as exported."""
    m = p["meta"]
    q = p["icp_corr"] >= 0
    Pi = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][q]
    Pj = np.c_[p["Xj_x"], p["Xj_y"], p["Xj_z"]][q]
    lab = p["icp_class"][q]
    n_rej = int((lab == "rejected").sum())
    thr = m["config"]["icp"]["threshold_m"]
    reg = {}
    G = np.linalg.inv(p["T_j_from_i_gt"])
    for state, T in (("before", np.eye(4)), ("icp", p["T_i_from_j_icp_native"]), ("gt", G)):
        if T is None or not q.any():
            reg[state] = None
            continue
        r = np.linalg.norm(Pi - (Pj @ T[:3, :3].T + T[:3, 3]), axis=1)
        inl = r[lab == "inlier"]
        reg[state] = {"inlier_median_m": float(np.median(inl)) if len(inl) else None,
                      "within_threshold": int((r < thr).sum())}
    T = lambda key: None if p[key] is None else p[key].tolist()
    nan = lambda a: [None if not math.isfinite(x) else round(float(x), 2) for x in a]
    return {
        "pair": m["pair"], "i": m["frame_i"], "j": m["frame_j"], "t_i": m["t_rgb_i"], "t_j": m["t_rgb_j"],
        "counts": m["counts"], "status": m["status"], "icp_threshold_m": thr, "config": m["config"],
        "err": {k: (None if row is None or not math.isfinite(float(row[k])) else float(row[k]))
                for k in ("pnp_rot_err_deg", "pnp_trans_err_m", "icp_rot_err_deg", "icp_trans_err_m",
                          "gt_motion_deg", "gt_motion_m")},
        "kp_i": r4(p["kp_i_xy"], 2), "kp_j": r4(p["kp_j_xy"], 2),
        "m": {"ui": r4(p["u_i"], 2), "vi": r4(p["v_i"], 2), "uj": r4(p["u_j"], 2), "vj": r4(p["v_j"], 2),
              "kpj": [int(x) for x in p["kp_j"]], "cls": "".join(CODE[c] for c in p["pnp_class"]),
              "depth_i": [int(x) for x in p["depth_raw_i"]], "depth_j": [int(x) for x in p["depth_raw_j"]],
              "pnp_u": nan(p["pnp_proj_u"]), "pnp_v": nan(p["pnp_proj_v"]), "gt_u": nan(p["gt_proj_u"]),
              "gt_v": nan(p["gt_proj_v"]), "pnp_res": nan(p["pnp_res_px"]), "gt_res": nan(p["gt_res_px"])},
        "icp": {"pi": r4(Pi, 5), "pj": r4(Pj, 5), "cls": "".join("I" if c == "inlier" else "R" for c in lab),
                "res": r4(p["icp_res_m"][q], 5), "rej_every": max(1, int(math.ceil(n_rej / 20)))},
        "T_gt_ji": T("T_j_from_i_gt"), "T_pnp_ji": T("T_j_from_i_pnp"), "T_icp_ji": T("T_j_from_i_icp"),
        "T_icp_ij": T("T_i_from_j_icp_native"), "reg": reg,
        "refits": m.get("export_only_pnp_refits_on_inliers", {}).get("attempts", [])}


def missing_export(meta, pair):
    ds = meta["dataset"] or "data/<sequence>"
    out = f"results/runs/diagnostics/{meta['sequence'] or '<sequence>'}/{meta['name']}_export/evaluator"
    fr = meta["frames_requested"] if meta["frames_requested"] is not None else "<N>"
    st = meta["stride"] if meta["stride"] is not None else "<S>"
    return {"available": False,
            "reason": f"run {meta['id']} has no match-level export, and no run with byte-identical results has one",
            "command": f"build/tum_trajectory_eval {ds} --frames {fr} --stride {st} --max-dt 0.02 --max-gt-gap 0.05 "
                       f"--out-dir {out} --export-pairs {pair}",
            "note": "The export is opt-in and does not change the estimate (pairs.csv / trajectory.csv are identical "
                    "with and without it). The dashboard picks the new run up on reload."}


def _evidence(ind, run_rows):
    """Measured facts and, separately, hypotheses for a pair; rules are stated with their numbers."""
    c = ind["counts"]
    e = ind["pose_errors_from_exported_transforms"]
    inl = [float(r["pnp_inliers"]) for r in run_rows.values()]
    facts, hyp = [], []
    facts.append(f"PnP: {c['pnp_inliers']} inliers of {c['pnp_correspondences']} correspondences "
                 f"({100 * (ind['pnp_inlier_ratio'] or 0):.1f} %); run median {np.median(inl):.0f} inliers.")
    facts.append(f"Distinct keypoints among the PnP inliers: {ind['pnp_inliers_distinct_kp_i']} in frame i, "
                 f"{ind['pnp_inliers_distinct_kp_j']} in frame j; {ind['matches_sharing_a_kp_j']} of {c['matches']} "
                 f"matches share a frame-j keypoint (at most {ind['max_matches_on_one_kp_j']} on one).")
    sp = ind["pnp_inlier_spread_i_px"]
    facts.append(f"PnP inliers span {sp['bbox_w']:.0f} x {sp['bbox_h']:.0f} px in frame i"
                 + (f" ({ind['pnp_inlier_3d_max_pairwise_m']:.3f} m in 3D)." if ind.get("pnp_inlier_3d_max_pairwise_m") is not None else "."))
    facts.append(f"Correspondences within 8 px under the TRUE pose: {ind['gt_consistent_pnp_correspondences']} of "
                 f"{c['pnp_correspondences']}; PnP inliers within 8 px of the true pose: "
                 f"{ind['pnp_inliers_that_are_gt_consistent']} of {c['pnp_inliers']}.")
    if ind.get("within_8px_of_returned_pnp_pose") is not None:
        facts.append(f"Correspondences within 8 px of the RETURNED PnP pose: {ind['within_8px_of_returned_pnp_pose']}.")
    facts.append(f"Depth: {c['depth_valid_i']} of {c['matches']} matched keypoints valid in frame i, "
                 f"{c['depth_valid_j']} in frame j; {100 * ind['depth_valid_fraction_image_i']:.0f} % of frame i's pixels.")
    facts.append(f"Sharpness (Laplacian variance) frame i {ind['sharpness_laplacian_var_i']:.0f}, frame j "
                 f"{ind['sharpness_laplacian_var_j']:.0f}; keypoints {c['keypoints_i']} / {c['keypoints_j']} (cap 500).")
    if e.get("pnp"):
        facts.append(f"PnP error vs ground truth {e['pnp']['rot_err_deg']:.2f} deg / {e['pnp']['trans_err_m']:.3f} m; "
                     + (f"ICP {e['icp']['rot_err_deg']:.2f} deg / {e['icp']['trans_err_m']:.3f} m." if e.get("icp") else "ICP: no pose."))
    p = ind.get("p_no_all_gt_consistent_5_sample_in_100")
    if p is not None:
        facts.append(f"P(no all-correct 5-point sample in 100 uniform draws) = {p:.3f} (assumes OpenCV's 5-point EPnP "
                     "minimal set and all 100 iterations; from the OpenCV source, not measured).")
    ni, dj = c["pnp_inliers"], ind["pnp_inliers_distinct_kp_j"]
    if ni and dj <= max(1, ni // 2):
        hyp.append(f"Many-to-one: {ni} inlier rows reduce to {dj} distinct frame-j keypoint(s), so the inlier set "
                   "carries far fewer independent constraints than its size suggests (observed geometry; whether "
                   "this caused the failure is a hypothesis).")
    if ind["gt_consistent_fraction"] is not None and ind["gt_consistent_fraction"] < 0.3:
        hyp.append("Few correct correspondences: with this fraction RANSAC is unlikely to sample an all-correct "
                   "minimal set (estimate above), consistent with the selected hypothesis being wrong.")
    if ind["sharpness_laplacian_var_i"] < 0.7 * ind["sharpness_laplacian_var_j"]:
        hyp.append("Frame i is markedly less sharp than frame j: motion blur may reduce correct matches "
                   "(correlation only; not tested).")
    if not hyp:
        hyp.append("No rule-based concern triggered for this pair.")
    return facts, hyp


def pair(run_id, k):
    meta = _run_meta(run_id)
    src = meta["export_source"]
    if not src or k not in _run_meta(src)["exports"]:
        return _safe({"run": run_id, "pair": k, "export": missing_export(meta, k)})
    sd = run_dir(src)
    pdir = sd / "diagnostics" / f"pair_{k}_{k + 1}"

    def build():
        p = C.load_pair(pdir)
        rows = C.pairs_rows(sd)
        problems = C.validate_pair(p, rows.get(k))
        mm = p["meta"]
        imgs = {s: C.read_rgb(dataset_dir(meta["dataset"]), mm[f"rgb_{s}"]) for s in "ij"}
        deps = {s: C.read_depth_raw(dataset_dir(meta["dataset"]), mm[f"depth_{s}"]) for s in "ij"}
        ind = PR.indicators(p, imgs, deps, rows.get(k))
        facts, hyp = _evidence(ind, rows)
        page = sd.parent / "pairs" / f"pair_{k}_{k + 1}" / "index.html"
        return {"run": run_id, "pair": k, "export": {"available": True, "source_run": src,
                                                     "equivalent": src != run_id, "dir": pdir.relative_to(ROOT).as_posix(),
                                                     "problems": problems},
                "data": pair_payload(p, rows.get(k)), "indicators": ind, "facts": facts, "hypotheses": hyp,
                "solver_status": {"pnp_solver_returned": mm["status"]["pnp_solver_returned"],
                                  "pnp_counted_successful": mm["status"]["pnp_ok"],
                                  "icp_ransac_ok": mm["status"]["icp_ransac_ok"],
                                  "icp_counted_successful": mm["status"]["icp_ok"]},
                "tracking_validity": ({"pnp_accepted": mm["status"]["pnp_accepted"],
                                       "reason": mm["status"].get("pnp_reject_reason") or None}
                                      if "pnp_accepted" in mm["status"] else None),
                "tracking_validity_note": ("PnP pose accepted only if its inliers contain >= the configured number of "
                                           "distinct keypoints in each frame (evaluator --pnp-min-distinct-inliers)"
                                           if "pnp_accepted" in mm["status"] else
                                           "not available in this export: made before the evaluator had a validity "
                                           "check (a returned pose counted as success)"),
                "raw_matches": None, "raw_matches_note": "not exported (only the filtered matches are)",
                "page": page.relative_to(OUTPUT).as_posix() if page.is_file() else None}
    return _safe(_cached(("pair", src, k), [pdir / "matches.csv", pdir / "pair.json"], build))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def pack_cloud(xyz, rgb):
    mm = np.clip(np.round(xyz * 1000), -32767, 32767).astype("<i2")
    return {"n": int(len(mm)), "xyz": b64(mm), "rgb": b64(np.asarray(rgb, np.uint8))}


def pose_list(R, p):
    return [[round(float(x), 7) for x in list(np.asarray(R[i]).reshape(9)) + list(np.asarray(p[i]))]
            for i in range(len(p))]


SCENE_MAX_FRAMES = 20   # depth frames in the scene cloud (evenly spaced, deterministic) + the requested ones
SCENE_STEP_PX = 8       # pixel grid of the scene cloud


def scene(run_id, extra=()):
    meta = _run_meta(run_id)
    d = run_dir(run_id)

    def build():
        run = tm.load_run(d)
        n = run["n"]
        K = C.run_intrinsics(d, meta["dataset"])
        poses, align = {}, {}
        for m in ("gt", "pnp", "icp"):
            poses[m] = {"raw": pose_list(run[f"{m}_R"], run[f"{m}_p"]), "aligned": None}
            if m != "gt":
                R, t = tm.umeyama_rigid(run[f"{m}_p"], run["gt_p"])
                poses[m]["aligned"] = pose_list(np.einsum("ij,njk->nik", R, run[f"{m}_R"]), run[f"{m}_p"] @ R.T + t)
                align[m] = {"rotation_deg": C.rot_deg(R), "translation_m": float(np.linalg.norm(t))}
        clouds, sel = [], []
        if meta["dataset_available"]:
            sel = sorted(set(np.linspace(0, n - 1, min(SCENE_MAX_FRAMES, n)).round().astype(int).tolist())
                         | {k for k in extra if 0 <= k < n})
            for k in sel:
                xyz, px = C.backproject(C.read_depth_raw(dataset_dir(meta["dataset"]), run["depth"][k]), K,
                                        step=SCENE_STEP_PX, offset=SCENE_STEP_PX // 2)
                rgb = C.read_rgb(dataset_dir(meta["dataset"]), run["rgb"][k])[px[:, 1], px[:, 0]]
                clouds.append(dict(pack_cloud(xyz, rgb), k=int(k)))
        axis = run["gt_R"][:, :, 2].mean(0)
        axis /= np.linalg.norm(axis)
        centre = run["gt_p"].mean(0) + 0.6 * axis
        eye = run["gt_p"].mean(0) - 1.6 * axis + np.array([0, 0, 0.8])
        dv = eye - centre
        dist = float(np.linalg.norm(dv))
        return {"run": run_id, "K": {k: K[k] for k in ("fx", "fy", "cx", "cy", "width", "height", "depth_factor")} if K else None,
                "frames": [{"k": i, "t": float(run["t"][i])} for i in range(n)], "poses": poses, "alignment": align,
                "held": {m: [bool(v) for v in run[f"{m}_held"]] for m in ("pnp", "icp")} | {"gt": [False] * n},
                "clouds": clouds, "cloud_frames": sel, "cloud_step_px": SCENE_STEP_PX,
                "dataset_available": meta["dataset_available"],
                "view": {"yaw": math.degrees(math.atan2(dv[1], dv[0])), "pitch": math.degrees(math.asin(dv[2] / dist)),
                         "dist": dist, "tx": float(centre[0]), "ty": float(centre[1]), "tz": float(centre[2])}}
    deps = [d / "trajectory.csv", d / "frames.csv"]
    return _cached(("scene", run_id, tuple(sorted(extra))), deps, build)


REG_STEP_PX = 5


def registration(run_id, k):
    pr = pair(run_id, k)
    if not pr["export"]["available"]:
        return pr
    meta = _run_meta(run_id)
    src = run_dir(pr["export"]["source_run"])

    def build():
        p = C.load_pair(src / "diagnostics" / f"pair_{k}_{k + 1}")
        K = C.intrinsics(p)
        clouds = {}
        for s in "ij":
            xyz, px = C.backproject(C.read_depth_raw(dataset_dir(meta["dataset"]), p["meta"][f"depth_{s}"]), K,
                                    step=REG_STEP_PX, offset=REG_STEP_PX // 2)
            rgb = C.read_rgb(dataset_dir(meta["dataset"]), p["meta"][f"rgb_{s}"])[px[:, 1], px[:, 0]]
            clouds[s] = pack_cloud(xyz, rgb)
        return {"clouds": clouds, "step_px": REG_STEP_PX}
    pdir = src / "diagnostics" / f"pair_{k}_{k + 1}"
    return {**pr, "registration": _cached(("reg", str(src), k), [pdir / "pair.json"], build)}


def compare(a, b):
    sa, sb = run_summary(a), run_summary(b)
    same = sa["meta"]["fingerprint"] == sb["meta"]["fingerprint"]
    same_cfg = all(sa["meta"][k] == sb["meta"][k] for k in ("dataset", "frames_requested", "stride"))
    return {"a": sa, "b": sb, "identical_results": same, "same_configuration": same_cfg}


def groups():
    """Runs grouped by configuration (dataset, frames, stride); distinct result fingerprints within a group."""
    runs, _ = discover()
    g = {}
    for r in runs:
        fr = {None: "not recorded", 0: "all"}.get(r["frames_requested"], r["frames_requested"])
        key = f"{r['sequence']} | frames {fr} | stride {r['stride'] if r['stride'] is not None else 'not recorded'}"
        g.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(g.items()):
        fps = sorted({r["fingerprint"] for r in rs})
        out.append({"configuration": key, "runs": [{"id": r["id"], "fingerprint": r["fingerprint"],
                                                    "git_commit": r["git_commit"], "git_dirty": r["git_dirty"],
                                                    "started": r["started"]} for r in rs],
                    "distinct_results": len(fps)})
    return out
