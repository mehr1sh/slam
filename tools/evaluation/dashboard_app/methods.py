"""Method sources for side-by-side comparison: one estimated trajectory of one implementation in one run.

Sources (each read only from its own files; a missing or malformed file is an error, never a fallback):
  reference  tum_trajectory_eval runs (frames.csv, pairs.csv, trajectory.csv): the reference pipeline.
             Features: project FAST + rotated BRIEF via find_feature_matches (NOT OpenCV ORB); poses:
             ref_pnp = OpenCV solvePnPRansac (+ the evaluator's PnP validity check where recorded),
             ref_icp = EstimateRigidRansac + g2o refinement.
  scratch    scratch_pipeline --tum-frames runs (production pipeline, OpenCV-free): command.txt naming
             scratch_pipeline, frames.csv (the evaluator's, copied), pairs.csv, trajectory.csv (scratch schema,
             '#' comment lines first), correspondences/, keypoints/. Pose: scratch_pnp = project RANSAC around the
             linear PnP + LM refinement on project FAST / rotated BRIEF / 8-level pyramid features.
Ground truth is not a source: it is the frames.csv pose of each frame and is shown separately.

Comparability: two sources are compared only if their frames.csv rows (frame, RGB / depth timestamps and files)
are identical ("configuration key" = hash of those columns); every metric is then computed by the same code
(tum_metrics.compute_metrics) on the same timestamps and the same ground truth.
"""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

import data as D
import tum_metrics as tm

LABELS = {
    "ref_pnp": ("Reference PnP", "reference pipeline: project FAST + rotated BRIEF features (not OpenCV ORB), OpenCV "
                                 "solvePnPRansac (100 it, 8 px), evaluator PnP validity check where recorded"),
    "ref_icp": ("Reference ICP", "reference pipeline: same features, EstimateRigidRansac (1 cm) + g2o refinement"),
    "scratch_pnp": ("Scratch PnP", "pnp_from_scratch (OpenCV-free): project FAST + rotated BRIEF, 8-level pyramid, "
                                   "RANSAC (300 it, 8 px) around the linear PnP + LM refinement"),
}
COLORS = {"gt": "#2ca02c", "ref_pnp": "#ff7f0e", "ref_icp": "#9467bd", "scratch_pnp": "#17becf"}


class SourceError(ValueError):
    pass


def frames_key(frames_csv):
    """Hash of the frame selection: frame index, RGB / depth timestamps and files (not ground truth values)."""
    h = hashlib.sha256()
    with open(frames_csv, newline="") as f:
        for r in csv.DictReader(f):
            h.update(f"{r['frame']},{r['t_rgb']},{r['t_depth']},{r['rgb']},{r['depth']}\n".encode())
    return h.hexdigest()[:16]


def _is_scratch(d):
    cmd = d / "command.txt"
    return cmd.is_file() and "scratch_pipeline" in cmd.read_text() and (d / "frames.csv").is_file()


def discover():
    """All method sources under results/runs/: reference runs (via data.discover) and scratch TUM runs."""
    out = []
    runs, _ = D.discover()
    for r in runs:
        d = D.OUTPUT / r["id"]
        key = frames_key(d / "frames.csv")
        for m in ("pnp", "icp"):
            sid = f"ref_{m}"
            out.append({"id": f"{r['id']}::{sid}", "run": r["id"], "method": sid, "column": m, "family": "reference",
                        "label": LABELS[sid][0], "implementation": LABELS[sid][1], "frames_key": key,
                        "sequence": r["sequence"], "dataset": r["dataset"], "stride": r["stride"],
                        "frames_requested": r["frames_requested"], "export_source": r.get("export_source"),
                        "source_files": [f"results/runs/{r['id']}/{f}" for f in ("trajectory.csv", "pairs.csv", "frames.csv")]})
    for cmd in sorted(D.OUTPUT.rglob("command.txt")):
        d = cmd.parent
        if "archive" in d.relative_to(D.OUTPUT).parts or not _is_scratch(d):
            continue
        rid = d.relative_to(D.OUTPUT).as_posix()
        text = cmd.read_text().split()
        seq = text[text.index("--tum-sequence") + 1] if "--tum-sequence" in text else None
        out.append({"id": f"{rid}::scratch_pnp", "run": rid, "method": "scratch_pnp", "column": "scratch",
                    "family": "scratch", "label": LABELS["scratch_pnp"][0], "implementation": LABELS["scratch_pnp"][1],
                    "frames_key": frames_key(d / "frames.csv"), "sequence": Path(seq).name if seq else None,
                    "dataset": seq, "stride": None, "frames_requested": None, "export_source": None,
                    "command": cmd.read_text().strip(),
                    "source_files": [f"results/runs/{rid}/{f}" for f in ("trajectory.csv", "pairs.csv", "frames.csv")]})
    return out


def get(source_id):
    for s in discover():
        if s["id"] == source_id:
            return s
    raise KeyError(f"unknown method source {source_id!r}")


def _read_scratch_csv(path):
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines))


def _num(v):
    return float(v) if v not in ("", None) else float("nan")


def load_scratch_run(d):
    """A run dict (tum_metrics.load_run keys) with the single method 'scratch' from a scratch TUM output."""
    d = Path(d)
    for f in ("frames.csv", "pairs.csv", "trajectory.csv"):
        if not (d / f).is_file():
            raise SourceError(f"{d / f}: missing")
    fr = list(csv.DictReader(open(d / "frames.csv", newline="")))
    tr = _read_scratch_csv(d / "trajectory.csv")
    pr = _read_scratch_csv(d / "pairs.csv")
    n = len(fr)
    if len(tr) != n:
        raise SourceError(f"{d / 'trajectory.csv'}: {len(tr)} poses for {n} frames in frames.csv (incomplete run)")
    if len(pr) != n - 1:
        raise SourceError(f"{d / 'pairs.csv'}: {len(pr)} pairs for {n} frames")
    if [int(r["frame"]) for r in tr] != list(range(n)) or [int(r["frame"]) for r in fr] != list(range(n)):
        raise SourceError(f"{d}: frame indices are not 0..{n - 1} in order")
    if [(int(r["frame_i"]), int(r["frame_j"])) for r in pr] != [(k, k + 1) for k in range(n - 1)]:
        raise SourceError(f"{d / 'pairs.csv'}: not the consecutive pairs 0->1, 1->2, ...")
    run = {"n": n, "methods": ["scratch"], "dir": d}
    run["t"] = run["t_rgb"] = np.array([float(r["t_rgb"]) for r in fr])
    run["t_depth"] = np.array([float(r["t_depth"]) for r in fr])
    run["dt_ms"] = np.array([float(r["dt_ms"]) for r in fr])
    run["rgb"] = [r["rgb"] for r in fr]
    run["depth"] = [r["depth"] for r in fr]
    run["gt_p"] = np.array([[float(r[f"gt_{a}"]) for a in "xyz"] for r in fr])
    run["gt_R"] = np.array([tm.quat_to_R(np.array([float(r[f"gt_q{a}"]) for a in "xyzw"])) for r in fr])
    run["scratch_p"] = np.array([[float(r[f"t{a}"]) for a in "xyz"] for r in tr])
    run["scratch_R"] = np.array([[[float(r[f"r{i}{j}"]) for j in range(3)] for i in range(3)] for r in tr])
    if not (np.isfinite(run["scratch_p"]).all() and np.isfinite(run["scratch_R"]).all()):
        raise SourceError(f"{d / 'trajectory.csv'}: non-finite pose")
    ok = np.array([r["pnp_ok"] == "1" for r in pr])
    run["scratch_ok"] = ok
    run["scratch_held"] = np.r_[False, ~ok]
    src = [r["pose_source"] for r in tr]
    if [s.startswith("held") for s in src[1:]] != list(~ok):
        raise SourceError(f"{d}: trajectory pose_source disagrees with pairs.csv pnp_ok")
    run["scratch_rot_err_deg"] = np.array([_num(r["rot_err_deg"]) for r in pr])
    run["scratch_trans_err_m"] = np.array([_num(r["trans_err_m"]) for r in pr])
    run["scratch_correspondences"] = np.array([int(r["correspondences_3d2d"]) for r in pr])
    run["scratch_inliers"] = np.array([int(r["pnp_inliers"]) for r in pr])
    run["scratch_rel_R"] = np.array([[[_num(r[f"r{i}{j}"]) for j in range(3)] for i in range(3)] for r in pr])
    run["scratch_rel_t"] = np.array([[_num(r[f"t{a}"]) for a in "xyz"] for r in pr])
    run["matches"] = np.array([int(r["filtered_matches"]) for r in pr])
    run["raw_matches"] = np.array([int(r["raw_matches"]) for r in pr])
    run["keypoints_i"] = np.array([int(r["keypoints_i"]) for r in pr])
    run["keypoints_j"] = np.array([int(r["keypoints_j"]) for r in pr])
    run["scratch_file_terr"] = np.array([float(r["translation_error_m"]) for r in tr])
    run["scratch_file_rerr"] = np.array([float(r["rotation_error_deg"]) for r in tr])
    return run


def check_scratch(run, tol_m=1e-5):
    """The scratch files agree with themselves under T_wc (X_world = R X_cam + t) and T_j<-i = T_wc[j]^-1 T_wc[i]."""
    pr = []
    te = np.linalg.norm(run["scratch_p"] - run["gt_p"], axis=1)
    if np.abs(te - run["scratch_file_terr"]).max() > tol_m:
        pr.append("scratch trajectory position errors differ from |p - p_gt|")
    if np.linalg.norm(run["scratch_p"][0] - run["gt_p"][0]) > tol_m:
        pr.append("scratch frame 0 is not anchored to ground truth")
    for k in np.nonzero(run["scratch_ok"])[0]:
        Ri, Rj = run["scratch_R"][k], run["scratch_R"][k + 1]
        R = Rj.T @ Ri
        t = Rj.T @ (run["scratch_p"][k] - run["scratch_p"][k + 1])
        if np.abs(R - run["scratch_rel_R"][k]).max() > 1e-6 or np.abs(t - run["scratch_rel_t"][k]).max() > 1e-6:
            pr.append(f"pair {k}: trajectory T_wc[j]^-1 T_wc[i] differs from the pairs.csv pose")
            break
    for k in np.nonzero(run["scratch_held"])[0]:
        if np.abs(run["scratch_p"][k] - run["scratch_p"][k - 1]).max() > 0:
            pr.append(f"frame {k}: held pose differs from the previous pose")
            break
    return pr


def load(source):
    """(run dict, method column) for a source: reference via tum_metrics.load_run, scratch via load_scratch_run."""
    d = D.OUTPUT / source["run"]
    if source["family"] == "reference":
        return tm.load_run(d), source["column"]
    return load_scratch_run(d), "scratch"


def metrics(source_id):
    s = get(source_id)
    d = D.OUTPUT / s["run"]

    def build():
        run, col = load(s)
        problems = tm.check_consistency(run) if s["family"] == "reference" else check_scratch(run)
        sub = dict(run)
        sub["methods"] = [col]
        met = tm.compute_metrics(sub)
        mm = met[col]
        a = mm.get("alignment_rotation_deg")
        mm["aligned_ate_caution"] = (f"the single global rigid fit rotates the estimate by {a:.1f} deg (> "
                                     f"{D.ALIGN_CAUTION_DEG:g}): interpret the aligned ATE with care"
                                     if a is not None and a > D.ALIGN_CAUTION_DEG else None)
        return {"source": s, "frames": met["frames"], "attempted_pairs": met["attempted_pairs"],
                "time_span_s": met["time_span_s"], "rpe_interval_frames_for_seconds": met["rpe_interval_frames_for_seconds"],
                "matches_mean": met["matches_mean"], "matches_min": met["matches_min"], "method": mm,
                "notes": [x for x in met["notes"] if x.startswith(col + ":")], "consistency_problems": problems,
                "timestamps_first_last": [float(run["t"][0]), float(run["t"][-1])]}
    deps = [d / "trajectory.csv", d / "pairs.csv", d / "frames.csv"]
    return tm.to_json_safe(D._cached(("source_metrics", source_id), deps, build))


def configurations():
    """Sources grouped by identical frame selection (frames_key)."""
    g = {}
    for s in discover():
        g.setdefault(s["frames_key"], []).append(s)
    out = []
    for key, ss in g.items():
        n = None
        try:
            with open(D.OUTPUT / ss[0]["run"] / "frames.csv") as f:
                n = sum(1 for _ in f) - 1
        except OSError:
            pass
        refs = sorted({s["run"] for s in ss if s["family"] == "reference"})
        strides = sorted({s["stride"] for s in ss if s["stride"] is not None})
        out.append({"frames_key": key, "frames": n, "sequence": ss[0]["sequence"],
                    "stride": strides[0] if len(strides) == 1 else None,
                    "families": sorted({s["family"] for s in ss}), "reference_runs": refs,
                    "scratch_runs": sorted({s["run"] for s in ss if s["family"] == "scratch"}),
                    "sources": [s["id"] for s in ss]})
    out.sort(key=lambda c: (-(len(c["families"])), c["sequence"] or "", c["frames"] or 0))
    return out


def compare(source_ids):
    """Metrics of several sources; refused if their frame selections differ."""
    ss = [get(i) for i in source_ids]
    keys = {s["frames_key"] for s in ss}
    if len(keys) != 1:
        raise SourceError("sources use different frame selections (frames.csv differs): "
                          + ", ".join(f"{s['id']} [{s['frames_key']}]" for s in ss))
    return {"frames_key": keys.pop(), "sources": [metrics(i) for i in source_ids]}


def source_scene(source_id, extra=()):
    """Poses of GT and of ONE source, held flags, and the per-frame camera-frame clouds (shared sampling)."""
    s = get(source_id)
    run, col = load(s)
    ref_run = None
    for r in D.discover()[0]:  # the clouds come from the dataset images; reuse the reference scene sampler
        if (D.OUTPUT / r["id"] / "frames.csv").is_file() and frames_key(D.OUTPUT / r["id"] / "frames.csv") == s["frames_key"]:
            ref_run = r["id"]
            break
    if ref_run is None:
        raise SourceError(f"no reference run with the frame selection of {source_id}")
    base = D.scene(ref_run, extra)
    gt_ok = np.allclose([p[9:] for p in base["poses"]["gt"]["raw"]], run["gt_p"], atol=1e-6)
    if not gt_ok:
        raise SourceError(f"{source_id}: ground truth differs from the reference run with the same frames")
    R, t = tm.umeyama_rigid(run[f"{col}_p"], run["gt_p"])
    poses = {"gt": base["poses"]["gt"],
             s["method"]: {"raw": D.pose_list(run[f"{col}_R"], run[f"{col}_p"]),
                           "aligned": D.pose_list(np.einsum("ij,njk->nik", R, run[f"{col}_R"]), run[f"{col}_p"] @ R.T + t)}}
    return {**{k: v for k, v in base.items() if k not in ("poses", "alignment", "held")},
            "source": s, "method": s["method"], "poses": poses,
            "alignment": {s["method"]: {"rotation_deg": tm.rot_angle_deg(R), "translation_m": float(np.linalg.norm(t))}},
            "held": {"gt": [False] * run["n"], s["method"]: [bool(x) for x in run[f"{col}_held"]]},
            "colors": COLORS, "labels": {"gt": "ground truth", s["method"]: s["label"]}}


def _scratch_pair(s, k):
    d = D.OUTPUT / s["run"]
    f = d / "correspondences" / f"pair_{k}_{k + 1}.csv"
    if not f.is_file():
        return {"available": False, "reason": f"results/runs/{f.relative_to(D.OUTPUT)} missing"}
    run = load_scratch_run(d)
    rows = list(csv.DictReader(open(f, newline="")))
    ok = bool(run["scratch_ok"][k])

    def cls(r):
        if r["pnp_used"] != "1":
            return "D"   # no frame-i depth: not given to PnP
        if not ok:
            return "U"   # PnP failed: no classification
        return "I" if r["pnp_inlier"] == "1" else "O"
    kp = {}
    for side, fr in (("i", k), ("j", k + 1)):
        kf = d / "keypoints" / f"frame_{fr:06d}.csv"
        kp[side] = [round(float(x), 2) for r in csv.DictReader(open(kf, newline="")) for x in (r["x"], r["y"])] \
            if kf.is_file() else []
    pr = _read_scratch_csv(d / "pairs.csv")[k]
    c = [cls(r) for r in rows]
    trains = [int(r["train"]) for r in rows]
    inl = [t_ for t_, x in zip(trains, c) if x == "I"]
    return {"available": True, "source_file": "results/runs/" + f.relative_to(D.OUTPUT).as_posix(),
            "i": k, "j": k + 1, "t_i": float(run["t"][k]), "t_j": float(run["t"][k + 1]),
            "counts": {"keypoints_i": int(pr["keypoints_i"]), "keypoints_j": int(pr["keypoints_j"]),
                       "raw_matches": int(pr["raw_matches"]), "matches": int(pr["filtered_matches"]),
                       "pnp_correspondences": int(pr["correspondences_3d2d"]), "pnp_inliers": int(pr["pnp_inliers"]),
                       "distinct_kp_j_among_inliers": len(set(inl)),
                       "matches_sharing_a_kp_j": int(sum(v for v in np.unique(trains, return_counts=True)[1] if v > 1))},
            "solver_ok": ok, "rot_err_deg": _num(pr["rot_err_deg"]), "trans_err_m": _num(pr["trans_err_m"]),
            "kp_i": kp["i"], "kp_j": kp["j"],
            "m": {"ui": [float(r["u_i"]) for r in rows], "vi": [float(r["v_i"]) for r in rows],
                  "uj": [float(r["u_j"]) for r in rows], "vj": [float(r["v_j"]) for r in rows],
                  "kpj": trains, "cls": "".join(c)},
            "classification": "inlier / outlier = the scratch RANSAC's pnp_inlier flag of each PnP input; D = no "
                              "frame-i depth (not a PnP input); U = PnP failed (no classification)"}


def _reference_pair(s, k, tl_rows):
    r = tl_rows[k]
    pair = D.pair(s["run"], k)
    out = {"available": pair["export"]["available"], "export": pair["export"],
           "counts": {"matches": int(r["matches"]), "pnp_correspondences": int(r["pnp_correspondences"]),
                      "pnp_inliers": int(r["pnp_inliers"]), "icp_correspondences": int(r["icp_correspondences"]),
                      "icp_inliers": int(r["icp_inliers"])},
           "pnp_rot_err_deg": _num(r["pnp_rot_err_deg"]), "pnp_trans_err_m": _num(r["pnp_trans_err_m"]),
           "icp_rot_err_deg": _num(r["icp_rot_err_deg"]), "icp_trans_err_m": _num(r["icp_trans_err_m"]),
           "pnp_solver_returned": r.get("pnp_solver_returned"), "pnp_accepted": r.get("pnp_accepted"),
           "pnp_reject_reason": r.get("pnp_reject_reason") or None,
           "pnp_returned_rot_err_deg": _num(r.get("pnp_returned_rot_err_deg")),
           "pnp_returned_trans_err_m": _num(r.get("pnp_returned_trans_err_m")),
           "gt_motion_deg": _num(r["gt_motion_deg"]), "gt_motion_m": _num(r["gt_motion_m"]),
           "source_file": f"results/runs/{s['run']}/pairs.csv"}
    if pair["export"]["available"]:
        dd = pair["data"]
        out.update(i=dd["i"], j=dd["j"], t_i=dd["t_i"], t_j=dd["t_j"], kp_i=dd["kp_i"], kp_j=dd["kp_j"], m=dd["m"],
                   distinct_kp_j_among_inliers=pair["indicators"]["pnp_inliers_distinct_kp_j"],
                   matches_sharing_a_kp_j=pair["indicators"]["matches_sharing_a_kp_j"],
                   keypoints=[dd["counts"]["keypoints_i"], dd["counts"]["keypoints_j"]])
    return out


def pair_compare(reference_run, scratch_run, k):
    """The same pair k from a reference run and a scratch run with the same frame selection."""
    ref = get(f"{reference_run}::ref_pnp")
    scr = get(f"{scratch_run}::scratch_pnp")
    if ref["frames_key"] != scr["frames_key"]:
        raise SourceError("reference and scratch runs use different frame selections")
    rows = list(csv.DictReader(open(D.OUTPUT / reference_run / "pairs.csv", newline="")))
    if not 0 <= k < len(rows):
        raise KeyError(f"pair {k} outside 0..{len(rows) - 1}")
    return tm.to_json_safe({"pair": k, "frames_key": ref["frames_key"],
                            "reference": _reference_pair(ref, k, rows), "scratch": _scratch_pair(scr, k),
                            "labels": LABELS})


# ---------------------------------------------------------------------------------------------------------------
# OpenCV ORB correspondences (tests/tum/tum_orb_correspondences.cpp): a separate diagnostic path, compared pair by
# pair with the scratch pipeline's own correspondences on the SAME frames (frames.csv hash).
ORB_LABEL = ("OpenCV ORB", "cv::ORB::create(500) keypoints + descriptors (OpenCV defaults), brute-force Hamming, "
                           "distance <= max(2 d_min, 30); inlier / outlier by cv::solvePnPRansac (100 it, 8 px, 0.99) "
                           "on matches with frame-i depth; diagnostic only (no trajectory)")


def orb_runs():
    out = []
    for cmd in sorted(D.OUTPUT.rglob("command.txt")):
        d = cmd.parent
        if "archive" in d.relative_to(D.OUTPUT).parts or "tum_orb_correspondences" not in cmd.read_text():
            continue
        if not (d / "frames.csv").is_file():
            continue
        out.append({"run": d.relative_to(D.OUTPUT).as_posix(), "frames_key": frames_key(d / "frames.csv"),
                    "pairs": sorted(int(p.name.split("_")[1]) for p in d.glob("pair_*_*") if (p / "pair.json").is_file()),
                    "label": ORB_LABEL[0], "implementation": ORB_LABEL[1], "command": cmd.read_text().strip()})
    return out


def _frame_rows(d):
    with open(Path(d) / "frames.csv", newline="") as f:
        return list(csv.DictReader(f))


def _orb_pair(run, k):
    import json
    d = D.OUTPUT / run / f"pair_{k}_{k + 1}"
    if not (d / "pair.json").is_file() or not (d / "matches.csv").is_file():
        return {"available": False, "reason": f"results/runs/{run}/pair_{k}_{k + 1} not exported",
                "command": f"build/tum_orb_correspondences data/<sequence> --frames-csv results/runs/{run}/frames.csv "
                           f"--out <new dir> --pairs {k}"}
    meta = json.loads((d / "pair.json").read_text())
    rows = list(csv.DictReader(open(d / "matches.csv", newline="")))
    ok = meta["pnp_solver_returned"]

    def cls(r):
        if r["pnp_used"] != "1":
            return "D"
        if not ok:
            return "U"
        return "I" if r["pnp_inlier"] == "1" else "O"
    c = [cls(r) for r in rows]
    if sum(x == "I" for x in c) != meta["counts"]["pnp_inliers"]:
        raise SourceError(f"{d}: inlier rows disagree with pair.json")
    kp = {s: [round(float(x), 2) for r in csv.DictReader(open(d / f"keypoints_{s}.csv", newline="")) for x in (r["x"], r["y"])]
          for s in "ij"}
    return {"available": True, "source_file": "results/runs/" + (d / "matches.csv").relative_to(D.OUTPUT).as_posix(),
            "i": k, "j": k + 1, "t_i": float(meta["t_rgb_i"]), "t_j": float(meta["t_rgb_j"]),
            "rgb_i": meta["rgb_i"], "rgb_j": meta["rgb_j"], "counts": meta["counts"], "solver_ok": ok,
            "rot_err_deg": meta["pnp_rot_err_deg"], "trans_err_m": meta["pnp_trans_err_m"],
            "gt_motion_deg": meta["gt_motion_deg"], "gt_motion_m": meta["gt_motion_m"], "config": meta["config"],
            "kp_i": kp["i"], "kp_j": kp["j"],
            "m": {"ui": [float(r["u_i"]) for r in rows], "vi": [float(r["v_i"]) for r in rows],
                  "uj": [float(r["u_j"]) for r in rows], "vj": [float(r["v_j"]) for r in rows],
                  "kpj": [int(r["train"]) for r in rows], "cls": "".join(c)}}


def _with_counts(x):
    """Inlier / outlier / unclassified counts recomputed from the drawn class string (what the panel shows)."""
    if x.get("available"):
        c = x["m"]["cls"]
        x["drawn_counts"] = {"matches": len(c), "inliers": c.count("I"), "outliers": c.count("O"),
                             "no_depth": c.count("D"), "unclassified": c.count("U")}
    return x


def corr_compare(orb_run, scratch_run, k):
    """ORB and scratch-PnP correspondences of the same pair k; refused unless both use the same frames.csv."""
    orb = [r for r in orb_runs() if r["run"] == orb_run]
    scr = [s for s in discover() if s["run"] == scratch_run and s["family"] == "scratch"]
    if not orb:
        raise KeyError(f"unknown ORB correspondence run {orb_run!r}")
    if not scr:
        raise KeyError(f"unknown scratch run {scratch_run!r}")
    if orb[0]["frames_key"] != scr[0]["frames_key"]:
        raise SourceError("the ORB and scratch runs use different frames (frames.csv differs): no comparison")
    fr = _frame_rows(D.OUTPUT / orb_run)
    if not 0 <= k < len(fr) - 1:
        raise KeyError(f"pair {k} outside 0..{len(fr) - 2}")
    o, s = _with_counts(_orb_pair(orb_run, k)), _with_counts(_scratch_pair(scr[0], k))
    frame = {"pair": k, "frame_i": k, "frame_j": k + 1, "t_rgb_i": float(fr[k]["t_rgb"]), "t_rgb_j": float(fr[k + 1]["t_rgb"]),
             "rgb_i": fr[k]["rgb"], "rgb_j": fr[k + 1]["rgb"]}
    for x in (o, s):  # both panels must show exactly this frame pair
        if x.get("available") and (x["i"], x["j"]) != (k, k + 1) or \
                (x.get("available") and abs(x["t_i"] - frame["t_rgb_i"]) > 1e-6 or x.get("available") and abs(x["t_j"] - frame["t_rgb_j"]) > 1e-6):
            raise SourceError(f"pair {k}: a panel's frames differ from frames.csv")
    ref = [r for r in configurations() if r["frames_key"] == orb[0]["frames_key"]]
    image_run = (ref[0]["reference_runs"] or [None])[0] if ref else None
    return tm.to_json_safe({"pair": k, "frames_key": orb[0]["frames_key"], "frames": frame, "n_pairs": len(fr) - 1,
                            "image_run": image_run, "orb": o, "scratch": s,
                            "labels": {"orb": ORB_LABEL, "scratch": LABELS["scratch_pnp"]}})


def corr_index():
    """ORB / scratch run pairs with identical frames, and notable pairs from each method's own files."""
    out = []
    scr = [s for s in discover() if s["family"] == "scratch"]
    for o in orb_runs():
        for s in scr:
            if s["frames_key"] != o["frames_key"]:
                continue
            notable = {}
            try:
                run = load_scratch_run(D.OUTPUT / s["run"])
                e = np.nan_to_num(run["scratch_rot_err_deg"], nan=-1)
                for k in np.argsort(-e)[:3]:
                    notable.setdefault(int(k), []).append(f"scratch PnP error {e[k]:.2f} deg")
            except SourceError:
                pass
            with open(D.OUTPUT / o["run"] / "summary.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            er = sorted(((float(r["pnp_rot_err_deg"]) if r["pnp_rot_err_deg"] not in ("", "null") else 1e9, int(r["pair"]))
                         for r in rows), reverse=True)[:3]
            for e_, k in er:
                notable.setdefault(k, []).append("ORB PnP solver failed" if e_ == 1e9 else f"ORB PnP error {e_:.2f} deg")
            for r in [c for c in configurations() if c["frames_key"] == o["frames_key"]][:1]:
                for rr in r["reference_runs"]:
                    with open(D.OUTPUT / rr / "pairs.csv", newline="") as f:
                        for p in csv.DictReader(f):
                            if p.get("pnp_accepted") == "0" and "reference PnP rejected" not in notable.get(int(p["pair"]), []):
                                notable.setdefault(int(p["pair"]), []).append("reference PnP rejected")
            out.append({"orb_run": o["run"], "scratch_run": s["run"], "frames_key": o["frames_key"],
                        "orb_pairs": o["pairs"], "notable": {str(k): v for k, v in sorted(notable.items())}})
    return out
