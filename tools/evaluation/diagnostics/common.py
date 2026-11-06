"""Shared loading, validation and drawing helpers for the diagnostic figures.

Input: a tum_trajectory_eval output directory (frames.csv, pairs.csv, trajectory.csv) and,
for per-pair views, its diagnostics/pair_<i>_<j>/ export (--export-pairs). Conventions
(see pair.json "conventions"):
  T_j_from_i   X_j = R X_i + t, camera optical axes (x right, y down, z forward), metres
  T_wc         X_world = R X_cam + t (TUM ground truth and trajectory.csv)
  ICP native   T_i_from_j: p_i = R p_j + t
Nothing here estimates anything: every class, residual and transform is the exported one;
validate_pair() recomputes the residuals from the exported points and transforms and
reports any disagreement instead of correcting it.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "tools" / "pipeline_analysis"))
import tum_metrics as tm  # noqa: E402
from png_reader import read_png  # noqa: E402

# One colour per trajectory source in every view (as plot_tum_results.py).
METHOD_COLOR = {"gt": "#2ca02c", "pnp": "#ff7f0e", "icp": "#9467bd"}
METHOD_LABEL = {"gt": "ground truth", "pnp": "PnP (RANSAC)", "icp": "ICP (robust)"}
# One colour per correspondence class; deliberately not the method colours.
CLASS_STYLE = {
    "inlier": dict(color="#0072B2", lw=1.3, alpha=0.95, ls="-"),
    "outlier": dict(color="#CC3311", lw=0.6, alpha=0.45, ls="-"),
    "rejected": dict(color="#CC3311", lw=0.6, alpha=0.45, ls="-"),
    "no_depth_i": dict(color="#7f7f7f", lw=0.6, alpha=0.6, ls=":"),
    "no_depth_j": dict(color="#bbbbbb", lw=0.6, alpha=0.6, ls=":"),
    "unclassified": dict(color="#000000", lw=0.6, alpha=0.6, ls="--"),
}
PNP_CLASSES = ("inlier", "outlier", "no_depth_i", "unclassified")
ICP_CLASSES = ("inlier", "rejected", "no_depth_i", "no_depth_j", "unclassified")
MATCH_COLS = ["match", "kp_i", "kp_j", "hamming", "u_i", "v_i", "u_j", "v_j", "depth_raw_i", "depth_raw_j",
              "depth_m_i", "depth_m_j", "pnp_corr", "pnp_class", "X_i", "Y_i", "Z_i", "pnp_proj_u", "pnp_proj_v",
              "pnp_z_j", "pnp_res_px", "gt_proj_u", "gt_proj_v", "gt_z_j", "gt_res_px", "icp_corr", "icp_class",
              "Xj_x", "Xj_y", "Xj_z", "icp_res_m", "icp_ransac_res_m", "icp_gt_res_m"]
STR_COLS = {"pnp_class", "icp_class"}


class ExportError(ValueError):
    """The diagnostic export is missing, malformed or inconsistent."""


def _csv(path):
    if not path.is_file():
        raise ExportError(f"{path}: missing (run tum_trajectory_eval with --export-pairs)")
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        rows = list(r)
    return header, rows


def load_pair(pair_dir):
    """pair.json + matches.csv + keypoints_{i,j}.csv -> dict of arrays (NaN for empty cells)."""
    d = Path(pair_dir)
    if not (d / "pair.json").is_file():
        raise ExportError(f"{d / 'pair.json'}: missing (run tum_trajectory_eval with --export-pairs)")
    meta = json.loads((d / "pair.json").read_text())
    header, rows = _csv(d / "matches.csv")
    if header != MATCH_COLS:
        missing = [c for c in MATCH_COLS if c not in (header or [])]
        raise ExportError(f"{d / 'matches.csv'}: unexpected columns (missing {missing})")
    p = {"dir": d, "meta": meta, "n": len(rows)}
    for k, col in enumerate(MATCH_COLS):
        vals = [r[k] for r in rows]
        if col in STR_COLS:
            p[col] = np.array(vals, dtype=object)
        else:
            try:
                p[col] = np.array([float(v) if v != "" else np.nan for v in vals])
            except ValueError as e:
                raise ExportError(f"{d / 'matches.csv'}: column {col}: {e}")
    for side in ("i", "j"):
        h, kr = _csv(d / f"keypoints_{side}.csv")
        if h != ["kp", "x", "y", "angle_deg", "response"]:
            raise ExportError(f"{d / f'keypoints_{side}.csv'}: unexpected columns {h}")
        a = np.array([[float(v) for v in r] for r in kr]).reshape(-1, 5)
        p[f"kp_{side}_xy"] = a[:, 1:3]
        p[f"kp_{side}_response"] = a[:, 4]
    for key in ("T_j_from_i_gt", "T_j_from_i_pnp", "T_j_from_i_icp", "T_i_from_j_icp_native",
                "T_i_from_j_icp_ransac_native", "T_wc_gt_i", "T_wc_gt_j"):
        p[key] = None if meta.get(key) is None else np.array(meta[key], dtype=float)
    return p


def rot_deg(R):
    return tm.rot_angle_deg(R)


def project(T, X, K):
    """Pinhole projection of camera-i points X (N,3) with T_j_from_i; no distortion (as the estimator)."""
    Y = X @ T[:3, :3].T + T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.c_[K["fx"] * Y[:, 0] / Y[:, 2] + K["cx"], K["fy"] * Y[:, 1] / Y[:, 2] + K["cy"]], Y[:, 2]


def intrinsics(p):
    return p["meta"]["intrinsics"]


def validate_pair(p, pairs_row=None, tol_px=1e-3, tol_m=1e-6):
    """Consistency checks of one export; returns a list of problems (empty = consistent)."""
    pr, m, K = [], p["meta"], intrinsics(p)
    c = m["counts"]
    n_pnp = int((p["pnp_corr"] >= 0).sum())
    n_icp = int((p["icp_corr"] >= 0).sum())
    if p["n"] != c["matches"] or n_pnp != c["pnp_correspondences"] or n_icp != c["icp_correspondences"]:
        pr.append(f"row counts {p['n']}/{n_pnp}/{n_icp} differ from pair.json counts")
    if len(p["kp_i_xy"]) != c["keypoints_i"] or len(p["kp_j_xy"]) != c["keypoints_j"]:
        pr.append("keypoint file lengths differ from pair.json counts")
    if not set(p["pnp_class"]) <= set(PNP_CLASSES) or not set(p["icp_class"]) <= set(ICP_CLASSES):
        pr.append("unknown class labels")
    if not np.array_equal(p["pnp_corr"][p["pnp_corr"] >= 0], np.arange(n_pnp)):
        pr.append("pnp_corr is not 0..n-1 in match order")
    inl = sorted(int(x) for x in p["pnp_corr"][p["pnp_class"] == "inlier"])
    if m["status"]["pnp_solver_returned"] and inl != sorted(m["pnp_inlier_indices"]):
        pr.append("PnP inlier rows differ from pair.json pnp_inlier_indices")
    if (p["pnp_class"] == "no_depth_i").sum() != (p["depth_raw_i"] == 0).sum():
        pr.append("no_depth_i rows differ from depth_raw_i == 0")
    if np.abs(p["depth_m_i"] - p["depth_raw_i"] / K["depth_factor"]).max(initial=0) > 1e-6:
        pr.append("depth_m_i != depth_raw_i / depth_factor")
    # back-projection (float pixel coordinates, depth from the truncated pixel, float32 in the estimator)
    s = p["pnp_corr"] >= 0
    X = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][s]
    z = p["depth_m_i"][s]
    Xr = np.c_[(p["u_i"][s] - K["cx"]) / K["fx"] * z, (p["v_i"][s] - K["cy"]) / K["fy"] * z, z]
    if len(X) and np.abs(X - Xr).max() > 1e-5:
        pr.append(f"3D points differ from the back-projection by {np.abs(X - Xr).max():.2e} m")
    uv_j = np.c_[p["u_j"], p["v_j"]][s]
    for key, col in (("T_j_from_i_pnp", "pnp_res_px"), ("T_j_from_i_gt", "gt_res_px")):
        if p[key] is None:
            if np.isfinite(p[col][s]).any():
                pr.append(f"{col} present without {key}")
            continue
        uv, _ = project(p[key], X, K)
        err = np.abs(np.linalg.norm(uv - uv_j, axis=1) - p[col][s])
        if len(err) and np.nanmax(err) > tol_px:
            pr.append(f"{col} differs from the recomputed residual by {np.nanmax(err):.2e} px")
    # transform directions
    if p["T_wc_gt_i"] is not None:
        G = np.linalg.inv(p["T_wc_gt_j"]) @ p["T_wc_gt_i"]
        if np.abs(G - p["T_j_from_i_gt"]).max() > 1e-8:
            pr.append("T_j_from_i_gt != inv(T_wc_j) T_wc_i")
    if p["T_j_from_i_icp"] is not None and p["T_i_from_j_icp_native"] is not None:
        if np.abs(np.linalg.inv(p["T_i_from_j_icp_native"]) - p["T_j_from_i_icp"]).max() > 1e-8:
            pr.append("T_j_from_i_icp != inv(T_i_from_j_icp_native)")
    # ICP: residuals and the 1 cm classification rule of EstimateRigidRansac
    q = p["icp_corr"] >= 0
    Pi = np.c_[p["X_i"], p["Y_i"], p["Z_i"]][q]
    Pj = np.c_[p["Xj_x"], p["Xj_y"], p["Xj_z"]][q]
    for key, col in (("T_i_from_j_icp_native", "icp_res_m"), ("T_i_from_j_icp_ransac_native", "icp_ransac_res_m")):
        if p[key] is None or not q.any():
            continue
        r = np.linalg.norm(Pi - (Pj @ p[key][:3, :3].T + p[key][:3, 3]), axis=1)
        if np.abs(r - p[col][q]).max() > tol_m:
            pr.append(f"{col} differs from the recomputed residual by {np.abs(r - p[col][q]).max():.2e} m")
    thr = m["config"]["icp"]["threshold_m"]
    if m["status"]["icp_ransac_ok"] and q.any():
        rule = p["icp_ransac_res_m"][q] < thr
        lab = p["icp_class"][q] == "inlier"
        # the export's residual is recomputed in double from the same floats: allow ties at the threshold only
        bad = (rule != lab) & (np.abs(p["icp_ransac_res_m"][q] - thr) > 1e-9)
        if bad.any():
            pr.append(f"{int(bad.sum())} ICP labels disagree with residual < {thr} m under the RANSAC model")
        if int(lab.sum()) != c["icp_inliers"]:
            pr.append("ICP inlier rows differ from pair.json count")
    if pairs_row is not None:
        for key, ck in (("matches", "matches"), ("pnp_correspondences", "pnp_correspondences"),
                        ("pnp_inliers", "pnp_inliers"), ("icp_correspondences", "icp_correspondences"),
                        ("icp_inliers", "icp_inliers")):
            if int(float(pairs_row[key])) != c[ck]:
                pr.append(f"pairs.csv {key} {pairs_row[key]} != export {c[ck]}")
        G = p["T_j_from_i_gt"]
        for meth in ("pnp", "icp"):
            T = p[f"T_j_from_i_{meth}"]
            if T is None:
                continue
            re = rot_deg(G[:3, :3].T @ T[:3, :3])
            te = np.linalg.norm(G[:3, 3] - T[:3, 3])
            # the export holds the RETURNED pose; with the validity check pairs.csv keeps its error separately
            kr, kt = ((f"pnp_returned_rot_err_deg", f"pnp_returned_trans_err_m") if meth == "pnp" and
                      "pnp_returned_rot_err_deg" in pairs_row else (f"{meth}_rot_err_deg", f"{meth}_trans_err_m"))
            if abs(re - float(pairs_row[kr])) > 1e-3 or abs(te - float(pairs_row[kt])) > 1e-5:
                pr.append(f"{meth} errors from the exported transforms ({re:.4f} deg, {te:.5f} m) differ from pairs.csv")
    return pr


def pairs_rows(run_dir):
    with open(Path(run_dir) / "pairs.csv", newline="") as f:
        return {int(r["pair"]): r for r in csv.DictReader(f)}


def read_rgb(dataset, rel):
    img = read_png(Path(dataset) / rel)
    return img[..., :3] if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)


def read_depth_raw(dataset, rel):
    d = read_png(Path(dataset) / rel)
    if d.ndim != 2:
        raise ExportError(f"{rel}: depth image is not single-channel")
    return d


def backproject(depth_raw, K, step=1, offset=0):
    """Valid pixels on a regular grid (every step-th row/column from offset) -> camera-frame points (N,3), pixels (N,2)."""
    v, u = np.mgrid[offset:depth_raw.shape[0]:step, offset:depth_raw.shape[1]:step]
    raw = depth_raw[v, u]
    ok = raw > 0
    z = raw[ok].astype(float) / K["depth_factor"]
    u, v = u[ok], v[ok]
    return np.c_[(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], np.c_[u, v]


def frustum_lines(T_wc, K, scale):
    """Line segments (M,2,3) of a camera frustum at pose T_wc, depth `scale` metres."""
    w, h = K.get("width", 640), K.get("height", 480)
    cs = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.c_[(cs[:, 0] - K["cx"]) / K["fx"], (cs[:, 1] - K["cy"]) / K["fy"], np.ones(4)] * scale
    pts = rays @ T_wc[:3, :3].T + T_wc[:3, 3]
    c = T_wc[:3, 3]
    segs = [(c, p) for p in pts] + [(pts[k], pts[(k + 1) % 4]) for k in range(4)]
    return np.array(segs)


# The evaluator's official intrinsics per sensor (src/dataset/tum.cpp OfficialIntrinsics).
OFFICIAL_INTRINSICS = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
                       "freiburg3": (535.4, 539.2, 320.1, 247.6)}


def run_intrinsics(run_dir, dataset=None):
    """Intrinsics of a run: from an export's pair.json, else from the sequence name (--dataset or the
    runner's config.json), as the evaluator chooses them. None if neither is available."""
    run_dir = Path(run_dir)
    for pj in sorted((run_dir / "diagnostics").glob("pair_*_*/pair.json")):
        return json.loads(pj.read_text())["intrinsics"]
    name = Path(dataset).name if dataset else None
    if name is None and (run_dir / "config.json").is_file():
        name = Path(json.loads((run_dir / "config.json").read_text()).get("dataset", "")).name
    for sensor, (fx, fy, cx, cy) in OFFICIAL_INTRINSICS.items():
        if name and sensor in name:
            return {"sensor": sensor, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": 640, "height": 480,
                    "depth_factor": 5000.0}
    return None


def se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def save(fig, out_dir, name, made=None, dpi=170):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    if made is not None:
        made.append(str(out_dir / f"{name}.png"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(tm.to_json_safe(obj), indent=2) + "\n")


def pixel_figure(width_px, height_px, dpi=100):
    """A figure whose data coordinates are output pixels (x right, y down): images drawn 1:1, no resampling."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_axis_off()
    return fig, ax


def save_png(fig, path, made=None, dpi=100):
    """PNG only, exact pixel size (no tight bbox)."""
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    if made is not None:
        made.append(str(path))


def sample_rows(rows, cap):
    """Deterministic subsample: every k-th row (in match order) so that at most `cap` remain."""
    rows = np.asarray(rows, int)
    if cap is None or len(rows) <= cap:
        return rows
    return rows[:: int(np.ceil(len(rows) / cap))]
