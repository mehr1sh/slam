"""Independent RGB-D reconstructions of one TUM run, placed with GT, PnP or ICP poses (NumPy only).

The three reconstructions share EVERYTHING except the pose that places each frame:
  sampling   frames 0, every <every>-th and the last of the run; in each, the pixels (offset + step a,
             offset + step b) whose raw depth is > 0. Decided in the image only (no world-space filtering),
             so the sampled pixels are identical in all modes.
  camera     X_cam = ((u - cx) z / fx, (v - cy) z / fy, z), z = raw / depth_factor (official intrinsics,
             no undistortion: as the evaluator).
  placement  X_world = R_wc X_cam + t_wc with the T_wc of the chosen pose source for THAT frame
             (trajectory.csv columns gt_*, pnp_* or icp_*; quaternion order qx qy qz qw).
  exclusion  an estimated mode drops a frame whose pose is HELD (copied from the previous frame because the
             pair failed or its pose was rejected): that frame has no estimate of its own. Decided from the
             evaluator's own pair status only (never from ground truth). GT mode never excludes frames.
Nothing is aligned, smoothed, interpolated or corrected; ground truth is not read in an estimated mode.
"""

import hashlib

import numpy as np

MODES = ("gt", "pnp", "icp")


def frame_selection(n, every, include=()):
    """Frames 0, every <every>-th, the last, and `include` (the same list for every mode)."""
    return sorted(set(range(0, n, every)) | {n - 1} | {int(k) for k in include if 0 <= int(k) < n})


def sample_frame(depth_raw, rgb, K, step, offset=None):
    """Camera-frame points of one frame. Returns xyz (N,3), colours (N,3) uint8, pixel indices (N,2), valid count."""
    off = step // 2 if offset is None else offset
    v, u = np.mgrid[off:depth_raw.shape[0]:step, off:depth_raw.shape[1]:step]
    raw = depth_raw[v, u]
    ok = raw > 0
    z = raw[ok].astype(np.float64) / K["depth_factor"]
    u, v = u[ok], v[ok]
    xyz = np.c_[(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z]
    return xyz, rgb[v, u], np.c_[u, v], int((depth_raw > 0).sum())


def place(xyz_cam, R_wc, t_wc):
    """X_world = R_wc X_cam + t_wc (T_wc maps camera coordinates to world coordinates)."""
    return xyz_cam @ np.asarray(R_wc).T + np.asarray(t_wc)


def pose_source(run, mode):
    """(R (n,3,3), t (n,3), held (n,) bool) of one pose source. Estimated modes never touch the gt_* arrays."""
    if mode not in MODES:
        raise ValueError(f"unknown pose source {mode!r}")
    held = np.zeros(run["n"], bool) if mode == "gt" else np.asarray(run[f"{mode}_held"], bool)
    return np.asarray(run[f"{mode}_R"]), np.asarray(run[f"{mode}_p"]), held


def reconstruct(run, read_frame, K, every, step, modes=MODES, include=()):
    """read_frame(k) -> (depth_raw, rgb). Returns {mode: dict(xyz, rgb, frames_used, frames_skipped, ...)} and
    the shared per-frame sampling record (identical for every mode by construction)."""
    sel = frame_selection(run["n"], every, include)
    samples = {}
    for k in sel:  # sampled once, shared by every mode
        depth, rgb = read_frame(k)
        xyz, col, px, n_valid = sample_frame(depth, rgb, K, step)
        samples[k] = dict(xyz=xyz, rgb=col, px=px, valid_depth_pixels=n_valid,
                          px_sha256=hashlib.sha256(np.ascontiguousarray(px, np.int32).tobytes()).hexdigest())
    out = {}
    for mode in modes:
        R, t, held = pose_source(run, mode)
        P, C, used, skipped = [], [], [], []
        for k in sel:
            if held[k]:
                skipped.append({"frame": int(k), "t_rgb": float(run["t"][k]),
                                "reason": f"{mode} pose held: pair {k - 1}->{k} failed or its pose was rejected "
                                          "(no estimate of its own)"})
                continue
            P.append(place(samples[k]["xyz"], R[k], t[k]))
            C.append(samples[k]["rgb"])
            used.append(int(k))
        out[mode] = dict(xyz=np.concatenate(P) if P else np.zeros((0, 3)),
                         rgb=np.concatenate(C) if C else np.zeros((0, 3), np.uint8),
                         frames_used=used, frames_skipped=skipped,
                         points=int(sum(len(samples[k]["xyz"]) for k in used)),
                         valid_depth_pixels=int(sum(samples[k]["valid_depth_pixels"] for k in used)),
                         per_frame_points={int(k): int(len(samples[k]["xyz"])) for k in used})
    return out, sel, samples


def compare(recon, samples, ref="gt", voxel=0.01):
    """Numbers that do not depend on appearance:
    - displacement: for frames placed in both modes, distance of each sampled point from its position in the
      reference reconstruction (same pixel, same depth: only the pose differs);
    - occupied voxels: distinct voxel_m cells covered by the cloud (doubled / smeared surfaces occupy more cells
      for the same number of points); per point, so modes with excluded frames stay comparable."""
    res = {}
    for mode, r in recon.items():
        occ = len(np.unique(np.floor(r["xyz"] / voxel).astype(np.int64), axis=0)) if len(r["xyz"]) else 0
        d = {"points": r["points"], "occupied_voxels": occ, "voxel_m": voxel,
             "occupied_voxels_per_1000_points": 1000.0 * occ / max(r["points"], 1)}
        if mode != ref:
            common = [k for k in r["frames_used"] if k in recon[ref]["frames_used"]]
            off_m, off_r = 0, 0
            disp = []
            for k in recon[ref]["frames_used"]:
                n = len(samples[k]["xyz"])
                if k in r["frames_used"]:
                    disp.append(np.linalg.norm(r["xyz"][off_m:off_m + n] - recon[ref]["xyz"][off_r:off_r + n], axis=1))
                    off_m += n
                off_r += n
            disp = np.concatenate(disp) if disp else np.zeros(0)
            d.update(frames_compared=len(common),
                     displacement_vs_gt_m=({"median": float(np.median(disp)), "p90": float(np.percentile(disp, 90)),
                                            "max": float(disp.max()), "fraction_over_5cm": float((disp > 0.05).mean())}
                                           if len(disp) else None))
        res[mode] = d
    return res
