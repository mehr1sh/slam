"""Compare two exports of visualization/export_scene_state.py (an original and a rebuilt scene).

    python3 visualization/compare_scene_state.py OLD.json NEW.json [--tol 1e-9]
        [--expect PROP_PATH ...] [--expect-text NAME ...]

Every category is checked and reported separately; none is skipped:
  meta            frame range, fps, engine, resolution, render path, active camera (exact)
  object set      same object names, types and parents (exact)
  camera poses    for every *_Animated_Camera at every frame: position difference (m), orientation difference
                  (deg, from the world rotation), viewing-direction difference (deg, camera -Z axis)
  all objects     every world-matrix element at every frame (covers the Bunny, WorldRoot, error vectors, lights)
  bunny           Bunny / Bunny_Root: scale, dimensions, mesh vertex count and vertex-coordinate SHA-256
  trajectories    every control point of every *_Trajectory curve
  cameras         intrinsics of every camera object
  visibility      hide_viewport / hide_render of every object at every frame (exact)
  scene props     the slam_* custom properties (exact, numbers within --tol)
  texts           the embedded text blocks (exact)
  runtime         both scenes' visualization scripts register (slam_hud_panel); sidebar settings scene.slam_viz
                  equal; every frame re-evaluated with the frame-change handler active: matrices, curve points,
                  visibility (what a user sees)
Numeric values pass when |old - new| <= --tol (default 1e-9); everything else must be identical.
--expect PROP_PATH (e.g. slam_scratch_source or slam_corrected_source.trajectory) and --expect-text NAME list
differences that are intended (provenance paths, the autoload script); they are still printed with both
values, and they fail the comparison if they turn out to be identical (the change did not happen).
Exit status 0 only if every category passes.
"""
import argparse
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("old")
ap.add_argument("new")
ap.add_argument("--tol", type=float, default=1e-9)
ap.add_argument("--expect", action="append", default=[])
ap.add_argument("--expect-text", action="append", default=[])
a = ap.parse_args()
A, B = json.load(open(a.old)), json.load(open(a.new))
results = []  # (category, ok, detail)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def diff_values(x, y, path, out):
    """Collect (path, x, y, |diff| or None) for every differing leaf."""
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append((f"{path}.{k}", x.get(k, "<missing>"), y.get(k, "<missing>"), None))
            else:
                diff_values(x[k], y[k], f"{path}.{k}", out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path, f"<{len(x)} items>", f"<{len(y)} items>", None))
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                diff_values(u, v, f"{path}[{i}]", out)
    elif x != y:
        nx, ny = num(x), num(y)
        if isinstance(x, str) and isinstance(y, str) and nx is not None and ny is not None:
            out.append((path, x, y, abs(nx - ny)))
        else:
            out.append((path, x, y, None))


def numeric_check(category, pairs_iter):
    worst, bad, n = 0.0, [], 0
    for path, x, y in pairs_iter:
        n += 1
        d = []
        diff_values(x, y, path, d)
        for p, u, v, dd in d:
            if dd is None:
                bad.append((p, u, v))
            else:
                worst = max(worst, dd)
                if dd > a.tol:
                    bad.append((p, u, v))
    results.append((category, not bad and n > 0, f"{n} items compared, max |diff| {worst:.3e}, "
                    f"{len(bad)} over tolerance" + "".join(f"\n      {p}: {u} -> {v}" for p, u, v in bad[:8])))


def mat(m):
    v = [float(x) for x in m]
    return [v[0:4], v[4:8], v[8:12], v[12:16]]


def rot(M):
    cols = [[M[i][j] for i in range(3)] for j in range(3)]
    cols = [[c / math.sqrt(sum(t * t for t in col)) for c in col] for col in cols]
    return [[cols[j][i] for j in range(3)] for i in range(3)]  # rows


def angle_between(u, v):
    """Angle between two vectors (deg); atan2 form, accurate near 0."""
    cx = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return math.degrees(math.atan2(math.sqrt(sum(t * t for t in cx)), sum(p * q for p, q in zip(u, v))))


def rotation_angle(R1, R2):
    """Angle of R1^T R2 (deg) from the chord ||R1 - R2||_F = 2*sqrt(2)*sin(theta/2); accurate near 0."""
    chord = math.sqrt(sum((R1[i][j] - R2[i][j]) ** 2 for i in range(3) for j in range(3)))
    return math.degrees(2 * math.asin(min(1.0, chord / (2 * math.sqrt(2)))))


# meta, object set
mdiff = []
diff_values(A["meta"], B["meta"], "meta", mdiff)
mdiff = [d for d in mdiff if not d[0].startswith(("meta.blender", "meta.file_version"))]  # informational below
results.append(("meta", not mdiff, "identical" if not mdiff else "; ".join(f"{p}: {u} -> {v}" for p, u, v, _ in mdiff)))
results.append(("meta (informational)", True, f"Blender {A['meta']['blender']} file {A['meta']['file_version']} -> "
                f"Blender {B['meta']['blender']} file {B['meta']['file_version']}"))
sig = lambda S: {k: (o["type"], o["parent"]) for k, o in S["objects"].items()}
results.append(("object set", sig(A) == sig(B), f"{len(A['objects'])} vs {len(B['objects'])} objects"
                + ("" if sig(A) == sig(B) else f"; only old {sorted(set(sig(A)) - set(sig(B)))}, only new {sorted(set(sig(B)) - set(sig(A)))}")))
frames = sorted(set(A["frames"]) | set(B["frames"]), key=int)
results.append(("frame set", set(A["frames"]) == set(B["frames"]), f"{len(A['frames'])} vs {len(B['frames'])} frames"))

# camera poses
cams = sorted(k for k in A["objects"] if k.endswith("_Animated_Camera"))
lines, ok_all = [], bool(cams)
for c in cams:
    wp = wr = wd = 0.0
    for f in frames:
        if c not in A["frames"].get(f, {}) or c not in B["frames"].get(f, {}):
            ok_all = False
            continue
        M1, M2 = mat(A["frames"][f][c]["matrix"]), mat(B["frames"][f][c]["matrix"])
        wp = max(wp, math.sqrt(sum((M1[i][3] - M2[i][3]) ** 2 for i in range(3))))
        R1, R2 = rot(M1), rot(M2)
        wr = max(wr, rotation_angle(R1, R2))
        wd = max(wd, angle_between([-R1[i][2] for i in range(3)], [-R2[i][2] for i in range(3)]))
    ok = wp <= a.tol and wr <= 1e-6 and wd <= 1e-6
    ok_all &= ok
    lines.append(f"      {c:28s} position {wp:.3e} m, orientation {wr:.3e} deg, view direction {wd:.3e} deg")
results.append(("camera poses (every frame)", ok_all, f"{len(cams)} cameras x {len(frames)} frames "
                f"(tolerance {a.tol:g} m, 1e-6 deg)\n" + "\n".join(lines)))

# all objects, every frame
numeric_check("all object world matrices (every frame)",
              ((f"frame {f}/{o}", A["frames"][f][o]["matrix"], B["frames"][f][o]["matrix"])
               for f in frames for o in A["objects"] if o in A["frames"].get(f, {}) and o in B["frames"].get(f, {})))
# bunny
bun = [o for o in ("Bunny", "Bunny_Root") if o in A["objects"]]
f0 = str(min(int(f) for f in A["frames"]))
numeric_check("bunny position, scale, dimensions, mesh",
              ((o, {**{k: A["objects"][o].get(k) for k in ("scale", "dimensions", "mesh")},
                    "world_matrix_frame0": A["frames"][f0][o]["matrix"]},
                {**{k: B["objects"].get(o, {}).get(k) for k in ("scale", "dimensions", "mesh")},
                 "world_matrix_frame0": B["frames"].get(f0, {}).get(o, {}).get("matrix")}) for o in bun))
# trajectories
numeric_check("trajectory curves (control points)",
              ((o, A["objects"][o].get("curve_points"), B["objects"].get(o, {}).get("curve_points"))
               for o in A["objects"] if A["objects"][o]["type"] == "CURVE"))
# camera intrinsics, all objects' scale/dimensions
numeric_check("camera intrinsics", ((o, A["objects"][o].get("camera"), B["objects"].get(o, {}).get("camera"))
                                    for o in A["objects"] if A["objects"][o]["type"] == "CAMERA"))
numeric_check("object scale and dimensions", ((o, [A["objects"][o]["scale"], A["objects"][o]["dimensions"]],
                                               [B["objects"].get(o, {}).get("scale"), B["objects"].get(o, {}).get("dimensions")])
                                              for o in A["objects"]))
# visibility
vis_bad = [(f, o) for f in frames for o in A["objects"]
           if (A["frames"].get(f, {}).get(o, {}).get("hide_viewport"), A["frames"].get(f, {}).get(o, {}).get("hide_render"))
           != (B["frames"].get(f, {}).get(o, {}).get("hide_viewport"), B["frames"].get(f, {}).get(o, {}).get("hide_render"))]
results.append(("visibility flags (every frame)", not vis_bad, f"{len(vis_bad)} differences" + (f": {vis_bad[:6]}" if vis_bad else "")))

# scene properties
pd = []
diff_values(A["props"], B["props"], "props", pd)
expected = {"props." + e for e in a.expect}
# Blender 5 stores bpy.props groups (scene.slam_viz) outside scene.keys(); when one side lacks it there, its
# values are compared through the API in the runtime category below (stated, not silently skipped).
moved = [d for d in pd if d[0] == "props.slam_viz" and "<missing>" in (d[1], d[2])
         and A.get("runtime", {}).get("slam_viz") is not None and B.get("runtime", {}).get("slam_viz") is not None]
for d in moved:
    results.append(("scene properties: slam_viz storage", True, "absent from scene.keys() on one side (Blender 5 system "
                    "property storage); values compared in 'sidebar settings scene.slam_viz'"))
unexpected = [d for d in pd if d[0] not in expected and d not in moved and not (d[3] is not None and d[3] <= a.tol)]
seen_expected = {d[0] for d in pd if d[0] in expected}
results.append(("scene properties", not unexpected, f"{len(pd)} differing leaves; unexpected {len(unexpected)}"
                + "".join(f"\n      UNEXPECTED {p}: {u} -> {v}" for p, u, v, _ in unexpected[:8])))
for e in sorted(expected):
    d = next((d for d in pd if d[0] == e), None)
    results.append((f"intended change {e}", d is not None, f"{d[1]!r} -> {d[2]!r}" if d else "NOT changed"))
# runtime: the scene's visualization scripts registered (export pass 2)
ra, rb = A.get("runtime", {}), B.get("runtime", {})
results.append(("visualization scripts load (slam_hud_panel.register)", ra.get("registered") is True and rb.get("registered") is True,
                f"old {ra.get('registered')} {ra.get('error', '')}, new {rb.get('registered')} {rb.get('error', '')}".strip()))
vd = []
diff_values(ra.get("slam_viz"), rb.get("slam_viz"), "slam_viz", vd)
results.append(("sidebar settings scene.slam_viz", ra.get("slam_viz") is not None and not vd,
                "identical" if ra.get("slam_viz") is not None and not vd else f"{vd[:6]}"))
LA, LB = A.get("frames_live", {}), B.get("frames_live", {})
results.append(("live frames present", bool(LA) and set(LA) == set(LB), f"{len(LA)} vs {len(LB)} frames"))
numeric_check("live (handler active): world matrices and curve points",
              ((f"live {f}/{o}", {k: v for k, v in LA[f][o].items() if not k.startswith("hide")},
                {k: v for k, v in LB.get(f, {}).get(o, {}).items() if not k.startswith("hide")})
               for f in sorted(LA, key=int) for o in LA[f]))
lv = [(f, o) for f in LA for o in LA[f]
      if (LA[f][o]["hide_viewport"], LA[f][o]["hide_render"]) != (LB.get(f, {}).get(o, {}).get("hide_viewport"), LB.get(f, {}).get(o, {}).get("hide_render"))]
results.append(("live (handler active): visibility flags", bool(LA) and not lv, f"{len(lv)} differences" + (f": {lv[:6]}" if lv else "")))
# texts
td = sorted(k for k in set(A["texts"]) | set(B["texts"]) if A["texts"].get(k) != B["texts"].get(k))
results.append(("text blocks", all(k in a.expect_text for k in td), f"differing: {td or 'none'}"))
for k in a.expect_text:
    results.append((f"intended change text {k}", k in td, "changed" if k in td else "NOT changed"))

ok = True
for cat, good, detail in results:
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {cat}: {detail}")
print("RESULT:", "EQUIVALENT (all categories pass)" if ok else "NOT EQUIVALENT")
sys.exit(0 if ok else 1)
