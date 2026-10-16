# Usage (repository root): blender -b visualization/scenes/bunny_slam_demo.blend --python docs/migration/04_scratch_refinement/blender_orbit_check.py
# Read-only: orbit direction, per-frame azimuth step, forward and up vectors of
# GT_Animated_Camera vs ScratchPnP_Animated_Camera, measured in the saved scene.
import bpy, math, mathutils
sc = bpy.context.scene
root = bpy.data.objects["WorldRoot"].matrix_world.copy()
c = root @ mathutils.Vector((-0.01684, 0.11015, -0.00154))   # bunny centre (GT optical-axis intersection), CV world
up = (root.to_3x3() @ mathutils.Vector((0, 1, 0))).normalized()  # dataset world up (+Y) in Blender
def state(name, f):
    sc.frame_set(f); m = bpy.data.objects[name].matrix_world.copy()
    p = m.translation; fwd = (m.to_3x3() @ mathutils.Vector((0, 0, -1))).normalized()
    cu = (m.to_3x3() @ mathutils.Vector((0, 1, 0))).normalized()
    r = p - c; r = r - r.dot(up) * up
    return p, fwd, cu, r
def az(r, r0):  # signed angle about `up` from r0 to r
    return math.degrees(math.atan2(up.dot(r0.cross(r)), r0.dot(r)))
for name in ("GT_Animated_Camera", "ScratchPnP_Animated_Camera"):
    steps = []
    _, _, _, r0 = state(name, 0)
    prev = r0
    for f in range(1, 36):
        _, _, _, r = state(name, f); steps.append(az(r, prev)); prev = r
    print(f"ORBIT {name:28s} steps: min {min(steps):+.2f} max {max(steps):+.2f} mean {sum(steps)/35:+.3f} deg; "
          f"same sign as GT on {sum(s > 0 for s in steps) if steps[0] > 0 else sum(s < 0 for s in steps)}/35; total {sum(steps):+.1f}")
print("frame | azimuth GT / scratch (deg) | step GT / scratch | forward·to-centre scratch | up·worldup GT / scratch | |dpos| scratch vs GT (m)")
g0 = state("GT_Animated_Camera", 0)[3]
for f in (0, 1, 2, 10, 11, 35):
    pg, fg, ug, rg = state("GT_Animated_Camera", f); ps, fs, us, rs = state("ScratchPnP_Animated_Camera", f)
    sg = az(rg, state("GT_Animated_Camera", f - 1)[3]) if f else 0.0
    ss = az(rs, state("ScratchPnP_Animated_Camera", f - 1)[3]) if f else 0.0
    print(f"{f:5d} | {az(rg, g0):+8.2f} / {az(rs, g0):+8.2f} | {sg:+6.2f} / {ss:+6.2f} | "
          f"{fs.dot((c - ps).normalized()):.5f} | {ug.dot(up):.4f} / {us.dot(up):.4f} | {(ps - pg).length:.4f}")
