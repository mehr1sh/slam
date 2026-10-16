"""Shared coordinate-convention constants AND the only pose-conversion helpers
used by build_scene.py, slam_animation.py and slam_hud_panel.py. Kept in one
place so the scripts' conversions cannot drift out of sync -- the HUD panel
must undo exactly the same transforms build_scene.py applies, or its
displayed R/t would be wrong.

=== Pose convention of every trajectory this package reads ===
groundtruth.txt, pnp_trajectory.txt, icp_trajectory.txt and the per-method
columns of slam_trajectory.csv all store T_wc (camera -> world), exactly as
written by tests/synthetic/slam_trajectory_test.cpp:
  - translation = camera center in the C++ world frame (Y-up, metres)
  - quaternion  = rotation mapping CV camera-axis vectors
                  (+X right, +Y down, +Z forward) into world vectors
Nothing here inverts, transposes or re-composes those poses; they are only
re-expressed for Blender (see R_FIX and WORLD_ROOT_ROTATION below).
"""

import math

import mathutils

# Local 180-degree-about-X rotation: converts a CV-convention camera
# rotation (+X right, +Y down, +Z forward) into Blender's camera convention
# (+X right, +Y up, -Z forward). Self-inverse (R_FIX @ R_FIX == identity).
# Applied only inside trajectory_to_blender_pose(), to every GT/PnP/ICP
# Blender Camera object.
R_FIX = mathutils.Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

# Fixed whole-rig display rotation: the C++/PLY world is Y-up (see
# include/render/trajectory.hpp's world_up default), but Blender's own
# viewport floor grid and "up" convention is Z-up. Every visualization
# object (bunny, camera, trajectory, ghosts, ground plane) is parented
# under one "WorldRoot" empty carrying this fixed rotation, so each
# object's OWN local transform stays exactly the raw C++ output -- this
# only changes how the whole rig is oriented relative to Blender's floor,
# never any trajectory/camera math. (x,y,z) -> (x,-z,y).
WORLD_ROOT_ROTATION = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90))


def pose_from_row(row, prefix):
    """(pos, quat_cv) T_wc pose of one method from one slam_trajectory.csv row
    (a dict, or the scene["slam_trajectory"] ID-property copy of one).
    prefix is the CSV column prefix: "gt", "pnp" or "icp". Pure read, no math.
    """
    pos = mathutils.Vector((row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]))
    quat_cv = mathutils.Quaternion((row[f"{prefix}_qw"], row[f"{prefix}_qx"],
                                    row[f"{prefix}_qy"], row[f"{prefix}_qz"]))
    return pos, quat_cv


def trajectory_to_blender_pose(pos, quat_cv):
    """THE visualization-only conversion from a trajectory pose to the local
    transform (under WorldRoot) of a Blender Camera object.

    Input:  T_wc exactly as stored in the trajectory files -- camera center
            `pos` and camera-to-world rotation `quat_cv` in CV camera axes.
    Output: (pos, quat) with pos unchanged and quat = R_wc @ R_FIX.

    What it does: relabels the camera's LOCAL axes only (a fixed 180-degree
    rotation about the camera's own X axis), because Blender's camera axes
    look down local -Z with +Y up, whereas the CV convention looks down +Z
    with +Y down. It does NOT invert, transpose, or re-compose the pose, and
    does not touch the world frame. (Validated earlier by rendering through
    a Blender camera placed this way along ground truth: the bunny
    silhouette matched the dataset images, IoU 0.999.)
    """
    R_blender = quat_cv.to_matrix() @ R_FIX
    return pos, R_blender.to_quaternion()
