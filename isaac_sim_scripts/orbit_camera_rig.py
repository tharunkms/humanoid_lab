"""
orbit_camera_rig.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Moves the ENTIRE D435i camera rig (parent prim) around the object as one
rigid unit, like a planet orbiting a star -- RGB, Depth, IR, and IMU all
move and re-aim together, since they're all children of the same parent
and simply inherit its motion.

Key fix: Camera_RGB has a FIXED LOCAL ROTATION relative to its parent
(not just a position offset) -- this is read once and corrected for, so
that setting the PARENT's orientation still results in Camera_RGB (and
Camera_Depth) pointing exactly at the target, not at some rotated-off
direction. Without this correction, rotating the parent directly using
the "-Z forward" camera convention points the whole rig in the wrong
direction, because a plain Xform parent has no such convention of its
own -- only the actual Camera-typed children do.

Usage: run everything down through the function definition ONCE, then
type single clean calls like `goto_view(0)` on their own line
(Ctrl+Enter) -- avoids the paste-indentation issue some editors
introduce with multi-line blocks.
"""
import math
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

# ---- EDIT THESE TO MATCH YOUR SCENE -----------------------------------
CAMERA_PARENT_PATH = "/World/d435i_camera"          # the whole rig moves as this ONE prim
CAMERA_RGB_PATH = "/World/d435i_camera/Camera_RGB"  # used only to read the fixed local rotation offset
TARGET = Gf.Vec3d(1.93, 1.65, 0.05)                 # object's approx world x, y, z (metres)
RADIUS = 0.4                                        # distance from object, metres
AZIMUTHS_DEG = [-40.0, 0.0, 40.0]                   # left / center / right, front-facing only
ELEVATIONS_DEG = [0.0, 35.0]                        # level and from above
# Total views = len(AZIMUTHS_DEG) * len(ELEVATIONS_DEG) = 6 by default
# -------------------------------------------------------------------------

parent_prim = stage.GetPrimAtPath(CAMERA_PARENT_PATH)
if not parent_prim.IsValid():
    raise RuntimeError(f"No prim at {CAMERA_PARENT_PATH} -- check the Stage tree for the real path")
parent_xform = UsdGeom.Xformable(parent_prim)

rgb_prim = stage.GetPrimAtPath(CAMERA_RGB_PATH)
if not rgb_prim.IsValid():
    raise RuntimeError(f"No prim at {CAMERA_RGB_PATH} -- check the Stage tree for the real path")
rgb_xform = UsdGeom.Xformable(rgb_prim)

# Read Camera_RGB's fixed local rotation relative to the parent, ONCE,
# before we touch anything. We'll correct for this below so that setting
# the PARENT's orientation still results in Camera_RGB pointing exactly
# at the target -- not off in whatever direction this offset would
# otherwise introduce.
_rgb_local_matrix = rgb_xform.GetLocalTransformation()
_rgb_local_rotation = _rgb_local_matrix.ExtractRotationQuat()
print(f"Camera_RGB local rotation relative to parent: {_rgb_local_rotation}")

# Read the STAGE's actual up-axis instead of assuming Z-up.
stage_up_axis = UsdGeom.GetStageUpAxis(stage)  # returns "Y" or "Z"
if stage_up_axis == UsdGeom.Tokens.y:
    WORLD_UP = Gf.Vec3d(0, 1, 0)
    print("Stage up-axis: Y -- using (0,1,0) as world_up")
else:
    WORLD_UP = Gf.Vec3d(0, 0, 1)
    print("Stage up-axis: Z -- using (0,0,1) as world_up")

_waypoints = [(az, el) for el in ELEVATIONS_DEG for az in AZIMUTHS_DEG]


def goto_view(i, radius=RADIUS, target=TARGET):
    """Move the WHOLE RIG to waypoint i, orbiting around `target`. Call
    once per view, then switch to the GUI, re-click the object, and
    press 'p' to capture."""
    azimuth, elevation = _waypoints[i]
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    if stage_up_axis == UsdGeom.Tokens.y:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)
    else:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0)

    offset = (radius * math.cos(el_rad) * math.cos(az_rad) * horiz_a
              + radius * math.cos(el_rad) * math.sin(az_rad) * horiz_b
              + radius * math.sin(el_rad) * WORLD_UP)
    pos = target + offset

    # Desired WORLD orientation for Camera_RGB (the "-Z forward" look-at,
    # same math as before -- this convention is correct for the actual
    # Camera-typed child, which is what we're ultimately aiming).
    forward = (target - pos).GetNormalized()
    cam_z = -forward
    cam_x = Gf.Cross(WORLD_UP, cam_z).GetNormalized()
    cam_y = Gf.Cross(cam_z, cam_x)
    rot_matrix = Gf.Matrix3d(
        cam_x[0], cam_x[1], cam_x[2],
        cam_y[0], cam_y[1], cam_y[2],
        cam_z[0], cam_z[1], cam_z[2],
    )
    desired_rgb_world_quat = rot_matrix.ExtractRotation().GetQuat()

    # Correct for Camera_RGB's fixed local rotation: since world_rot =
    # parent_rot * local_rot, solve parent_rot = world_rot * inverse(local_rot)
    # so that after composing with the UNCHANGED child local rotation,
    # Camera_RGB ends up at exactly the desired world orientation.
    parent_quat = desired_rgb_world_quat * _rgb_local_rotation.GetInverse()

    parent_xform.ClearXformOpOrder()
    parent_xform.AddTranslateOp().Set(pos)
    parent_xform.AddOrientOp().Set(Gf.Quatf(parent_quat))

    print(f"view {i}: azimuth={azimuth:.0f}deg elevation={elevation:.0f}deg "
          f"-> pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  "
          f"-- now switch to the GUI, re-click the object, press 'p'")


print(f"Total waypoints: {len(_waypoints)} (call goto_view(0) through goto_view({len(_waypoints)-1}))")

# After running the block above once, call these one at a time (each on
# its own clean Ctrl+Enter, no leading whitespace):
#
#   goto_view(0)
#   goto_view(1)
#   ...
#   goto_view(5)
