"""
orbit_camera_hemisphere.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Moves the D435i camera prim to a handful of positions across a
FRONT-FACING HEMISPHERE around the object -- not a full 360deg orbit.

For grasp-quality partial reconstruction, full 360deg coverage isn't
needed: a robot approaching from one side only ever needs the object's
visible surface from that general direction, and in real life the object
may be occluded from other angles anyway. This matches how GPD/AnyGrasp/
VGN (hr07_manipulation_4.pdf) all operate -- from a partial, single-
approach-side point cloud, not a complete 3D model.

What actually matters is that the views are NOT collinear -- varying
both azimuth (left/center/right) AND elevation (level/above) guarantees
real 3D spread, unlike a manual drag (tends to slide along one axis) or
a single-height ring (zero vertical coverage).

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
CAMERA_PARENT_PATH = "/World/d435i_camera"       # translate this -- moves RGB+Depth+IR together
CAMERA_RGB_PATH = "/World/d435i_camera/Camera_RGB"      # rotate THIS -- actual Camera prim,
CAMERA_DEPTH_PATH = "/World/d435i_camera/Camera_Depth"  # ...and THIS -- must aim both, not just RGB,
                                                         # or depth data won't match wherever RGB is looking
TARGET = Gf.Vec3d(1.93, 1.65, 0.05)             # object's approx world x, y, z (metres) -- eyeball from viewport
RADIUS = 0.4                                    # distance from object, metres
AZIMUTHS_DEG = [-40.0, 0.0, 40.0]               # left / center / right, front-facing only
ELEVATIONS_DEG = [0.0, 35.0]                    # level and from above
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

depth_prim = stage.GetPrimAtPath(CAMERA_DEPTH_PATH)
if not depth_prim.IsValid():
    raise RuntimeError(f"No prim at {CAMERA_DEPTH_PATH} -- check the Stage tree for the real path")
depth_xform = UsdGeom.Xformable(depth_prim)

# Read the STAGE's actual up-axis instead of assuming Z-up. USD stages can
# be authored either Y-up or Z-up depending on how they were created --
# hardcoding one caused a 90deg flip when it didn't match this stage's
# real convention.
stage_up_axis = UsdGeom.GetStageUpAxis(stage)  # returns "Y" or "Z"
if stage_up_axis == UsdGeom.Tokens.y:
    WORLD_UP = Gf.Vec3d(0, 1, 0)
    print("Stage up-axis: Y -- using (0,1,0) as world_up")
else:
    WORLD_UP = Gf.Vec3d(0, 0, 1)
    print("Stage up-axis: Z -- using (0,0,1) as world_up")

# Every (azimuth, elevation) combination -- genuine spread across both
# axes, not a single-height ring or a collinear slide.
_waypoints = [(az, el) for el in ELEVATIONS_DEG for az in AZIMUTHS_DEG]


def goto_view(i, radius=RADIUS, target=TARGET):
    """Move the camera to waypoint i. Call once per view, then switch to
    the GUI, re-click the object, and press 'p' to capture."""
    azimuth, elevation = _waypoints[i]
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Two horizontal basis vectors perpendicular to WORLD_UP, so this
    # works correctly regardless of whether the stage is Y-up or Z-up.
    if stage_up_axis == UsdGeom.Tokens.y:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)  # X-Z plane, Y is up
    else:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0)  # X-Y plane, Z is up

    offset = (radius * math.cos(el_rad) * math.cos(az_rad) * horiz_a
              + radius * math.cos(el_rad) * math.sin(az_rad) * horiz_b
              + radius * math.sin(el_rad) * WORLD_UP)
    pos = target + offset

    # Full look-at (yaw AND pitch) via a rotation matrix.
    forward = (target - pos).GetNormalized()  # direction from camera to target
    cam_z = -forward  # USD camera looks down local -Z, so local +Z = -forward
    cam_x = Gf.Cross(WORLD_UP, cam_z).GetNormalized()  # local right axis
    cam_y = Gf.Cross(cam_z, cam_x)  # local up axis
    # USD's Gf.Matrix3d takes basis vectors as ROWS, not columns.
    rot_matrix = Gf.Matrix3d(
        cam_x[0], cam_x[1], cam_x[2],
        cam_y[0], cam_y[1], cam_y[2],
        cam_z[0], cam_z[1], cam_z[2],
    )
    quat = rot_matrix.ExtractRotation().GetQuat()

    parent_xform.ClearXformOpOrder()
    parent_xform.AddTranslateOp().Set(pos)

    rgb_xform.ClearXformOpOrder()
    rgb_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))  # sit at parent's position -- real D435i's
                                                         # RGB-to-depth baseline is mm-scale, negligible
                                                         # against our much larger RADIUS
    rgb_xform.AddOrientOp().Set(Gf.Quatf(quat))

    depth_xform.ClearXformOpOrder()
    depth_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    depth_xform.AddOrientOp().Set(Gf.Quatf(quat))  # same aim as RGB -- otherwise depth keeps
                                                     # pointing wherever it originally was

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
