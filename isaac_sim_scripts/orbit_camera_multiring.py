"""
orbit_camera_multiring.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Moves the D435i camera prim through a systematic ORBIT around the
object: a full 360deg azimuth sweep, repeated at several different
elevations (e.g. slightly below, level, and above the object). This
guarantees real coverage on all sides AND from different heights --
unlike a single flat ring (only sees the object from one elevation
angle) or a manual drag (tends to slide along one axis by accident).

Usage: run everything down through the function definition ONCE, then
type single clean calls like `goto_view(0)` on their own line
(Ctrl+Enter) -- avoids the paste-indentation issue some editors
introduce with multi-line blocks.
"""
import math
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

# ---- EDIT THESE THREE THINGS -----------------------------------------
CAMERA_PATH = "/World/d435i_camera/Camera_RGB"  # exact path from the Stage tree -- confirm this
TARGET = Gf.Vec3d(1.93, 1.65, 0.05)             # object's approx world x, y, z (metres) -- eyeball from viewport
RADIUS = 0.4                                    # orbit distance from object, metres
# ------------------------------------------------------------------------
ELEVATIONS_DEG = [-15.0, 20.0, 55.0]  # each ring's height angle -- low, level, high
VIEWS_PER_RING = 4                    # azimuth steps per ring (4 = every 90deg)
# Total views = len(ELEVATIONS_DEG) * VIEWS_PER_RING = 12 by default

camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
if not camera_prim.IsValid():
    raise RuntimeError(f"No prim at {CAMERA_PATH} -- check the Stage tree for the real camera path")
xform = UsdGeom.Xformable(camera_prim)

# Read the STAGE's actual up-axis instead of assuming Z-up. USD stages can
# be authored either Y-up or Z-up depending on how they were created --
# hardcoding one caused a 90deg flip when it didn't match this stage's
# real convention. TF/ROS world frame is Z-up by our project's setup, but
# the raw USD prim transform we're writing here needs to match the
# STAGE's own axis, which may differ.
stage_up_axis = UsdGeom.GetStageUpAxis(stage)  # returns "Y" or "Z"
if stage_up_axis == UsdGeom.Tokens.y:
    WORLD_UP = Gf.Vec3d(0, 1, 0)
    print("Stage up-axis: Y -- using (0,1,0) as world_up")
else:
    WORLD_UP = Gf.Vec3d(0, 0, 1)
    print("Stage up-axis: Z -- using (0,0,1) as world_up")

# Precompute the full waypoint list: ring 0 sweeps all its azimuths first,
# then ring 1, then ring 2 -- so goto_view(0..3) is the low ring,
# goto_view(4..7) is the level ring, goto_view(8..11) is the high ring.
_waypoints = []
for elevation in ELEVATIONS_DEG:
    for step in range(VIEWS_PER_RING):
        azimuth = 360.0 * step / VIEWS_PER_RING
        _waypoints.append((azimuth, elevation))


def goto_view(i, radius=RADIUS, target=TARGET):
    """Move the camera to waypoint i of the precomputed orbit (see
    _waypoints above). Call once per view, then switch to the GUI,
    re-click the object, and press 'p' to capture."""
    azimuth, elevation = _waypoints[i]
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Two horizontal basis vectors perpendicular to WORLD_UP, so the orbit
    # sweeps the correct plane regardless of whether this stage is Y-up
    # or Z-up (hardcoding X/Y for azimuth + Z for elevation was the actual
    # cause of the 90deg flip on a Y-up stage).
    if stage_up_axis == UsdGeom.Tokens.y:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1)  # X-Z plane, Y is up
    else:
        horiz_a, horiz_b = Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0)  # X-Y plane, Z is up

    offset = (radius * math.cos(el_rad) * math.cos(az_rad) * horiz_a
              + radius * math.cos(el_rad) * math.sin(az_rad) * horiz_b
              + radius * math.sin(el_rad) * WORLD_UP)
    pos = target + offset

    # Full look-at (yaw AND pitch) via a rotation matrix, not yaw-only --
    # yaw-only was the bug: at any nonzero elevation the camera was
    # positioned correctly but pointing sideways instead of angled toward
    # the object, since it never tilted to compensate for height.
    forward = (target - pos).GetNormalized()  # direction from camera to target
    cam_z = -forward  # USD camera looks down local -Z, so local +Z = -forward
    cam_x = Gf.Cross(WORLD_UP, cam_z).GetNormalized()  # local right axis
    cam_y = Gf.Cross(cam_z, cam_x)  # local up axis
    # USD's Gf.Matrix3d takes basis vectors as ROWS, not columns -- the
    # column-based construction was causing a 90deg roll (floor rendering
    # as a vertical wall instead of the ground plane).
    rot_matrix = Gf.Matrix3d(
        cam_x[0], cam_x[1], cam_x[2],
        cam_y[0], cam_y[1], cam_y[2],
        cam_z[0], cam_z[1], cam_z[2],
    )
    quat = rot_matrix.ExtractRotation().GetQuat()

    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(pos)
    xform.AddOrientOp().Set(Gf.Quatf(quat))

    ring = i // VIEWS_PER_RING
    print(f"view {i} (ring {ring}, elevation={elevation:.0f}deg, azimuth={azimuth:.0f}deg) "
          f"-> pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  -- now switch to the GUI, re-click the object, press 'p'")


print(f"Total waypoints: {len(_waypoints)} (call goto_view(0) through goto_view({len(_waypoints)-1}))")

# After running the block above once, call these one at a time (each on
# its own clean Ctrl+Enter, no leading whitespace):
#
#   goto_view(0)
#   goto_view(1)
#   ...
#   goto_view(11)
