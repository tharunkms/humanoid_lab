"""
orbit_camera_waypoints.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Moves the D435i camera prim to N evenly-spaced positions around an object,
each looking roughly at the object's centre. Run one call to goto_view(i) at
a time -- after each move, switch to isaac_sim_segment_gui.py and press 'p'
to capture that view, then come back and call the next index.

This gives clean, reproducible spacing for point cloud fusion, which
freehand mouse-dragging the viewport can't reliably provide.
"""
import math
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

# ---- EDIT THESE THREE THINGS -----------------------------------------
CAMERA_PATH = "/World/d435i_camera"   # exact path from the Stage tree -- confirm this
TARGET = Gf.Vec3d(1.93, 1.65, 0.05)   # object's approx world x, y, z (metres) -- eyeball from viewport
RADIUS = 0.4                          # distance from object, metres -- stay outside D435i's ~0.15-0.3m blind zone
# ------------------------------------------------------------------------
HEIGHT = 0.3                          # camera height above the object, metres
NUM_VIEWS = 8                         # 8 gives good overlap; use 4 for a quick front/left/right/top-style plan

camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
if not camera_prim.IsValid():
    raise RuntimeError(f"No prim at {CAMERA_PATH} -- check the Stage tree for the real camera path")
xform = UsdGeom.Xformable(camera_prim)


def goto_view(i, num_views=NUM_VIEWS, radius=RADIUS, height=HEIGHT, target=TARGET):
    """Move the camera to waypoint i of num_views around `target`. Call this
    once per view, then switch to the GUI and press 'p' to capture."""
    angle = 2 * math.pi * i / num_views
    x = target[0] + radius * math.cos(angle)
    y = target[1] + radius * math.sin(angle)
    z = target[2] + height

    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    # simple yaw-only look-at -- fine for a roughly-horizontal ring of views;
    # doesn't tilt pitch down toward the object, adjust manually in the
    # viewport afterward if the object drifts out of frame at this height
    dx, dy = target[0] - x, target[1] - y
    yaw = math.degrees(math.atan2(dy, dx)) + 90.0  # +90 because camera looks along local -Y in USD by convention; flip to +/-90 or 180 if the object isn't centered
    xform.AddRotateZOp().Set(yaw)

    print(f"view {i}/{num_views - 1}: pos=({x:.2f},{y:.2f},{z:.2f})  yaw={yaw:.1f}deg "
          f"-- now switch to the GUI and press 'p' to capture")


# Example: run these one at a time (not all at once), pressing 'p' in the
# GUI between each call:
#
#   goto_view(0)
#   goto_view(1)
#   goto_view(2)
#   ...
#   goto_view(NUM_VIEWS - 1)
