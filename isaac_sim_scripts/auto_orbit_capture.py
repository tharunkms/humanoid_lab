"""
auto_orbit_capture.py -- paste into Isaac Sim's Script Editor AFTER
isaac_sim_native_gui.py has been run and you've already clicked an
object in that GUI at least once (so `_gui` exists with a selected
object and `_gui.mask`/`_gui.depth` are populated).

Teleports the camera RIG (parent prim) around the currently-selected
object -- using its live 3D world position as the center of curvature --
sweeping a hemisphere of azimuth/elevation angles. At each waypoint it:
  1. Teleports the rig (translate parent + rotate to face the target,
     using the same parent-translate + child-rotation-correction fix
     validated earlier -- moving the whole rig as one unit).
  2. Forces a render update so the new frame is actually rendered.
  3. Projects the target's known 3D position into the NEW camera view
     (inverse of the pinhole deprojection) to get a pixel coordinate,
     and auto-clicks there -- no manual re-clicking needed.
  4. Captures the view via the GUI's existing _capture_pose().

Depends on `_gui` (the running NativeSegmentGUI instance) already
existing in this Python session.
"""
import math
import time
import numpy as np
import omni.usd
import omni.kit.app
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

# ---- EDIT THESE ----------------------------------------------------------
CAMERA_PARENT_PATH = "/World/d435i_camera"
CAMERA_RGB_PATH = "/World/d435i_camera/Camera_RGB"
RADIUS = 0.4
AZIMUTHS_DEG = [-40.0, 0.0, 40.0]
ELEVATIONS_DEG = [0.0, 35.0]
RENDER_SETTLE_FRAMES = 5  # how many app updates to force after each teleport,
                           # so the renderer actually produces the new frame
                           # before we try to project/click/capture
# ---------------------------------------------------------------------------

parent_prim = stage.GetPrimAtPath(CAMERA_PARENT_PATH)
parent_xform = UsdGeom.Xformable(parent_prim)
rgb_prim = stage.GetPrimAtPath(CAMERA_RGB_PATH)
rgb_xform = UsdGeom.Xformable(rgb_prim)

_rgb_local_rotation = rgb_xform.GetLocalTransformation().ExtractRotationQuat()

stage_up_axis = UsdGeom.GetStageUpAxis(stage)
if stage_up_axis == UsdGeom.Tokens.y:
    WORLD_UP = Gf.Vec3d(0, 1, 0)
else:
    WORLD_UP = Gf.Vec3d(0, 0, 1)

_waypoints = [(az, el) for el in ELEVATIONS_DEG for az in AZIMUTHS_DEG]


def _teleport_rig(target, azimuth, elevation, radius=RADIUS):
    """Move the whole rig to orbit position (azimuth, elevation) around
    `target`, correcting for Camera_RGB's fixed local rotation offset so
    it (and its siblings) actually face the target after teleporting."""
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
    parent_quat = desired_rgb_world_quat * _rgb_local_rotation.GetInverse()

    parent_xform.ClearXformOpOrder()
    parent_xform.AddTranslateOp().Set(pos)
    parent_xform.AddOrientOp().Set(Gf.Quatf(parent_quat))
    return pos


def _get_camera_pose_in_world(camera_path, world_prim_path="/World"):
    """Same logic as in isaac_sim_native_gui.py, duplicated here since
    that script was pasted as a temp file, not an importable module."""
    cam_prim = stage.GetPrimAtPath(camera_path)
    world_prim = stage.GetPrimAtPath(world_prim_path)
    cam_xformable = UsdGeom.Xformable(cam_prim)
    world_xformable = UsdGeom.Xformable(world_prim)
    cam_to_stage = cam_xformable.ComputeLocalToWorldTransform(0)
    world_to_stage = world_xformable.ComputeLocalToWorldTransform(0)
    cam_to_world = cam_to_stage * world_to_stage.GetInverse()
    m = np.array(cam_to_world).reshape(4, 4).T
    return m


def _project_target_to_pixel(target):
    """Project the known 3D world target into the CURRENT camera's pixel
    coordinates -- inverse of the pinhole deprojection formula. Lets us
    auto-click on the object after teleporting, instead of needing a
    person to manually re-click at each new angle."""
    T = _get_camera_pose_in_world(CAMERA_RGB_PATH)
    R, t = T[:3, :3], T[:3, 3]
    # world -> camera frame: inverse of camera-to-world
    cam_point = R.T @ (np.array(target) - t)
    if cam_point[2] <= 0.05:
        return None  # target is behind or too close to the camera
    fx, fy = _gui.intrinsics["fx"], _gui.intrinsics["fy"]
    cx, cy = _gui.intrinsics["cx"], _gui.intrinsics["cy"]
    u = fx * cam_point[0] / cam_point[2] + cx
    v = fy * cam_point[1] / cam_point[2] + cy
    w, h = _gui.intrinsics["width"], _gui.intrinsics["height"]
    if not (0 <= u < w and 0 <= v < h):
        return None  # projected outside the frame -- object not actually visible from here
    return int(u), int(v)


def orbit_and_capture():
    """Run the full automated sweep: get the currently-selected object's
    live world position, then teleport + auto-click + capture at every
    waypoint."""
    pos_result = _gui._object_3d_position()
    if pos_result is None:
        print("No object currently selected in the GUI -- click one first.")
        return
    _, target_np = pos_result
    target = Gf.Vec3d(*target_np.tolist())
    print(f"Orbiting around target={target}, {len(_waypoints)} waypoints")

    app = omni.kit.app.get_app()
    for i, (azimuth, elevation) in enumerate(_waypoints):
        pos = _teleport_rig(target, azimuth, elevation)
        print(f"view {i}: azimuth={azimuth:.0f} elevation={elevation:.0f} pos={pos}")

        for _ in range(RENDER_SETTLE_FRAMES):
            app.update()

        pixel = _project_target_to_pixel(target)
        if pixel is None:
            print(f"  SKIPPED -- target not visible from this angle (check RADIUS/waypoint bounds)")
            continue
        u, v = pixel
        print(f"  auto-clicking at pixel ({u},{v})")
        _gui._send_click(u, v, label=1)

        for _ in range(2):
            app.update()

        _gui._capture_pose()
        print(f"  captured (total views: {_gui.pose_count})")

    print(f"Orbit complete. {_gui.pose_count} views captured. "
          f"Switch to the GUI and press 'Fuse Cloud' when ready.")


print(f"Ready. {len(_waypoints)} waypoints configured. Call orbit_and_capture() to run the full sweep.")
