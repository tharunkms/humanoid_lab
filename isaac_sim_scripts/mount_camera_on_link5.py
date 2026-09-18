"""
mount_camera_on_link5.py -- ONE-TIME: physically parent the D435i rig onto
the OpenManipulator-X's link5 in USD. Paste into the Script Editor, timeline
STOPPED, run once.

Before this, the camera rig (RIG_PATH, a free top-level Xform) was moved
every frame by arm_view_poser.py writing its world transform, computed from
a guessed CAM_MOUNT_XYZ / CAM_MOUNT_PITCH_DEG offset in robot_view_poser.py
(marked "PLACEHOLDER, refine on hardware" there).

After this script:
  - the rig is a REAL child of link5 in the stage tree, at a fixed local
    xformOp equal to that same CAM_MOUNT_XYZ/PITCH-derived offset -- so
    nothing visually jumps;
  - USD's own parent/child inheritance carries the rig along for free
    whenever link5 moves -- no more per-frame rig write needed;
  - arm_view_poser.ArmViewPoser detects the real parenting on its next
    construction and reads the mount straight back out of USD instead of
    recomputing it from the placeholder constants, so there is exactly one
    source of truth for the mount from here on.

Refine CAM_MOUNT_XYZ / CAM_MOUNT_PITCH_DEG in robot_view_poser.py FIRST to
match your actual hardware bracket, THEN run this script -- it bakes
whatever those constants say in at the moment it runs.

After running: update RIG_PATH in isaac_sim_native_gui.py (and in
robot_view_poser.py, if you also use the Go1+arm combined poser) to the
NEW_RIG_PATH this prints, then re-paste isaac_sim_native_gui.py.
"""
import omni.usd
import omni.kit.commands
from pxr import UsdGeom

from robot_view_poser import ARM_PATH, CAMERA_LINK, RIG_PATH, _Tree, _to_gf
from arm_view_poser import ArmViewPoser

stage = omni.usd.get_context().get_stage()

rig = stage.GetPrimAtPath(RIG_PATH)
if not rig.IsValid():
    raise RuntimeError(f"{RIG_PATH} not found -- edit RIG_PATH at the top of robot_view_poser.py "
                       f"if your rig prim is somewhere else")

arm = _Tree(stage, ARM_PATH)
link5_path = arm.find_link(CAMERA_LINK)
if str(rig.GetParent().GetPath()) == link5_path:
    raise RuntimeError(f"{RIG_PATH} is already a child of {link5_path} -- nothing to do")

# Compute the mount offset the SAME way ArmViewPoser does (free-floating
# path, since the rig hasn't moved yet), before touching the stage.
poser = ArmViewPoser(stage, verbose=False)
local_offset = poser.rig_in_link
assert not poser.physically_mounted

new_path = f"{link5_path}/{rig.GetName()}"
ok, _ = omni.kit.commands.execute("MovePrim", path_from=RIG_PATH, path_to=new_path)
if not ok:
    raise RuntimeError(f"MovePrim {RIG_PATH} -> {new_path} failed")

moved = stage.GetPrimAtPath(new_path)
xf = UsdGeom.Xformable(moved)
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(_to_gf(local_offset))

print(f"[mount] {RIG_PATH} -> {new_path}")
print(f"[mount] local mount (rig-in-link5): translate {local_offset[:3, 3].round(4)}")
print(f'[mount] NEW_RIG_PATH = "{new_path}"')
print("[mount] Update RIG_PATH to this value in isaac_sim_native_gui.py "
     "(and robot_view_poser.py if you use the Go1 combined poser), then re-paste.")
print("[mount] Save the stage (File > Save) so the mount survives a reload.")
