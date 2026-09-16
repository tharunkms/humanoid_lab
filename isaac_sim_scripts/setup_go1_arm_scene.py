"""
setup_go1_arm_scene.py  --  run ONCE in the Isaac Sim Script Editor (timeline STOPPED)
====================================================================================

Adds the Unitree Go1 and the OpenManipulator-X to the current table scene,
switches them to kinematic puppet mode, mounts the arm on the Go1 trunk and the
existing D435i rig (/World/d435i_camera) on the gripper link (link5), then
poses the robot so the camera matches where it is right now.

Afterwards: File > Save As ... (new file, keep the old scene as backup).
"""

import sys
import importlib
import numpy as np
import omni.usd
import omni.client
import omni.kit.app
import omni.kit.commands

SCRIPTS_DIR = "/home/user/kamarajmagadapallt1/projects/quadruped-inspection/isaac_sim/scripts"
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import robot_view_poser as rvp
importlib.reload(rvp)

# ---- asset sources (first one that exists is used) --------------------------
GO1_USD = None   # set an absolute .usd path to override the Isaac asset library
GO1_URDF = None  # e.g. ".../go1_description/urdf/go1.urdf" if no USD is available
ARM_USD = None   # e.g. the arm USD you copied earlier
ARM_URDF = ("/home/user/kamarajmagadapallt1/Documents/lab-project-2026/open_manipulator/"
            "open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf")


def _exists(url):
    return bool(url) and omni.client.stat(url)[0] == omni.client.Result.OK


def _library_go1():
    root = None
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
    except Exception:
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
            root = get_assets_root_path()
        except Exception:
            pass
    return f"{root}/Isaac/Robots/Unitree/Go1/go1.usd" if root else None


def _reference(stage, path, usd):
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(usd)
    print(f"[setup] referenced {usd} -> {path}")


def _import_urdf(stage, path, urdf):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.importer.urdf", True)
    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = False
    cfg.make_default_prim = False
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    _, imported = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf, import_config=cfg, get_articulation_root=False)
    if not imported or not stage.GetPrimAtPath(imported).IsValid():
        raise RuntimeError(f"URDF import failed for {urdf}")
    if str(imported) != path:
        omni.kit.commands.execute("MovePrim", path_from=str(imported), path_to=path)
    print(f"[setup] imported {urdf} -> {path}")


def _add(stage, path, usd, urdf, label):
    if stage.GetPrimAtPath(path).IsValid():
        print(f"[setup] {label} already at {path}, keeping it")
        return
    for u in (usd,):
        if _exists(u):
            _reference(stage, path, u)
            return
    if urdf and _exists(urdf):
        _import_urdf(stage, path, urdf)
        return
    raise RuntimeError(f"no source found for {label}: usd={usd} urdf={urdf} -- set the path at the top")


stage = omni.usd.get_context().get_stage()

# 1. remember where the camera is now (= the view the GUI used before)
cam = stage.GetPrimAtPath(f"{rvp.RIG_PATH}/{rvp.CAM_PRIM_NAME}")
C0 = rvp._world(cam)
p0 = C0[:3, 3].copy()
look0 = p0 + (-C0[:3, 2]) * 1.0
print(f"[setup] current camera at {p0.round(3)}")

# 2. add robots
_add(stage, rvp.GO1_PATH, GO1_USD or _library_go1(), GO1_URDF, "Go1")
_add(stage, rvp.ARM_PATH, ARM_USD, ARM_URDF, "OpenManipulator-X")

# 3. physics off on the robots (perception-only puppet)
rvp.make_puppet(stage, rvp.GO1_PATH)
rvp.make_puppet(stage, rvp.ARM_PATH)

# 4. build poser (prints the height/reach report) and match the old view
poser = rvp.RobotViewPoser(stage)
info = poser.set_view(p0, look0)
print(f"[setup] requested {info['requested_pos']}  achieved {info['achieved_pos']}")
print("[setup] done -- check the viewport, then File > Save As")
