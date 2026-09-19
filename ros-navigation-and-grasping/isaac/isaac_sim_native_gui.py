"""
isaac_sim_native_gui.py -- paste into Isaac Sim's Script Editor (Window > Script Editor).

Runs the SAM2 click-to-segment GUI INSIDE Isaac Sim itself, as a native
omni.ui window, instead of as an external process in the ROS1 container.

Key architectural change from isaac_sim_segment_gui.py:
  - Camera intrinsics: read DIRECTLY from the Camera prim's USD attributes
    (focalLength, horizontalAperture, etc.) -- no camera_info topic, no
    NaN-intrinsics bugs, no frame_id mismatches.
  - Camera pose: read DIRECTLY from the prim's world transform via USD
    -- no /tf, no timestamp-sync issues, no TF lookup failures.
  - RGB/Depth frames: captured via omni.replicator annotators attached
    directly to the camera prim -- no ROS image topics needed at all.

Process B (sam2_service.py) is UNCHANGED -- same ZeroMQ wire protocol
(ipc_common.py), same session file format on disk. Only Process A moves
from an external cv2 window into this native Isaac Sim window.

ipc_common.py needs to be importable from wherever you run this --
either copy it next to your Isaac Sim launch location and add that to
sys.path, or paste its contents directly above this script.

Usage: paste into Script Editor, Ctrl+Enter. A floating window titled
"SAM2 Segmentation" should appear inside Isaac Sim.

Step-1 additions (auto-detect + menu):
  - "Detect props" button: one detect_all round trip, every found object
    becomes a button in the right-hand "Detected now" list. Clicking one
    hands its mask to SAM2 tracking (select_mask) -- same downstream path
    as a mouse click on the image.
  - "Search for" list: fixed common categories. Clicking one enters search
    mode: detect_all is re-run every SEARCH_INTERVAL_S while the camera
    moves, each candidate is scored against the category by CLIP, and the
    first one above SEARCH_MATCH_PROB is locked and tracked automatically.
  - External command interface (ZeroMQ REP on COMMAND_BIND): another
    module -- a task planner, a teammate's locomotion node, a test script
    -- sends {"cmd": "inspect", "target": "cereal box", "goal": {...}}
    and the GUI runs the whole chain unattended: (teleport to goal as the
    locomotion stand-in) -> search for the target -> lock -> plan and
    capture views -> fuse -> report the .pcd path via {"cmd": "status"}.
    The GUI buttons and the free-text search box call the same code.
  - Shadow filter (GUI side, uses depth): a candidate whose points are
    planar AND coplanar with the surface right around it is a shadow, not
    an object, and is dropped. Floating / far-away objects are kept.
"""
import sys
import os
import json
import time
import array
import numpy as np
import zmq
import cv2

import omni.ui as ui
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom

# ---- EDIT THESE ----------------------------------------------------------
CAMERA_RGB_PATH = "/World/go1_sensor/open_manipulator_x/link5/D435i/d435i_camera/Camera_RGB"
CAMERA_DEPTH_PATH = "/World/d435i_camera/Camera_Depth"
ZMQ_ADDR = "tcp://131.220.7.222:5555"
COMMAND_BIND = "tcp://*:5556"   # external entry point: other modules send inspect/status/abort here (ZeroMQ REP)
SESSION_ROOT = os.path.expanduser("~/isaac_native_pc_sessions")
WORLD_FRAME_PRIM = "/World"  # pose is reported relative to this prim
IPC_COMMON_DIR = "/home/user/kamarajmagadapallt1/sam2_service"  # folder containing ipc_common.py

# Depth annotator. "distance_to_image_plane" is the pinhole Z that the
# reprojection math below (X = (u-cx)*z/fx) and the saved depth PNGs
# assume. The previous "distance_to_camera" is the Euclidean ray length,
# which overstates z by up to ~20% at the image edges of a 69 deg FOV.
# If fused clouds look WORSE after this change, set it back to
# "distance_to_camera" and tell me -- that would mean the assumption about
# what the annotator returns is wrong for this build.
DEPTH_ANNOTATOR = "distance_to_image_plane"

# Step-1 menu / search settings
SEARCH_CATEGORIES = ["bottle", "cup", "mug", "can", "plate", "bowl",
                     "box", "book", "ball", "wrench", "screwdriver"]
SEARCH_INTERVAL_S = 2.0      # gap between detect_all calls in search mode (measured from end of previous call)
SEARCH_MATCH_PROB = 0.25     # lock when CLIP prob for the target >= this (absolute rule) ...
SEARCH_MATCH_MIN = 0.08      # ... or when it is >= this AND
SEARCH_MATCH_RATIO = 3.0     #     >= this many times the next-best candidate's score (relative rule)
                             # or when the argmax label equals the target. Absolute CLIP softmax values are low
                             # on untextured sim objects and split between near-synonyms ("box"/"cube"), so
                             # "clearly the best of what is on the table" is the more reliable signal.
DETECT_MAX_RANGE_M = 5.0     # candidates need some valid depth closer than this to be listed
SHADOW_PLANAR_RMS_M = 0.005  # candidate counts as "flat" if its points fit a plane within 5 mm RMS
SHADOW_RING_DIST_M = 0.006   # ...and as a shadow if the surrounding ring is within 6 mm of that plane

# ---- steps 3-5: teleport view planner (stand-ins for Go1 + OpenManipulator-X) ----
RIG_PATH = "/World/go1_sensor/open_manipulator_x/link5/D435i"          # Xform that gets teleported (all sensor children move with it)
TABLETOP_PATH = "/World/PropTable/Tabletop"  # Cylinder prim: gives table centre, radius, top height
BODY_EDGE_CLEARANCE_M = 0.25   # Go1 body front stays this far outside the table edge
ARM_FORWARD_REACH_M = 0.25     # camera can be this far in front of the body front in the approach pose
ARM_MAX_REACH_M = 0.38         # OpenManipulator-X horizontal reach for the top-down view
MIN_VIEW_DIST_M = 0.35         # never put the camera closer than this (horizontally) in approach/side views
APPROACH_CAM_HEIGHT_M = 0.35   # camera height above the tabletop, approach view
SIDE_CAM_HEIGHT_M = 0.25       # camera height above the tabletop, side views
SIDE_AZIMUTH_DEG = 60.0        # body repositions this far around the target for the two side views
TOPDOWN_HEIGHT_M = 0.35        # camera height above the target for the top-down view
TOPDOWN_MIN_OFFSET_M = 0.08    # small horizontal offset so the look-at is never exactly vertical
SETTLE_FRAMES = 10             # frames to wait after a teleport before trusting the render/depth
GOAL_PITCH_DEG = 20.0          # camera pitch (down) when placed at an external goal pose
CAPTURE_MIN_VALID_FRAC_DEFAULT = 0.15   # capture gate for a normal object
CAPTURE_MIN_VALID_FRAC_FLOOR = 0.03     # never accept a view with less than this, however bad the target is
CAPTURE_ADAPTIVE_MARGIN = 0.6           # per-plan threshold = margin * (valid frac seen at plan start)

# ---- D435i depth degradation (see depth_noise.py) ----
# Applied to every depth frame before ANY consumer sees it, so detection,
# 3D positions, capture and fusion all run on realistic depth. Toggle with
# the "Depth noise" button; severity 1.0 = nominal D435i, 2.0 = pessimistic.
DEPTH_NOISE = dict(enabled=True, severity=1.0)
NOISE_RECOMPUTE_STRIDE = 89    # sample stride used to detect that the clean depth changed (camera moved)

# ---- tabletop voxel belief (see voxel_belief.py) ----
# Every captured view's FULL depth frame is fused into one occupancy grid
# over the table and the space above it -- not just the target's mask. It
# tracks free / occupied / unknown, so a later next-best-view step can ask
# "which voxels near the target have I never seen?" and so incidental
# observations of other objects are kept for a later inspection request.
VOXEL_BELIEF = True
VOXEL_RES = 0.005              # 5 mm
VOXEL_HEIGHT = 0.5             # metres of space above the tabletop to model
VOXEL_RAY_STRIDE = 3           # depth-pixel stride for ray casting (~300 ms/view at 3)
VOXEL_FRONTIER_RADIUS = 0.25   # radius around the target for the unknown-voxel query

# ---- next-best-view planning (uses the voxel belief above) ----
# Instead of a fixed 4-pose plan, score a ring of reachable candidate poses
# by how much of the REACHABLE unknown space near the target each would
# actually observe (FOV + range + line of sight through the belief), take
# the best, integrate it, and re-score. Stops when the best remaining
# candidate is not worth the move.
NBV_ENABLED = True
NBV_N_AZIMUTH = 12             # candidate poses per ring (body + arm at each azimuth)
NBV_MAX_VIEWS = 6              # hard cap on views per plan
NBV_MIN_GAIN = 0.08            # stop when the best candidate sees < this share of the weighted frontier
NBV_FRONTIER_RADIUS = 0.15     # frontier considered "near the target"
NBV_SURFACE_SIGMA = 0.05       # weight frontier voxels by distance to the known object surface
NBV_HALF_FOV_DEG = 32.0
NBV_MAX_RANGE = 0.9
NBV_MIN_INCIDENCE_COS = 0.34   # ~70 deg: a frontier voxel whose surface the camera meets more edge-on
                               # than this is not counted as observable (real depth fails at grazing
                               # angles, so counting it made predicted gain systematically optimistic)
PLAN_MAX_MASK_FRAC = 0.15      # a re-seeded mask bigger than this fraction of the image is not a table prop
TABLE_CLIP_M = 0.004           # at capture, drop mask pixels whose world z is within this of the tabletop (shadows, fringe)
PLAN_MIN_TARGET_TOL_M = 0.08   # re-seeded mask centroid must be within max(this, 0.75*object size) of the target
# ---------------------------------------------------------------------------

if IPC_COMMON_DIR not in sys.path:
    sys.path.insert(0, IPC_COMMON_DIR)
from depth_noise import apply_d435i_noise, DEFAULT_CFG as NOISE_DEFAULTS   # same folder as ipc_common.py
from voxel_belief import VoxelBelief
from ipc_common import (
    build_request, parse_response,
    REQ_CLICK, REQ_TRACK, REQ_RESET, REQ_CLASSIFY, REQ_GENERATE_PC, REQ_VIEW_PC,
    REQ_DETECT_ALL, REQ_SELECT_MASK,
    STATUS_OK, STATUS_NO_OBJECT, PAYLOAD_LABEL_MAP,
)

os.makedirs(SESSION_ROOT, exist_ok=True)

stage = omni.usd.get_context().get_stage()


# ---------------------------------------------------------------------- #
# Direct USD camera intrinsics/pose (replaces camera_info / TF entirely)
# ---------------------------------------------------------------------- #
def get_camera_intrinsics(camera_path, width, height):
    """Read real intrinsics directly from the Camera prim -- no ROS
    camera_info topic, no NaN-focal-length bugs possible."""
    cam_prim = stage.GetPrimAtPath(camera_path)
    cam = UsdGeom.Camera(cam_prim)
    focal_length = cam.GetFocalLengthAttr().Get()
    h_aperture = cam.GetHorizontalApertureAttr().Get()
    v_aperture = cam.GetVerticalApertureAttr().Get()
    fx = width * focal_length / h_aperture
    fy = height * focal_length / v_aperture
    cx, cy = width / 2.0, height / 2.0
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


# OpenCV/pinhole camera frame (x right, y down, z FORWARD -- what the
# depth unprojection X=(u-cx)*z/fx produces) vs USD Camera prim frame
# (x right, y up, z BACKWARD). Same origin, y and z flipped. Every point we
# compute is in the OpenCV frame, so the matrix we apply to it must be
# OpenCV-cam -> world, not USD-cam -> world. Without this, a point 1 m in
# front of the camera lands 1 m BEHIND it in world coordinates (observed:
# box reported at y=3.6 while camera y=4.8 and table y=5.9).
OPENCV_TO_USD_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def get_camera_pose_in_world(camera_path, world_prim_path=WORLD_FRAME_PRIM):
    """4x4 OpenCV-optical-frame -> world matrix, read directly from USD --
    no /tf, no timestamp sync issues, no lookup failures. Apply directly
    to points from the pinhole unprojection."""
    cam_prim = stage.GetPrimAtPath(camera_path)
    world_prim = stage.GetPrimAtPath(world_prim_path)
    cam_xformable = UsdGeom.Xformable(cam_prim)
    world_xformable = UsdGeom.Xformable(world_prim)
    cam_to_stage = cam_xformable.ComputeLocalToWorldTransform(0)
    world_to_stage = world_xformable.ComputeLocalToWorldTransform(0)
    cam_to_world = cam_to_stage * world_to_stage.GetInverse()
    # Gf.Matrix4d is row-vector convention (v' = v * M) -- transpose to
    # get a standard column-vector 4x4 for our own math/JSON storage.
    m_usd = np.array(cam_to_world).reshape(4, 4).T
    return m_usd @ OPENCV_TO_USD_CAM


# ---------------------------------------------------------------------- #
# Teleport helpers (steps 3-5). All in stage/world coordinates, Z-up.
# ---------------------------------------------------------------------- #
def np_to_gf(m_np):
    """column-vector 4x4 numpy -> Gf.Matrix4d (row-vector convention).
    UNTESTED ASSUMPTION: Gf.Matrix4d(*16 floats) takes them row-major; the
    row-vector convention means we hand it the TRANSPOSE of our matrix.
    (Same convention the Gf.Matrix3d lesson in the brief was about.)"""
    return Gf.Matrix4d(*[float(v) for v in np.asarray(m_np).T.flatten()])


def read_table_geometry():
    """(centre_xyz, radius, top_z) of the Tabletop cylinder, world frame."""
    prim = stage.GetPrimAtPath(TABLETOP_PATH)
    cyl = UsdGeom.Cylinder(prim)
    radius = float(cyl.GetRadiusAttr().Get())
    height = float(cyl.GetHeightAttr().Get())
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    ctr = np.array(m.ExtractTranslation())
    # world scale along the cylinder's own X (radius) and Z (height) axes
    sx = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetLength()
    sz = Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetLength()
    return ctr, radius * sx, float(ctr[2] + 0.5 * height * sz)


def usd_camera_lookat(cam_pos, target, up_hint=(0.0, 0.0, 1.0)):
    """4x4 USD-camera -> world matrix (column convention) for a camera at
    cam_pos looking at target. USD Camera: x right, y up, z BACKWARD."""
    cam_pos = np.asarray(cam_pos, float)
    f = np.asarray(target, float) - cam_pos
    f /= np.linalg.norm(f)
    up = np.asarray(up_hint, float)
    if abs(np.dot(f, up)) > 0.98:          # looking straight up/down: pick another up
        up = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = r, u, -f    # columns = camera axes in world
    T[:3, 3] = cam_pos
    return T


def teleport_rig_so_camera_is(cam_T_usd_np, rig_path=RIG_PATH, camera_path=CAMERA_RGB_PATH):
    """Move the RIG (parent Xform) so that camera_path ends up at the given
    USD-camera->world pose. Moving the parent keeps every sensor child
    (RGB, Depth, IR) rigidly together -- the 'depth stayed frozen' lesson.

    Row-vector algebra: cam_world = cam_local * rig_world
      => rig_world = cam_local^-1 * cam_world_desired
    cam_local is read live from USD so no hard-coded child offset is needed."""
    rig_prim = stage.GetPrimAtPath(rig_path)
    cam_prim = stage.GetPrimAtPath(camera_path)
    rig_world = UsdGeom.Xformable(rig_prim).ComputeLocalToWorldTransform(0)
    cam_world = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(0)
    cam_local = cam_world * rig_world.GetInverse()
    rig_world_new = cam_local.GetInverse() * np_to_gf(cam_T_usd_np)
    # the rig's xformOps are relative to ITS parent (normally /World = identity)
    parent_world = UsdGeom.Xformable(rig_prim.GetParent()).ComputeLocalToWorldTransform(0)
    rig_local_new = rig_world_new * parent_world.GetInverse()

    t = rig_local_new.ExtractTranslation()
    q = rig_local_new.ExtractRotationQuat()     # Gf.Quatd (scale must be 1 -- it is now)
    t_attr = rig_prim.GetAttribute("xformOp:translate")
    o_attr = rig_prim.GetAttribute("xformOp:orient")
    if not t_attr or not o_attr:
        raise RuntimeError(f"{rig_path} needs xformOp:translate + xformOp:orient "
                           f"(has: {[a.GetName() for a in rig_prim.GetAttributes() if a.GetName().startswith('xformOp')]})")
    t_attr.Set(Gf.Vec3f(t) if t_attr.GetTypeName() == "float3" else Gf.Vec3d(t))
    if o_attr.GetTypeName() == "quatf":
        o_attr.Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(q.GetImaginary())))
    else:
        o_attr.Set(q)
    return rig_local_new


def set_rig_local_matrix(rig_local_gf, rig_path=RIG_PATH):
    """Restore a previously saved rig pose (Gf.Matrix4d, local)."""
    rig_prim = stage.GetPrimAtPath(rig_path)
    t = rig_local_gf.ExtractTranslation()
    q = rig_local_gf.ExtractRotationQuat()
    t_attr = rig_prim.GetAttribute("xformOp:translate")
    o_attr = rig_prim.GetAttribute("xformOp:orient")
    t_attr.Set(Gf.Vec3f(t) if t_attr.GetTypeName() == "float3" else Gf.Vec3d(t))
    if o_attr.GetTypeName() == "quatf":
        o_attr.Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(q.GetImaginary())))
    else:
        o_attr.Set(q)


def enforce_unit_scale(path=RIG_PATH):
    """Any scale on the rig is inherited by the camera prims and shears
    every unprojected point (observed: xformOp:scale (1.34,1,1) crept back
    after a stage reload -> 8 mm teleport error + distorted fused box).
    Force it to (1,1,1) here so the pipeline never runs on a scaled rig."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"enforce_unit_scale: no prim at {path} -- did the rig get reparented?")
    attr = prim.GetAttribute("xformOp:scale")
    if not attr:
        return
    sc = attr.Get()
    if sc is None or all(abs(float(v) - 1.0) < 1e-6 for v in sc):
        return
    print(f"[isaac_sim_native_gui] WARNING: {path} had xformOp:scale={tuple(sc)} -- forcing to (1,1,1). "
          f"Save the stage (File > Save) so it stays that way.")
    attr.Set(Gf.Vec3f(1, 1, 1) if attr.GetTypeName() == "float3" else Gf.Vec3d(1, 1, 1))


def camera_transform_is_rigid(camera_path=CAMERA_RGB_PATH, tol=1e-3):
    """True if the camera's world rotation block is a pure rotation."""
    m = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(camera_path)).ComputeLocalToWorldTransform(0)).reshape(4, 4)
    sv = np.linalg.svd(m[:3, :3], compute_uv=False)
    return bool(np.all(np.abs(sv - 1.0) < tol)), sv


enforce_unit_scale()


# ---------------------------------------------------------------------- #
# Replicator capture -- RGB + depth directly from the render product
# ---------------------------------------------------------------------- #
# Each paste of this script into the Script Editor runs in a fresh module
# namespace, but replicator state lives in the Isaac Sim process. Without
# cleanup, every run leaks a render product + 2 annotators; after enough
# runs the depth annotator started returning an EMPTY (1-D) array while
# RGB still worked. Keep the handles on `builtins` (survives re-runs) and
# tear the previous set down first. UNTESTED API DETAIL: annotator.detach()
# and render_product.destroy() -- both wrapped, a failure only prints.
import builtins
_prev = getattr(builtins, "_native_gui_capture", None)
if _prev is not None:
    for ann in (_prev.get("rgb"), _prev.get("depth")):
        try:
            ann.detach()
        except Exception as exc:
            print(f"[isaac_sim_native_gui] previous annotator detach failed: {exc}")
    try:
        _prev["rp"].destroy()
    except Exception as exc:
        print(f"[isaac_sim_native_gui] previous render product destroy failed: {exc}")
    print("[isaac_sim_native_gui] tore down previous run's render product/annotators")

render_product = rep.create.render_product(CAMERA_RGB_PATH, resolution=(1280, 720))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])
depth_annotator = rep.AnnotatorRegistry.get_annotator(DEPTH_ANNOTATOR)
depth_annotator.attach([render_product])
builtins._native_gui_capture = {"rp": render_product, "rgb": rgb_annotator, "depth": depth_annotator}


# ---------------------------------------------------------------------- #
# GUI state + logic
# ---------------------------------------------------------------------- #
class NativeSegmentGUI:
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(ZMQ_ADDR)

        # external entry point (see docstring). Polled non-blocking once per
        # frame so it never stalls the UI; replies immediately with an ack,
        # the caller polls "status" for progress.
        self.cmd_sock = self.ctx.socket(zmq.REP)
        self.cmd_sock.bind(COMMAND_BIND)
        self.task = {"state": "idle", "target": None, "goal": None, "session": None,
                     "pcd": None, "message": ""}
        self.auto_plan = False      # set by an "inspect" command: lock -> plan -> fuse without clicks
        self.auto_fuse = False
        self.rig_start = None       # rig pose at script start, for goal poses given relative to it

        # depth noise: cache so a static camera doesn't pay ~100 ms/frame
        self.noise_cfg = dict(NOISE_DEFAULTS); self.noise_cfg.update(DEPTH_NOISE)
        self.noise_rng = np.random.default_rng()
        self._clean_depth = None    # last ideal depth (kept for the *_depth_clean.png sidecar)
        self._noise_key = None
        self._noisy_cache = None

        self.belief = None          # VoxelBelief, created on the first capture of a session
        self.nbv_candidates = []    # candidate poses for next-best-view selection
        self.nbv_used = set()

        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.mask = None
        self.tracking_active = False
        self.last_label = None
        self.last_shape = None
        self.status_text = ""

        self.session_dir = None
        self.pose_count = 0
        self.captured_poses = []

        # ---- step 1: detected candidates + search mode ----
        self.candidates = []          # list of dicts from detect_all (after shadow filter)
        self.label_map = None         # uint8 HxW, pixel = candidate id
        self.selected_id = None       # candidate id currently being tracked (None if via mouse click)
        self.search_target = None     # category string while in search mode, else None
        self.search_best = None       # (id, prob) of best partial match so far, for the status line
        self._last_detect_end = 0.0
        self._detect_count = 0

        # ---- steps 3-5: view plan state machine ----
        self.plan = []                # list of {"name", "cam_pos", "look_at"}
        self.plan_idx = -1
        self.plan_state = None        # None | "settle" | "reseed"
        self.plan_settle = 0
        self.plan_target = None       # target world xyz captured when the plan started
        self.plan_target_tol = PLAN_MIN_TARGET_TOL_M
        self.rig_home = None          # Gf.Matrix4d local pose of the rig before the plan

        self._build_window()
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="native_segment_gui_update")

    # ---- UI construction ----
    def _build_window(self):
        self.window = ui.Window("SAM2 Segmentation", width=980, height=680)
        self.window.set_visibility_changed_fn(self._on_visibility_changed)
        with self.window.frame:
            with ui.HStack(spacing=6):
                # ---- left: image + controls (unchanged behaviour) ----
                with ui.VStack(spacing=4, width=660):
                    self.status_label = ui.Label("waiting for first frame...", height=20)
                    self.image_provider = ui.ByteImageProvider()
                    self.image_widget = ui.ImageWithProvider(
                        self.image_provider, width=1280 // 2, height=720 // 2)
                    self.image_widget.set_mouse_pressed_fn(self._on_image_clicked)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("Detect props (d)", clicked_fn=self._detect_props)
                        ui.Button("Classify", clicked_fn=self._classify)
                        ui.Button("Capture View (p)", clicked_fn=self._capture_pose)
                        ui.Button("Fuse Cloud (g)", clicked_fn=self._generate_pointcloud)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("View Cloud (v)", clicked_fn=self._view_pointcloud)
                        ui.Button("New Session (n)", clicked_fn=self._start_session)
                        ui.Button("Stop search", clicked_fn=self._stop_search)
                        ui.Button("Reset (r)", clicked_fn=self._reset)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("Plan + capture views", clicked_fn=self._start_view_plan)
                        ui.Button("Return rig home", clicked_fn=self._return_rig_home)
                        ui.Button("Abort plan", clicked_fn=self._abort_plan)
                        ui.Button("Depth noise on/off", clicked_fn=self._toggle_noise)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("Save belief", clicked_fn=self._save_belief)
                        ui.Button("View belief", clicked_fn=self._view_belief)
                    self.info_label = ui.Label("", height=20, word_wrap=True)
                    self.search_label = ui.Label("", height=20, word_wrap=True)
                # ---- right: menu ----
                with ui.VStack(spacing=4):
                    ui.Label("Detected now", height=20)
                    # ui.Frame with a build function is the documented omni.ui
                    # way to have a dynamic list: call .rebuild() after
                    # self.candidates changes. (Untested here -- if
                    # set_build_fn/rebuild don't exist in this build, the
                    # fallback is a ui.VStack and .clear() + re-adding buttons.)
                    with ui.ScrollingFrame(height=300):
                        self.detected_frame = ui.Frame()
                        self.detected_frame.set_build_fn(self._build_detected_menu)
                    ui.Label("Search for (any object name)", height=20)
                    with ui.HStack(height=24, spacing=4):
                        # UNTESTED omni.ui detail: StringField + model.get_value_as_string().
                        # Fallback if it errors: ui.StringField(ui.SimpleStringModel()).
                        self.search_field = ui.StringField()
                        ui.Button("Search", width=70,
                                  clicked_fn=lambda: self._start_search(self.search_field.model.get_value_as_string().strip()))
                    ui.Label("or pick one:", height=18)
                    with ui.ScrollingFrame():
                        with ui.VStack(spacing=2):
                            for cat in SEARCH_CATEGORIES:
                                ui.Button(cat, height=22,
                                          clicked_fn=lambda c=cat: self._start_search(c))

    def _build_detected_menu(self):
        """Build fn for the dynamic candidate list (runs inside the Frame)."""
        with ui.VStack(spacing=2):
            if not self.candidates:
                ui.Label("(press Detect props)", height=20)
                return
            for c in self.candidates:
                txt = f"{c['id']}: {c['label']} ({c['shape']} {c['confidence']:.2f})"
                if "target_prob" in c:
                    txt += f"  ~{self.search_target}: {c['target_prob']:.2f}"
                if c.get("depth_warning"):
                    txt += "  [low depth]"
                if c["id"] == self.selected_id:
                    txt = "> " + txt
                ui.Button(txt, height=22, clicked_fn=lambda cid=c["id"]: self._select_candidate(cid))

    def _refresh_detected_menu(self):
        try:
            self.detected_frame.rebuild()
        except AttributeError as exc:
            print(f"[isaac_sim_native_gui] Frame.rebuild unavailable ({exc}) -- menu not refreshed")

    # ---- cleanup when the window is closed --------------------------
    def _on_visibility_changed(self, visible):
        if not visible:
            # user clicked the window's close button -- stop the
            # per-frame update loop and disconnect, instead of leaving
            # both running invisibly in the background.
            if self._update_sub is not None:
                self._update_sub = None
            for sk in (self.sock, self.cmd_sock):
                try:
                    sk.close(linger=0)
                except Exception:
                    pass
            print("[isaac_sim_native_gui] window closed -- update loop and ZMQ sockets stopped.")

    # ---- per-frame update ----
    def _on_update(self, event):
        self._poll_commands()
        rgba = rgb_annotator.get_data()
        depth = depth_annotator.get_data()
        if rgba is None or rgba.size == 0:
            return
        self.rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        h, w = self.rgb.shape[:2]
        # validate depth every frame: an empty/1-D array here is what
        # crashed detect/plan with "too many indices for array"
        depth_ok = depth is not None and getattr(depth, "ndim", 0) == 2 and depth.shape == (h, w)
        if depth_ok != getattr(self, "_depth_ok", None):
            self._depth_ok = depth_ok
            shape = None if depth is None else getattr(depth, "shape", None)
            print(f"[frame] rgb {w}x{h}; depth {'OK' if depth_ok else 'INVALID'} shape={shape} "
                  f"dtype={None if depth is None else getattr(depth, 'dtype', None)}")
        self.depth = depth.astype(np.float32) if depth_ok else None  # float32 metres, pinhole Z
        self._clean_depth = self.depth
        if self.depth is not None and self.noise_cfg.get("enabled") and self.intrinsics is not None:
            self.depth = self._noisy_depth(self.depth)
        if self.intrinsics is None:
            self.intrinsics = get_camera_intrinsics(CAMERA_RGB_PATH, w, h)

        if self.plan_state is not None:
            self._plan_step()
        elif self.tracking_active:
            self._track()
        elif self.search_target is not None:
            # search mode: only poll while nothing is locked; the interval
            # is measured from the END of the last detect_all so the UI
            # gets SEARCH_INTERVAL_S of normal frames between the ~5 s
            # blocking calls.
            if time.time() - self._last_detect_end >= SEARCH_INTERVAL_S:
                self._detect_props(target=self.search_target)
                self._try_lock_search_target()

        self._render_frame()

    # ---- voxel belief -----------------------------------------------------
    def _save_belief(self):
        """Write the belief next to the session: .npz to reload, .ply to look at."""
        if self.belief is None or self.session_dir is None:
            self.status_text = "no belief yet -- capture a view first"
            return
        npz = os.path.join(self.session_dir, "voxel_belief.npz")
        ply = os.path.join(self.session_dir, "voxel_belief.ply")
        self.belief.save_npz(npz)
        n = self.belief.save_ply(ply, what="occupied")
        st = self.belief.stats()
        self.status_text = (f"belief: {st['views']} views, {st['occupied']} occupied, "
                            f"{st['observed_frac']:.0%} observed -> voxel_belief.ply")
        print(f"[belief] saved {n} occupied voxels -> {ply}")
        print(f"[belief] {st}")

    def _view_belief(self):
        """Open the belief's occupied voxels in the same viewer as the clouds."""
        if self.belief is None or self.session_dir is None:
            self.status_text = "no belief yet"
            return
        ply = os.path.join(self.session_dir, "voxel_belief.ply")
        self.belief.save_ply(ply, what="occupied")
        meta, payload = build_request(REQ_VIEW_PC, ply_path=ply)
        self.sock.send_multipart([meta, payload])
        self.sock.recv_multipart()

    # ---- depth noise ------------------------------------------------------
    def _noisy_depth(self, clean):
        """D435i-style degraded copy of `clean`. Recomputed only when the
        clean depth changed (camera moved / scene changed); a static view
        reuses the last draw."""
        key = clean[::NOISE_RECOMPUTE_STRIDE, ::NOISE_RECOMPUTE_STRIDE].tobytes()
        if key == self._noise_key and self._noisy_cache is not None:
            return self._noisy_cache
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
        noisy = apply_d435i_noise(clean, self.rgb, fx, fy, cx, cy, cfg=self.noise_cfg, rng=self.noise_rng)
        self._noise_key, self._noisy_cache = key, noisy
        return noisy

    def _noise_sigma_at(self, z):
        """Expected depth noise std (m) at range z under the current model,
        0 when noise is off. Used to widen geometric thresholds that were
        tuned for ideal depth."""
        if not self.noise_cfg.get("enabled") or self.intrinsics is None:
            return 0.0
        fb = self.intrinsics["fx"] * self.noise_cfg["baseline_m"]
        return float(z * z / fb * self.noise_cfg["disp_sigma_px"] * self.noise_cfg["severity"])

    def _toggle_noise(self):
        self.noise_cfg["enabled"] = not self.noise_cfg.get("enabled")
        self._noise_key = None
        self.status_text = f"depth noise {'ON' if self.noise_cfg['enabled'] else 'OFF'} (severity {self.noise_cfg['severity']})"
        print(f"[noise] {self.status_text}")

    def _render_frame(self):
        if self.rgb is None:
            return
        vis = self.rgb.copy()
        if self.mask is not None:
            overlay = vis.copy()
            overlay[self.mask] = (0, 255, 0)
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        elif self.candidates:
            # nothing locked yet: show every candidate with its id so the
            # menu entries are easy to match to the picture
            for c in self.candidates:
                x, y, w, h = c["bbox"]
                col = (0, 186, 252) if c["id"] != (self.search_best or (None,))[0] else (0, 0, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, f"{c['id']}:{c['label']}", (x, max(14, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        vis_rgba = cv2.cvtColor(vis, cv2.COLOR_BGR2RGBA)
        # Resize to the WIDGET's actual display size before sending --
        # we were sending the full 1280x720 buffer even though the
        # widget only shows it at 640x360, wasting 4x the data/frame.
        disp_w = self.image_widget.width.value if hasattr(self.image_widget.width, "value") else 640
        disp_h = self.image_widget.height.value if hasattr(self.image_widget.height, "value") else 360
        vis_rgba_small = cv2.resize(vis_rgba, (int(disp_w), int(disp_h)))
        self.image_provider.set_bytes_data(
            array.array('B', vis_rgba_small.tobytes()), [int(disp_w), int(disp_h)])

        pos = self._object_3d_position()
        if pos is not None:
            cam_pt, world_pt = pos
            self.status_label.text = (
                f"cam: [{cam_pt[0]:.2f},{cam_pt[1]:.2f},{cam_pt[2]:.2f}]m  "
                f"world: [{world_pt[0]:.2f},{world_pt[1]:.2f},{world_pt[2]:.2f}]m")
        else:
            self.status_label.text = "no object selected" + ("" if self.depth is not None else "   [NO DEPTH DATA]")
        label_txt = f"{self.last_label} ({self.last_shape})" if self.last_label else ""
        sess_txt = f"session: {os.path.basename(self.session_dir)} views:{self.pose_count}" if self.session_dir else ""
        self.info_label.text = f"{label_txt}   {sess_txt}   {self.status_text}"
        if self.search_target is None:
            self.search_label.text = ""
        elif self.tracking_active:
            self.search_label.text = f"search '{self.search_target}': LOCKED on id {self.selected_id}"
        elif self.search_best is not None:
            bid, bp = self.search_best
            self.search_label.text = (f"search '{self.search_target}': looking... best so far id {bid} "
                                      f"(prob {bp:.2f}, need >= {SEARCH_MATCH_PROB}) -- move camera")
        else:
            self.search_label.text = f"search '{self.search_target}': looking... (scans: {self._detect_count})"

    # ---- mouse ----
    def _on_image_clicked(self, x, y, button, modifier):
        if self.rgb is None:
            return
        img_h, img_w = self.rgb.shape[:2]
        widget_w, widget_h = self.image_widget.computed_width, self.image_widget.computed_height
        if not widget_w or not widget_h:
            return
        # omni.ui mouse callbacks often report position relative to the
        # WINDOW/SCREEN, not the specific widget -- subtract the widget's
        # own screen position first. If clicks are still landing wrong
        # after this, the debug print below shows the real raw numbers
        # so we can see exactly what's being received.
        try:
            offset_x = self.image_widget.screen_position_x
            offset_y = self.image_widget.screen_position_y
            local_x = x - offset_x
            local_y = y - offset_y
        except AttributeError:
            local_x, local_y = x, y  # this build may already give widget-local coords
        print(f"[click debug] raw=({x},{y}) widget_size=({widget_w},{widget_h}) "
              f"local=({local_x},{local_y}) img_size=({img_w},{img_h})")
        px = int(local_x / widget_w * img_w)
        py = int(local_y / widget_h * img_h)
        px = max(0, min(img_w - 1, px))
        py = max(0, min(img_h - 1, py))
        label = 1 if button == 0 else 0  # left=positive, right=negative
        self._send_click(px, py, label)

    def _send_click(self, x, y, label):
        meta, payload = build_request(REQ_CLICK, frame_bgr=self.rgb, x=x, y=y, label=label)
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, mask = parse_response(resp_meta, resp_payload)
        if meta_d["status"] == STATUS_OK:
            self.mask = mask
            self.tracking_active = True
            self._classify()
        else:
            self.status_text = meta_d.get("message", "click failed")

    def _track(self):
        if not self.tracking_active or self.rgb is None:
            return
        meta, payload = build_request(REQ_TRACK, frame_bgr=self.rgb)
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, mask = parse_response(resp_meta, resp_payload)
        if meta_d["status"] == STATUS_OK:
            self.mask = mask
        elif meta_d["status"] == STATUS_NO_OBJECT:
            self.mask = None
            self.tracking_active = False
            self.selected_id = None   # search mode (if on) resumes polling automatically

    def _classify(self):
        if self.mask is None or self.rgb is None:
            return
        ys, xs = np.where(self.mask)
        if len(xs) == 0:
            return
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = self.rgb[y0:y1 + 1, x0:x1 + 1]
        meta, payload = build_request(REQ_CLASSIFY, frame_bgr=crop)
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, _ = parse_response(resp_meta, resp_payload)
        if meta_d["status"] == STATUS_OK:
            self.last_label = meta_d["label"]
            self.last_shape = meta_d["shape"]

    def _reset(self):
        meta, payload = build_request(REQ_RESET)
        self.sock.send_multipart([meta, payload])
        self.sock.recv_multipart()
        self.mask = None
        self.tracking_active = False
        self.last_label = None
        self.last_shape = None
        self.selected_id = None
        self.search_target = None
        self.search_best = None
        self.candidates = []
        self.label_map = None
        self.auto_plan = False      # GUI-driven work is manual again after a Reset
        self.auto_fuse = False
        self.capture_min_valid_frac = CAPTURE_MIN_VALID_FRAC_DEFAULT
        self._refresh_detected_menu()

    # ---- step 1: detect all props -> menu ----------------------------
    def _detect_props(self, target=None):
        """One detect_all round trip. Blocks the UI for the call (~5 s;
        longer the very first time after a Process B restart while it
        loads the image model + CLIP)."""
        if self.rgb is None:
            return
        if self.depth is None:
            self.status_text = "no depth data -- restart Isaac Sim and re-run the script"
            print(f"[detect] {self.status_text}")
            return
        extra = {"target_label": target} if target else {}
        meta, payload = build_request(REQ_DETECT_ALL, frame_bgr=self.rgb, **extra)
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, label_map = parse_response(resp_meta, resp_payload)
        self._last_detect_end = time.time()
        self._detect_count += 1
        if meta_d["status"] != STATUS_OK or meta_d.get("payload") != PAYLOAD_LABEL_MAP:
            self.status_text = f"detect failed: {meta_d.get('message', '')}"
            print(f"[detect] {self.status_text}")
            return
        raw = meta_d.get("candidates", [])
        depth = self.depth  # snapshot: same frame the RGB came from
        kept, dropped, flagged = [], [], 0
        for c in raw:
            drop_reason, warn = self._reject_reason(c, label_map, depth)
            if drop_reason:
                dropped.append((c, drop_reason))
            else:
                if warn:
                    c["depth_warning"] = warn
                    flagged += 1
                kept.append((c, warn))
        self.candidates = [c for c, _ in kept]
        self.label_map = label_map
        self.detect_id = meta_d.get("detect_id")
        self.selected_id = None
        self.status_text = f"detected {len(self.candidates)} props ({len(dropped)} filtered" +                            (f", {flagged} low-depth" if flagged else "") + ")"
        print(f"[detect] service: {meta_d.get('message')}")
        for c, r in dropped:
            print(f"[detect] dropped id {c['id']} {c['label']}: {r}")
        for c, w in kept:
            tp = f" target_prob={c['target_prob']:.2f}" if "target_prob" in c else ""
            note = f"  [WARNING: {w}]" if w else ""
            print(f"[detect] kept id {c['id']} {c['label']} ({c['shape']} {c['confidence']:.2f}) bbox={c['bbox']}{tp}{note}")
        self._refresh_detected_menu()

    def _unproject(self, mask, depth, step=3):
        """Camera-frame XYZ for every step-th valid pixel of a bool mask."""
        ys, xs = np.nonzero(mask)
        ys, xs = ys[::step], xs[::step]
        z = depth[ys, xs]
        ok = np.isfinite(z) & (z > 0.05) & (z < DETECT_MAX_RANGE_M)
        xs, ys, z = xs[ok], ys[ok], z[ok]
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
        return np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)

    def _reject_reason(self, c, label_map, depth):
        """Returns (drop_reason, warning) -- drop_reason is a string to drop
        the candidate entirely, or None to keep it (with an optional
        warning string when depth is too thin to run the geometric shadow
        test but the mask itself looks like a real object).

        A candidate with almost no valid depth is ambiguous: it could be a
        sliver of mask noise, or it could be a real (often dark/reflective)
        object the sensor just failed to range -- dropping the second case
        silently makes objects disappear from the menu with no way to tell
        the two apart. So: too few RGB mask pixels to be a real object ->
        drop. Enough mask pixels but too few valid depth points -> KEEP,
        flagged, since the object is visibly there even if unmeasured; the
        shadow/plane test that needs depth is simply skipped for it."""
        if depth is None or self.intrinsics is None:
            return None, None
        m = label_map == c["id"]
        mask_px = int(m.sum())
        pts = self._unproject(m, depth)
        if len(pts) < 30:
            if mask_px < 200:
                return f"only {len(pts)} valid depth points and a {mask_px} px mask -- too small to trust", None
            return None, f"only {len(pts)} of {mask_px} mask px have valid depth -- shadow test skipped, position may be wrong"
        # plane fit to the candidate's own points (SVD: normal = smallest singular vector)
        ctr = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        n = vt[-1]
        rms = float(np.sqrt(np.mean(((pts - ctr) @ n) ** 2)))
        # with realistic depth noise a flat surface is no longer flat to 5 mm:
        # widen both thresholds to ~2.5 sigma at this range
        sig = self._noise_sigma_at(float(np.median(pts[:, 2])))
        rms_thresh = max(SHADOW_PLANAR_RMS_M, 2.5 * sig)
        ring_thresh = max(SHADOW_RING_DIST_M, 2.5 * sig)
        if rms > rms_thresh:
            return None, None  # has real 3D relief -> an object
        # flat: is the surrounding surface on the SAME plane? then it's a shadow
        ring = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~m
        ring_pts = self._unproject(ring, depth, step=2)
        if len(ring_pts) < 30:
            return None, None
        ring_d = float(np.median(np.abs((ring_pts - ctr) @ n)))
        if ring_d < ring_thresh:
            return (f"shadow (flat, rms {rms*1000:.1f} mm; surroundings {ring_d*1000:.1f} mm from same plane)"), None
        return None, None  # flat but raised above its surroundings (plate, book) -> keep

    def _select_candidate(self, cid):
        """Menu click: hand the candidate's mask (cached in Process B) to the
        tracker. Same downstream path as a mouse click on the image."""
        cand = next((c for c in self.candidates if c["id"] == cid), None)
        if cand is None or self.rgb is None:
            return
        meta, payload = build_request(REQ_SELECT_MASK, frame_bgr=self.rgb, candidate_id=cid,
                                      detect_id=getattr(self, "detect_id", None))
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, mask = parse_response(resp_meta, resp_payload)
        if meta_d["status"] == STATUS_OK and mask is not None:
            self.mask = mask
            self.tracking_active = True
            self.selected_id = cid
            self.last_label = cand["label"]     # already classified by detect_all
            self.last_shape = cand["shape"]
            self.status_text = f"tracking id {cid}: {cand['label']}"
            print(f"[select] id {cid} {cand['label']} -> tracking, mask px={int(mask.sum())}")
        else:
            self.status_text = f"select failed: {meta_d.get('message', '')}"
            print(f"[select] {self.status_text}")
        self._refresh_detected_menu()

    # ---- step 1: search mode -------------------------------------------
    def _start_search(self, category):
        category = (category or "").strip()
        if not category:
            self.status_text = "type an object name first"
            return
        if self.tracking_active:
            self._reset()
        self.search_target = category
        self.search_best = None
        self._last_detect_end = 0.0   # poll immediately on the next frame
        self.status_text = f"searching for '{category}'"
        self.task.update({"state": "searching", "target": category, "pcd": None, "message": ""})
        print(f"[search] started for '{category}'")

    def _stop_search(self):
        if self.search_target is not None:
            print(f"[search] stopped ('{self.search_target}')")
        self.search_target = None
        self.search_best = None

    def _try_lock_search_target(self):
        """After a search-mode detect_all: lock onto the best candidate if it
        clears the threshold, else remember it for the status line."""
        if self.search_target is None or not self.candidates:
            self.search_best = None
            return
        scored = [c for c in self.candidates if "target_prob" in c]
        if not scored:
            self.status_text = "Process B returned no target scores -- is it the current sam2_service.py?"
            return
        ranked = sorted(scored, key=lambda c: c["target_prob"], reverse=True)
        best = ranked[0]
        second = ranked[1]["target_prob"] if len(ranked) > 1 else 0.0
        self.search_best = (best["id"], best["target_prob"])
        absolute = best["target_prob"] >= SEARCH_MATCH_PROB
        relative = best["target_prob"] >= SEARCH_MATCH_MIN and best["target_prob"] >= SEARCH_MATCH_RATIO * max(second, 1e-3)
        by_label = best["label"] == self.search_target
        print(f"[search] best id {best['id']} {best['label']} prob={best['target_prob']:.2f} (next {second:.2f}) "
              f"abs={absolute} rel={relative} label={by_label}")
        if absolute or relative or by_label:
            print(f"[search] match: id {best['id']} {best['label']} prob={best['target_prob']:.2f} -> locking")
            self._select_candidate(best["id"])
            if self.tracking_active:
                self.task.update({"state": "locked", "message": f"locked on id {best['id']} ({best['label']})"})
                if self.auto_plan:
                    self._start_view_plan()

    # ---- 3D position (direct USD intrinsics, no ROS) ----
    def _object_3d_position(self):
        if self.mask is None or self.depth is None or self.intrinsics is None:
            return None
        ys, xs = np.where(self.mask)
        if len(xs) == 0:
            return None
        z = self.depth[ys, xs]
        valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
        if valid.sum() < 20:
            return None
        xs, ys, z = xs[valid], ys[valid], z[valid]
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        cam_point = np.array([np.median(X), np.median(Y), np.median(z)])
        T = get_camera_pose_in_world(CAMERA_RGB_PATH)
        world_point = T[:3, :3] @ cam_point + T[:3, 3]
        return cam_point, world_point

    # ---- steps 3-5: view plan (teleport stands in for Go1 walk + arm move) ----
    # ---- next-best-view ---------------------------------------------------
    def _nbv_candidates(self, P, rig_xy):
        """Ring of reachable candidate poses -- same standoff/reach rules as
        the fixed plan, but NBV_N_AZIMUTH azimuths, each with a body-level
        and an arm-only pose."""
        ctr, R, top_z = read_table_geometry()

        def edge_dist(az):
            d = np.array([np.cos(az), np.sin(az)])
            rel = P[:2] - ctr[:2]
            b = 2 * np.dot(rel, d); c = np.dot(rel, rel) - R * R
            disc = b * b - 4 * c
            return max((-b + np.sqrt(disc)) / 2, 0.0) if disc >= 0 else 0.0

        out = []
        for i in range(NBV_N_AZIMUTH):
            az = 2 * np.pi * i / NBV_N_AZIMUTH
            d = np.array([np.cos(az), np.sin(az)])
            body_front = edge_dist(az) + BODY_EDGE_CLEARANCE_M
            cam_dist = max(body_front - ARM_FORWARD_REACH_M, MIN_VIEW_DIST_M)
            xy = P[:2] + cam_dist * d
            out.append({"name": f"body_{i}", "kind": "body",
                        "cam_pos": np.array([xy[0], xy[1], top_z + SIDE_CAM_HEIGHT_M]), "look_at": P.copy()})
            top_off = max(body_front - ARM_MAX_REACH_M, TOPDOWN_MIN_OFFSET_M)
            xy2 = P[:2] + top_off * d
            out.append({"name": f"arm_{i}", "kind": "arm",
                        "cam_pos": np.array([xy2[0], xy2[1], P[2] + TOPDOWN_HEIGHT_M]), "look_at": P.copy()})
        return out

    def _frontier_weights(self, frontier, surface):
        """Weight unknown voxels by proximity to the object's known surface,
        so the score reflects the TARGET's unseen surface rather than the
        much larger volume of empty space around it."""
        if surface is None or len(surface) == 0 or len(frontier) == 0:
            return np.ones(len(frontier))
        d2 = np.empty(len(frontier))
        step = max(1, 2_000_000 // max(len(surface), 1))
        for i in range(0, len(frontier), step):
            blk = frontier[i:i + step]
            d2[i:i + step] = ((blk[:, None, :] - surface[None, :, :]) ** 2).sum(-1).min(axis=1)
        return np.exp(-d2 / (2.0 * NBV_SURFACE_SIGMA ** 2))

    def _object_surface_points(self, P, radius=0.12):
        """Known occupied voxels close to the target -- its observed shell."""
        occ = self.belief.occupied_points()
        if len(occ) == 0:
            return occ
        return occ[np.linalg.norm(occ - np.asarray(P)[None, :], axis=1) <= radius]

    def _nbv_pick(self):
        """Score every unused candidate against the current belief and return
        (best_candidate, gain, n_frontier) or (None, 0, n) to stop."""
        P = self.plan_target
        top_z = read_table_geometry()[2]
        frontier = self.belief.frontier_near(P, NBV_FRONTIER_RADIUS, exclude_below_z=top_z)
        if len(frontier) < 50:
            return None, 0.0, len(frontier)
        surface = self._object_surface_points(P)
        w = self._frontier_weights(frontier, surface)
        total = float(w.sum())
        # normals are the same for every candidate -- compute once per step
        normals = self.belief._frontier_normals(frontier)
        self._nbv_normals = normals
        best, best_s = None, -1.0
        for c in self.nbv_candidates:
            if c["name"] in self.nbv_used:
                continue
            vis = self.belief.visible_from(c["cam_pos"], frontier, look_at=c["look_at"],
                                           half_fov_deg=NBV_HALF_FOV_DEG, max_range=NBV_MAX_RANGE,
                                           surface_normals=normals, min_incidence_cos=NBV_MIN_INCIDENCE_COS)
            sc = float(w[vis].sum())
            if sc > best_s:
                best, best_s = c, sc
        raw_gain = best_s / max(total, 1e-9) if best is not None else 0.0
        # Geometric visibility assumes every ray returns depth. In practice a
        # large share of pixels come back invalid (noise dropout, range, dark
        # surfaces), so only a fraction of "visible" voxels actually get
        # resolved. Scale the prediction by the yield this scene has actually
        # delivered so far (fraction of previously predicted-visible voxels
        # that really became observed), which keeps the stop rule meaningful.
        yield_est = getattr(self, "nbv_yield", None)
        gain = raw_gain * yield_est if yield_est is not None else raw_gain
        return best, gain, len(frontier)

    def _build_view_plan(self, P, rig_xy):
        """Four camera poses around target P (world xyz), given where the
        robot currently is (rig_xy). Returns list of {name, cam_pos, look_at}."""
        ctr, R, top_z = read_table_geometry()
        az0 = np.arctan2(rig_xy[1] - P[1], rig_xy[0] - P[0])   # direction target -> robot

        def edge_dist(az):
            """distance from the target, along azimuth az, to the table edge"""
            d = np.array([np.cos(az), np.sin(az)])
            rel = P[:2] - ctr[:2]
            b = 2 * np.dot(rel, d); c = np.dot(rel, rel) - R * R
            disc = b * b - 4 * c
            if disc < 0:               # target outside the table circle: treat edge as here
                return 0.0
            t = (-b + np.sqrt(disc)) / 2
            return max(t, 0.0)

        def body_view(name, az, cam_h):
            d = np.array([np.cos(az), np.sin(az)])
            body_front = edge_dist(az) + BODY_EDGE_CLEARANCE_M
            cam_dist = max(body_front - ARM_FORWARD_REACH_M, MIN_VIEW_DIST_M)
            xy = P[:2] + cam_dist * d
            return {"name": name, "cam_pos": np.array([xy[0], xy[1], top_z + cam_h]), "look_at": P.copy(),
                    "body_front_dist": body_front}

        d0 = np.array([np.cos(az0), np.sin(az0)])
        body_front0 = edge_dist(az0) + BODY_EDGE_CLEARANCE_M
        top_off = max(body_front0 - ARM_MAX_REACH_M, TOPDOWN_MIN_OFFSET_M)
        top_xy = P[:2] + top_off * d0
        plan = [
            body_view("approach", az0, APPROACH_CAM_HEIGHT_M),
            {"name": "arm_topdown", "cam_pos": np.array([top_xy[0], top_xy[1], P[2] + TOPDOWN_HEIGHT_M]),
             "look_at": P.copy(), "body_front_dist": body_front0},
            body_view("body_left", az0 + np.deg2rad(SIDE_AZIMUTH_DEG), SIDE_CAM_HEIGHT_M),
            body_view("body_right", az0 - np.deg2rad(SIDE_AZIMUTH_DEG), SIDE_CAM_HEIGHT_M),
        ]
        print(f"[plan] table ctr={ctr.round(3)} R={R:.3f} top_z={top_z:.3f}  target={P.round(3)}  az0={np.degrees(az0):.1f} deg")
        for v in plan:
            print(f"[plan]   {v['name']:<12} cam={v['cam_pos'].round(3)}  body_front={v['body_front_dist']:.2f} m from target")
        return plan

    def _start_view_plan(self):
        enforce_unit_scale()
        rigid, sv = camera_transform_is_rigid()
        if not rigid:
            self.status_text = f"REFUSING plan: camera transform has scale/shear (singular values {np.round(sv, 3)})"
            print(f"[plan] {self.status_text}")
            return
        pos = self._object_3d_position()
        if pos is None:
            self.status_text = "select/track an object first"
            return
        _, P = pos
        rig_prim = stage.GetPrimAtPath(RIG_PATH)
        rig_world = UsdGeom.Xformable(rig_prim).ComputeLocalToWorldTransform(0)
        parent_world = UsdGeom.Xformable(rig_prim.GetParent()).ComputeLocalToWorldTransform(0)
        self.rig_home = rig_world * parent_world.GetInverse()
        rig_xy = np.array(rig_world.ExtractTranslation())[:2]
        self.plan_target = np.asarray(P, float)
        self.task.update({"state": "capturing", "message": "view plan running"})
        # object size from the current mask -> tolerance for the re-seed guard
        pts = self._unproject(self.mask, self.depth, step=2)
        size = float(np.max(pts.max(axis=0) - pts.min(axis=0))) if len(pts) >= 30 else 0.0
        self.plan_target_tol = max(PLAN_MIN_TARGET_TOL_M, 0.75 * size)
        print(f"[plan] object size ~{size*100:.1f} cm -> re-seed tolerance {self.plan_target_tol*100:.1f} cm")
        # capture gate: some targets (dark/reflective) simply never return much
        # valid depth from ANY angle (real D435i behaviour, not a bug -- see
        # depth_noise.py's dark-surface dropout). A fixed 15% threshold then
        # refuses every single view for them. Scale the threshold to what was
        # actually achievable at the angle where this object was selected,
        # floored so a near-fully-invalid view is still never accepted.
        ys, xs = np.where(self.mask)
        md = self.depth[ys, xs]
        seen_frac = float((np.isfinite(md) & (md > 0.05) & (md < 5.0)).mean()) if len(md) else 0.0
        self.capture_min_valid_frac = max(CAPTURE_MIN_VALID_FRAC_FLOOR,
                                          min(CAPTURE_MIN_VALID_FRAC_DEFAULT, CAPTURE_ADAPTIVE_MARGIN * seen_frac))
        print(f"[plan] target had {seen_frac:.0%} valid depth at selection -> capture threshold "
              f"{self.capture_min_valid_frac:.0%} for this plan")
        self.nbv_used = set()
        self.nbv_yield = None       # learned per plan from verify measurements
        self.nbv_candidates = self._nbv_candidates(self.plan_target, rig_xy) if NBV_ENABLED else []
        if NBV_ENABLED:
            # First view still comes from the fixed rules: the belief is empty
            # at this point, so there is nothing to score against yet. Every
            # view after it is chosen from the belief.
            self.plan = self._build_view_plan(self.plan_target, rig_xy)[:1]
            print(f"[nbv] enabled: {len(self.nbv_candidates)} candidate poses, "
                  f"max {NBV_MAX_VIEWS} views, stop below {NBV_MIN_GAIN:.0%} gain")
        else:
            self.plan = self._build_view_plan(self.plan_target, rig_xy)
        self._start_session()          # fresh session for this object
        self.plan_idx = -1
        self._advance_plan()

    def _advance_plan(self):
        self.plan_idx += 1
        # next-best-view: when the precomputed list runs out, pick the next
        # pose from the CURRENT belief rather than stopping.
        if (NBV_ENABLED and self.plan_idx >= len(self.plan) and self.belief is not None
                and self.pose_count > 0 and len(self.plan) < NBV_MAX_VIEWS):
            try:
                cand, gain, n_fr = self._nbv_pick()
                if cand is None:
                    print(f"[nbv] only {n_fr} reachable frontier voxels left -> done")
                elif gain < NBV_MIN_GAIN:
                    print(f"[nbv] best remaining candidate would see {gain:.1%} of the frontier "
                          f"(< {NBV_MIN_GAIN:.0%}) -> done after {len(self.plan)} views")
                else:
                    az = np.degrees(np.arctan2(cand["cam_pos"][1] - self.plan_target[1],
                                               cand["cam_pos"][0] - self.plan_target[0]))
                    print(f"[nbv] view {len(self.plan)+1}: {cand['name']} ({cand['kind']}) az={az:.0f} deg, "
                          f"predicted gain {gain:.1%} of {n_fr} frontier voxels")
                    self.nbv_used.add(cand["name"])
                    # keep the predicted-visible set to check against reality after capture
                    try:
                        top_z2 = read_table_geometry()[2]
                        fr2 = self.belief.frontier_near(self.plan_target, NBV_FRONTIER_RADIUS, exclude_below_z=top_z2)
                        nrm2 = self.belief._frontier_normals(fr2)
                        vis2 = self.belief.visible_from(cand["cam_pos"], fr2, look_at=cand["look_at"],
                                                        half_fov_deg=NBV_HALF_FOV_DEG, max_range=NBV_MAX_RANGE,
                                                        surface_normals=nrm2, min_incidence_cos=NBV_MIN_INCIDENCE_COS)
                        self.nbv_predicted = fr2[vis2]
                    except Exception:
                        self.nbv_predicted = None
                    self.plan.append({"name": cand["name"], "cam_pos": cand["cam_pos"],
                                      "look_at": cand["look_at"], "body_front_dist": float("nan")})
            except Exception as exc:
                print(f"[nbv] selection failed ({exc}) -- ending plan")
        if self.plan_idx >= len(self.plan):
            self.plan_state = None
            self.status_text = f"plan done: {self.pose_count} views captured -- press Fuse Cloud"
            print(f"[plan] finished, {self.pose_count} views captured in {self.session_dir}")
            self.task.update({"state": "captured", "session": self.session_dir,
                              "message": f"{self.pose_count} views captured"})
            self._save_belief()
            if self.auto_fuse:
                self._generate_pointcloud()
            return
        v = self.plan[self.plan_idx]
        # stop the 2-frame tracker: a teleport is far too big a jump for it
        self.tracking_active = False
        self.mask = None
        try:
            teleport_rig_so_camera_is(usd_camera_lookat(v["cam_pos"], v["look_at"]))
        except Exception as exc:
            self.status_text = f"teleport failed: {exc}"
            print(f"[plan] teleport failed on {v['name']}: {exc}")
            self.plan_state = None
            return
        self.plan_state = "settle"
        self.plan_settle = 0
        self.status_text = f"view {self.plan_idx + 1}/{len(self.plan)} '{v['name']}': settling"
        print(f"[plan] teleported for '{v['name']}' -> cam {v['cam_pos'].round(3)}")

    def _plan_step(self):
        """Called once per frame while a plan is running."""
        if self.plan_state == "settle":
            self.plan_settle += 1
            if self.plan_settle >= SETTLE_FRAMES:
                self.plan_state = "reseed"
            return
        if self.plan_state == "reseed":
            v = self.plan[self.plan_idx]
            T = get_camera_pose_in_world(CAMERA_RGB_PATH)        # opencv-cam -> world (new pose)
            print(f"[plan] '{v['name']}' camera now at {T[:3, 3].round(3)} (wanted {v['cam_pos'].round(3)})")
            uv = self._project_world_point(self.plan_target, T)
            if uv is None:
                print(f"[plan] '{v['name']}': target projects outside the image -- skipping view")
                self._advance_plan()
                return
            u, vv = uv
            self._send_click(u, vv, 1)      # re-seed SAM2 on the known target pixel
            ok, why = self._reseed_ok(u, vv)
            if not ok:
                print(f"[plan] '{v['name']}': click re-seed rejected ({why}) -- trying detect_all fallback")
                self._reset_tracking_only()
                cid, dist = self._nearest_candidate_to_target()
                if cid is not None:
                    print(f"[plan] '{v['name']}': candidate id {cid} is {dist*100:.1f} cm from target -> selecting")
                    self._select_candidate(cid)
                    ok, why = self._reseed_ok(None, None)
                else:
                    why = f"no detect_all candidate within {self.plan_target_tol*100:.0f} cm of target"
            if not ok:
                print(f"[plan] '{v['name']}': SKIPPED ({why})")
                self._reset_tracking_only()
                self._advance_plan()
                return
            before = self.pose_count
            self._capture_pose()             # has its own depth-validity refusal
            if self.pose_count == before:
                print(f"[plan] '{v['name']}': capture refused ({self.status_text})")
            else:
                print(f"[plan] '{v['name']}': captured as pose_{before:02d}")
            self._advance_plan()

    def _reseed_ok(self, u, vv):
        """Sanity-check the mask obtained after a teleport. Returns (ok, reason)."""
        if self.mask is None or not self.tracking_active:
            return False, "no mask"
        h, w = self.mask.shape
        frac = float(self.mask.sum()) / (h * w)
        if frac > PLAN_MAX_MASK_FRAC:
            return False, f"mask covers {frac:.0%} of image (table/background, not the object)"
        if u is not None and not self.mask[vv, u]:
            return False, "mask does not contain the click pixel"
        pos = self._object_3d_position()
        if pos is None:
            return False, "no valid depth under the mask"
        dist = float(np.linalg.norm(pos[1] - self.plan_target))
        if dist > self.plan_target_tol:
            return False, f"mask centroid {dist*100:.1f} cm from target (tol {self.plan_target_tol*100:.0f} cm)"
        return True, f"ok ({frac:.1%} of image, {dist*100:.1f} cm from target)"

    def _reset_tracking_only(self):
        """Drop the current mask/tracking without touching plan state or menu."""
        meta, payload = build_request(REQ_RESET)
        self.sock.send_multipart([meta, payload])
        self.sock.recv_multipart()
        self.mask = None
        self.tracking_active = False
        self.selected_id = None

    def _nearest_candidate_to_target(self):
        """Run detect_all on the current view and return (id, dist) of the
        candidate whose 3D position is closest to the plan target, or
        (None, None) if none is within tolerance."""
        self._detect_props()
        best, best_d = None, None
        T = get_camera_pose_in_world(CAMERA_RGB_PATH)
        for c in self.candidates:
            pts = self._unproject(self.label_map == c["id"], self.depth, step=2)
            if len(pts) < 20:
                continue
            cam_pt = np.median(pts, axis=0)
            world_pt = T[:3, :3] @ cam_pt + T[:3, 3]
            d = float(np.linalg.norm(world_pt - self.plan_target))
            print(f"[plan]   candidate id {c['id']} {c['label']}: {d*100:.1f} cm from target")
            if d <= self.plan_target_tol and (best_d is None or d < best_d):
                best, best_d = c["id"], d
        return best, best_d

    def _project_world_point(self, P, T_cam_to_world):
        """world xyz -> (u, v) pixel in the current RGB image, or None."""
        p = np.linalg.inv(T_cam_to_world) @ np.array([P[0], P[1], P[2], 1.0])
        if p[2] <= 0.05:
            return None
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
        u = int(round(fx * p[0] / p[2] + cx)); vv = int(round(fy * p[1] / p[2] + cy))
        h, w = self.rgb.shape[:2]
        if not (0 <= u < w and 0 <= vv < h):
            return None
        return u, vv

    def _abort_plan(self):
        if self.plan_state is not None:
            print(f"[plan] aborted at view {self.plan_idx + 1}")
        self.plan_state = None

    def _return_rig_home(self):
        if self.rig_home is None:
            self.status_text = "no saved home pose (run a plan first)"
            return
        self.plan_state = None
        self.tracking_active = False
        self.mask = None
        set_rig_local_matrix(self.rig_home)
        self.status_text = "rig returned to pre-plan pose"

    # ---- external command interface ----------------------------------
    def _poll_commands(self):
        """Non-blocking check of the command socket; at most one command per frame."""
        try:
            parts = self.cmd_sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception as exc:
            print(f"[cmd] recv failed: {exc}")
            return
        try:
            req = json.loads(parts[0].decode("utf-8")) if parts else {}
            reply = self._handle_command(req)
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[cmd] error handling {parts[:1]}: {exc}")
        try:
            self.cmd_sock.send_multipart([json.dumps(reply).encode("utf-8")])
        except Exception as exc:
            print(f"[cmd] reply failed: {exc}")

    def _handle_command(self, req):
        cmd = req.get("cmd")
        if cmd == "status":
            return {"ok": True, "task": dict(self.task), "tracking": self.tracking_active,
                    "views": self.pose_count, "status_text": self.status_text}
        if cmd == "abort":
            self._abort_plan()
            self._stop_search()
            self._reset()
            self.task.update({"state": "idle", "message": "aborted"})
            return {"ok": True}
        if cmd == "inspect":
            target = (req.get("target") or "").strip()
            if not target:
                return {"ok": False, "error": "'target' (object name) is required"}
            if self.rgb is None or self.depth is None:
                return {"ok": False, "error": "no camera frames yet"}
            self._abort_plan(); self._stop_search(); self._reset()
            goal = req.get("goal")          # {"x":..,"y":..,"theta_deg":..,"frame":"world"|"start"} or None
            if goal is not None and req.get("teleport_to_goal", True):
                self._teleport_to_goal_pose(goal)
            self.auto_plan = bool(req.get("auto_plan", True))
            self.auto_fuse = bool(req.get("auto_fuse", True))
            self._start_search(target)
            self.task.update({"goal": goal})
            print(f"[cmd] inspect target='{target}' goal={goal} auto_plan={self.auto_plan} auto_fuse={self.auto_fuse}")
            return {"ok": True, "task": dict(self.task)}
        return {"ok": False, "error": f"unknown cmd '{cmd}' (use inspect / status / abort)"}

    def _teleport_to_goal_pose(self, goal):
        """Locomotion stand-in: put the rig at (x, y, current height) with
        heading theta. 'frame': 'world' (default) or 'start' = relative to
        the rig pose when this script was started (x forward, y left)."""
        rig_prim = stage.GetPrimAtPath(RIG_PATH)
        rig_world = UsdGeom.Xformable(rig_prim).ComputeLocalToWorldTransform(0)
        pos_now = np.array(rig_world.ExtractTranslation())
        if self.rig_start is None:
            self.rig_start = (pos_now.copy(), self._rig_yaw())
        x, y = float(goal["x"]), float(goal["y"])
        theta = np.deg2rad(float(goal.get("theta_deg", 0.0)))
        if goal.get("frame", "world") == "start":
            p0, yaw0 = self.rig_start
            c, s_ = np.cos(yaw0), np.sin(yaw0)
            x, y = p0[0] + c * x - s_ * y, p0[1] + s_ * x + c * y
            theta = yaw0 + theta
        cam_pos = np.array([x, y, pos_now[2]])
        look_at = cam_pos + np.array([np.cos(theta), np.sin(theta), -np.tan(np.deg2rad(GOAL_PITCH_DEG))])
        teleport_rig_so_camera_is(usd_camera_lookat(cam_pos, look_at))
        self.tracking_active = False; self.mask = None
        print(f"[cmd] teleported rig to goal ({x:.2f}, {y:.2f}, yaw {np.degrees(theta):.0f} deg)")

    def _rig_yaw(self):
        """Heading of the camera's forward axis projected on the ground (rad)."""
        T = get_camera_pose_in_world(CAMERA_RGB_PATH)     # OpenCV frame: forward = +z
        f = T[:3, 2]
        return float(np.arctan2(f[1], f[0]))

    # ---- multi-view capture ----
    def _start_session(self):
        self.belief = None      # one belief per session (per target inspection)
        if self.last_label is None:
            self._classify()
        name = (self.last_label or "object").replace(" ", "_")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(SESSION_ROOT, f"{name}_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.pose_count = 0
        self.captured_poses = []
        meta = {"intrinsics": self.intrinsics, "label": self.last_label, "shape": self.last_shape,
                "depth_noise": {k: self.noise_cfg[k] for k in ("enabled", "severity", "disp_sigma_px", "baseline_m")}}
        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _capture_pose(self):
        if self.mask is None or self.rgb is None or self.depth is None:
            self.status_text = "nothing selected -- click an object first"
            return
        ys, xs = np.where(self.mask)
        mask_depth = self.depth[ys, xs]
        valid = np.isfinite(mask_depth) & (mask_depth > 0.05) & (mask_depth < 5.0)
        valid_frac = valid.sum() / max(len(mask_depth), 1)
        min_frac = getattr(self, "capture_min_valid_frac", CAPTURE_MIN_VALID_FRAC_DEFAULT)
        if valid_frac < min_frac:
            self.status_text = f"REFUSING capture -- only {valid_frac:.0%} valid depth (need {min_frac:.0%}), re-click and retry"
            return
        rigid, sv = camera_transform_is_rigid()
        if not rigid:
            self.status_text = f"REFUSING capture: camera transform has scale/shear (singular values {np.round(sv, 3)})"
            print(f"[capture] {self.status_text}")
            return
        if self.session_dir is None:
            self._start_session()

        # Table-plane clip: a shadow or a strip of mask fringe lies ON the
        # tabletop; the object itself never does (its bottom face is
        # unseen). Unproject the mask to world and drop pixels within
        # TABLE_CLIP_M of the tabletop plane before saving.
        mask_save = self.mask.copy()
        try:
            _, _, top_z = read_table_geometry()
            T = get_camera_pose_in_world(CAMERA_RGB_PATH)
            fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
            cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
            zv = np.where(valid, mask_depth, np.nan)
            wz = (T[2, 0] * (xs - cx) * zv / fx + T[2, 1] * (ys - cy) * zv / fy + T[2, 2] * zv + T[2, 3])
            clip_m = max(TABLE_CLIP_M, 2.0 * self._noise_sigma_at(float(np.nanmedian(zv))))
            on_table = np.isfinite(wz) & (wz < top_z + clip_m)
            if on_table.any():
                mask_save[ys[on_table], xs[on_table]] = False
            print(f"[capture] pose {self.pose_count}: clipped {int(on_table.sum())} of {len(xs)} mask px "
                  f"({on_table.mean():.0%}) lying on the table plane")
        except Exception as exc:
            print(f"[capture] table-plane clip skipped: {exc}")

        idx = self.pose_count
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_rgb.png"), self.rgb)
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_mask.png"),
                    mask_save.astype(np.uint8) * 255)
        depth_mm = np.clip(np.nan_to_num(self.depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_depth.png"), depth_mm)
        # ---- voxel belief: fuse the WHOLE frame, not just the object ----
        if VOXEL_BELIEF:
            try:
                if self.belief is None:
                    ctr, radius, top_z = read_table_geometry()
                    self.belief = VoxelBelief.around_table(ctr, radius, top_z, res=VOXEL_RES, height=VOXEL_HEIGHT)
                T = get_camera_pose_in_world(CAMERA_RGB_PATH)
                fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]
                t0 = time.time()
                frame_valid = float((np.isfinite(self.depth) & (self.depth > 0.05) & (self.depth < 5.0)).mean())
                n = self.belief.integrate_depth(self.depth, T, fx, fy, cx, cy, stride=VOXEL_RAY_STRIDE)
                st = self.belief.stats()
                print(f"[belief] view {idx}: {n} rays ({frame_valid:.0%} of frame has valid depth) in {(time.time()-t0)*1000:.0f} ms -> "
                      f"occupied {st['occupied']} free {st['free']} unknown {st['unknown']} "
                      f"({st['observed_frac']:.1%} observed)")
                if self.plan_target is not None:
                    top_z = read_table_geometry()[2]
                    unk = self.belief.unknown_near(self.plan_target, VOXEL_FRONTIER_RADIUS, exclude_below_z=top_z)
                    fr = self.belief.frontier_near(self.plan_target, NBV_FRONTIER_RADIUS, exclude_below_z=top_z)
                    # frontier can GROW after a view: opening up free space turns
                    # previously sealed unknown voxels into reachable ones.
                    print(f"[belief] near target: {len(unk)} unknown within {VOXEL_FRONTIER_RADIUS*100:.0f} cm, "
                          f"{len(fr)} REACHABLE frontier within {NBV_FRONTIER_RADIUS*100:.0f} cm")
                    pred = getattr(self, "nbv_predicted", None)
                    if pred is not None and len(pred):
                        pi = self.belief.world_to_idx(pred)
                        ok = self.belief.in_bounds(pi)
                        pi = pi[ok]
                        now_obs = self.belief.observed[pi[:, 0], pi[:, 1], pi[:, 2]]
                        obs_frac = float(now_obs.mean())
                        prev = getattr(self, "nbv_yield", None)
                        # running estimate, seeded by the first measurement
                        self.nbv_yield = obs_frac if prev is None else 0.5 * prev + 0.5 * obs_frac
                        print(f"[nbv] verify: {int(now_obs.sum())}/{len(pi)} predicted-visible voxels "
                              f"actually became observed ({obs_frac:.0%}); "
                              f"yield estimate now {self.nbv_yield:.0%}")
                        self.nbv_predicted = None
            except Exception as exc:
                print(f"[belief] integration failed: {exc}")

        if self.noise_cfg.get("enabled") and self._clean_depth is not None:
            # ideal depth alongside the degraded one, so fusion can be evaluated with/without noise
            clean_mm = np.clip(np.nan_to_num(self._clean_depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_depth_clean.png"), clean_mm)

        T = get_camera_pose_in_world(CAMERA_RGB_PATH)
        pose_entry = {"index": idx, "camera_frame": "Camera_RGB_opencv_optical", "cam_to_world": T.tolist()}
        self.captured_poses.append(pose_entry)
        meta_path = os.path.join(self.session_dir, "session_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["poses"] = self.captured_poses
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.pose_count += 1
        self.status_text = f"captured view {idx}"

    def _generate_pointcloud(self):
        if self.session_dir is None or self.pose_count < 2:
            self.status_text = "need >=2 captured views before fusing"
            return
        meta, payload = build_request(REQ_GENERATE_PC, session_dir=self.session_dir)
        self.sock.send_multipart([meta, payload])
        resp_meta, resp_payload = self.sock.recv_multipart()
        meta_d, _ = parse_response(resp_meta, resp_payload)
        self.status_text = f"{meta_d.get('status')}: {meta_d.get('message', '')}"
        if meta_d.get("status") == STATUS_OK:
            self.task.update({"state": "done", "session": self.session_dir,
                              "pcd": os.path.join(self.session_dir, "fused_pointcloud.pcd"),
                              "message": meta_d.get("message", "")})
        else:
            self.task.update({"state": "failed", "message": f"fusion failed: {meta_d.get('message', '')}"})

    def _view_pointcloud(self):
        if self.session_dir is None:
            return
        pcd_path = os.path.join(self.session_dir, "fused_pointcloud.pcd")
        meta, payload = build_request(REQ_VIEW_PC, ply_path=pcd_path)
        self.sock.send_multipart([meta, payload])
        self.sock.recv_multipart()


import omni.kit.app
_gui = NativeSegmentGUI()
print("Native SAM2 Segmentation GUI window created inside Isaac Sim.")
