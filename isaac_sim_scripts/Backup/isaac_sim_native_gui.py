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
CAMERA_RGB_PATH = "/World/d435i_camera/Camera_RGB"
CAMERA_DEPTH_PATH = "/World/d435i_camera/Camera_Depth"
ZMQ_ADDR = "tcp://131.220.7.222:5555"
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
SEARCH_MATCH_PROB = 0.25     # lock onto a candidate when CLIP prob for the target >= this (or argmax label == target)
DETECT_MAX_RANGE_M = 5.0     # candidates need some valid depth closer than this to be listed
SHADOW_PLANAR_RMS_M = 0.005  # candidate counts as "flat" if its points fit a plane within 5 mm RMS
SHADOW_RING_DIST_M = 0.006   # ...and as a shadow if the surrounding ring is within 6 mm of that plane

# ---- steps 3-5: teleport view planner (stand-ins for Go1 + OpenManipulator-X) ----
RIG_PATH = "/World/d435i_camera"          # Xform that gets teleported (all sensor children move with it)
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
PLAN_MAX_MASK_FRAC = 0.15      # a re-seeded mask bigger than this fraction of the image is not a table prop
TABLE_CLIP_M = 0.004           # at capture, drop mask pixels whose world z is within this of the tabletop (shadows, fringe)
PLAN_MIN_TARGET_TOL_M = 0.08   # re-seeded mask centroid must be within max(this, 0.75*object size) of the target
# ---------------------------------------------------------------------------

if IPC_COMMON_DIR not in sys.path:
    sys.path.insert(0, IPC_COMMON_DIR)
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
                    ui.Label("Search for", height=20)
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
            try:
                self.sock.close()
            except Exception:
                pass
            print("[isaac_sim_native_gui] window closed -- update loop and ZMQ socket stopped.")

    # ---- per-frame update ----
    def _on_update(self, event):
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
        kept, dropped = [], []
        for c in raw:
            reason = self._reject_reason(c, label_map, depth)
            (dropped if reason else kept).append((c, reason))
        self.candidates = [c for c, _ in kept]
        self.label_map = label_map
        self.selected_id = None
        self.status_text = f"detected {len(self.candidates)} props ({len(dropped)} filtered)"
        print(f"[detect] service: {meta_d.get('message')}")
        for c, r in dropped:
            print(f"[detect] dropped id {c['id']} {c['label']}: {r}")
        for c in self.candidates:
            tp = f" target_prob={c['target_prob']:.2f}" if "target_prob" in c else ""
            print(f"[detect] kept id {c['id']} {c['label']} ({c['shape']} {c['confidence']:.2f}) bbox={c['bbox']}{tp}")
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
        """Return a string reason to drop this candidate, or None to keep it.
        Only two reasons: no usable depth, or 'this is a shadow'.
        Floating / far objects are deliberately kept (user wants them listed)."""
        if depth is None or self.intrinsics is None:
            return None
        m = label_map == c["id"]
        pts = self._unproject(m, depth)
        if len(pts) < 30:
            return f"only {len(pts)} valid depth points within {DETECT_MAX_RANGE_M} m"
        # plane fit to the candidate's own points (SVD: normal = smallest singular vector)
        ctr = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        n = vt[-1]
        rms = float(np.sqrt(np.mean(((pts - ctr) @ n) ** 2)))
        if rms > SHADOW_PLANAR_RMS_M:
            return None  # has real 3D relief -> an object
        # flat: is the surrounding surface on the SAME plane? then it's a shadow
        ring = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~m
        ring_pts = self._unproject(ring, depth, step=2)
        if len(ring_pts) < 30:
            return None
        ring_d = float(np.median(np.abs((ring_pts - ctr) @ n)))
        if ring_d < SHADOW_RING_DIST_M:
            return f"shadow (flat, rms {rms*1000:.1f} mm; surroundings {ring_d*1000:.1f} mm from same plane)"
        return None  # flat but raised above its surroundings (plate, book) -> keep

    def _select_candidate(self, cid):
        """Menu click: hand the candidate's mask (cached in Process B) to the
        tracker. Same downstream path as a mouse click on the image."""
        cand = next((c for c in self.candidates if c["id"] == cid), None)
        if cand is None or self.rgb is None:
            return
        meta, payload = build_request(REQ_SELECT_MASK, frame_bgr=self.rgb, candidate_id=cid)
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
        if self.tracking_active:
            self._reset()
        self.search_target = category
        self.search_best = None
        self._last_detect_end = 0.0   # poll immediately on the next frame
        self.status_text = f"searching for '{category}'"
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
            self.status_text = f"'{self.search_target}' not in CLIP vocabulary on Process B"
            return
        best = max(scored, key=lambda c: c["target_prob"])
        self.search_best = (best["id"], best["target_prob"])
        if best["target_prob"] >= SEARCH_MATCH_PROB or best["label"] == self.search_target:
            print(f"[search] match: id {best['id']} {best['label']} prob={best['target_prob']:.2f} -> locking")
            self._select_candidate(best["id"])

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
        # object size from the current mask -> tolerance for the re-seed guard
        pts = self._unproject(self.mask, self.depth, step=2)
        size = float(np.max(pts.max(axis=0) - pts.min(axis=0))) if len(pts) >= 30 else 0.0
        self.plan_target_tol = max(PLAN_MIN_TARGET_TOL_M, 0.75 * size)
        print(f"[plan] object size ~{size*100:.1f} cm -> re-seed tolerance {self.plan_target_tol*100:.1f} cm")
        self.plan = self._build_view_plan(self.plan_target, rig_xy)
        self._start_session()          # fresh session for this object
        self.plan_idx = -1
        self._advance_plan()

    def _advance_plan(self):
        self.plan_idx += 1
        if self.plan_idx >= len(self.plan):
            self.plan_state = None
            self.status_text = f"plan done: {self.pose_count} views captured -- press Fuse Cloud"
            print(f"[plan] finished, {self.pose_count} views captured in {self.session_dir}")
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

    # ---- multi-view capture ----
    def _start_session(self):
        if self.last_label is None:
            self._classify()
        name = (self.last_label or "object").replace(" ", "_")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(SESSION_ROOT, f"{name}_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.pose_count = 0
        self.captured_poses = []
        meta = {"intrinsics": self.intrinsics, "label": self.last_label, "shape": self.last_shape}
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
        if valid_frac < 0.15:
            self.status_text = f"REFUSING capture -- only {valid_frac:.0%} valid depth, re-click and retry"
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
            on_table = np.isfinite(wz) & (wz < top_z + TABLE_CLIP_M)
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
