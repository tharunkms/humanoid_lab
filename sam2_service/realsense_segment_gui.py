#!/usr/bin/env python3
"""
realsense_segment_gui.py -- Process A for a REAL Intel RealSense D435i (handheld).

Same two-process design and wire protocol as the Isaac Sim GUI: this
window captures frames and owns geometry; sam2_service.py (Process B)
does SAM2 / CLIP / fusion. What differs from the simulator variant:

  frames       pyrealsense2, colour 1280x720 + depth aligned to colour
  intrinsics   read from the device (colour stream after alignment)
  camera pose  RGB-D visual odometry (Open3D, keyframe-based) on the whole
               frame; world = first camera frame. Poses go into the session
               so fusion uses the same direct-pose path as the simulator.
               (Registering the object alone fails: a cylinder/ball/box
               face is degenerate for shape matching.) On the robot, TF
               from kinematics becomes the primary pose source and this
               a cross-check.
  table plane  estimated per frame from depth (RANSAC) instead of USD,
               used for the shadow test and the table-plane clip
  view plan    none -- you move the camera; press 'p' at each view.
               The re-seed guard (mask size, click containment) still
               applies before a view is saved.
  depth noise  real. No model applied; the session is tagged so Process B
               uses its noise-aware fusion filters.

Run (host, sam2 venv, Process B already listening):
    pip install pyrealsense2          # once; see NOTE below if it fails
    python3 realsense_segment_gui.py --zmq tcp://localhost:5555

Interface: one window -- camera view (left), control panel (right) with
clickable buttons, a search box (click it or press 's', type, Enter), a
clickable list of detected objects, live object / tracking / session
readouts, and a log strip. Keyboard shortcuts:
    left click   positive point -> segment + track      right click  negative point
    d detect   1..9 select   s search box   x stop search   c classify   p capture
    g fuse     v view cloud  n new session  t RGB/depth view  r reset     q quit

NOTE pyrealsense2 wheels exist for Python 3.12 in recent librealsense
releases (>= 2.55). If `pip install pyrealsense2` finds no wheel for the
venv's Python, install it into the system python3 instead and run this
script with that interpreter (only numpy, opencv-python and pyzmq are
needed here; SAM2/torch stay in Process B).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipc_common import (
    build_request, parse_response,
    REQ_CLICK, REQ_TRACK, REQ_RESET, REQ_CLASSIFY, REQ_GENERATE_PC, REQ_VIEW_PC,
    REQ_DETECT_ALL, REQ_SELECT_MASK,
    STATUS_OK, STATUS_NO_OBJECT, PAYLOAD_LABEL_MAP,
)

# ---- settings (mirror the Isaac GUI where the concept is the same) ----
SESSION_ROOT = os.path.expanduser("~/realsense_pc_sessions")
WIDTH, HEIGHT, FPS = 1280, 720, 30
DEPTH_MIN_M, DEPTH_MAX_M = 0.20, 4.0
SEARCH_INTERVAL_S = 2.0
SEARCH_MATCH_PROB, SEARCH_MATCH_MIN, SEARCH_MATCH_RATIO = 0.25, 0.08, 3.0
SHADOW_PLANAR_RMS_M, SHADOW_RING_DIST_M = 0.005, 0.006
TABLE_CLIP_M = 0.004
TABLE_PLANE_MIN_INLIERS = 0.20   # RANSAC plane must explain this share of depth points to be used for clipping
CAPTURE_MIN_VALID_FRAC = 0.15
PLAN_MAX_MASK_FRAC = 0.15
# real D435i noise level, for the noise-aware thresholds: sigma_z = z^2/(f b) * disp_sigma
NOISE = dict(enabled=True, severity=1.0, disp_sigma_px=0.08, baseline_m=0.050, source="real D435i")


# ---- visual odometry settings ----
ODOM_SCALE = 0.5            # run odometry on a half-resolution frame (speed)
ODOM_KEYFRAME_TRANS_M = 0.12   # start a new keyframe after moving this far ...
ODOM_KEYFRAME_ROT_DEG = 12.0   # ... or rotating this much (keeps drift low, overlap high)
ODOM_DEPTH_MIN, ODOM_DEPTH_MAX = 0.2, 3.0


class RGBDOdometry:
    """
    Camera pose from the camera itself (Open3D RGB-D odometry, hybrid
    photometric + geometric term). World frame = the first frame's camera
    frame (OpenCV convention: x right, y down, z forward -- the same
    convention the simulator sessions use, so fusion is unchanged).

    Keyframe-based: every frame is registered against the last keyframe
    (not the previous frame), which keeps drift low; a new keyframe is
    taken when the camera has moved/rotated enough. Uses the WHOLE frame
    (desk, walls, cables, labels) -- a bare cylinder is degenerate for
    registration, the scene around it is not.
    UNTESTED API DETAILS (Open3D 0.17+): compute_rgbd_odometry signature
    and OdometryOption field names -- both wrapped; a failure only prints.
    """

    def __init__(self, intr, width, height):
        import open3d as o3d
        self.o3d = o3d
        sw, sh = int(width * ODOM_SCALE), int(height * ODOM_SCALE)
        self.size = (sw, sh)
        self.K = o3d.camera.PinholeCameraIntrinsic(sw, sh, intr["fx"] * ODOM_SCALE, intr["fy"] * ODOM_SCALE,
                                                   intr["cx"] * ODOM_SCALE, intr["cy"] * ODOM_SCALE)
        self.opt = o3d.pipelines.odometry.OdometryOption()
        try:
            self.opt.depth_diff_max = 0.05
            self.opt.depth_min = ODOM_DEPTH_MIN
            self.opt.depth_max = ODOM_DEPTH_MAX
        except Exception as exc:
            print(f"[odom] OdometryOption fields not settable ({exc}) -- using defaults")
        self.jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self.key_rgbd = None
        self.T_key = np.eye(4)          # keyframe cam -> world
        self.T_rel = np.eye(4)          # current cam -> keyframe cam (last estimate, used as init)
        self.T = np.eye(4)              # current cam -> world
        self.ok = False
        self.n_key = 0
        self.fails = 0
        self.last_ms = 0.0

    def _rgbd(self, rgb_bgr, depth_m):
        o3d = self.o3d
        small_rgb = cv2.resize(rgb_bgr, self.size, interpolation=cv2.INTER_AREA)
        small_d = cv2.resize(depth_m, self.size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        color = o3d.geometry.Image(np.ascontiguousarray(small_rgb[:, :, ::-1]))
        depth = o3d.geometry.Image(np.ascontiguousarray(small_d))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=ODOM_DEPTH_MAX, convert_rgb_to_intensity=True)

    def update(self, rgb_bgr, depth_m):
        t0 = time.time()
        cur = self._rgbd(rgb_bgr, depth_m)
        if self.key_rgbd is None:
            self.key_rgbd, self.T_key, self.T_rel, self.T, self.ok = cur, np.eye(4), np.eye(4), np.eye(4), True
            self.n_key = 1
            return self.T
        try:
            success, T_cur_to_key, _ = self.o3d.pipelines.odometry.compute_rgbd_odometry(
                cur, self.key_rgbd, self.K, self.T_rel, self.jac, self.opt)
        except Exception as exc:
            print(f"[odom] compute_rgbd_odometry failed: {exc}")
            success, T_cur_to_key = False, None
        self.last_ms = (time.time() - t0) * 1000.0
        if not success:
            self.fails += 1
            self.ok = False
            if self.fails >= 5:        # lost against this keyframe: re-anchor on the current frame at the last good pose
                self.key_rgbd, self.T_key, self.T_rel, self.fails = cur, self.T.copy(), np.eye(4), 0
                self.n_key += 1
                print("[odom] re-anchored keyframe at last good pose (tracking had failed 5x)")
            return self.T
        self.fails = 0
        self.ok = True
        self.T_rel = np.asarray(T_cur_to_key)
        self.T = self.T_key @ self.T_rel
        # new keyframe when we've moved enough from the current one
        t = np.linalg.norm(self.T_rel[:3, 3])
        ang = np.degrees(np.arccos(np.clip((np.trace(self.T_rel[:3, :3]) - 1) / 2, -1, 1)))
        if t > ODOM_KEYFRAME_TRANS_M or ang > ODOM_KEYFRAME_ROT_DEG:
            self.key_rgbd, self.T_key, self.T_rel = cur, self.T.copy(), np.eye(4)
            self.n_key += 1
        return self.T


class FrameGrabber:
    """
    Pulls frames from the camera and runs odometry at the camera's full
    frame rate in a background thread, independent of the GUI loop (which
    stalls 50-150 ms per SAM2 call -- too much motion between odometry
    steps when the two ran in the same loop). The GUI takes the latest
    frame TOGETHER with the pose that belongs to it.
    """

    def __init__(self, pipe, align, depth_scale, filters, odom):
        import threading
        self.pipe, self.align, self.depth_scale, self.filters, self.odom = pipe, align, depth_scale, filters, odom
        self.lock = threading.Lock()
        self.latest = None          # (rgb, depth, T_cam_to_world, odom_ok, seq)
        self.seq = 0
        self.fps = 0.0
        self._t = time.time()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        rs = None
        while not self._stop:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=2000)
            except Exception as exc:
                print(f"[rs] wait_for_frames: {exc}"); continue
            frames = self.align.process(frames)
            d = frames.get_depth_frame(); c = frames.get_color_frame()
            if not d or not c:
                continue
            for f in self.filters:
                d = f.process(d)
            rgb = np.asanyarray(c.get_data()).copy()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            depth[(depth < DEPTH_MIN_M) | (depth > DEPTH_MAX_M)] = 0.0
            T, ok = np.eye(4), False
            if self.odom is not None:
                T = self.odom.update(rgb, depth).copy(); ok = self.odom.ok
            now = time.time(); self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-3); self._t = now
            with self.lock:
                self.seq += 1
                self.latest = (rgb, depth, T, ok, self.seq)

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self._stop = True
        self.thread.join(timeout=3.0)


class RealSenseGUI:
    def __init__(self, zmq_addr, rs_filters=False, odometry=True):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        profile = self.pipe.start(cfg)
        dev = profile.get_device()
        self.depth_scale = float(dev.first_depth_sensor().get_depth_scale())   # metres per unit (0.001)
        self.align = rs.align(rs.stream.color)
        # intrinsics of the COLOUR stream = intrinsics of the aligned depth
        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = cs.get_intrinsics()
        self.intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                           "width": intr.width, "height": intr.height, "model": str(intr.model),
                           "coeffs": list(intr.coeffs)}
        ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_fx = float(ds.get_intrinsics().fx)     # stereo geometry uses the DEPTH imager's focal length
        ext = ds.get_extrinsics_to(cs)
        self.depth_to_color = {"rotation": list(ext.rotation), "translation": list(ext.translation)}
        print(f"[rs] {dev.get_info(rs.camera_info.name)} S/N {dev.get_info(rs.camera_info.serial_number)} "
              f"fw {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"[rs] colour intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f} depth fx={self.depth_fx:.1f} "
              f"depth scale={self.depth_scale}  depth->colour t={np.round(ext.translation, 4)} m")
        self.filters = []
        if rs_filters:
            self.filters = [rs.spatial_filter(), rs.temporal_filter()]
            print("[rs] spatial + temporal post-processing filters ON")

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(zmq_addr)

        self.rgb = None; self.depth = None
        self.mask = None; self.tracking_active = False
        self.last_label = None; self.last_shape = None
        self.status = ""
        self.session_dir = None; self.pose_count = 0
        self.candidates = []; self.label_map = None; self.detect_id = None; self.selected_id = None
        self.search_target = None; self.search_best = None; self._last_detect_end = 0.0
        self.table_plane = None      # (normal toward camera, d) in camera frame, from RANSAC
        self.click_uv = None
        self.odom = None
        if odometry:
            try:
                self.odom = RGBDOdometry(self.intrinsics, WIDTH, HEIGHT)
                print("[odom] RGB-D visual odometry ON (world = first frame), running in the grabber thread at camera rate")
            except Exception as exc:
                print(f"[odom] disabled: {exc}")
        self.grabber = FrameGrabber(self.pipe, self.align, self.depth_scale, self.filters, self.odom)
        self.frame_T, self.frame_odom_ok, self._frame_seq = np.eye(4), False, -1

    # ---------------------------------------------------------------- frames
    def grab(self):
        """Latest frame from the grabber thread, with the pose that belongs to it."""
        latest = self.grabber.get()
        if latest is None:
            time.sleep(0.005); return False
        rgb, depth, T, ok, seq = latest
        if seq == self._frame_seq:
            time.sleep(0.002); return False       # nothing new yet
        self._frame_seq = seq
        self.rgb, self.depth, self.frame_T, self.frame_odom_ok = rgb, depth, T, ok
        return True

    # ---------------------------------------------------------------- geometry
    def _unproject(self, mask, step=3):
        ys, xs = np.nonzero(mask)
        ys, xs = ys[::step], xs[::step]
        z = self.depth[ys, xs]
        ok = z > 0
        xs, ys, z = xs[ok], ys[ok], z[ok]
        K = self.intrinsics
        return np.stack([(xs - K["cx"]) * z / K["fx"], (ys - K["cy"]) * z / K["fy"], z], axis=1)

    def _noise_sigma_at(self, z):
        fb = getattr(self, "depth_fx", self.intrinsics["fx"]) * NOISE["baseline_m"]
        return float(z * z / fb * NOISE["disp_sigma_px"] * NOISE["severity"])

    def _fit_table_plane(self, iters=150, thresh=0.01):
        """Dominant plane in the depth image (camera frame) by RANSAC.
        Returns (n, d) with n oriented toward the camera so that points
        ON the table have signed distance ~0 and objects on it > 0."""
        full = np.ones(self.depth.shape, bool)
        pts = self._unproject(full, step=8)
        if len(pts) < 500:
            return None
        rng = np.random.default_rng(0)
        best_n, best_d, best_cnt = None, None, 0
        for _ in range(iters):
            i = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[i]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -np.dot(n, p0)
            cnt = int((np.abs(pts @ n + d) < thresh).sum())
            if cnt > best_cnt:
                best_n, best_d, best_cnt = n, d, cnt
        if best_n is None:
            return None
        inl = np.abs(pts @ best_n + best_d) < thresh
        ctr = pts[inl].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[inl] - ctr, full_matrices=False)
        n = vt[-1]
        if np.dot(n, -ctr) < 0:       # orient toward the camera (origin)
            n = -n
        d = -np.dot(n, ctr)
        self.table_plane = (n, d, best_cnt / len(pts))
        return self.table_plane

    def _object_cam_position(self):
        if self.mask is None:
            return None
        pts = self._unproject(self.mask, step=2)
        if len(pts) < 20:
            return None
        return np.median(pts, axis=0)

    def _reject_reason(self, c):
        """Shadow test, camera frame (same logic as the Isaac GUI)."""
        m = self.label_map == c["id"]
        pts = self._unproject(m)
        if len(pts) < 30:
            return f"only {len(pts)} valid depth points"
        ctr = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        n = vt[-1]
        rms = float(np.sqrt(np.mean(((pts - ctr) @ n) ** 2)))
        sig = self._noise_sigma_at(float(np.median(pts[:, 2])))
        rms_thresh = max(SHADOW_PLANAR_RMS_M, 2.5 * sig)
        ring_thresh = max(SHADOW_RING_DIST_M, 2.5 * sig)
        if rms > rms_thresh:
            return None
        ring = cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool) & ~m
        ring_pts = self._unproject(ring, step=2)
        if len(ring_pts) < 30:
            return None
        ring_d = float(np.median(np.abs((ring_pts - ctr) @ n)))
        if ring_d < ring_thresh:
            return f"shadow (flat rms {rms*1000:.1f} mm, ring {ring_d*1000:.1f} mm, sigma {sig*1000:.1f} mm)"
        return None

    # ---------------------------------------------------------------- Process B calls
    def _req(self, req_type, frame=None, **meta):
        m, p = build_request(req_type, frame_bgr=frame, **meta)
        self.sock.send_multipart([m, p])
        return parse_response(*self.sock.recv_multipart())

    def click(self, u, v, label):
        meta, mask = self._req(REQ_CLICK, self.rgb, x=u, y=v, label=label)
        if meta["status"] == STATUS_OK and mask is not None:
            self.mask, self.tracking_active = mask, True
            self.status = f"segmented at ({u},{v})"
        else:
            self.status = f"click failed: {meta.get('message','')}"

    def track(self):
        meta, mask = self._req(REQ_TRACK, self.rgb)
        note = meta.get("message", "")
        if meta["status"] == STATUS_OK and mask is not None:
            self.mask = mask
            if note:
                self.status = f"tracking: {note}"     # mask held, not updated this frame
        elif meta["status"] == STATUS_NO_OBJECT:
            self.mask, self.tracking_active, self.selected_id = None, False, None
            self.status = f"tracking lost ({note}) -- click or detect again"
            self.log(self.status)

    def reset(self):
        self._req(REQ_RESET)
        self.mask = None; self.tracking_active = False; self.selected_id = None
        self.last_label = self.last_shape = None
        self.search_target = None; self.search_best = None
        self.candidates = []; self.label_map = None
        self.status = "reset"

    def classify(self):
        if self.mask is None:
            return
        ys, xs = np.nonzero(self.mask)
        crop = self.rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        meta, _ = self._req(REQ_CLASSIFY, crop)
        if meta["status"] == STATUS_OK:
            self.last_label, self.last_shape = meta["label"], meta["shape"]
            self.status = f"classified: {meta['label']} ({meta['shape']} {meta['confidence']:.2f})"

    def detect(self, target=None):
        extra = {"target_label": target} if target else {}
        meta, lm = self._req(REQ_DETECT_ALL, self.rgb, **extra)
        self._last_detect_end = time.time()
        if meta["status"] != STATUS_OK or meta.get("payload") != PAYLOAD_LABEL_MAP:
            self.status = f"detect failed: {meta.get('message','')}"; return
        self.label_map = lm
        kept, dropped = [], []
        for c in meta.get("candidates", []):
            r = self._reject_reason(c)
            (dropped if r else kept).append((c, r))
        self.candidates = [c for c, _ in kept]
        self.detect_id = meta.get("detect_id"); self.selected_id = None
        self.status = f"detected {len(self.candidates)} props ({len(dropped)} filtered)"
        self.log(self.status)
        for c, r in dropped:
            print(f"[detect] dropped id {c['id']} {c['label']}: {r}")
        for c in self.candidates:
            tp = f" target_prob={c['target_prob']:.2f}" if "target_prob" in c else ""
            print(f"[detect] kept id {c['id']} {c['label']} ({c['shape']} {c['confidence']:.2f}) bbox={c['bbox']}{tp}")

    def select(self, cid):
        cand = next((c for c in self.candidates if c["id"] == cid), None)
        if cand is None:
            self.status = f"no candidate {cid}"; return
        meta, mask = self._req(REQ_SELECT_MASK, self.rgb, candidate_id=cid, detect_id=self.detect_id)
        if meta["status"] == STATUS_OK and mask is not None:
            self.mask, self.tracking_active, self.selected_id = mask, True, cid
            self.last_label, self.last_shape = cand["label"], cand["shape"]
            self.status = f"tracking id {cid}: {cand['label']}"
        else:
            self.status = f"select failed: {meta.get('message','')}"

    def try_lock(self):
        scored = [c for c in self.candidates if "target_prob" in c]
        if not scored:
            self.search_best = None; return
        ranked = sorted(scored, key=lambda c: c["target_prob"], reverse=True)
        best = ranked[0]; second = ranked[1]["target_prob"] if len(ranked) > 1 else 0.0
        self.search_best = (best["id"], best["target_prob"])
        ok = (best["target_prob"] >= SEARCH_MATCH_PROB or best["label"] == self.search_target or
              (best["target_prob"] >= SEARCH_MATCH_MIN and best["target_prob"] >= SEARCH_MATCH_RATIO * max(second, 1e-3)))
        print(f"[search] best id {best['id']} {best['label']} prob={best['target_prob']:.2f} (next {second:.2f}) lock={ok}")
        if ok:
            self.select(best["id"])

    # ---------------------------------------------------------------- sessions
    def new_session(self):
        os.makedirs(SESSION_ROOT, exist_ok=True)
        lbl = (self.last_label or "object").replace(" ", "_")
        self.session_dir = os.path.join(SESSION_ROOT, f"{lbl}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.pose_count = 0
        meta = {"intrinsics": {k: self.intrinsics[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
                "intrinsics_full": self.intrinsics, "depth_to_color_extrinsics": self.depth_to_color,
                "label": self.last_label, "shape": self.last_shape,
                "camera": "Intel RealSense D435i (real, handheld)",
                "pose_source": ("rgbd_odometry (Open3D hybrid, keyframe-based; world = first camera frame, "
                                "OpenCV optical convention)") if self.odom is not None else "none",
                "depth_noise": NOISE, "poses": []}
        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.status = f"new session {os.path.basename(self.session_dir)}"
        self.last_capture_info = ""
        self.log(self.status)

    def capture(self):
        if self.mask is None or not self.tracking_active:
            self.status = "nothing tracked -- click or select first"; return
        h, w = self.mask.shape
        frac = self.mask.sum() / (h * w)
        if frac > PLAN_MAX_MASK_FRAC:
            self.status = f"refusing: mask covers {frac:.0%} of the image (table/background?)"; return
        ys, xs = np.nonzero(self.mask)
        z = self.depth[ys, xs]; valid = z > 0
        if valid.mean() < CAPTURE_MIN_VALID_FRAC:
            self.status = f"refusing: only {valid.mean():.0%} of mask has valid depth"; return
        if self.session_dir is None:
            self.new_session()
        # table-plane clip using the RANSAC plane (camera frame)
        mask_save = self.mask.copy()
        plane = self._fit_table_plane()
        clipped = 0
        if plane is not None and plane[2] < TABLE_PLANE_MIN_INLIERS:
            print(f"[capture] table plane only {plane[2]:.0%} inliers -- not trusted, no clip this view")
            plane = None
        if plane is not None:
            n, d, frac_inl = plane
            pts = np.stack([(xs - self.intrinsics["cx"]) * z / self.intrinsics["fx"],
                            (ys - self.intrinsics["cy"]) * z / self.intrinsics["fy"], z], axis=1)
            sd = pts @ n + d                     # >0 toward camera = above the table
            clip_m = max(TABLE_CLIP_M, 2.0 * self._noise_sigma_at(float(np.median(z[valid]))))
            on_table = valid & (sd < clip_m)
            mask_save[ys[on_table], xs[on_table]] = False
            clipped = int(on_table.sum())
            self.plane_inliers = frac_inl
            print(f"[capture] table plane inliers {frac_inl:.0%}; clipped {clipped} px within {clip_m*1000:.1f} mm of it")
        idx = self.pose_count
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_rgb.png"), self.rgb)
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_mask.png"), mask_save.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(self.session_dir, f"pose_{idx:02d}_depth.png"),
                    np.clip(self.depth * 1000.0, 0, 65535).astype(np.uint16))
        if self.odom is not None:
            if not self.frame_odom_ok:
                self.log(f"WARNING view {idx}: odometry not tracking on this frame -- pose may be stale")
            mp = os.path.join(self.session_dir, "session_meta.json")
            meta = json.load(open(mp))
            meta.setdefault("poses", []).append({"index": idx, "camera_frame": "Camera_RGB_opencv_optical",
                                                 "cam_to_world": self.frame_T.tolist(), "odometry_ok": bool(self.frame_odom_ok)})
            json.dump(meta, open(mp, "w"), indent=2)
        self.pose_count += 1
        self.status = f"captured view {idx} ({valid.mean():.0%} valid depth, {clipped} px clipped)"
        self.last_capture_info = f"last: view {idx}, {valid.mean():.0%} valid, {clipped} px table-clipped"
        self.log(self.status)
        print(f"[capture] pose_{idx:02d} saved to {self.session_dir}")

    def fuse(self):
        if self.session_dir is None or self.pose_count < 2:
            self.status = "need >= 2 captured views"; return
        meta, _ = self._req(REQ_GENERATE_PC, session_dir=self.session_dir)
        self.status = f"{meta.get('status')}: {meta.get('message','')}"
        self.log(self.status)

    def view(self):
        if self.session_dir:
            self._req(REQ_VIEW_PC, ply_path=os.path.join(self.session_dir, "fused_pointcloud.pcd"))

    # ---------------------------------------------------------------- UI
    # Layout (one OpenCV canvas): camera view left, control panel right,
    # log strip below the view. Everything in the panel is clickable; the
    # keyboard shortcuts still work. No extra dependencies.
    CANVAS_W, CANVAS_H = 1600, 900
    VIEW = (10, 10, 1152, 648)              # x, y, w, h of the camera view (0.9 x 1280x720)
    PANEL_X, PANEL_W = 1175, 415
    LOG = (10, 670, 1152, 220)
    C_BG, C_PANEL, C_TXT, C_DIM = (28, 28, 28), (40, 40, 40), (235, 235, 235), (150, 150, 150)
    C_BTN, C_BTN_HI, C_ACC, C_OK, C_WARN, C_BAD = (70, 70, 70), (95, 95, 95), (252, 186, 0), (80, 200, 80), (0, 200, 255), (60, 60, 230)

    def _ui_init(self):
        from collections import deque
        self.log_lines = deque(maxlen=9)
        self.buttons = []               # (x, y, w, h, label, callback, hotkey)
        self.cand_rows = []             # (y0, y1, cid)
        self.search_text = ""
        self.search_focus = False
        self.show_depth = False
        self.fps = 0.0; self._t_last = time.time()
        self.hover = None
        self.last_capture_info = ""
        self.plane_inliers = None
        bx, bw, bh, gap = self.PANEL_X + 10, 128, 34, 7
        rows = [[("Detect (d)", self.detect, 'd'), ("Classify (c)", self.classify, 'c'), ("Capture (p)", self.capture, 'p')],
                [("Fuse (g)", self.fuse, 'g'), ("View cloud (v)", self.view, 'v'), ("New session (n)", self.new_session, 'n')],
                [("Stop search (x)", self._stop_search, 'x'), ("RGB / Depth (t)", self._toggle_depth, 't'), ("Reset (r)", self.reset, 'r')]]
        y = 96
        for row in rows:
            for i, (lab, cb, key) in enumerate(row):
                self.buttons.append((bx + i * (bw + gap), y, bw, bh, lab, cb, key))
            y += bh + gap
        self.search_box = (bx, y + 8, bw * 2 + gap, 30)
        self.buttons.append((bx + 2 * (bw + gap), y + 8, bw, 30, "Search", self._start_search_from_box, None))
        self.cand_top = y + 8 + 30 + 52
        self.buttons.append((self.PANEL_X + self.PANEL_W - 90, self.CANVAS_H - 40, 80, 28, "Quit (q)", None, 'q'))

    def log(self, msg):
        self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        print(f"[gui] {msg}")

    def _stop_search(self):
        self.search_target = None; self.search_best = None; self.log("search stopped")

    def _toggle_depth(self):
        self.show_depth = not self.show_depth

    def _start_search_from_box(self):
        t = self.search_text.strip()
        if not t:
            self.status = "type an object name in the search box first"; return
        if self.tracking_active:
            self.reset()
        self.search_target, self.search_best, self._last_detect_end = t, None, 0.0
        self.search_focus = False
        self.log(f"searching for '{t}'")

    def _view_to_image(self, x, y):
        vx, vy, vw, vh = self.VIEW
        if vx <= x < vx + vw and vy <= y < vy + vh:
            return int((x - vx) * WIDTH / vw), int((y - vy) * HEIGHT / vh)
        return None

    def on_mouse(self, event, x, y, flags, param):
        self.hover = (x, y)
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        uv = self._view_to_image(x, y)
        if uv is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click(*uv, 1); self.log(f"positive click at {uv}")
            elif self.tracking_active:
                self.click(*uv, 0); self.log(f"negative click at {uv}")
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        sx, sy, sw, sh = self.search_box
        self.search_focus = sx <= x < sx + sw and sy <= y < sy + sh
        for (bx, by, bw, bh, lab, cb, key) in self.buttons:
            if bx <= x < bx + bw and by <= y < by + bh:
                if lab.startswith("Quit"):
                    self._quit = True
                elif cb:
                    cb()
                return
        for (y0, y1, cid) in self.cand_rows:
            if self.PANEL_X <= x < self.PANEL_X + self.PANEL_W and y0 <= y < y1:
                self.select(cid); self.log(f"selected candidate {cid}"); return

    def _txt(self, img, text, x, y, size=0.5, color=None, bold=False):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color or self.C_TXT, 2 if bold else 1, cv2.LINE_AA)

    def _depth_vis(self):
        d = self.depth
        valid = d > 0
        vmax = np.percentile(d[valid], 99) if valid.any() else 1.0
        x = np.clip(d / max(vmax, 1e-6), 0, 1)
        img = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        return img

    def render(self):
        canvas = np.full((self.CANVAS_H, self.CANVAS_W, 3), self.C_BG, np.uint8)
        # ---- camera view ----
        base = self._depth_vis() if self.show_depth else self.rgb
        vis = base.copy()
        if self.mask is not None:
            ov = vis.copy(); ov[self.mask] = (0, 255, 0)
            vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
            cnts, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        elif self.candidates:
            for c in self.candidates:
                x, y, w, h = c["bbox"]
                col = self.C_BAD if self.search_best and c["id"] == self.search_best[0] else self.C_WARN
                cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
                cv2.putText(vis, f"{c['id']}", (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        vx, vy, vw, vh = self.VIEW
        canvas[vy:vy + vh, vx:vx + vw] = cv2.resize(vis, (vw, vh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (vx - 1, vy - 1), (vx + vw, vy + vh), (80, 80, 80), 1)
        # crosshair + hover pixel readout
        if self.hover:
            uv = self._view_to_image(*self.hover)
            if uv is not None:
                u, v = uv; z = float(self.depth[v, u])
                self._txt(canvas, f"({u},{v})  z={z:.3f} m" if z > 0 else f"({u},{v})  z=invalid",
                          vx + 8, vy + vh - 10, 0.5, self.C_ACC)
        # ---- panel ----
        px, pw = self.PANEL_X, self.PANEL_W
        cv2.rectangle(canvas, (px, 10), (px + pw, self.CANVAS_H - 10), self.C_PANEL, -1)
        valid_frac = float((self.depth > 0).mean())
        self._txt(canvas, "Intel RealSense D435i  (handheld)", px + 10, 34, 0.6, self.C_ACC, True)
        K = self.intrinsics
        self._txt(canvas, f"{WIDTH}x{HEIGHT}  camera {self.grabber.fps:4.1f} fps  gui {self.fps:4.1f} fps   fx {K['fx']:.0f}", px + 10, 58, 0.48, self.C_DIM)
        col = self.C_OK if valid_frac > 0.7 else self.C_WARN if valid_frac > 0.4 else self.C_BAD
        self._txt(canvas, f"depth valid {valid_frac:.0%}", px + 10, 80, 0.5, col)
        if self.odom is not None:
            T = self.frame_T; t = T[:3, 3]
            ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
            oc = self.C_OK if self.frame_odom_ok else self.C_BAD
            self._txt(canvas, f"odometry {'OK' if self.frame_odom_ok else 'LOST'}  pos {t[0]:+.2f} {t[1]:+.2f} {t[2]:+.2f} m  "
                              f"rot {ang:4.1f} deg  kf {self.odom.n_key}  {self.odom.last_ms:.0f} ms", px + 10, self.CANVAS_H - 22, 0.42, oc)
        cv2.rectangle(canvas, (px + 150, 68), (px + 150 + int(240 * valid_frac), 82), col, -1)
        cv2.rectangle(canvas, (px + 150, 68), (px + 390, 82), self.C_DIM, 1)
        # buttons
        for (bx, by, bw, bh, lab, cb, key) in self.buttons:
            hov = self.hover and bx <= self.hover[0] < bx + bw and by <= self.hover[1] < by + bh
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), self.C_BTN_HI if hov else self.C_BTN, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)
            self._txt(canvas, lab, bx + 8, by + 22, 0.45)
        # search box
        sx, sy, sw, sh = self.search_box
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (20, 20, 20), -1)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), self.C_ACC if self.search_focus else (110, 110, 110), 1)
        txt = self.search_text + ("_" if self.search_focus and int(time.time() * 2) % 2 == 0 else "")
        self._txt(canvas, txt if (self.search_text or self.search_focus) else "type object name, Enter to search", sx + 8, sy + 21, 0.5,
                  self.C_TXT if (self.search_text or self.search_focus) else self.C_DIM)
        y = self.cand_top - 26
        if self.search_target:
            if self.tracking_active:
                st = f"search '{self.search_target}': LOCKED on id {self.selected_id}"; col = self.C_OK
            elif self.search_best:
                st = f"search '{self.search_target}': best id {self.search_best[0]} p={self.search_best[1]:.2f}"; col = self.C_WARN
            else:
                st = f"search '{self.search_target}': scanning..."; col = self.C_WARN
            self._txt(canvas, st, px + 10, y, 0.48, col)
        # candidates
        y = self.cand_top
        self._txt(canvas, f"Detected ({len(self.candidates)})  -- click a row or press 1-9", px + 10, y, 0.5, self.C_ACC, True)
        y += 10
        self.cand_rows = []
        for c in self.candidates[:9]:
            y0 = y; y += 24
            sel = c["id"] == self.selected_id
            best = self.search_best and c["id"] == self.search_best[0]
            if sel:
                cv2.rectangle(canvas, (px + 6, y0 + 4), (px + pw - 6, y0 + 26), (60, 90, 60), -1)
            tp = f"  ~{c['target_prob']:.2f}" if "target_prob" in c else ""
            self._txt(canvas, f"{c['id']}  {c['label']}  ({c['shape']} {c['confidence']:.2f}){tp}", px + 12, y0 + 20, 0.48,
                      self.C_OK if sel else self.C_BAD if best else self.C_TXT)
            self.cand_rows.append((y0 + 4, y0 + 26, c["id"]))
        y = self.cand_top + 10 + 24 * 9 + 16
        # object readout
        self._txt(canvas, "Object", px + 10, y, 0.5, self.C_ACC, True); y += 22
        if self.mask is not None:
            pos = self._object_cam_position()
            mpx = int(self.mask.sum()); mv = float((self.depth[self.mask] > 0).mean()) if mpx else 0.0
            self._txt(canvas, f"{self.last_label or 'unlabelled'} {('(' + self.last_shape + ')') if self.last_shape else ''}", px + 12, y, 0.48); y += 20
            if pos is not None:
                self._txt(canvas, f"cam xyz  {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m   dist {np.linalg.norm(pos):.2f} m", px + 12, y, 0.45); y += 20
            self._txt(canvas, f"mask {mpx} px   valid depth {mv:.0%}", px + 12, y, 0.45); y += 20
            tr = "tracking" if not self.status.startswith("tracking: held") else "tracking (held)"
            col = self.C_OK if tr == "tracking" else self.C_WARN
        else:
            self._txt(canvas, "none selected -- click the image, or Detect", px + 12, y, 0.45, self.C_DIM); y += 20
            tr, col = "idle", self.C_DIM
        cv2.circle(canvas, (px + 18, y - 4), 6, col, -1); self._txt(canvas, tr, px + 32, y, 0.45, col); y += 26
        # session readout
        self._txt(canvas, "Session", px + 10, y, 0.5, self.C_ACC, True); y += 22
        self._txt(canvas, os.path.basename(self.session_dir) if self.session_dir else "none (Capture starts one)", px + 12, y, 0.45); y += 20
        self._txt(canvas, f"views captured: {self.pose_count}   {'(>=2 needed to fuse)' if self.pose_count < 2 else 'ready to fuse'}", px + 12, y, 0.45); y += 20
        if self.last_capture_info:
            self._txt(canvas, self.last_capture_info, px + 12, y, 0.42, self.C_DIM); y += 20
        if self.plane_inliers is not None:
            self._txt(canvas, f"table plane: {self.plane_inliers:.0%} of depth points", px + 12, y, 0.42, self.C_DIM); y += 20
        # status + log
        lx, ly, lw, lh = self.LOG
        cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), self.C_PANEL, -1)
        self._txt(canvas, self.status, lx + 10, ly + 24, 0.55, self.C_ACC, True)
        for i, line in enumerate(self.log_lines):
            self._txt(canvas, line, lx + 10, ly + 50 + 19 * i, 0.45, self.C_TXT if i == len(self.log_lines) - 1 else self.C_DIM)
        self._txt(canvas, "left click: segment   right click: negative point   keys: d c p g v n x t r q   1-9 select", lx + 10, ly + lh - 8, 0.42, self.C_DIM)
        return canvas

    def run(self):
        self._ui_init(); self._quit = False
        win = "RealSense D435i -- SAM2 perception"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, self.CANVAS_W, self.CANVAS_H)
        cv2.setMouseCallback(win, self.on_mouse)
        self.log("ready -- Process B at " + str(self.sock.getsockopt(zmq.LAST_ENDPOINT) if hasattr(zmq, 'LAST_ENDPOINT') else 'zmq'))
        try:
            while not self._quit:
                if not self.grab():
                    continue
                now = time.time(); self.fps = 0.9 * self.fps + 0.1 / max(now - self._t_last, 1e-3); self._t_last = now
                if self.tracking_active:
                    self.track()
                elif self.search_target and time.time() - self._last_detect_end >= SEARCH_INTERVAL_S:
                    self.detect(target=self.search_target); self.try_lock()
                cv2.imshow(win, self.render())
                k = cv2.waitKey(1) & 0xFF
                if k == 255:
                    continue
                if self.search_focus:
                    if k in (13, 10): self._start_search_from_box()
                    elif k == 27: self.search_focus = False
                    elif k == 8: self.search_text = self.search_text[:-1]
                    elif 32 <= k <= 126: self.search_text += chr(k)
                    continue
                if k == ord('q'): break
                elif ord('1') <= k <= ord('9'): self.select(k - ord('0'))
                elif k == ord('s'): self.search_focus = True
                else:
                    for (_, _, _, _, lab, cb, key) in self.buttons:
                        if key and k == ord(key) and cb:
                            cb(); break
        finally:
            self.grabber.stop(); self.pipe.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmq", default="tcp://localhost:5555")
    ap.add_argument("--rs-filters", action="store_true", help="enable RealSense spatial+temporal filters (default: raw depth)")
    ap.add_argument("--no-odometry", action="store_true", help="disable RGB-D visual odometry (poses will be unknown)")
    a = ap.parse_args()
    RealSenseGUI(a.zmq, rs_filters=a.rs_filters, odometry=not a.no_odometry).run()
