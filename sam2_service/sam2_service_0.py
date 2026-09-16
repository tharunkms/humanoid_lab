#!/usr/bin/env python3
"""
sam2_service.py -- Process B: SAM2 inference service.

Runs in ITS OWN Python 3.10+ virtualenv (NOT the ROS1 Noetic container).
See requirements_sam2.txt for what to install there.

Responsibilities:
  - Load the SAM2 model + video/streaming predictor ONCE at startup.
  - Listen on a ZeroMQ REP socket for requests from Process A
    (rgbd_segment_gui.py): "click" (new point prompt), "track" (propagate
    to the next frame), "reset" (clear tracking state).
  - Maintain SAM2's internal memory-bank / session state across calls so
    tracking persists frame-to-frame without re-prompting.
  - Never crash the caller: any inference exception is caught and reported
    back as a STATUS_ERROR response so Process A can show "segmentation
    unavailable" instead of dying.

IMPORTANT -- read before running (confirmed against the actual installed
version, commit 2b90b9f, `pip install "git+https://github.com/facebookresearch/sam2.git"`):

  This install ONLY ships `sam2.build_sam.build_sam2_video_predictor`.
  There is no camera/streaming predictor -- upstream has explicitly said
  live-stream frame-by-frame tracking isn't natively supported
  (facebookresearch/sam2 issue #134). `init_state()` requires a directory
  of JPEG frames on disk; you cannot hand it one live frame at a time and
  keep extending the same inference_state.

  Workaround used below (`SAM2Session`): every `track()` call writes just
  TWO frames to a small temp directory -- the previous frame (with its
  already-known mask re-injected via `add_new_mask`) and the new frame --
  then calls `init_state()` + `propagate_in_video()` over that pair only,
  and keeps the new frame's mask for next time. This bounds the cost of
  each call to ~2 frames instead of growing with every frame seen, which
  is what makes this usable at anything close to real time. The trade-off:
  SAM2's longer-range memory bank isn't used, so tracking can drift after
  occlusions or fast motion -- press `r` in the GUI and re-click if that
  happens.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np
import zmq

from ipc_common import (
    parse_request, build_response,
    REQ_CLICK, REQ_TRACK, REQ_RESET, REQ_CLASSIFY, REQ_GENERATE_PC, REQ_VIEW_PC,
    STATUS_OK, STATUS_NO_OBJECT, STATUS_ERROR,
)


def remap_path(path, container_root, host_root):
    """
    The GUI (Process A) runs inside the ROS1 Noetic container and only
    knows container-side paths (e.g. /root/catkin_ws/...); this service
    (Process B) runs natively on the host and needs the equivalent host
    path (e.g. /home/user/.../catkin_ws/...) into the SAME underlying
    directory (they're the same disk location via the Docker bind mount --
    just two different path prefixes for it). If the given path doesn't
    start with the known container prefix, it's passed through unchanged
    (covers running Process A directly on the host too, e.g. --backend webcam).
    """
    if container_root and path.startswith(container_root):
        return host_root + path[len(container_root):]
    return path

# SAM 2.1 checkpoints/configs -- confirmed present in this install via
# `find .../sam2 -iname "*.yaml"`. Checkpoint URLs (092824 build):
#   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_<size>.pt
CHECKPOINTS = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}
CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


class SAM2Session:
    """
    Wraps SAM2's video predictor and adapts it to a live camera feed via a
    2-frame-at-a-time re-conditioning trick (see module docstring for why).
    Only one object is tracked at a time, matching the v1 scope in the spec.
    """

    def __init__(self, checkpoint_dir, checkpoint_key, device="cuda"):
        self.device = device
        self.predictor = self._load_predictor(checkpoint_dir, checkpoint_key, device)
        self.frame_dir = tempfile.mkdtemp(prefix="sam2_pair_")
        self.anchor_frame = None    # last frame we have a mask for (numpy BGR)
        self.anchor_mask = None     # mask aligned to anchor_frame
        self.active = False

    def _load_predictor(self, checkpoint_dir, checkpoint_key, device):
        from sam2.build_sam import build_sam2_video_predictor
        cfg = CONFIGS[checkpoint_key]
        ckpt = os.path.join(checkpoint_dir, CHECKPOINTS[checkpoint_key])
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt} -- download it first, see requirements_sam2.txt"
            )
        print(f"[sam2_service] loading checkpoint={ckpt} cfg={cfg} device={device}", file=sys.stderr)
        predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
        print("[sam2_service] model loaded.", file=sys.stderr)
        return predictor

    def reset(self):
        self.active = False
        self.anchor_frame = None
        self.anchor_mask = None
        self._clear_dir()

    def _clear_dir(self):
        for f in os.listdir(self.frame_dir):
            os.remove(os.path.join(self.frame_dir, f))

    def click(self, frame_bgr, x, y, label):
        """First click (or a refinement click on the current newest frame)."""
        self._clear_dir()
        cv2.imwrite(os.path.join(self.frame_dir, "00000.jpg"), frame_bgr)

        state = self.predictor.init_state(video_path=self.frame_dir)
        _, _, mask_logits = self.predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([[x, y]], dtype=np.float32),
            labels=np.array([label], dtype=np.int32),
        )
        mask = self._logits_to_mask(mask_logits)
        self.anchor_frame = frame_bgr
        self.anchor_mask = mask
        self.active = True
        return mask

    def track(self, frame_bgr):
        """Propagate the tracked object's mask from the anchor frame onto frame_bgr."""
        if not self.active or self.anchor_frame is None:
            return None

        self._clear_dir()
        cv2.imwrite(os.path.join(self.frame_dir, "00000.jpg"), self.anchor_frame)
        cv2.imwrite(os.path.join(self.frame_dir, "00001.jpg"), frame_bgr)

        state = self.predictor.init_state(video_path=self.frame_dir)
        self.predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=self.anchor_mask)

        mask = None
        for out_frame_idx, out_obj_ids, mask_logits in self.predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=2
        ):
            if out_frame_idx == 1:
                mask = self._logits_to_mask(mask_logits)

        if mask is not None:
            self.anchor_frame = frame_bgr
            self.anchor_mask = mask
        return mask

    @staticmethod
    def _logits_to_mask(mask_logits):
        mask = (mask_logits[0] > 0.0)
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        return np.asarray(mask).squeeze().astype(bool)


# Deliberately small and coarse -- "just say it's a bottle, not a detailed
# description" per the brief. Each label maps to one of four primary
# geometric shapes, which is what actually drives the view-planning logic
# in the GUI. Extend this dict with more lab-relevant objects as needed;
# it's the only place shape assumptions live.
CLIP_SHAPE_LABELS = {
    "bottle": "cylinder", "cup": "cylinder", "mug": "cylinder", "can": "cylinder",
    "jar": "cylinder", "pipe": "cylinder", "cylinder": "cylinder", "bucket": "cylinder",
    "box": "cuboid", "book": "cuboid", "carton": "cuboid", "container": "cuboid",
    "cube": "cuboid", "block": "cuboid", "case": "cuboid",
    "ball": "sphere", "sphere": "sphere", "orange": "sphere", "orb": "sphere",
    "wrench": "irregular", "screwdriver": "irregular", "tool": "irregular",
    "cable": "irregular", "bracket": "irregular", "robot part": "irregular",
    "toy": "irregular", "plant": "irregular", "unknown object": "irregular",
}


class Classifier:
    """
    Coarse zero-shot object recognition via CLIP -- deliberately not a
    trained detector, since the brief only needs "it's roughly a bottle"
    plus the primary geometric shape that follows from that, not precise
    categorization. Runs once per newly-selected object, not per frame.
    """

    def __init__(self, device="cuda"):
        import torch
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.labels = list(CLIP_SHAPE_LABELS.keys())
        prompts = [f"a photo of a {label}" for label in self.labels]
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(device)
            text_features = self.model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(f"[sam2_service] classifier loaded ({len(self.labels)} labels)", file=sys.stderr)

    def classify(self, crop_bgr):
        import torch
        from PIL import Image
        rgb = crop_bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]
        idx = int(sims.argmax().item())
        label = self.labels[idx]
        confidence = float(sims[idx].item())
        shape = CLIP_SHAPE_LABELS[label]
        return label, shape, confidence


class PointCloudBuilder:
    """
    Fuses a captured pose session (RGB + mask + depth per pose, saved by
    the GUI) into one point cloud. Needs Open3D -- imported lazily so a
    missing install doesn't break click-to-track or classification.

    Honest limitation: there is no known relative transform between poses
    (no AR marker, no arm forward-kinematics feed) -- this estimates each
    pose's alignment to the previous one via feature-based global
    registration (FPFH + RANSAC) followed by point-to-plane ICP refinement,
    then chains those pairwise transforms into one global frame. This is a
    standard approach for unknown-viewpoint fusion, but it depends on
    consecutive views actually overlapping enough to find correspondences
    -- textureless objects or very large jumps between poses can make a
    pair fail to register well. Each pair's ICP fitness score (0-1, higher
    is better) is reported back so a poor fusion isn't presented as if it
    were trustworthy.
    """

    def __init__(self, voxel_size=0.004):
        self.voxel_size = voxel_size

    def _load_pose_cloud(self, session_dir, idx, intr):
        import open3d as o3d
        rgb = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_rgb.png"))
        mask = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_mask.png"), cv2.IMREAD_GRAYSCALE)
        depth_path = os.path.join(session_dir, f"pose_{idx:02d}_depth.png")
        if rgb is None or mask is None or not os.path.isfile(depth_path):
            return None
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        masked_depth = depth.copy()
        masked_depth[mask < 128] = 0

        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd.estimate_normals()
        return pcd

    def _preprocess_for_global_reg(self, pcd):
        import open3d as o3d
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
        return pcd_down, fpfh

    def _register_pair(self, src, tgt):
        import open3d as o3d
        src_down, src_fpfh = self._preprocess_for_global_reg(src)
        tgt_down, tgt_fpfh = self._preprocess_for_global_reg(tgt)
        dist_thresh = self.voxel_size * 1.5
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down, tgt_down, src_fpfh, tgt_fpfh, True, dist_thresh,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        icp = o3d.pipelines.registration.registration_icp(
            src, tgt, self.voxel_size * 0.8, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return icp.transformation, icp.fitness

    def fuse_session(self, session_dir, intr):
        import open3d as o3d
        pose_files = sorted(glob.glob(os.path.join(session_dir, "pose_*_rgb.png")))
        indices = sorted(int(os.path.basename(f).split("_")[1]) for f in pose_files)

        clouds = {}
        for idx in indices:
            pcd = self._load_pose_cloud(session_dir, idx, intr)
            if pcd is not None and len(pcd.points) > 100:
                clouds[idx] = pcd
        if len(clouds) < 2:
            raise RuntimeError(f"only {len(clouds)} usable views had depth data -- need at least 2 "
                                f"(views captured too close, inside the sensor's blind range, get dropped)")

        ordered = sorted(clouds.keys())
        global_tf = {ordered[0]: np.eye(4)}
        merged = clouds[ordered[0]]
        fitness_report = [(ordered[0], 1.0)]
        for i in range(1, len(ordered)):
            src, tgt = clouds[ordered[i]], clouds[ordered[i - 1]]
            tf_rel, fitness = self._register_pair(src, tgt)
            global_tf[ordered[i]] = global_tf[ordered[i - 1]] @ tf_rel
            transformed = o3d.geometry.PointCloud(src).transform(global_tf[ordered[i]])
            merged = merged + transformed
            fitness_report.append((ordered[i], fitness))

        merged = merged.voxel_down_sample(self.voxel_size)
        cl, ind = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        merged = merged.select_by_index(ind)
        return merged, fitness_report

    @staticmethod
    def render_thumbnail(pcd, out_path, elev=20, azim=-60):
        """Static snapshot via matplotlib (Agg backend) -- deliberately not
        Open3D's own offscreen renderer, which needs a working GL context
        that isn't reliably available headless; matplotlib's software
        rasterizer always works and is plenty for a side-panel preview."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pts = np.asarray(pcd.points)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        if len(pts) > 20000:
            idx = np.random.choice(len(pts), 20000, replace=False)
            pts = pts[idx]
            cols = cols[idx] if cols is not None else None

        fig = plt.figure(figsize=(3.4, 3.4), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=(cols if cols is not None else "gray"), s=1.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)


def serve(bind_addr, checkpoint_dir, checkpoint_key, device, container_root, host_root):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(bind_addr)
    print(f"[sam2_service] listening on {bind_addr}", file=sys.stderr)

    try:
        session = SAM2Session(checkpoint_dir, checkpoint_key, device=device)
        model_ready = True
    except Exception:
        # Model failed to load (missing CUDA, bad checkpoint path, etc).
        # Keep serving so Process A gets a clean STATUS_ERROR instead of a
        # connection-refused / hang, per the spec's failure-handling section.
        traceback.print_exc()
        session = None
        model_ready = False

    # Classifier and PointCloudBuilder load lazily / stay stateless-cheap,
    # so a missing open_clip or open3d install can't take down SAM2 itself
    # -- distance/tracking keep working even if these other features can't.
    classifier = {"instance": None, "load_failed": False}
    pc_builder = PointCloudBuilder()

    while True:
        meta_bytes, payload_bytes = sock.recv_multipart()
        t0 = time.time()
        try:
            meta, frame = parse_request(meta_bytes, payload_bytes)
            req_type = meta.get("type")

            if not model_ready:
                raise RuntimeError("SAM2 model failed to load at startup; see server logs")

            if req_type == REQ_RESET:
                session.reset()
                resp = build_response(STATUS_NO_OBJECT)

            elif req_type == REQ_CLICK:
                mask = session.click(frame, meta["x"], meta["y"], meta["label"])
                resp = build_response(STATUS_OK, mask_bool=mask)

            elif req_type == REQ_TRACK:
                mask = session.track(frame)
                if mask is None:
                    resp = build_response(STATUS_NO_OBJECT)
                else:
                    resp = build_response(STATUS_OK, mask_bool=mask)

            elif req_type == REQ_CLASSIFY:
                if classifier["load_failed"]:
                    raise RuntimeError("classifier previously failed to load -- see earlier server logs")
                if classifier["instance"] is None:
                    try:
                        classifier["instance"] = Classifier(device=device)
                    except Exception as load_exc:
                        classifier["load_failed"] = True
                        raise RuntimeError(f"classifier failed to load: {load_exc}") from load_exc
                label, shape, confidence = classifier["instance"].classify(frame)
                resp = build_response(STATUS_OK, label=label, shape=shape, confidence=confidence)

            elif req_type == REQ_GENERATE_PC:
                session_dir = remap_path(meta["session_dir"], container_root, host_root)
                meta_path = os.path.join(session_dir, "session_meta.json")
                if not os.path.isfile(meta_path):
                    raise RuntimeError("session_meta.json missing -- camera intrinsics were never captured "
                                        "(camera_info topic wasn't available when this object was selected)")
                with open(meta_path) as f:
                    session_meta = json.load(f)
                merged, fitness_report = pc_builder.fuse_session(session_dir, session_meta["intrinsics"])
                ply_path = os.path.join(session_dir, "fused_pointcloud.ply")
                thumb_path = os.path.join(session_dir, "thumbnail.png")

                import open3d as o3d
                o3d.io.write_point_cloud(ply_path, merged)
                pc_builder.render_thumbnail(merged, thumb_path)

                avg_fitness = float(np.mean([f for _, f in fitness_report]))
                low_fitness_views = [idx for idx, f in fitness_report if f < 0.3]
                msg = f"fused {len(fitness_report)} views, avg registration fitness {avg_fitness:.2f}"
                if low_fitness_views:
                    msg += f" (poor alignment on views {low_fitness_views} -- treat fusion as approximate)"
                resp = build_response(STATUS_OK, message=msg, num_points=len(merged.points),
                                       avg_fitness=avg_fitness)

            elif req_type == REQ_VIEW_PC:
                ply_path = remap_path(meta["ply_path"], container_root, host_root)
                if not os.path.isfile(ply_path):
                    raise RuntimeError(f"point cloud not found at {ply_path}")
                # Launches Open3D's OWN native interactive window (real
                # mouse-driven rotate/pan/zoom, maintained by Open3D) on
                # whatever display this service's environment is using --
                # NOT embedded in the GUI's OpenCV canvas. Detached
                # subprocess so it can't block or crash this service.
                subprocess.Popen([
                    sys.executable, "-c",
                    "import open3d as o3d, sys; "
                    "pcd = o3d.io.read_point_cloud(sys.argv[1]); "
                    "o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Viewer')",
                    ply_path,
                ])
                resp = build_response(STATUS_OK, message="viewer launched on host display")

            else:
                resp = build_response(STATUS_ERROR, message=f"unknown request type {req_type!r}")

        except Exception as exc:  # noqa: BLE001 -- must never take the server down
            traceback.print_exc()
            resp = build_response(STATUS_ERROR, message=str(exc))

        sock.send_multipart(resp)
        dt = time.time() - t0
        if dt > 0.2:
            print(f"[sam2_service] slow request: {dt*1000:.0f} ms", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bind", default="tcp://*:5555", help="ZeroMQ bind address")
    ap.add_argument("--checkpoint", choices=list(CHECKPOINTS), default="base_plus",
                     help="SAM2 checkpoint size -- pick based on nvidia-smi VRAM (see README)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                     help="Directory containing the downloaded sam2.1_hiera_*.pt file")
    ap.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--container-catkin-root", default="/root/catkin_ws",
                     help="Where the catkin workspace appears INSIDE the GUI's container "
                          "(only matters for point cloud generation/viewing -- see remap_path())")
    ap.add_argument("--host-catkin-root",
                     default="/home/user/kamarajmagadapallt1/projects/quadruped-inspection/catkin_ws",
                     help="Where that SAME directory appears on this host (via the Docker bind mount)")
    args = ap.parse_args()
    serve(args.bind, args.checkpoint_dir, args.checkpoint, args.device,
          args.container_catkin_root, args.host_catkin_root)


if __name__ == "__main__":
    main()
