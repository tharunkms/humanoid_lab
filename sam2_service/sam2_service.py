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
    REQ_DETECT_ALL, REQ_SELECT_MASK,
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
        self.track_fails = 0
        self.last_track_note = ""
        return mask

    def seed_mask(self, frame_bgr, mask_bool):
        """Start tracking from a KNOWN mask (e.g. one candidate out of
        detect_all) instead of a click. The mask itself becomes the anchor;
        the next track() call re-injects it via add_new_mask exactly like
        the click path does, so nothing downstream changes."""
        self._clear_dir()
        self.anchor_frame = frame_bgr
        self.anchor_mask = np.asarray(mask_bool).astype(bool)
        self.active = True
        self.track_fails = 0
        self.last_track_note = ""
        return self.anchor_mask

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

        if mask is None:
            return self._track_failed()

        # ---- guarded anchor roll -------------------------------------------
        # The anchor rolls to the current frame so propagation is always
        # frame-to-frame (a static anchor goes stale on a moving camera).
        # But an UNGUARDED roll means one bad propagation -- e.g. the mask
        # spilling onto a plain wall behind the object, which SAM2 does on
        # low-contrast real footage -- becomes the new anchor and is tracked
        # faithfully from then on. So the new mask must stay consistent with
        # the previous one before it is accepted: similar area, overlapping
        # it, and not a large fraction of the image. Otherwise the previous
        # mask is held; after too many consecutive rejections the object is
        # declared lost so the client re-selects.
        h, w = mask.shape
        prev = self.anchor_mask
        area_prev = int(prev.sum()); area_new = int(mask.sum())
        ratio = area_new / max(area_prev, 1)
        union = int((mask | prev).sum()); iou = int((mask & prev).sum()) / max(union, 1)
        frac = area_new / float(h * w)
        ok = (TRACK_AREA_RATIO_MIN <= ratio <= TRACK_AREA_RATIO_MAX and iou >= TRACK_MIN_IOU
              and frac <= TRACK_MAX_FRAC and area_new >= TRACK_MIN_PX)
        if ok:
            self.anchor_frame = frame_bgr
            self.anchor_mask = mask
            self.track_fails = 0
            self.last_track_note = ""
            return mask
        self.last_track_note = f"held (area x{ratio:.2f}, iou {iou:.2f}, {frac:.0%} of image)"
        return self._track_failed(hold=True)

    def _track_failed(self, hold=False):
        self.track_fails = getattr(self, "track_fails", 0) + 1
        if self.track_fails > TRACK_MAX_FAILS:
            self.active = False
            self.last_track_note = f"lost after {self.track_fails} rejected frames"
            return None
        return self.anchor_mask if hold else None

    @staticmethod
    def _logits_to_mask(mask_logits):
        mask = (mask_logits[0] > 0.0)
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        return np.asarray(mask).squeeze().astype(bool)


# ---- tracker roll guard ----
TRACK_AREA_RATIO_MIN = 0.5   # new mask area vs previous: reject if it halves ...
TRACK_AREA_RATIO_MAX = 2.0   # ... or doubles between consecutive tracked frames
TRACK_MIN_IOU = 0.3          # new mask must overlap the previous one
TRACK_MAX_FRAC = 0.30        # a mask covering more of the image than this is background, not a prop
TRACK_MIN_PX = 50
TRACK_MAX_FAILS = 20         # consecutive rejected frames before the object is declared lost

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
    "plate": "irregular", "bowl": "irregular",
    "wrench": "irregular", "screwdriver": "irregular", "tool": "irregular",
    "cable": "irregular", "bracket": "irregular", "robot part": "irregular",
    "toy": "irregular", "plant": "irregular", "unknown object": "irregular",
    # "background" shape: detect_all drops these so the table surface,
    # floor and walls never show up as selectable props in the menu.
    "table": "background", "tabletop": "background", "floor": "background",
    "wall": "background", "shadow": "background",
}

# ---- detect_all tuning -------------------------------------------------
DETECT_POINTS_PER_SIDE = 32     # SAM2 prompt grid density; 32 = ~3 s on RTX 4080 at 720p
DETECT_MIN_AREA_FRAC = 0.0005   # drop masks < 0.05% of the image (noise/specks)
DETECT_MAX_AREA_FRAC = 0.30     # drop masks > 30% of the image (table, background)
DETECT_MAX_BORDER_TOUCHES = 1   # drop masks whose bbox touches >=2 image borders
DETECT_CONTAIN_THRESH = 0.80    # drop a mask if >=80% of it lies inside an already-kept (bigger) one
DETECT_MAX_CANDIDATES = 255     # label map is uint8; 0 is reserved for "none"


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

        self._extra_text_features = {}   # open-vocabulary cache: label -> normalised text embedding

    def _text_feature(self, label):
        """Normalised CLIP text embedding for an arbitrary label string.
        CLIP is open-vocabulary: any phrase works ("cereal box", "newspaper").
        Encoded once per new label and cached."""
        import torch
        if label in self._extra_text_features:
            return self._extra_text_features[label]
        with torch.no_grad():
            tokens = self.tokenizer([f"a photo of a {label}"]).to(self.device)
            f = self.model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
        self._extra_text_features[label] = f
        return f

    def _probs(self, crop_bgr, extra_label=None):
        """Softmax over the fixed vocabulary (plus one optional extra label,
        appended as the LAST entry) for one crop. Returns numpy 1-D."""
        import torch
        from PIL import Image
        rgb = crop_bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text = self.text_features
            if extra_label is not None:
                text = torch.cat([text, self._text_feature(extra_label)], dim=0)
            sims = (100.0 * image_features @ text.T).softmax(dim=-1)[0]
        return sims.cpu().numpy()

    def classify(self, crop_bgr):
        label, shape, confidence, _ = self.classify_full(crop_bgr)
        return label, shape, confidence

    def classify_full(self, crop_bgr, target_label=None):
        """(label, shape, confidence, probs, target_prob).

        probs is aligned with self.labels (+ target_label as the last entry
        when it is not already in the vocabulary). target_prob is the
        softmax mass on target_label, or None when no target was given.
        A target that is NOT in the fixed vocabulary is scored just the
        same -- it competes in the softmax like any other label -- so the
        caller can ask for anything."""
        in_vocab = target_label in self.labels if target_label else False
        extra = None if (target_label is None or in_vocab) else target_label
        probs = self._probs(crop_bgr, extra_label=extra)
        labels = self.labels + ([extra] if extra else [])
        idx = int(probs.argmax())
        label = labels[idx]
        confidence = float(probs[idx])
        shape = CLIP_SHAPE_LABELS.get(label, "irregular")   # unknown-label shape fallback
        target_prob = None
        if target_label is not None:
            target_prob = float(probs[labels.index(target_label)])
        return label, shape, confidence, probs, target_prob


class PropDetector:
    """
    Prompt-free "find every prop on the table" step (pipeline step 1).

    Uses SAM2's automatic mask generator (a dense grid of point prompts
    through the IMAGE predictor) rather than the video predictor -- a
    separate build_sam2() model instance, so the click-to-track session
    above is untouched. Then filters the raw masks down to plausible props
    (area, image-border, part-of-a-bigger-mask) and CLIP-classifies each
    survivor; anything CLIP calls "background" (table/floor/wall) is
    dropped too.

    Keeps the last frame + the kept masks in memory so that a follow-up
    REQ_SELECT_MASK can hand one of them straight to SAM2Session.seed_mask
    without the GUI having to ship the mask back over the wire.
    """

    def __init__(self, checkpoint_dir, checkpoint_key, device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        cfg = CONFIGS[checkpoint_key]
        ckpt = os.path.join(checkpoint_dir, CHECKPOINTS[checkpoint_key])
        print(f"[sam2_service] loading image model for detect_all ({checkpoint_key})", file=sys.stderr)
        model = build_sam2(cfg, ckpt, device=device)
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=DETECT_POINTS_PER_SIDE,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=100,   # px; removes tiny disconnected islands inside a mask
        )
        self.last_frame = None      # BGR frame the candidates were computed on
        self.last_masks = {}        # candidate id (int) -> bool HxW mask
        self.detect_id = 0          # token: incremented per detect_all; select_mask must quote it
        print("[sam2_service] detect_all ready.", file=sys.stderr)

    def detect(self, frame_bgr, classifier, target_label=None):
        h, w = frame_bgr.shape[:2]
        img_area = float(h * w)
        target_label = target_label or None
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        raw = self.generator.generate(rgb)   # list of dicts: segmentation, area, bbox [x,y,w,h], ...
        raw.sort(key=lambda m: m["area"], reverse=True)

        kept = []  # (mask, bbox)
        n_area = n_border = n_contained = 0
        for m in raw:
            seg = np.asarray(m["segmentation"]).astype(bool)
            area = int(seg.sum())
            frac = area / img_area
            if frac < DETECT_MIN_AREA_FRAC or frac > DETECT_MAX_AREA_FRAC:
                n_area += 1
                continue
            x, y, bw, bh = [int(v) for v in m["bbox"]]
            touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 2) + int(y + bh >= h - 2)
            if touches > DETECT_MAX_BORDER_TOUCHES:
                n_border += 1
                continue
            contained = False
            for k_seg, _ in kept:
                if np.logical_and(seg, k_seg).sum() / area >= DETECT_CONTAIN_THRESH:
                    contained = True
                    break
            if contained:
                n_contained += 1
                continue
            kept.append((seg, (x, y, bw, bh)))

        # Classify survivors; drop background-class ones; assign ids only
        # to what remains so the menu never has gaps.
        label_map = np.zeros((h, w), dtype=np.uint8)
        candidates = []
        masks = {}
        n_background = 0
        # Paint large->small so a partially overlapping smaller prop wins
        # the shared pixels (kept is already area-descending).
        for seg, (x, y, bw, bh) in kept:
            crop = frame_bgr[y:y + bh, x:x + bw]
            if crop.size == 0:
                continue
            label, shape, conf, probs, target_prob = classifier.classify_full(crop, target_label)
            if shape == "background":
                n_background += 1
                top2 = np.argsort(probs)[::-1][:2]
                _labels = classifier.labels + ([target_label] if target_label and target_label not in classifier.labels else [])
                alt = ", ".join(f"{_labels[i]}={probs[i]:.2f}" for i in top2)
                print(f"[sam2_service] detect_all: dropped as background bbox=[{x},{y},{bw},{bh}] "
                      f"area={int(seg.sum())} top2: {alt}", file=sys.stderr)
                continue
            cid = len(candidates) + 1
            if cid > DETECT_MAX_CANDIDATES:
                break
            ys, xs = np.nonzero(seg)
            label_map[seg] = cid
            masks[cid] = seg
            cand = {
                "id": cid, "label": label, "shape": shape, "confidence": round(conf, 3),
                "bbox": [x, y, bw, bh], "area": int(seg.sum()),
                "centroid": [int(xs.mean()), int(ys.mean())],
            }
            if target_prob is not None:
                # how strongly THIS crop looks like the requested category,
                # independent of what the argmax label was. Works for any
                # string, not only the fixed vocabulary.
                cand["target_prob"] = round(target_prob, 3)
            candidates.append(cand)

        self.last_frame = frame_bgr
        self.last_masks = masks
        self.detect_id += 1
        stats = (f"raw={len(raw)} dropped: area={n_area} border={n_border} "
                 f"contained={n_contained} background={n_background} -> {len(candidates)} props")
        print(f"[sam2_service] detect_all: {stats}", file=sys.stderr)
        return candidates, label_map, stats


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

    def __init__(self, voxel_size=0.001):
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

    def fuse_session(self, session_dir, intr, poses=None, depth_noise=None):
        """
        poses: optional list of {"index": int, "cam_to_world": 4x4} dicts,
        e.g. from Isaac Sim's /tf (ground truth). When every captured view
        has one, that pose is used directly as the global alignment instead
        of blind FPFH+RANSAC feature matching -- ICP still runs afterwards,
        but only as a light local refinement (small search window) rather
        than as the thing finding the alignment from scratch. This is both
        faster and far more reliable than feature-based registration on
        synthetic-render textures, which can be flat/repetitive.

        Falls back to the original real-camera behaviour (FPFH+RANSAC+ICP
        chain, no known relative transform) when poses are missing or only
        partially available -- e.g. a `/tf` lookup failed for some view.
        """
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
        pose_by_idx = {p["index"]: np.array(p["cam_to_world"])
                        for p in (poses or []) if "cam_to_world" in p}
        use_gt_poses = all(idx in pose_by_idx for idx in ordered)

        global_tf = {ordered[0]: np.eye(4)}
        merged = clouds[ordered[0]]
        fitness_report = [(ordered[0], 1.0)]

        if use_gt_poses:
            # Apply the known TF pose DIRECTLY -- no ICP refinement.
            #
            # ICP is a technique for estimating an UNKNOWN relative
            # transform from point correspondences (hr01/hr02 coordinate-
            # transform material). We have the opposite situation: Isaac
            # Sim's /tf gives an exact, noise-free camera pose. Letting
            # ICP "refine" on top of that doesn't improve it -- it lets a
            # spurious local minimum override a pose that was already
            # exact, which is exactly what was happening here: our clouds
            # are extremely sparse (~70 points/view), so point-to-point
            # ICP was locking onto a handful of coincidentally-nearby
            # points and reporting fitness=1.00 while actually producing
            # 3+ disjoint clusters (confirmed via DBSCAN). This matches
            # hr02_perception_2.pdf's SDF/TSDF fusion approach: known-pose
            # multi-view fusion combines measurements using the given
            # pose directly, letting noise cancel out across views --
            # it does not re-derive alignment from the noisy geometry.
            #
            # Output frame: WORLD (the frame the poses are expressed in),
            # not the first view's camera frame. Each view's cam_to_world
            # (OpenCV optical frame -> world, see the GUI) is applied
            # directly, including view 0. The downstream grasp planner and
            # the arm IK both live in world coordinates, and it makes
            # checking the result against the USD scene trivial.
            merged = o3d.geometry.PointCloud()
            fitness_report = []
            for idx in ordered:
                T = pose_by_idx[idx]                 # view idx -> world
                global_tf[idx] = T
                merged = merged + o3d.geometry.PointCloud(clouds[idx]).transform(T)
                fitness_report.append((idx, 1.0))  # trusted pose, not ICP-derived
        else:
            for i in range(1, len(ordered)):
                src, tgt = clouds[ordered[i]], clouds[ordered[i - 1]]
                tf_rel, fitness = self._register_pair(src, tgt)
                global_tf[ordered[i]] = global_tf[ordered[i - 1]] @ tf_rel
                transformed = o3d.geometry.PointCloud(src).transform(global_tf[ordered[i]])
                merged = merged + transformed
                fitness_report.append((ordered[i], fitness))

        # --- noise-aware filter settings ------------------------------------
        # Depth noise (simulated D435i model, or a real camera -- both tag the
        # session with "depth_noise") has ~sigma_z = z^2/(f b) * sigma_d of
        # scatter plus holes. Thresholds tuned for ideal depth (4 mm cluster
        # radius, 1 mm voxel) would fragment a noisy object into many small
        # clusters and "keep the largest" would throw most of it away. Scale
        # them with the expected noise at capture range.
        sigma = 0.0
        if depth_noise and depth_noise.get("enabled"):
            capture_z = 0.5   # typical capture range (m); views are 0.35-0.6 m from the target
            fb = float(intr["fx"]) * float(depth_noise.get("baseline_m", 0.05))
            sigma = capture_z ** 2 / fb * float(depth_noise.get("disp_sigma_px", 0.08)) * float(depth_noise.get("severity", 1.0))
        voxel = max(self.voxel_size, 1.5 * sigma)
        eps = max(self.voxel_size * 4, 6.0 * sigma + 2 * voxel)
        std_ratio = 2.0 if sigma == 0 else 2.5
        if sigma > 0:
            print(f"[sam2_service] fuse: noise-aware filters sigma={sigma*1000:.1f} mm voxel={voxel*1000:.1f} mm "
                  f"dbscan eps={eps*1000:.1f} mm", file=sys.stderr)

        merged = merged.voxel_down_sample(voxel)
        cl, ind = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        merged = merged.select_by_index(ind)

        # Statistical outlier removal only catches points that are locally
        # sparse -- it does NOT catch a small stray cluster that's tight
        # among itself but far from the main object (e.g. a few pixels of
        # mask leakage in one view, or a reflection artifact). Keep the
        # largest connected component -- and, with noise, any other
        # component at least a quarter of its size: a hole band (dropped
        # edge pixels) can legitimately split one object's faces into two
        # components, and dropping a whole face is worse than keeping a
        # rare stray blob.
        if len(merged.points) > 20:
            labels = np.array(merged.cluster_dbscan(eps=eps, min_points=5))
            if labels.max() >= 0:  # at least one real cluster found (not all noise)
                sizes = np.bincount(labels[labels >= 0])
                largest = sizes.argmax()
                keep = [largest]
                if sigma > 0:
                    keep = [i for i, n in enumerate(sizes) if n >= 0.25 * sizes[largest]]
                merged = merged.select_by_index(np.where(np.isin(labels, keep))[0])
                if len(keep) > 1:
                    print(f"[sam2_service] fuse: kept {len(keep)} components of sizes {[int(sizes[i]) for i in keep]}",
                          file=sys.stderr)

        return merged, fitness_report, use_gt_poses

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
    detector = {"instance": None, "load_failed": False}
    pc_builder = PointCloudBuilder()

    def get_classifier():
        if classifier["load_failed"]:
            raise RuntimeError("classifier previously failed to load -- see earlier server logs")
        if classifier["instance"] is None:
            try:
                classifier["instance"] = Classifier(device=device)
            except Exception as load_exc:
                classifier["load_failed"] = True
                raise RuntimeError(f"classifier failed to load: {load_exc}") from load_exc
        return classifier["instance"]

    def get_detector():
        if detector["load_failed"]:
            raise RuntimeError("prop detector previously failed to load -- see earlier server logs")
        if detector["instance"] is None:
            try:
                detector["instance"] = PropDetector(checkpoint_dir, checkpoint_key, device=device)
            except Exception as load_exc:
                detector["load_failed"] = True
                raise RuntimeError(f"prop detector failed to load: {load_exc}") from load_exc
        return detector["instance"]

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
                note = getattr(session, "last_track_note", "")
                if mask is None:
                    resp = build_response(STATUS_NO_OBJECT, message=note)
                else:
                    resp = build_response(STATUS_OK, mask_bool=mask, message=note)

            elif req_type == REQ_CLASSIFY:
                label, shape, confidence = get_classifier().classify(frame)
                resp = build_response(STATUS_OK, label=label, shape=shape, confidence=confidence)

            elif req_type == REQ_DETECT_ALL:
                if frame is None:
                    raise RuntimeError("detect_all needs a frame in the payload")
                candidates, label_map, stats = get_detector().detect(
                    frame, get_classifier(), target_label=meta.get("target_label"))
                resp = build_response(STATUS_OK, message=stats, label_map=label_map,
                                       candidates=candidates, target_label=meta.get("target_label"),
                                       detect_id=get_detector().detect_id)

            elif req_type == REQ_SELECT_MASK:
                det = detector["instance"]
                if det is None or not det.last_masks:
                    raise RuntimeError("no detect_all candidates cached -- run detect_all first")
                cid = int(meta["candidate_id"])
                want = meta.get("detect_id")
                if want is not None and int(want) != det.detect_id:
                    # another client ran detect_all in between -- the ids no longer mean the same masks
                    raise RuntimeError(f"stale candidate ids: you selected from detection #{want}, "
                                       f"current is #{det.detect_id} -- re-run detect_all")
                if cid not in det.last_masks:
                    raise RuntimeError(f"candidate id {cid} not in last detect_all result "
                                        f"(have {sorted(det.last_masks)})")
                # Prefer the caller's CURRENT frame as the anchor if it sent
                # one (live feed may have advanced a few frames since
                # detect_all); fall back to the frame the mask was made on.
                anchor = frame if frame is not None else det.last_frame
                mask = session.seed_mask(anchor, det.last_masks[cid])
                resp = build_response(STATUS_OK, mask_bool=mask, candidate_id=cid)

            elif req_type == REQ_GENERATE_PC:
                session_dir = remap_path(meta["session_dir"], container_root, host_root)
                meta_path = os.path.join(session_dir, "session_meta.json")
                if not os.path.isfile(meta_path):
                    raise RuntimeError("session_meta.json missing -- camera intrinsics were never captured "
                                        "(camera_info topic wasn't available when this object was selected)")
                with open(meta_path) as f:
                    session_meta = json.load(f)
                merged, fitness_report, used_gt_poses = pc_builder.fuse_session(
                    session_dir, session_meta["intrinsics"], poses=session_meta.get("poses"),
                    depth_noise=session_meta.get("depth_noise"))
                # .pcd, not .ply -- this is the format the grasping pipeline
                # (Phase after this one) expects to load directly.
                pcd_path = os.path.join(session_dir, "fused_pointcloud.pcd")
                thumb_path = os.path.join(session_dir, "thumbnail.png")

                import open3d as o3d
                o3d.io.write_point_cloud(pcd_path, merged)
                pc_builder.render_thumbnail(merged, thumb_path)

                avg_fitness = float(np.mean([f for _, f in fitness_report]))
                low_fitness_views = [idx for idx, f in fitness_report if f < 0.3]
                method = "ground-truth pose (direct, no ICP), WORLD frame" if used_gt_poses \
                    else "FPFH+RANSAC+ICP (no pose available), view-0 camera frame"
                msg = f"fused {len(fitness_report)} views via {method}, avg fitness {avg_fitness:.2f} -> {pcd_path}"
                # record the output frame next to the poses so a consumer never has to guess
                session_meta["fused_pointcloud_frame"] = "world" if used_gt_poses else "view0_opencv_optical"
                bb = merged.get_axis_aligned_bounding_box()
                session_meta["fused_pointcloud_aabb"] = {"min": [round(v, 4) for v in bb.get_min_bound()],
                                                         "max": [round(v, 4) for v in bb.get_max_bound()]}
                with open(meta_path, "w") as f:
                    json.dump(session_meta, f, indent=2)
                if low_fitness_views:
                    msg += f" (poor alignment on views {low_fitness_views} -- treat fusion as approximate)"
                resp = build_response(STATUS_OK, message=msg, num_points=len(merged.points),
                                       avg_fitness=avg_fitness, used_gt_poses=used_gt_poses)

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
