#!/usr/bin/env python3
"""
refine_session_poses.py -- check and refine the camera poses of a handheld
RealSense session using the WHOLE scene, then re-fuse.

Why: the object alone (a can, a ball, a box face) is degenerate for
registration, but the full frame -- desk edge, wall, cables, other objects
-- is not. Each view's full-scene point cloud is registered to the previous
view's with point-to-plane ICP, starting from the odometry estimate. The
size of the correction ICP applies is the diagnostic: a few mm / <2 deg
means odometry was fine; centimetres means it drifted; tens of cm means it
was wrong.

Usage (host, sam2 venv):
    python3 refine_session_poses.py <session_dir>            # report + write refined poses
    python3 refine_session_poses.py <session_dir> --dry-run  # report only
Then press Fuse in the GUI again (same session) -- fusion reads the
refined poses from session_meta.json. The original odometry poses are kept
under "poses_odometry".
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
import open3d as o3d

VOXEL = 0.006          # scene downsampling for ICP (m)
DEPTH_TRUNC = 2.5      # ignore scene depth beyond this (m)


def load_scene_cloud(session_dir, idx, intr):
    rgb = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_depth.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return None, None
    K = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"], intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    color = o3d.geometry.Image(np.ascontiguousarray(rgb[:, :, ::-1]))
    d = o3d.geometry.Image(depth.astype(np.uint16))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, d, depth_scale=1000.0, depth_trunc=DEPTH_TRUNC,
                                                              convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K)
    pcd = pcd.voxel_down_sample(VOXEL)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    mask = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_mask.png"), cv2.IMREAD_GRAYSCALE)
    return pcd, mask


def object_centroid_cam(session_dir, idx, intr, mask):
    depth = cv2.imread(os.path.join(session_dir, f"pose_{idx:02d}_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    ys, xs = np.nonzero(mask > 127)
    z = depth[ys, xs]; ok = z > 0
    if ok.sum() < 20:
        return None
    xs, ys, z = xs[ok], ys[ok], z[ok]
    return np.median(np.stack([(xs - intr["cx"]) * z / intr["fx"], (ys - intr["cy"]) * z / intr["fy"], z], 1), axis=0)


def rot_deg(T):
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--to-first", action="store_true", help="register every view to view 0 instead of to the previous view")
    a = ap.parse_args()
    mp = os.path.join(a.session_dir, "session_meta.json")
    meta = json.load(open(mp))
    intr = meta["intrinsics"]
    poses = {p["index"]: np.array(p["cam_to_world"]) for p in meta.get("poses", [])}
    idxs = sorted(poses)
    if len(idxs) < 2:
        sys.exit("need >= 2 views with poses")
    print(f"session {os.path.basename(a.session_dir)}: {len(idxs)} views, pose source: {meta.get('pose_source')}")

    clouds, masks = {}, {}
    for i in idxs:
        clouds[i], masks[i] = load_scene_cloud(a.session_dir, i, intr)
        print(f"  view {i}: scene cloud {len(clouds[i].points)} pts")

    # object centroid spread BEFORE refinement (all views' centroids should coincide in world)
    def spread(P):
        cs = []
        for i in idxs:
            c = object_centroid_cam(a.session_dir, i, intr, masks[i])
            if c is not None:
                cs.append(P[i][:3, :3] @ c + P[i][:3, 3])
        cs = np.array(cs)
        return float(np.linalg.norm(cs - cs.mean(0), axis=1).max()) if len(cs) > 1 else float("nan")
    print(f"\nobject centroid spread across views, odometry poses: {spread(poses)*100:.1f} cm (ideal: ~0-1 cm)\n")

    refined = {idxs[0]: poses[idxs[0]].copy()}
    print(f"{'pair':>8} {'init dt cm':>11} {'ICP fit':>8} {'RMSE mm':>8} {'corr dt cm':>11} {'corr rot deg':>13}")
    for k in range(1, len(idxs)):
        i = idxs[k]; j = idxs[0] if a.to_first else idxs[k - 1]
        src, tgt = clouds[i], clouds[j]
        T_init = np.linalg.inv(poses[j]) @ poses[i]          # view i -> view j, from odometry
        T = T_init
        for thresh in (0.05, 0.02, 0.008):                    # coarse -> fine
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, thresh, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = reg.transformation
        corr = np.linalg.inv(T_init) @ T
        print(f"{i:>3}->{j:<3} {np.linalg.norm(T_init[:3,3])*100:>11.1f} {reg.fitness:>8.2f} {reg.inlier_rmse*1000:>8.1f} "
              f"{np.linalg.norm(corr[:3,3])*100:>11.1f} {rot_deg(corr):>13.1f}")
        refined[i] = refined[j] @ T
    print(f"\nobject centroid spread across views, refined poses:  {spread(refined)*100:.1f} cm")

    if a.dry_run:
        print("\n--dry-run: session_meta.json not modified"); return
    meta.setdefault("poses_odometry", meta["poses"])
    meta["poses"] = [{"index": i, "camera_frame": "Camera_RGB_opencv_optical", "cam_to_world": refined[i].tolist(),
                      "refined": True} for i in idxs]
    meta["pose_source"] = (meta.get("pose_source", "") + " + scene-level point-to-plane ICP refinement").strip()
    json.dump(meta, open(mp, "w"), indent=2)
    print(f"\nrefined poses written to {mp} (odometry poses kept as 'poses_odometry'). Press Fuse again.")


if __name__ == "__main__":
    main()
