#!/usr/bin/env python3
"""
depth_to_pointcloud_node.py -- standalone ROS1 node.

Subscribes to a depth image + its camera_info, converts every valid
depth pixel into a 3D point using the pinhole camera model (same
formula as hr02_perception_2.pdf: X=(u-cx)*z/fx, Y=(v-cy)*z/fy, Z=z),
and saves the result as a .ply point cloud via Open3D.

Unlike the SAM2 segmentation pipeline, this converts the WHOLE visible
scene from a single depth frame -- no masking, no multi-view fusion.
Matches the original Phase 4 spec: "Subscribes to depth + camera_info,
converts depth image -> 3D point cloud using Open3D, applies camera
intrinsics, saves output as .ply file."

Run inside the ROS1 Noetic container:
    python3 depth_to_pointcloud_node.py \
        --depth-topic /camera/depth/image_raw \
        --info-topic /camera/camera_info \
        --output ~/pointcloud_output.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


class DepthToPointCloudNode:
    def __init__(self, depth_topic, info_topic, rgb_topic, output_path,
                 continuous, min_depth, max_depth):
        self.bridge = CvBridge()
        self.output_path = os.path.expanduser(output_path)
        self.continuous = continuous
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.saved_once = False

        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        subs = [depth_sub, info_sub]

        self.rgb = None
        if rgb_topic:
            rgb_sub = message_filters.Subscriber(rgb_topic, Image)
            subs.append(rgb_sub)

        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.05)
        ts.registerCallback(self._on_frame)

        rospy.loginfo(f"[depth_to_pointcloud] waiting for {depth_topic} + {info_topic}"
                       + (f" + {rgb_topic}" if rgb_topic else "") + " ...")
        rospy.loginfo(f"[depth_to_pointcloud] output: {self.output_path}  "
                       f"mode: {'continuous (overwrites each frame)' if continuous else 'single-shot'}")

    def _on_frame(self, depth_msg, info_msg, rgb_msg=None):
        if self.saved_once and not self.continuous:
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")  # metres
        width, height = info_msg.width, info_msg.height
        K = info_msg.K
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        if not np.isfinite(fx) or fx == 0:
            rospy.logwarn_throttle(5, "[depth_to_pointcloud] invalid intrinsics (fx is nan/zero) "
                                       "-- skipping this frame, check camera_info")
            return

        rgb = None
        if rgb_msg is not None:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        pcd = self._depth_to_pointcloud(depth, fx, fy, cx, cy, rgb)
        if pcd is None or len(pcd.points) == 0:
            rospy.logwarn_throttle(5, "[depth_to_pointcloud] no valid depth points in this frame")
            return

        o3d.io.write_point_cloud(self.output_path, pcd)
        rospy.loginfo(f"[depth_to_pointcloud] saved {len(pcd.points)} points -> {self.output_path}")
        self.saved_once = True

        if not self.continuous:
            rospy.loginfo("[depth_to_pointcloud] single-shot mode -- done, shutting down.")
            rospy.signal_shutdown("capture complete")

    def _depth_to_pointcloud(self, depth, fx, fy, cx, cy, rgb=None):
        """Standard pinhole deprojection (hr02_perception_2.pdf):
        X=(u-cx)*z/fx, Y=(v-cy)*z/fy, Z=z -- applied to every pixel with
        valid depth, not just a masked region."""
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        valid = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        if valid.sum() == 0:
            return None

        us, vs, z = us[valid], vs[valid], z[valid]
        X = (us - cx) * z / fx
        Y = (vs - cy) * z / fy
        points = np.stack([X, Y, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if rgb is not None:
            colors = rgb[valid].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--info-topic", default="/camera/camera_info")
    ap.add_argument("--rgb-topic", default="/camera/rgb/image_raw",
                     help="optional -- adds color to the point cloud. Pass empty string to skip.")
    ap.add_argument("--output", default="~/pointcloud_output.ply")
    ap.add_argument("--continuous", action="store_true",
                     help="keep saving every frame (overwrites output each time) instead of "
                          "single-shot (save once, then exit)")
    ap.add_argument("--min-depth", type=float, default=0.05, help="metres")
    ap.add_argument("--max-depth", type=float, default=5.0, help="metres")
    args, _ = ap.parse_known_args()

    rospy.init_node("depth_to_pointcloud_node", anonymous=True)
    node = DepthToPointCloudNode(
        args.depth_topic, args.info_topic, args.rgb_topic or None,
        args.output, args.continuous, args.min_depth, args.max_depth)
    rospy.spin()


if __name__ == "__main__":
    main()
