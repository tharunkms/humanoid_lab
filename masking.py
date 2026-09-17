def process_segmented_point_cloud(self, mask):
        if self.current_depth is None or self.intrinsics is None:
            return

        # Fetch Intrinsic Parameters
        fx = self.intrinsics.K[0]
        fy = self.intrinsics.K[4]
        cx = self.intrinsics.K[2]
        cy = self.intrinsics.K[5]

        # Filter the depth image using the boolean mask
        segmented_depth = np.where(mask, self.current_depth, 0)
        
        # Extract non-zero index rows/cols (the pixels belonging to the target object)
        v_indices, u_indices = np.where(segmented_depth > 0)
        z_values = segmented_depth[v_indices, u_indices] / 1000.0  # Convert mm to meters

        if len(z_values) == 0:
            return

        # Back-project pixels to 3D Camera coordinates
        x_values = (u_indices - cx) * z_values / fx
        y_values = (v_indices - cy) * z_values / fy
        points_3d = np.vstack((x_values, y_values, z_values)).T

        # Send this target cloud downstream to your grasp planner
        self.publish_to_isaac_arm(points_3d)