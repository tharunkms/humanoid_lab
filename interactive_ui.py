def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_point = np.array([[x, y]], dtype=np.float32)
            print(f"Target selected at image coordinates: X={x}, Y={y}")

    def run_gui(self):
        cv2.namedWindow("RealSense SAM 2 Grasping GUI")
        cv2.setMouseCallback("RealSense SAM 2 Grasping GUI", self.mouse_click_event)

        while not rospy.is_shutdown():
            if self.current_frame is None:
                continue
            
            display_frame = self.current_frame.copy()
            
            # If a point is selected, run SAM 2 inference
            if self.selected_point is not None:
                # Set image context for SAM
                self.predictor.set_image(self.current_frame)
                
                # Label '1' indicates a foreground point prompt
                masks, scores, _ = self.predictor.predict(
                    point_coords=self.selected_point,
                    point_labels=np.array([1])
                )
                
                # Get the highest-scoring mask
                best_mask = masks[np.argmax(scores)]
                
                # Overlay mask on display frame (colored translucent green)
                mask_visual = np.zeros_like(display_frame)
                mask_visual[best_mask] = [0, 255, 0] 
                display_frame = cv2.addWeighted(display_frame, 1.0, mask_visual, 0.4, 0)
                
                # Draw the clicked point
                cv2.circle(display_frame, tuple(self.selected_point[0].astype(int)), 5, (0, 0, 255), -1)
                
                # Extract point cloud once mask stabilizes
                self.process_segmented_point_cloud(best_mask)
            
            cv2.imshow("RealSense SAM 2 Grasping GUI", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()