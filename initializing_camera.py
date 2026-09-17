import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class RealSenseSAMGUI:
    def __init__(self):
        rospy.init_node('s_a_m_grasp_gui', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_callback)
        
        self.current_frame = None
        self.current_depth = None
        self.intrinsics = None
        self.selected_point = None  # Stores (x, y) from mouse click

        # Initialize SAM 2 Predictor
        # from sam2.sam2_image_predictor import SAM2ImagePredictor
        # self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")