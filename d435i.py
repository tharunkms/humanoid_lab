import numpy as np
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import IMUSensor

ROOT = "/World/open_manipulator_x/link5/realsense2_camera/base_link"

def add_camera(prim_path, width, height, h_fov_deg, freq=30):
    cam = Camera(prim_path=prim_path, resolution=(width, height), frequency=freq)
    cam.initialize()
    focal_length = 1.88
    h_fov = np.deg2rad(h_fov_deg)
    h_aperture = 2 * focal_length * np.tan(h_fov / 2)
    cam.set_focal_length(focal_length)
    cam.set_horizontal_aperture(h_aperture)
    cam.set_vertical_aperture(h_aperture * height / width)
    cam.set_clipping_range(0.05, 10.0)
    return cam

rgb_cam   = add_camera(f"{ROOT}/camera_color_frame/camera_color_optical_frame/rgb_cam", 1920, 1080, 69)
left_ir   = add_camera(f"{ROOT}/camera_infra1_frame/camera_infra1_optical_frame/left_ir_cam", 1280, 720, 87)
right_ir  = add_camera(f"{ROOT}/camera_infra2_frame/camera_infra2_optical_frame/right_ir_cam", 1280, 720, 87)
depth_cam = add_camera(f"{ROOT}/camera_depth_frame/camera_depth_optical_frame/depth_cam", 1280, 720, 87)
depth_cam.add_distance_to_camera_to_frame()

imu = IMUSensor(prim_path=f"{ROOT}/camera_gyro_frame/imu", frequency=200)
imu.initialize()


