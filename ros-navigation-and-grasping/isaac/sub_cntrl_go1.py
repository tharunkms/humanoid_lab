# Launch the Isaac Sim application first (required for standalone scripts)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Enable ROS1 extension
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros_bridge")

import rospy
from geometry_msgs.msg import Twist

import os
import numpy as np
import math
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.bounds import create_bbox_cache, compute_combined_aabb
from omni.physx import get_physx_scene_query_interface

class KinematicQuadrupedController:
    def __init__(self, robot, smoothing_factor=0.15):
        self.robot = robot
        self.smoothing_factor = smoothing_factor
        self._bbox_cache = create_bbox_cache()

        # State Tracking
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_dyaw = 0.0
        self.walk_phase = 0.0

        # Initialize Pose variables
        self.target_pos = np.zeros(3)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_z = 0.0
        self.target_yaw = 0.0

    def _compute_stance_joint_positions(self):
        """Computes the trot-gait joint targets for the current walk_phase."""
        d1_thigh = 0.8 + math.sin(self.walk_phase) * 0.3
        d1_calf = -1.5 - max(0, math.sin(self.walk_phase)) * 0.4

        d2_thigh = 0.8 + math.sin(self.walk_phase + math.pi) * 0.3
        d2_calf = -1.5 - max(0, math.sin(self.walk_phase + math.pi)) * 0.4

        joint_positions = np.zeros(self.robot.num_dof)
        for i, name in enumerate(self.robot.dof_names):
            if "hip" in name:
                joint_positions[i] = 0.0
            elif "thigh" in name:
                if "FL" in name or "RR" in name: joint_positions[i] = d1_thigh
                else: joint_positions[i] = d2_thigh
            elif "calf" in name:
                if "FL" in name or "RR" in name: joint_positions[i] = d1_calf
                else: joint_positions[i] = d2_calf
        return joint_positions

    def _find_ground_z(self, x, y, start_z, max_distance=10.0):
        """Raycasts straight down from (x, y, start_z) to find the floor height,
        ignoring hits on the robot's own colliders. Falls back to start_z if
        nothing is hit."""
        hits = []

        def report_hit(hit):
            if not str(hit.collision).startswith(self.robot.prim_path):
                hits.append(hit.position[2])
            return True  # keep collecting hits along the ray

        get_physx_scene_query_interface().raycast_all(
            (float(x), float(y), float(start_z)), (0.0, 0.0, -1.0), max_distance, report_hit
        )
        return max(hits) if hits else start_z

    def sync_pose(self):
        """Captures the robot's current pose. Call this when Play is pressed."""
        # Reset momentum and snap the legs into the nominal standing pose
        # BEFORE measuring height, so the height we lock in matches the pose
        # the animation will actually hold (otherwise the legs jump into
        # this stance on the first update() and the feet no longer line up
        # with whatever height happened to be captured).
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_dyaw = 0.0
        self.walk_phase = 0.0
        self.robot.set_joint_positions(self._compute_stance_joint_positions())

        self.target_pos, self.target_quat = self.robot.get_world_pose()
        self.target_yaw = quat_to_euler_angles(self.target_quat)[2]

        # Calibrate the standing height from the robot's actual geometry
        # instead of trusting wherever it was placed in the GUI: measure the
        # lowest point of its world-space bounding box in this stance, find
        # the actual floor beneath the robot via raycast (works regardless
        # of scene layout/naming), and shift the body so the feet land
        # exactly on it.
        aabb = compute_combined_aabb(self._bbox_cache, [self.robot.prim_path])
        lowest_point_z = aabb[2]
        feet_offset = lowest_point_z - self.target_pos[2]
        ground_z = self._find_ground_z(
            self.target_pos[0], self.target_pos[1], self.target_pos[2] + 2.0
        )
        self.base_z = ground_z - feet_offset
        self.target_pos[2] = self.base_z
        self.robot.set_world_pose(position=self.target_pos, orientation=self.target_quat)

    def update(self, target_dx, target_dy, target_dyaw):
        """Applies smoothing, procedural animation, and teleportation."""
        # 1. Apply Inertia/Smoothing
        self.current_dx += (target_dx - self.current_dx) * self.smoothing_factor
        self.current_dy += (target_dy - self.current_dy) * self.smoothing_factor
        self.current_dyaw += (target_dyaw - self.current_dyaw) * self.smoothing_factor

        # 2. Calculate Speed Magnitude for Animation
        speed_mag = math.sqrt(self.current_dx**2 + self.current_dy**2)
        if speed_mag > 0.001:
            self.walk_phase += speed_mag * 45.0
        # else: hold the current phase. Snapping it back to 0 here caused a
        # visible pop every time the robot slowed down or a /cmd_vel
        # message dropped out, since the legs would jump straight from
        # mid-stride to the stance pose in a single frame.

        # 3. Calculate Procedural Walking Animation (Trot Gait)
        self.robot.set_joint_positions(self._compute_stance_joint_positions())

        # 4. Calculate Movement and Z-Bobbing
        if abs(self.current_dx) > 0.0001 or abs(self.current_dy) > 0.0001 or abs(self.current_dyaw) > 0.0001:
            self.target_yaw += self.current_dyaw
            
            dX = self.current_dx * np.cos(self.target_yaw) - self.current_dy * np.sin(self.target_yaw)
            dY = self.current_dx * np.sin(self.target_yaw) + self.current_dy * np.cos(self.target_yaw)
            
            self.target_pos += np.array([dX, dY, 0.0])
            
            # Apply bobbing on top of the captured base height
            z_bob = abs(math.sin(self.walk_phase)) * 0.015
            self.target_pos[2] = self.base_z + z_bob
            
            self.target_quat = euler_angles_to_quat(np.array([0.0, 0.0, self.target_yaw]))

        # 5. Apply the Pose
        self.robot.set_world_pose(position=self.target_pos, orientation=self.target_quat)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))

# ==========================================
# ROS 1 Setup
# ==========================================
latest_cmd_vel = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}
physics_fps = 60.0 

def cmd_vel_callback(msg):
    latest_cmd_vel["dx"] = msg.linear.x / physics_fps
    latest_cmd_vel["dy"] = msg.linear.y / physics_fps
    latest_cmd_vel["dyaw"] = msg.angular.z / physics_fps

try:
    rospy.init_node('isaac_sim_go1_controller', anonymous=True)
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    print("Successfully connected to ROS. Listening to /cmd_vel...")
except Exception as e:
    print(f"Failed to initialize ROS: {e}. Is roscore running?")

# ==========================================
# Isaac Sim Setup
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
open_stage(usd_path=os.path.join(script_dir, "base_scene_quadruped_inspection.usd"))

world = World(physics_dt=1.0/physics_fps, rendering_dt=1.0/physics_fps)
go1 = Robot(prim_path="/World/go1_sensor", name="go1_robot")
world.scene.add(go1)
world.reset()

go1_controller = KinematicQuadrupedController(robot=go1)

# State tracker for the Play button
was_playing = False

# 6. Main Simulation Loop
while simulation_app.is_running() and not rospy.is_shutdown():
    world.step(render=True)
    
    # Check if the play button is pressed in the GUI
    is_playing = world.is_playing()
    
    # If the user JUST clicked Play, sync the controller's target to the robot's current pose in the GUI
    if is_playing and not was_playing:
        go1_controller.sync_pose()
        
    # Only animate and move the robot if the simulation is actively playing
    if is_playing:
        go1_controller.update(
            latest_cmd_vel["dx"], 
            latest_cmd_vel["dy"], 
            latest_cmd_vel["dyaw"]
        )
        
    # Update state for the next frame
    was_playing = is_playing

simulation_app.close()