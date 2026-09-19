"""
Go1 kinematic quadruped controller -- Script Editor version.

Run this by pasting into Isaac Sim's Script Editor while a stage containing
/World/go1_sensor is already open and the Play button is available. Unlike
the original standalone script, this does NOT launch SimulationApp, does NOT
open a fixed .usd file, and does NOT block in a while-loop -- it registers a
physics callback and returns immediately, exactly like the render-product /
annotator pattern used in isaac_sim_native_gui.py.

Re-running this script (paste again after an edit) is safe: it tears down
the previous run's ROS subscriber and physics callback first, the same way
isaac_sim_native_gui.py tears down its previous render product/annotators.
"""

import builtins
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.physx import get_physx_scene_query_interface

# Make sure the ROS1 extension is enabled -- harmless if it already is.
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros_bridge")

ROBOT_PRIM_PATH = "/World/go1_sensor"
PHYSICS_FPS = 60.0
CALLBACK_NAME = "go1_cmd_vel_controller"


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
                if "FL" in name or "RR" in name:
                    joint_positions[i] = d1_thigh
                else:
                    joint_positions[i] = d2_thigh
            elif "calf" in name:
                if "FL" in name or "RR" in name:
                    joint_positions[i] = d1_calf
                else:
                    joint_positions[i] = d2_calf
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
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_dyaw = 0.0
        self.walk_phase = 0.0
        self.robot.set_joint_positions(self._compute_stance_joint_positions())

        self.target_pos, self.target_quat = self.robot.get_world_pose()
        self.target_yaw = quat_to_euler_angles(self.target_quat)[2]

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
        self.current_dx += (target_dx - self.current_dx) * self.smoothing_factor
        self.current_dy += (target_dy - self.current_dy) * self.smoothing_factor
        self.current_dyaw += (target_dyaw - self.current_dyaw) * self.smoothing_factor

        speed_mag = math.sqrt(self.current_dx ** 2 + self.current_dy ** 2)
        if speed_mag > 0.001:
            self.walk_phase += speed_mag * 45.0

        self.robot.set_joint_positions(self._compute_stance_joint_positions())

        if abs(self.current_dx) > 0.0001 or abs(self.current_dy) > 0.0001 or abs(self.current_dyaw) > 0.0001:
            self.target_yaw += self.current_dyaw

            dX = self.current_dx * np.cos(self.target_yaw) - self.current_dy * np.sin(self.target_yaw)
            dY = self.current_dx * np.sin(self.target_yaw) + self.current_dy * np.cos(self.target_yaw)

            self.target_pos += np.array([dX, dY, 0.0])

            z_bob = abs(math.sin(self.walk_phase)) * 0.015
            self.target_pos[2] = self.base_z + z_bob

            self.target_quat = euler_angles_to_quat(np.array([0.0, 0.0, self.target_yaw]))

        self.robot.set_world_pose(position=self.target_pos, orientation=self.target_quat)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))


# ---------------------------------------------------------------------- #
# Tear down the previous run (subscriber + physics callback) before
# creating new ones -- same reasoning as the render product/annotator
# cleanup in isaac_sim_native_gui.py: re-pasting this script must not
# stack a second callback or leak a second subscriber.
# ---------------------------------------------------------------------- #
_prev = getattr(builtins, "_go1_controller_state", None)
if _prev is not None:
    try:
        _prev["sub"].unregister()
    except Exception as exc:
        print(f"[go1_controller] previous subscriber unregister failed: {exc}")
    try:
        _prev["world"].remove_physics_callback(_prev["callback_name"])
    except Exception as exc:
        print(f"[go1_controller] previous physics callback removal failed: {exc}")
    print("[go1_controller] tore down previous run's ROS subscriber + physics callback")

# ---------------------------------------------------------------------- #
# ROS 1 setup
# ---------------------------------------------------------------------- #
latest_cmd_vel = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}


def cmd_vel_callback(msg):
    latest_cmd_vel["dx"] = msg.linear.x / PHYSICS_FPS
    latest_cmd_vel["dy"] = msg.linear.y / PHYSICS_FPS
    latest_cmd_vel["dyaw"] = msg.angular.z / PHYSICS_FPS


if not rospy.core.is_initialized():
    try:
        rospy.init_node("isaac_sim_go1_controller", anonymous=True)
        print("[go1_controller] ROS node initialized")
    except Exception as exc:
        print(f"[go1_controller] Failed to initialize ROS: {exc}. Is roscore running?")

subscriber = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)

# ---------------------------------------------------------------------- #
# Isaac Sim setup -- uses the stage that's ALREADY open in the editor.
# Does not call open_stage(); does not create a second World if one
# already exists for this stage.
# ---------------------------------------------------------------------- #
world = World.instance()
if world is None:
    world = World(physics_dt=1.0 / PHYSICS_FPS, rendering_dt=1.0 / PHYSICS_FPS)

if world.scene.object_exists("go1_robot"):
    go1 = world.scene.get_object("go1_robot")
else:
    go1 = Robot(prim_path=ROBOT_PRIM_PATH, name="go1_robot")
    world.scene.add(go1)

# add_physics_callback requires an initialized physics simulation context
# (world._physics_context / _physx_interface). That's only created once
# world.reset() has run -- a freshly constructed World, or one that exists
# but was never reset in this session, has it as None, which is what threw
# "'NoneType' object has no attribute '_physx_interface'" here.
world.reset()

go1_controller = KinematicQuadrupedController(robot=go1)

# Closure state replacing the old was_playing local variable, since there's
# no enclosing while-loop frame for it to live in anymore.
_state = {"was_playing": False}


def on_physics_step(step_size):
    is_playing = world.is_playing()

    if is_playing and not _state["was_playing"]:
        go1_controller.sync_pose()

    if is_playing:
        go1_controller.update(
            latest_cmd_vel["dx"],
            latest_cmd_vel["dy"],
            latest_cmd_vel["dyaw"],
        )

    _state["was_playing"] = is_playing


world.add_physics_callback(CALLBACK_NAME, on_physics_step)

builtins._go1_controller_state = {
    "sub": subscriber,
    "world": world,
    "callback_name": CALLBACK_NAME,
}

print("[go1_controller] Ready -- press Play, then publish Twist messages to /cmd_vel")