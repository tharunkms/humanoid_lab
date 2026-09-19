#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import sys
import os
import math
import threading
import actionlib
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseFeedback, MoveBaseResult

# Add the mpc directory to sys.path so we can import mpc_controller
mpc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mpc'))
if mpc_dir not in sys.path:
    sys.path.append(mpc_dir)

from mpc_controller import MPCComponent

class MPCROSNode:
    def __init__(self):
        # Fixed name so the action (/mpc_ros_node/navigate) and ~params have a known namespace
        rospy.init_node('mpc_ros_node')
        
        # Initialize the MPC Component
        self.mpc = MPCComponent(N=20, sim_time=20, step_horizon=0.1)
        
        rospy.loginfo("Initializing MPC Solver and Constraints...")
        self.mpc.init_symbolic_vars()
        self.mpc.init_cost_fn_and_g_constraints()
        self.mpc.init_solver()
        self.mpc.init_constraint_args()
        
        # State variables
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.target_state = np.array([0.0, 0.0, 0.0])
        self.global_path = None
        self.obstacles = []
        self.lookahead_dist = 1.5 # meters

        # Align mode: turn toward the carrot before handing control to the MPC.
        # Hysteresis (enter > exit) plus a committed turn direction stop the
        # left/right flip-flopping when the carrot is ~180 deg behind the robot.
        self.align_enter_angle = math.radians(90)
        self.align_exit_angle = math.radians(20)
        self.align_omega = self.mpc.omega_max
        self.align_backup_speed = 0.15 # m/s, used to make room when there's no space to spin
        self.align_lookahead = 0.3 # s, how far ahead align-mode motions are collision checked
        self.align_max_shuffle = 0.4 # m, max travel in one direction before trying the other way
        self.align_timeout = 15.0 # s, give up and let the MPC try (e.g. reverse down a corridor)
        self.align_cooldown = 5.0 # s, how long align mode stays off after giving up
        self.aligning = False
        self.align_dir = 1.0
        self.align_start_time = 0.0
        self.align_cooldown_until = 0.0
        self.shuffle_dir = 0.0 # 0 = not shuffling, otherwise +1 forward / -1 backward
        self.shuffle_start = None
        self.shuffle_flips = 0 # direction reversals since the last successful spin

        # Goal handling. A goal comes from the ~/navigate action (for other nodes, e.g. the
        # arm) or from /mpc_target (RViz). move_base is only used as a planner via its
        # make_plan service, so never send goals to move_base directly: while it has an
        # active goal it refuses make_plan requests.
        self.xy_goal_tolerance = rospy.get_param('~xy_goal_tolerance', 0.05) # m, latched once reached
        self.yaw_goal_tolerance = math.radians(rospy.get_param('~yaw_goal_tolerance_deg', 5.0))
        self.settle_time = rospy.get_param('~settle_time', 0.8) # s standing still before reporting success
        self.final_approach_dist = rospy.get_param('~final_approach_dist', 0.3) # m, ignore path heading inside this
        self.goal_snap_tolerance = rospy.get_param('~goal_snap_tolerance', 0.1) # m, how far the planner may move the goal
        self.replan_period = rospy.get_param('~replan_period', 1.0) # s
        self.plan_patience = rospy.get_param('~plan_patience', 5.0) # s without a valid plan before aborting
        self.progress_timeout = rospy.get_param('~progress_timeout', 20.0) # s without getting 5 cm closer before aborting
        self.final_spin_min_omega = 0.15 # rad/s, floor so the P controller doesn't stall in the Go1's deadband
        self.final_approach_max_angle = math.radians(10) # turn in place first if the goal is further off-axis
        self.mpc.goal_tolerance = 0.5 * self.xy_goal_tolerance

        # All goal state changes and action server calls happen in the main loop. Other threads
        # (the /mpc_target subscriber, the replan timer) only leave requests under this lock.
        self.goal_lock = threading.Lock()
        self.pending_topic_goal = None # PoseStamped from /mpc_target, picked up by the main loop
        self.pending_abort = None # reason string set by the replan timer
        self.goal = None # np.array([x, y, yaw]) in map, or None when idle
        self.goal_is_action = False # True if the goal came from the action server
        self.phase = 'idle' # idle -> track -> final_align -> settle -> idle
        self.phase_start = 0.0
        self.last_plan_ok_time = 0.0
        self.best_goal_dist = float('inf')
        self.last_progress_time = 0.0
        
        self.mpc.prepare_step(self.current_state)
        self.mpc.init_sim_params()
        
        # ROS TF Listener
        self.tf_listener = tf.TransformListener()
        
        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # GUI Target Subscriber (RViz "2D Nav Goal" publishes here)
        rospy.Subscriber('/mpc_target', PoseStamped, self.gui_target_callback)

        # Global planning through move_base's planner + costmaps
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.target_marker_pub = rospy.Publisher('/mpc_target_marker', Marker, queue_size=1)
        self.trajectory_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=1)
        self.obstacles_pub = rospy.Publisher('/mpc_obstacles', MarkerArray, queue_size=1)
        self.footprint_pub = rospy.Publisher('/mpc_footprint', MarkerArray, queue_size=1)
        
        self.rate = rospy.Rate(int(1.0 / self.mpc.step_horizon))
        rospy.sleep(1.0)

        # Action server for other nodes: send a MoveBaseGoal, get SUCCEEDED once the robot is
        # at the pose (position + heading) and has settled, ABORTED with a reason otherwise.
        # No callbacks are registered: SimpleActionServer runs them while holding its own lock,
        # so the main loop polls it instead (see process_goal_requests).
        self.action_server = actionlib.SimpleActionServer('~navigate', MoveBaseAction, auto_start=False)
        self.action_server.start()

        rospy.Timer(rospy.Duration(self.replan_period), self.replan_timer_callback)

    # ------------------------------------------------------------------ goal handling

    def gui_target_callback(self, msg):
        with self.goal_lock:
            self.pending_topic_goal = msg

    def process_goal_requests(self):
        """Handle new/cancelled goals from the action server and /mpc_target (main loop only)."""
        with self.goal_lock:
            topic_goal, self.pending_topic_goal = self.pending_topic_goal, None
            abort_reason, self.pending_abort = self.pending_abort, None

        if abort_reason is not None and self.goal is not None:
            self.finish_goal(False, abort_reason)

        if self.action_server.is_new_goal_available():
            # accept_new_goal() also cancels a still-active previous action goal
            if self.goal is not None and not self.goal_is_action:
                rospy.loginfo("/mpc_target goal replaced by an action goal")
            goal = self.action_server.accept_new_goal()
            rospy.loginfo("New goal from action client")
            self.start_goal(goal.target_pose, is_action=True)
        elif self.action_server.is_preempt_requested():
            rospy.loginfo("Goal cancelled by action client")
            self.clear_goal()
            self.action_server.set_preempted(MoveBaseResult(), "Cancelled by client")

        if topic_goal is not None:
            rospy.loginfo("New goal from /mpc_target")
            if self.action_server.is_active():
                self.action_server.set_preempted(MoveBaseResult(), "Replaced by a goal on /mpc_target")
            self.start_goal(topic_goal, is_action=False)

    def start_goal(self, pose_msg, is_action):
        self.goal_is_action = is_action
        q = pose_msg.pose.orientation
        if math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2) < 1e-6:
            self.finish_goal(False, "Goal has an invalid (all-zero) quaternion")
            return
        try:
            pose_msg.header.stamp = rospy.Time(0) # use the latest transform
            if not pose_msg.header.frame_id:
                pose_msg.header.frame_id = 'map'
            pose_map = self.tf_listener.transformPose('map', pose_msg)
        except Exception as e:
            self.finish_goal(False, f"Could not transform goal from '{pose_msg.header.frame_id}' to map: {e}")
            return

        q = pose_map.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        now = rospy.get_time()
        with self.goal_lock:
            self.goal = np.array([pose_map.pose.position.x, pose_map.pose.position.y, yaw])
            self.global_path = None
            self.last_plan_ok_time = now
            self.pending_abort = None
        self.aligning = False
        self.set_phase('track', now)
        self.best_goal_dist = float('inf')
        self.last_progress_time = now
        self.mpc.prepare_step(self.current_state)
        rospy.loginfo(f"Navigating to x={self.goal[0]:.2f} y={self.goal[1]:.2f} "
                      f"yaw={math.degrees(self.goal[2]):.0f} deg")

        # Plan right away rather than waiting for the timer
        error = self.replan()
        if error is not None:
            self.finish_goal(False, error)

    def clear_goal(self):
        with self.goal_lock:
            self.goal = None
            self.global_path = None
            self.pending_abort = None
        self.phase = 'idle'
        self.aligning = False

    def finish_goal(self, success, text):
        if success:
            rospy.loginfo(f"Goal reached: {text}")
        else:
            rospy.logwarn(f"Goal aborted: {text}")
        self.clear_goal()
        if self.goal_is_action and self.action_server.is_active():
            if success:
                self.action_server.set_succeeded(MoveBaseResult(), text)
            else:
                self.action_server.set_aborted(MoveBaseResult(), text)

    def set_phase(self, phase, now):
        if phase != self.phase:
            rospy.loginfo(f"Navigation phase: {self.phase} -> {phase}")
        self.phase = phase
        self.phase_start = now

    def replan_timer_callback(self, _event):
        if self.goal is not None and self.phase == 'track':
            error = self.replan()
            if error is not None:
                with self.goal_lock:
                    self.pending_abort = error

    def replan(self):
        """Ask move_base's global planner for a path from the robot to the goal.

        Stores the path in self.global_path. Returns None, or a reason string if the goal
        should be aborted.
        """
        with self.goal_lock:
            goal_state = self.goal
        if goal_state is None:
            return None
        robot_state = self.current_state

        start = PoseStamped()
        start.header.frame_id = 'map'
        start.header.stamp = rospy.Time.now()
        start.pose.position.x, start.pose.position.y = robot_state[0], robot_state[1]
        q = tf.transformations.quaternion_from_euler(0, 0, robot_state[2])
        start.pose.orientation.x, start.pose.orientation.y, start.pose.orientation.z, start.pose.orientation.w = q

        goal = PoseStamped()
        goal.header = start.header
        goal.pose.position.x, goal.pose.position.y = goal_state[0], goal_state[1]
        q = tf.transformations.quaternion_from_euler(0, 0, goal_state[2])
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = q

        poses = []
        try:
            poses = self.make_plan(start=start, goal=goal, tolerance=0.0).plan.poses
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn_throttle(2.0, f"make_plan call failed: {e}")

        now = rospy.get_time()
        with self.goal_lock:
            if self.goal is not goal_state:
                return None # goal changed while we were planning; this result is stale
            if poses:
                # The planner may quietly end the path near the goal when the goal itself is in an
                # obstacle/inflation (GlobalPlanner then appends the requested goal after that
                # point), so check that the path really arrives at the goal.
                end = poses[-1].pose.position
                end_gap = math.hypot(end.x - goal_state[0], end.y - goal_state[1])
                last_step = 0.0
                if len(poses) >= 2:
                    prev = poses[-2].pose.position
                    last_step = math.hypot(end.x - prev.x, end.y - prev.y)
                if end_gap > self.goal_snap_tolerance or last_step > self.goal_snap_tolerance:
                    return "Goal is not reachable (inside an obstacle or its inflation)"
                self.global_path = poses
                self.last_plan_ok_time = now
            elif now - self.last_plan_ok_time > self.plan_patience:
                return f"No valid global plan for {self.plan_patience:.0f} s"
        return None

    def scan_callback(self, msg):
        # Convert LaserScan to (X, Y) in the map frame
        points = []
        angle = msg.angle_min
        
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', msg.header.frame_id, rospy.Time(0))
            laser_transform = self.tf_listener.fromTranslationRotation(trans, rot)
        except Exception:
            return
            
        for r in msg.ranges:
            if not np.isinf(r) and not np.isnan(r) and r >= msg.range_min and r <= msg.range_max:
                x_s = r * math.cos(angle)
                y_s = r * math.sin(angle)
                point_s = np.array([x_s, y_s, 0.0, 1.0])
                point_m = np.dot(laser_transform, point_s)
                points.append([point_m[0], point_m[1]])
            angle += msg.angle_increment
            
        if len(points) == 0:
            self.obstacles = []
            return
            
        # Voxel Grid Downsampling
        voxel_size = 0.2 # Resolution of the grid (was 0.3)
        voxels = {}
        for p in points:
            # Snap point to grid index
            grid_x = math.floor(p[0] / voxel_size)
            grid_y = math.floor(p[1] / voxel_size)
            voxel_idx = (grid_x, grid_y)
            
            if voxel_idx not in voxels:
                voxels[voxel_idx] = []
            voxels[voxel_idx].append(p)
            
        detected_obstacles = []
        for voxel_idx, voxel_points in voxels.items():
            # Find centroid of the points in this voxel
            voxel_points_arr = np.array(voxel_points)
            centroid = np.mean(voxel_points_arr, axis=0)
            
            # The obstacle diameter is the voxel size with minimal padding
            diameter = voxel_size + 0.05
            
            dist_to_robot = math.hypot(centroid[0] - self.current_state[0], centroid[1] - self.current_state[1])
            
            detected_obstacles.append({
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "diameter": float(diameter),
                "dist": dist_to_robot
            })
            
        # Sort by distance to robot and keep only top MAX_OBS
        detected_obstacles.sort(key=lambda x: x["dist"])
        self.obstacles = detected_obstacles[:self.mpc.MAX_OBS]
        
        # Publish obstacles for RViz visualization
        marker_array = MarkerArray()
        for i in range(self.mpc.MAX_OBS):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "map"
            marker.ns = "mpc_obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            
            if i < len(self.obstacles):
                obs = self.obstacles[i]
                marker.action = Marker.ADD
                marker.pose.position.x = obs["x"]
                marker.pose.position.y = obs["y"]
                marker.pose.position.z = 0.5
                marker.pose.orientation.w = 1.0
                marker.scale.x = obs["diameter"]
                marker.scale.y = obs["diameter"]
                marker.scale.z = 1.0
                marker.color.a = 0.5
                marker.color.r = 1.0 # Orange
                marker.color.g = 0.5
                marker.color.b = 0.0
            else:
                marker.action = Marker.DELETE
                
            marker_array.markers.append(marker)
            
        self.obstacles_pub.publish(marker_array)

    def get_current_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            x, y = trans[0], trans[1]
            euler = tf.transformations.euler_from_quaternion(rot)
            theta = euler[2]
            return np.array([x, y, theta])
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"TF Error: {e}")
            return None
            
    def get_carrot_target(self, path):
        if not path:
            rospy.loginfo_throttle(2.0, "Waiting for a global path...")
            return None
            
        # Find closest point on path to robot
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose_stamped in enumerate(path):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = math.hypot(px - self.current_state[0], py - self.current_state[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead from the closest point
        target_idx = closest_idx
        accumulated_dist = 0.0
        
        while accumulated_dist < self.lookahead_dist and target_idx < len(path) - 1:
            p1 = path[target_idx].pose.position
            p2 = path[target_idx+1].pose.position
            dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
            accumulated_dist += dist
            target_idx += 1
            
        carrot = path[target_idx].pose.position
        target_theta = math.atan2(carrot.y - self.current_state[1], carrot.x - self.current_state[0])
        return np.array([carrot.x, carrot.y, target_theta])

    def publish_marker(self, state):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "mpc_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = state[0]
        marker.pose.position.y = state[1]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.target_marker_pub.publish(marker)

    def publish_footprint(self, current_pose):
        marker_array = MarkerArray()
        x, y, theta = current_pose
        
        offset = self.mpc.rob_circle_offset
        diam = self.mpc.rob_circle_diameter
        
        centers = [
            (x + offset * math.cos(theta), y + offset * math.sin(theta)),
            (x, y),
            (x - offset * math.cos(theta), y - offset * math.sin(theta))
        ]
        
        for i, (cx, cy) in enumerate(centers):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "map"
            marker.ns = "mpc_footprint"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = diam
            marker.scale.y = diam
            marker.scale.z = 0.05
            marker.color.a = 0.4
            marker.color.r = 0.0
            marker.color.g = 0.5
            marker.color.b = 1.0 # Light blue
            marker_array.markers.append(marker)
        
        self.footprint_pub.publish(marker_array)

    def footprint_clearance(self, x, y, theta):
        # Smallest gap between any footprint circle and any obstacle (negative = overlapping)
        offset = self.mpc.rob_circle_offset
        rob_r = self.mpc.rob_circle_diameter / 2
        clearance = float('inf')
        for d in (offset, 0.0, -offset):
            cx = x + d * math.cos(theta)
            cy = y + d * math.sin(theta)
            for obs in self.obstacles:
                gap = math.hypot(obs["x"] - cx, obs["y"] - cy) - rob_r - obs["diameter"] / 2
                clearance = min(clearance, gap)
        return clearance

    def compute_align_cmd(self, heading_err):
        """Turn toward the carrot before letting the MPC drive.

        Rotates in place when that's collision free over a short lookahead; otherwise
        shuffles forward/backward to make room (a multi-point turn). Returns (v, w),
        or None once the heading is aligned and the MPC should take over.
        """
        now_t = rospy.get_time()
        if (not self.aligning and abs(heading_err) > self.align_enter_angle
                and now_t >= self.align_cooldown_until):
            self.aligning = True
            self.align_start_time = now_t
            self.shuffle_dir = 0.0
            self.shuffle_flips = 0
            # Commit to the shorter direction now; don't re-decide every cycle
            self.align_dir = 1.0 if heading_err >= 0 else -1.0
            rospy.loginfo(f"Align mode: turning {'left' if self.align_dir > 0 else 'right'} "
                          f"({math.degrees(heading_err):.0f} deg)")
        elif self.aligning and abs(heading_err) < self.align_exit_angle:
            self.aligning = False
            # The MPC's warm start is from before the turn; start it fresh
            self.mpc.prepare_step(self.current_state)
            rospy.loginfo("Align mode: done, handing over to MPC")
        elif self.aligning and (now_t - self.align_start_time > self.align_timeout or self.shuffle_flips >= 2):
            # Timed out, or shuffled both ways without gaining room to spin (e.g. narrow corridor)
            self.aligning = False
            self.align_cooldown_until = now_t + self.align_cooldown
            self.mpc.prepare_step(self.current_state)
            rospy.logwarn("Align mode: no room to turn around here, letting the MPC try")

        if not self.aligning:
            return None

        x, y, theta = self.current_state
        now = self.footprint_clearance(x, y, theta)
        # A move is allowed if it doesn't create new overlap or deepen an existing one
        min_allowed = min(0.0, now)

        spin_theta = theta + self.align_dir * self.align_omega * self.align_lookahead
        if self.footprint_clearance(x, y, spin_theta) >= min_allowed:
            self.shuffle_dir = 0.0
            self.shuffle_flips = 0
            return 0.0, self.align_dir * self.align_omega

        # No room to spin: move along the heading to make room (a multi-point turn)
        step = self.align_backup_speed * self.align_lookahead

        def can_move(direction):
            nx = x + direction * step * math.cos(theta)
            ny = y + direction * step * math.sin(theta)
            return self.footprint_clearance(nx, ny, theta) >= min_allowed

        # Keep shuffling the same way (prevents fwd/back dithering) until blocked or far enough
        if self.shuffle_dir != 0.0:
            travelled = math.hypot(x - self.shuffle_start[0], y - self.shuffle_start[1])
            if travelled > self.align_max_shuffle or not can_move(self.shuffle_dir):
                self.shuffle_dir = -self.shuffle_dir
                self.shuffle_start = (x, y)
                self.shuffle_flips += 1
        else:
            # Start in whichever direction gives the next spin more room
            best = None
            for direction in (-1.0, 1.0):
                if not can_move(direction):
                    continue
                nx = x + direction * step * math.cos(theta)
                ny = y + direction * step * math.sin(theta)
                score = self.footprint_clearance(nx, ny, spin_theta)
                if best is None or score > best[0]:
                    best = (score, direction)
            if best is not None:
                self.shuffle_dir = best[1]
                self.shuffle_start = (x, y)

        if self.shuffle_dir != 0.0 and can_move(self.shuffle_dir):
            return self.shuffle_dir * self.align_backup_speed, 0.0

        rospy.logwarn_throttle(2.0, "Align mode: no room to turn or move, stopping")
        return 0.0, 0.0

    def spin_is_safe(self, omega):
        # True if rotating in place at omega for align_lookahead doesn't create/deepen overlap
        x, y, theta = self.current_state
        min_allowed = min(0.0, self.footprint_clearance(x, y, theta))
        return self.footprint_clearance(x, y, theta + omega * self.align_lookahead) >= min_allowed

    def mpc_cmd(self, target):
        """Run one MPC step toward target and publish the predicted trajectory. Returns (v, w)."""
        self.target_state = target
        self.publish_marker(target)
        try:
            u = self.mpc.step(self.current_state, target, self.obstacles)
            if not self.mpc.last_solve_ok:
                rospy.logwarn_throttle(2.0, "MPC solve failed, stopping this cycle")
            elif self.mpc.max_slack > 1e-3:
                rospy.logwarn_throttle(2.0, f"MPC plan violates obstacle margin by {self.mpc.max_slack:.3f} m")

            # Publish trajectory prediction
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            states = self.mpc.DM2Arr(self.mpc.X0)
            for i in range(states.shape[1]):
                p = PoseStamped()
                p.header = path_msg.header
                p.pose.position.x = float(states[0, i])
                p.pose.position.y = float(states[1, i])
                p.pose.orientation.w = 1.0 
                path_msg.poses.append(p)
            self.trajectory_pub.publish(path_msg)

            return float(u[0, 0]), float(u[1, 0])
        except Exception as e:
            rospy.logerr_throttle(2.0, f"MPC Step failed: {e}")
            return 0.0, 0.0

    def track_step(self, dist):
        """Drive along the global path (align mode + MPC). Returns (v, w)."""
        x, y, theta = self.current_state
        if dist < self.final_approach_dist:
            # Close to the goal the path heading is meaningless and a unicycle can't fix
            # sideways error by creeping, so turn-drive-turn: point the front or back
            # (whichever is closer) at the goal, drive straight, fix heading afterwards.
            self.aligning = False
            bearing = math.atan2(self.goal[1] - y, self.goal[0] - x)
            err_fwd = (bearing - theta + math.pi) % (2 * math.pi) - math.pi
            err_back = (bearing + math.pi - theta + math.pi) % (2 * math.pi) - math.pi
            err = err_fwd if abs(err_fwd) <= abs(err_back) else err_back
            if abs(err) > self.final_approach_max_angle:
                w = math.copysign(max(min(1.5 * abs(err), self.mpc.omega_max), self.final_spin_min_omega), err)
                if self.spin_is_safe(w):
                    self.publish_marker(np.array([self.goal[0], self.goal[1], theta + err]))
                    return 0.0, w
            return self.mpc_cmd(np.array([self.goal[0], self.goal[1], theta + err]))

        with self.goal_lock:
            path = self.global_path
        carrot = self.get_carrot_target(path)
        if carrot is None:
            self.aligning = False
            return 0.0, 0.0

        # Normalize target theta to prevent wrap-around infinite spinning
        theta_err = (carrot[2] - theta + math.pi) % (2 * math.pi) - math.pi
        carrot[2] = theta + theta_err

        align_cmd = self.compute_align_cmd(theta_err)
        if align_cmd is not None:
            self.target_state = carrot
            self.publish_marker(carrot)
            return align_cmd
        return self.mpc_cmd(carrot)

    def navigate_step(self):
        """One control cycle for the active goal. Returns (v, w)."""
        now = rospy.get_time()
        x, y, theta = self.current_state
        gx, gy, gyaw = self.goal
        dist = math.hypot(gx - x, gy - y)
        yaw_err = (gyaw - theta + math.pi) % (2 * math.pi) - math.pi

        if self.phase == 'track':
            if dist > self.xy_goal_tolerance:
                # Watchdog: abort if we stop getting closer (blocked, oscillating, ...)
                if dist < self.best_goal_dist - 0.05:
                    self.best_goal_dist = dist
                    self.last_progress_time = now
                elif now - self.last_progress_time > self.progress_timeout:
                    self.finish_goal(False, f"No progress toward the goal for {self.progress_timeout:.0f} s "
                                            f"({dist:.2f} m away)")
                    return 0.0, 0.0
                return self.track_step(dist)
            # Position reached; it stays latched while we turn unless we get pushed well away
            self.set_phase('final_align', now)
        elif dist > 3 * self.xy_goal_tolerance:
            rospy.logwarn(f"Drifted {dist:.2f} m from the goal while turning/settling, driving back")
            self.set_phase('track', now)
            self.best_goal_dist = float('inf')
            self.last_progress_time = now
            self.mpc.prepare_step(self.current_state)
            return 0.0, 0.0

        if self.phase == 'final_align':
            # Aim for half the tolerance so body sway during settling stays inside it
            if abs(yaw_err) <= 0.5 * self.yaw_goal_tolerance:
                self.set_phase('settle', now)
                return 0.0, 0.0
            w = float(np.clip(1.5 * yaw_err, -self.mpc.omega_max, self.mpc.omega_max))
            if abs(w) < self.final_spin_min_omega:
                w = math.copysign(self.final_spin_min_omega, yaw_err)
            if not self.spin_is_safe(w):
                self.finish_goal(False, "No room to rotate to the goal heading")
                return 0.0, 0.0
            return 0.0, w

        if self.phase == 'settle':
            if abs(yaw_err) > self.yaw_goal_tolerance:
                self.set_phase('final_align', now)
                return 0.0, 0.0
            if now - self.phase_start >= self.settle_time:
                self.finish_goal(True, f"position error {dist:.3f} m, heading error "
                                       f"{math.degrees(yaw_err):.1f} deg")
            return 0.0, 0.0

        return 0.0, 0.0

    def publish_feedback(self):
        feedback = MoveBaseFeedback()
        feedback.base_position.header.stamp = rospy.Time.now()
        feedback.base_position.header.frame_id = 'map'
        feedback.base_position.pose.position.x = self.current_state[0]
        feedback.base_position.pose.position.y = self.current_state[1]
        q = tf.transformations.quaternion_from_euler(0, 0, self.current_state[2])
        o = feedback.base_position.pose.orientation
        o.x, o.y, o.z, o.w = q
        self.action_server.publish_feedback(feedback)

    def run(self):
        rospy.loginfo("Starting MPC loop. Waiting for a goal (~navigate action or /mpc_target)...")
        
        while not rospy.is_shutdown():
            pose = self.get_current_pose()
            if pose is not None:
                self.current_state = pose
                self.publish_footprint(pose)
            else:
                self.rate.sleep()
                continue

            self.process_goal_requests()

            v_cmd, w_cmd = 0.0, 0.0
            if self.goal is not None:
                if self.goal_is_action and self.action_server.is_active():
                    self.publish_feedback()
                v_cmd, w_cmd = self.navigate_step()
            
            twist_msg = Twist()
            twist_msg.linear.x = v_cmd
            twist_msg.angular.z = w_cmd
            self.cmd_pub.publish(twist_msg)
            
            self.rate.sleep()
            
        self.cmd_pub.publish(Twist())

if __name__ == '__main__':
    try:
        node = MPCROSNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
