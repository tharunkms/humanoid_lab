# ROS Workspace Source Directory

This directory contains the ROS packages for your navigation project.

To build the workspace, run the following command inside the Docker container:

```bash
catkin build
```

Then, source the environment:

```bash
source devel/setup.bash
```

## Navigation interface (for other nodes, e.g. the arm)

`mpc_ros_node` drives the base to a pose. Plans come from `move_base`'s global planner (via its
`make_plan` service) and the MPC follows them. Don't send goals to `move_base` directly
(`/move_base/goal`, `/move_base_simple/goal`): while `move_base` has a goal of its own it refuses
planning requests.

**Action:** `/mpc_ros_node/navigate` (`move_base_msgs/MoveBaseAction`)

- `target_pose`: any frame TF can transform to `map` (e.g. `map`, `odom`). Orientation is the final
  heading of `base_link`.
- `SUCCEEDED` means the robot is within `xy_goal_tolerance` and `yaw_goal_tolerance_deg` of the pose and
  has stood still for `settle_time` (so it's safe to take a camera picture). The result text gives the
  final error.
- `ABORTED` comes with a reason: goal inside an obstacle or its inflation, no plan, no room to rotate
  to the goal heading, no progress, or a TF failure.
- Cancelling stops the robot (`PREEMPTED`). A new goal replaces the current one.
- Feedback: `base_position`, the current robot pose in `map`.

```python
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

client = actionlib.SimpleActionClient('/mpc_ros_node/navigate', MoveBaseAction)
client.wait_for_server()
goal = MoveBaseGoal()
goal.target_pose.header.frame_id = 'map'
goal.target_pose.pose.position.x = 2.0
goal.target_pose.pose.orientation.w = 1.0
client.send_goal(goal)
client.wait_for_result()
print(client.get_state(), client.get_goal_status_text())  # 3 = SUCCEEDED, 4 = ABORTED
```

**RViz:** the "2D Nav Goal" tool publishes to `/mpc_target` (same behavior, no result reporting).

**Parameters** (private, `~name`, defaults in brackets): `xy_goal_tolerance` [0.05 m],
`yaw_goal_tolerance_deg` [5], `settle_time` [0.8 s], `final_approach_dist` [0.3 m],
`goal_snap_tolerance` [0.1 m], `replan_period` [1.0 s], `plan_patience` [5 s], `progress_timeout` [20 s].
