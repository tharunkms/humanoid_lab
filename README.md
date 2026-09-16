# humanoid_lab

Isaac Sim automation for a quadruped (Unitree Go1 + OpenManipulator-X arm)
tabletop-inspection pipeline: scene setup, camera-view generation, ROS1
integration, and a native SAM2-driven segmentation/next-best-view GUI.

The repo has two top-level directories:

- [`isaac_sim_scripts/`](isaac_sim_scripts/) — Isaac Sim side (Process A):
  scene setup, camera-view generation, ROS1 integration, kinematics, and
  the native GUI. Documented below.
- [`sam2_service/`](sam2_service/) — the SAM2/CLIP/Open3D perception
  service (Process B) that the GUI talks to over ZeroMQ, plus a
  handheld-RealSense variant of Process A. See its own
  [README](sam2_service/README.md) and [AGENT.md](sam2_service/AGENT.md).

## Big picture

The project simulates a Go1 quadruped with an arm-mounted RGB-D camera
(D435i) inspecting props on a tabletop. Two generations of the pipeline
coexist in this folder:

1. **Free-floating camera rig** (older, simpler) — a camera prim is
   teleported directly around the scene to generate views. Used by the
   `orbit_camera_*.py` family and `auto_orbit_capture.py`.
2. **Robot-mounted camera via IK** (current target architecture) — the
   camera is physically mounted on the arm, and `robot_view_poser.py`
   solves inverse kinematics for the Go1 base + arm joints so the camera
   reaches a requested pose. Wired up by `setup_go1_arm_scene.py`.

The main application, `isaac_sim_native_gui.py`, is a native Isaac Sim
(`omni.ui`) window that drives segmentation, tracking, classification,
point-cloud fusion, and next-best-view planning by talking to an external
SAM2 microservice over ZeroMQ. It can drive either camera-driving
mechanism above.

Most scripts here are **Script Editor snippets**, not an importable
package — see [AGENT.md](AGENT.md) for the one exception
(`robot_view_poser.py`) and other conventions to know before editing.

## Invocation modes

| Mode | Scripts | How it's run |
|---|---|---|
| Paste into Isaac Sim Script Editor, run against a live stage | `create_table_with_props.py`, `setup_go1_arm_scene.py`, `orbit_camera_*.py`, `auto_orbit_capture.py`, `build_ros1_action_graph.py`, `isaac_sim_native_gui.py` | Window → Script Editor, run top-to-bottom, then call functions from the REPL |
| Standalone Isaac Sim app | `test_script.py` | `python.sh test_script.py` |
| Standalone ROS1 node | `depth_to_pointcloud_node.py` | `python3 depth_to_pointcloud_node.py` inside a ROS1 Noetic container |
| Importable module | `robot_view_poser.py` | `import robot_view_poser as rvp` from another Script Editor snippet |

## Files

### Scene setup

- **`create_table_with_props.py`** — Builds a round pedestal table plus
  five props (cup, mug, bottle, cuboid, cylinder) from pure USD
  primitives under `/World/PropTable`. No external assets; idempotent
  (clears and rebuilds the prim tree on re-run). Config constants at the
  top (`TABLE_ROOT`, `TABLE_CENTER`/`RADIUS`/`HEIGHT`, `ENABLE_PHYSICS`).
  Its output path (`/World/PropTable/Tabletop`) is a hardcoded dependency
  of `robot_view_poser.py` and `isaac_sim_native_gui.py`.

- **`setup_go1_arm_scene.py`** — Run once, after the table script. Adds
  the Go1 (from the Isaac asset library or a local USD/URDF) and
  OpenManipulator-X (URDF import), switches both to kinematic "puppet"
  mode (disables articulation/rigid-body/collision APIs), and re-poses
  the robot so the mounted D435i camera matches wherever the free
  camera rig previously was, by calling
  `robot_view_poser.RobotViewPoser.set_view()`. Requires editing the
  hardcoded, machine-specific asset paths (`GO1_USD`/`GO1_URDF`,
  `ARM_URDF`) and `SCRIPTS_DIR` before running.

### Camera / view capture

Four `orbit_camera_*.py` scripts share one pattern: paste into Script
Editor, run once to define `goto_view(i)`, then call it once per view
from the REPL, manually re-clicking the target and pressing "p" in the
(external, older) segmentation GUI between calls. They represent an
increasingly-correct sequence of the same idea — each fixes a bug found
in the previous one (see inline comments):

1. **`orbit_camera_waypoints.py`** — flat horizontal ring, yaw-only
   look-at (imprecise; oldest/simplest).
2. **`orbit_camera_multiring.py`** — full 360° azimuth sweep at multiple
   elevations, proper yaw+pitch look-at, auto-detects stage up-axis.
3. **`orbit_camera_hemisphere.py`** — front-hemisphere-only sweep
   (azimuths × elevations) for grasp-pipeline-style partial coverage;
   rotates RGB/Depth child prims independently.
4. **`orbit_camera_rig.py`** — same hemisphere sweep, but teleports the
   whole camera rig as one rigid unit (with a local-rotation correction)
   so RGB/Depth/IR/IMU stay co-registered. This rig-teleport math is the
   pattern reused by `auto_orbit_capture.py`.

- **`auto_orbit_capture.py`** — the automated successor to the four
  scripts above. Must be run *after* `isaac_sim_native_gui.py`, since it
  reaches into the running GUI's global `_gui` instance for the
  currently-selected object. Reuses `orbit_camera_rig.py`'s rig-teleport
  math (duplicated, not imported — see AGENT.md), and adds: live 3D
  target tracking, render-settling waits, automatic re-projection of the
  tracked target into each new view to get a click pixel, auto-click,
  and pose capture — removing the manual click/keypress step.

  In the robot-mounted architecture, direct rig teleportation is
  superseded by `robot_view_poser.RobotViewPoser.set_view()`; the orbit
  scripts remain as simpler, standalone alternatives for the
  free-floating-camera setup.

### ROS1 integration

- **`build_ros1_action_graph.py`** — Builds an OmniGraph Action Graph at
  `/World/ROS_ActionGraph` publishing `/clock`, `/tf` + `/tf_static`,
  `/joint_states` (Go1), `/arm/joint_states` (arm), `/camera/rgb/image_raw`
  + `/camera/camera_info`, and `/camera/depth/image_raw`. Uses the
  `isaacsim.*` node namespace (Isaac Sim 4.5). This is a parallel,
  alternate publishing path — `isaac_sim_native_gui.py` bypasses ROS
  entirely and reads camera intrinsics/pose straight from USD.

- **`depth_to_pointcloud_node.py`** — Standalone ROS1 node (`rospy`, not
  an Isaac Sim script). Subscribes to depth + camera_info (+ optional
  RGB) via `message_filters.ApproximateTimeSynchronizer`, deprojects the
  frame into a colored point cloud, and writes a `.ply` via Open3D. CLI
  args: `--depth-topic`, `--info-topic`, `--rgb-topic`, `--output`,
  `--continuous`, `--min-depth`, `--max-depth`. Consumes the topics
  published by `build_ros1_action_graph.py`.

### GUI (main application)

- **`isaac_sim_native_gui.py`** (~1500 lines) — the de facto main app.
  A native `omni.ui` window (`NativeSegmentGUI`) that:
  - reads camera intrinsics/pose directly from the USD Camera prim, and
    captures RGB/depth via `omni.replicator.core` annotators;
  - talks to the **SAM2 segmentation microservice** in
    [`sam2_service/`](sam2_service/) over ZeroMQ, using its shared
    wire-protocol module (`sam2_service/ipc_common.py`) for
    click-to-segment, tracking, classification, and open-vocabulary
    detect-all;
  - runs its own ZeroMQ REP server so external processes (a task
    planner, a locomotion node, test scripts) can send commands like
    `{"cmd": "inspect", "target": ..., "goal": ...}` and get the whole
    capture pipeline run unattended;
  - applies simulated D435i depth degradation
    (`sam2_service/depth_noise.py`) before any consumer sees depth;
  - maintains a tabletop voxel occupancy belief
    (`sam2_service/voxel_belief.py`) fused across every captured view;
  - plans next-best-view poses by scoring candidate poses against
    unseen-voxel coverage, with a fallback to a fixed 4-pose plan
    (approach / side / top-down);
  - manages capture sessions under `SESSION_ROOT`, validating each view
    (valid-depth fraction, rigid camera transform) before saving, and
    can request fused point-cloud generation/viewing from the SAM2
    service.

  Instantiates a global `_gui = NativeSegmentGUI()` at import time —
  this is the object `auto_orbit_capture.py` expects to find in the same
  Script Editor session. Supersedes an older, external `cv2`-window GUI
  (`isaac_sim_segment_gui.py`) that is not part of this repo.

### Kinematics library

- **`robot_view_poser.py`** — not a GUI; a physics/kinematics module
  (`class RobotViewPoser`, plus internal `_Tree`/`_Joint` kinematic-tree
  helpers). Reads the Go1 and OpenManipulator-X USD physics-joint trees,
  builds forward-kinematics chains, and solves inverse kinematics
  (Levenberg–Marquardt) for the Go1 base (x, y, yaw) + arm joints so the
  arm-mounted camera reaches a requested world position/look-direction,
  subject to a soft "stay outside the table edge" constraint.
  `make_puppet()` disables physics so poses are purely kinematic.
  Public API: `RobotViewPoser(stage).set_view(cam_pos, look_at)`,
  `set_base()`, `camera_world()`. Deliberately written as a real,
  importable module (only a lazy `import omni.usd` inside `__init__`) so
  it can be imported and reloaded from other scripts, unlike everything
  else in this repo.

### Utilities

- **`test_script.py`** — minimal standalone Isaac Sim smoke test, run
  via `python.sh`: starts `SimulationApp`, loads a Go1 USD, builds a
  one-node ROS1 joint-state action graph, and steps the world. Uses the
  older `omni.isaac.*` extension namespace (inconsistent with the rest
  of the repo's `isaacsim.*`) and a hardcoded asset path — an early
  prototype, not integrated with anything else here.

## Known consistency risks

See [AGENT.md](AGENT.md) for the full list — in short: several
machine-specific absolute paths (ZMQ address, script/asset directories),
one namespace inconsistency (`test_script.py`), duplicated
constants/math between `isaac_sim_native_gui.py` and
`robot_view_poser.py` / `orbit_camera_rig.py`, and a fully duplicated
copy of `robot_view_poser.py` under `sam2_service/` — all of which must
be kept in sync by hand.
