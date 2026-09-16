# AGENT.md

Guidance for AI coding agents (and humans) working in this repo. Read
[README.md](README.md) first for the architecture overview. This file
covers the `isaac_sim_scripts/` (Isaac Sim / Process A) side only —
`sam2_service/` has its own [AGENT.md](sam2_service/AGENT.md).

## What this repo is

Isaac Sim scripts for a Go1-quadruped + arm tabletop-inspection pipeline,
paired with a separate SAM2 perception service. `isaac_sim_scripts/` is
**not** a Python package. Almost every file in it is a Script Editor
snippet meant to be pasted into a running Isaac Sim session and executed
against whatever stage is already open — there is no single entry point,
`main()`, or test suite to run. (`sam2_service/` is a real, importable
Python project with its own venv — see its own docs.)

## Before changing anything

- Figure out which of the two camera-driving mechanisms a script belongs
  to (free-floating rig teleport vs. robot-mounted IK via
  `robot_view_poser.py`) — see README. Don't mix the two approaches
  inside one script without saying so.
- Check whether the file is a Script Editor snippet or a real module.
  Only `robot_view_poser.py` is written to be `import`ed; everything
  else assumes it's being run top-to-bottom in the Script Editor REPL,
  so it's fine (and expected) for those files to use module-level
  globals, `sys.path.insert` hacks, and code that runs on import.
- The `orbit_camera_*.py` files are intentionally kept as separate,
  increasingly-correct variants (each one's docstring/comments explain
  the bug it fixes relative to the previous file) — this is a deliberate
  debugging trail, not accidental duplication. Don't "clean up" by
  deleting the earlier ones unless asked; if you fix a bug in one,
  check whether the same bug exists in `orbit_camera_rig.py`'s
  rig-teleport math as duplicated into `auto_orbit_capture.py`.

## Known duplication that must be kept in sync by hand

- Rig-teleport + local-rotation-correction math: duplicated across
  `orbit_camera_rig.py` and `auto_orbit_capture.py` (the latter can't
  import the former since both are pasted as temp scripts, not modules).
- Body-clearance / tabletop geometry constants
  (`BODY_HALF_LENGTH_M` and friends): duplicated between
  `robot_view_poser.py` and `isaac_sim_native_gui.py` — `robot_view_poser.py`
  has a comment noting they must be kept in sync with the GUI's planner
  constants. If you change one, change the other.
- `TABLE_PATH` / `TABLETOP_PATH` (`/World/PropTable/Tabletop`): hardcoded
  identically in `robot_view_poser.py` and `isaac_sim_native_gui.py`,
  and produced by `create_table_with_props.py`. Renaming the prim in one
  place breaks the others silently (no import-time check).
- `robot_view_poser.py` is **fully duplicated** at
  `sam2_service/robot_view_poser.py` (needed there because Process B
  runs in its own venv and can't import across the ZeroMQ process
  boundary). A fix to the IK solver, the table-clearance constraint, or
  the body-clearance constants must be applied to both copies.

## Machine-specific paths to expect (and not to "fix" generically)

These are real per-machine config, not bugs — but flag them if you're
asked to make the repo portable:

- `isaac_sim_native_gui.py`: `ZMQ_ADDR` (hardcoded IP), `IPC_COMMON_DIR`,
  `SESSION_ROOT`.
- `setup_go1_arm_scene.py`: `GO1_USD`/`GO1_URDF`, `ARM_URDF`,
  `SCRIPTS_DIR` (path to `robot_view_poser.py`).
- `test_script.py`: `GO1_USD_PATH`.

## Cross-directory dependency on sam2_service/

`isaac_sim_native_gui.py` imports `sam2_service.py` (the SAM2
microservice, run out-of-process in its own venv), `ipc_common.py`
(shared ZeroMQ wire protocol), `depth_noise.py`, and `voxel_belief.py`
via a `sys.path` hack pointing at `IPC_COMMON_DIR` — these now live in
[`sam2_service/`](sam2_service/) in this same repo, but are still a
**separate OS process** with its own Python environment, not an
in-process import. If you're asked to touch the GUI's segmentation,
tracking, depth-noise, or voxel-fusion behavior, make the change in
`sam2_service/` and treat `IPC_COMMON_DIR` as pointing at a checkout of
that directory — check with the user before assuming the two directories
should be merged or made importable across the process boundary.

## Namespace inconsistency

`test_script.py` uses the deprecated `omni.isaac.*` extension namespace
(e.g. `omni.isaac.ros1_bridge`). Every other script uses the current
`isaacsim.*` namespace (Isaac Sim 4.5+). Don't copy patterns from
`test_script.py` into new code; treat it as an old prototype.

## Environment assumptions

- Isaac Sim 4.5+, `isaacsim.*` extensions, ROS1 Noetic bridge (not
  ROS2).
- Script Editor snippets run inside Isaac Sim's own Python environment
  (has `omni.*`, `pxr`, numpy). `depth_to_pointcloud_node.py` runs in a
  *separate* ROS1 Noetic container Python environment (`rospy`,
  `cv_bridge`, `message_filters`, `sensor_msgs`, `open3d`) — don't assume
  it shares the Isaac Sim environment's dependencies.
- `isaac_sim_native_gui.py` additionally needs `zmq` and
  `omni.replicator.core` in the Isaac Sim Python environment.

## When adding a new script

- State up front, in a comment, whether it's a Script Editor snippet or
  a standalone `python.sh`/`python3` entry point, and what it assumes
  already exists in the stage (mirrors the style already used at the
  top of `auto_orbit_capture.py` and `setup_go1_arm_scene.py`).
- If it depends on prim paths from `create_table_with_props.py` or
  `setup_go1_arm_scene.py`, hardcode them as named constants at the top
  of the file (existing convention), not inline magic strings.
- Update [README.md](README.md)'s file table/sections and this file's
  "known duplication" / "machine-specific paths" lists if you introduce
  new hand-synced constants or new external/absolute-path dependencies.
