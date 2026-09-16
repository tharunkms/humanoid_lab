# sam2_service

Object-perception pipeline for a quadruped + arm inspection robot: click-to-segment
and track any object with SAM2, classify it open-vocabulary with CLIP, sweep
multiple camera views (Isaac Sim or a handheld RealSense), and fuse those views
into a single denoised point cloud for downstream grasp planning.

The pipeline exists as **two OS processes talking over ZeroMQ**, because the GUI
that owns the camera (ROS1 Noetic container, or Isaac Sim's bundled Python) and
the SAM2/CLIP/Open3D inference stack cannot share one Python environment.

```
Process A (GUI, owns the camera/scene)         Process B (this repo's core)
┌─────────────────────────────┐   ZeroMQ        ┌───────────────────────────┐
│ realsense_segment_gui.py     │  REQ/REP        │ sam2_service.py            │
│  - captures RGB(+D) frames   │◄───tcp://──────►│  - SAM2Session (track)      │
│  - clicks / search UI        │   :5555         │  - Classifier (CLIP)        │
│  - captures multi-view poses │                 │  - PropDetector (detect_all)│
│  - saves session_meta.json   │                 │  - PointCloudBuilder (fuse) │
└─────────────────────────────┘                 └───────────────────────────┘
```

Process A is whatever owns the camera for a given rig — this repo ships the
handheld Intel RealSense D435i variant (`realsense_segment_gui.py`). The Isaac
Sim simulator variant of Process A lives in the simulator project, not here;
`ipc_common.py` and `sam2_service.py` are written to be shared unmodified by
either.

## Repository layout

| File | Role |
|---|---|
| `ipc_common.py` | Wire protocol shared by both processes — request/response types, JPEG/PNG (de)coding. Zero heavy deps so both envs can import it. |
| `sam2_service.py` | **Process B.** ZeroMQ REP server: SAM2 tracking, CLIP classification, auto-detection, multi-view point cloud fusion. |
| `realsense_segment_gui.py` | **Process A** (handheld rig). OpenCV window: capture, click/search UI, RGB-D visual odometry, multi-view capture session. |
| `robot_view_poser.py` | Isaac Sim only. Inverse kinematics that drives a Go1 quadruped base + OpenManipulator-X arm so the mounted D435i reaches a requested camera pose. |
| `voxel_belief.py` | Log-odds occupancy grid (OctoMap-style) over the tabletop, built from each view's depth — tells a next-best-view planner which voxels near the target are still unknown. |
| `depth_noise.py` | Adds realistic D435i stereo-depth noise/dropout to Isaac Sim's ideal depth, so simulated sessions fuse the same way real ones do. |
| `refine_session_poses.py` | Offline tool: re-registers a handheld session's whole-scene point clouds with ICP to check/correct visual-odometry drift before re-fusing. |
| `inspect_client.py` | External client library + CLI for a task planner / locomotion node to drive the pipeline (`inspect` / `status` / `abort` over a separate ZeroMQ port, `5556`, exposed by the Isaac Sim GUI process — not part of this repo). |
| `test_detect_all.py` | Exercises `detect_all` + `select_mask` against a running `sam2_service.py` using a saved image. |
| `test_voxel_belief.py` | Unit-style checks for `voxel_belief.py`'s raycasting and log-odds updates. |
| `requirements_sam2.txt` | Dependencies for Process B's dedicated virtualenv. |

See [AGENT.md](AGENT.md) for the full architecture and data-flow reference.

## Setup

Process B needs its **own** Python 3.10+ environment, separate from whatever
Process A runs in (ROS1 Noetic container, or Isaac Sim's bundled Python):

```bash
python3.10 -m venv venv && source venv/bin/activate

# pick the torch build matching your host's CUDA version
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM2 itself, pinned to the commit sam2_service.py was written against
pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f"

pip install -r requirements_sam2.txt
```

Download a checkpoint from the [sam2 repo](https://github.com/facebookresearch/sam2)
into `checkpoints/` (pick by GPU VRAM): `tiny`, `small`, `base_plus` (default), `large`.

## Running

```bash
# Process B (sam2 venv)
python3 sam2_service.py --checkpoint base_plus --bind tcp://*:5555

# Process A, handheld RealSense rig (system python3 + pyrealsense2)
python3 realsense_segment_gui.py --zmq tcp://localhost:5555
```

GUI controls: left click = positive point, right click = negative point,
`d` detect-all, `1`-`9` select a detected candidate, `s` search box,
`c` classify, `p` capture pose, `g` fuse point cloud, `v` view cloud,
`n` new session, `t` toggle RGB/depth, `r` reset, `q` quit.

## Testing

```bash
python3 test_detect_all.py path/to/frame.png --target "bottle"   # needs sam2_service.py running
python3 test_voxel_belief.py                                     # standalone
```
