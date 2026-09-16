# AGENT.md — sam2_service architecture reference

Context for an agent (or a human) picking up work in this directory: what each
piece does, how data flows between them, and the non-obvious constraints that
shaped the design. Read this before changing the wire protocol, `SAM2Session`,
or the fusion pipeline.

## Why two processes

The GUI that owns the camera and scene (ROS1 Noetic container for the robot
rig, or Isaac Sim's bundled Python for simulation) cannot host SAM2 + CLIP +
Open3D in the same interpreter — conflicting Python versions and dependency
trees. So the pipeline is split into two OS processes that only ever exchange
JSON + JPEG/PNG bytes over a ZeroMQ `REQ`/`REP` socket:

- **Process A** — owns the camera, the click/search UI, and (for multi-view
  capture) the scene geometry or robot pose. This repo's Process A is
  `realsense_segment_gui.py` (handheld Intel RealSense D435i). An Isaac-Sim
  variant of Process A exists in the simulator project and is not part of
  this repo, but speaks the exact same protocol.
- **Process B** — `sam2_service.py`. Loads SAM2/CLIP once at startup and
  answers requests. Stateful across calls (tracking session, last `detect_all`
  candidates) but never crashes on inference errors — every exception is
  caught and returned as `STATUS_ERROR` so Process A can degrade gracefully.

`ipc_common.py` is imported unmodified by both sides and has zero heavy
dependencies (`numpy`, `cv2`, `json`, `zmq` only), so it can live in either
environment without dragging in torch/SAM2/Open3D.

## Wire protocol (`ipc_common.py`)

Every message is a 2-part ZMQ multipart frame: `[json_meta_bytes, binary_payload_bytes]`.

Request types (`meta["type"]`):

| Type | Sent when | Payload in | Payload out |
|---|---|---|---|
| `click` | user clicks a point on the frame | JPEG frame | PNG mask |
| `track` | next frame, propagate existing mask | JPEG frame | PNG mask (or `no_object`) |
| `reset` | clear tracking state | — | — |
| `classify` | identify category + shape of a crop | JPEG crop | — (label/shape/confidence in meta) |
| `detect_all` | auto-segment + classify every object in frame | JPEG frame | PNG label map (id per pixel) |
| `select_mask` | start tracking one `detect_all` candidate | optional current JPEG frame | PNG mask |
| `generate_pointcloud` | fuse a captured multi-view session | — (`session_dir` in meta) | — |
| `view_pointcloud` | open a native Open3D viewer on the host display | — (`ply_path` in meta) | — |

Response status: `ok`, `no_object` (idle/tracking lost), `error` (message has
details). The binary payload kind is self-describing via `meta["payload"]`
(`"mask"` vs `"label_map"`) so an older client that never sends `detect_all`
sees no protocol change.

`generate_pointcloud` / `view_pointcloud` carry filesystem paths instead of
frame bytes, and `remap_path()` in `sam2_service.py` translates a path from
Process A's view of the filesystem (inside its container) to Process B's view
(on the host), via `--container-catkin-root` / `--host-catkin-root`. Get this
wrong and fusion will report "session_meta.json missing" even though the
session directory clearly exists — on the wrong side of a bind mount.

## `sam2_service.py` (Process B) internals

### `SAM2Session` — the tracking workaround

The installed SAM2 build only ships `build_sam2_video_predictor`, which
requires `init_state()` to point at a **directory of JPEG frames already on
disk** — there is no true frame-by-frame streaming predictor
(facebookresearch/sam2#134).

Workaround: every `track()` call writes exactly **two** frames to a small temp
directory — the previous frame (with its last known mask re-injected via
`add_new_mask`) and the new frame — then runs `init_state()` +
`propagate_in_video()` over just that pair, and keeps the new mask for next
time. Cost per call is therefore ~constant (2 frames), not growing with
session length, which is what makes this usable near real time.

Trade-off: SAM2's longer-range memory bank is never used, so tracking can
drift after occlusion or fast motion. The GUI's `r` (reset) + re-click is the
expected recovery path, not a bug workaround to "fix" — don't try to extend
the temp directory to more frames to fix drift, it defeats the point of the
two-frame bound.

### `Classifier` — open-vocabulary CLIP

Zero-shot category + primary-shape (box/cylinder/sphere/irregular) classification
of a cropped region using `open_clip` (ViT-B/32). Stateless beyond model
weights, loaded lazily so a broken `open_clip` install doesn't take down SAM2
tracking.

### `PropDetector` — `detect_all` / `select_mask`

Auto-segments every object in a frame (no click prompt) and classifies each
one, returning a single-channel `label_map` (pixel value = candidate id, 0 =
background) plus per-candidate metadata. Caches `last_masks` / `last_frame` /
`detect_id` so a later `select_mask` can seed `SAM2Session` from a cached mask
without re-running detection. `detect_id` guards against **stale candidate
ids**: if another client runs `detect_all` again before this one calls
`select_mask`, the id no longer refers to the same mask, and the server
rejects the request rather than silently tracking the wrong object.

### `PointCloudBuilder` — multi-view fusion

Fuses a capture session (`pose_NN_rgb.png` / `pose_NN_depth.png` pairs +
`session_meta.json`) into one `.pcd`. Two registration paths:

- **Ground-truth pose available** (`session_meta["poses"]`, e.g. from Isaac
  Sim's known camera transforms or the robot's kinematic chain): views are
  fused directly in the **world frame**, no ICP needed.
- **No ground-truth pose** (handheld RealSense): pairwise **FPFH feature
  matching + RANSAC global registration + point-to-plane ICP**, output in the
  **first view's camera frame** (`view0_opencv_optical`). Registering the
  object crop alone is degenerate for symmetric shapes (a can, a ball, a box
  face all look the same from many angles) — this is why
  `realsense_segment_gui.py` runs whole-frame RGB-D visual odometry as its
  primary pose source, and `refine_session_poses.py` exists as an offline
  whole-scene ICP check/correction on top of that.

The output frame and axis-aligned bounding box are written back into
`session_meta.json` (`fused_pointcloud_frame`, `fused_pointcloud_aabb`) so a
downstream grasp planner never has to guess which convention it's reading.

`.pcd`, not `.ply`, because that's what the grasping pipeline stage after this
one consumes directly.

## `realsense_segment_gui.py` (Process A, handheld rig)

Captures 1280x720 color + depth (aligned) from a physical D435i via
`pyrealsense2`. Camera pose per frame comes from Open3D's keyframe RGB-D
odometry over the **whole frame** (world = first camera frame) — the object
alone is a degenerate registration target, as above. Table plane is estimated
per-frame from depth via RANSAC (used for shadow rejection and table-plane
clipping) instead of read from a known USD scene. Depth is real sensor noise,
so no noise model is applied, but the session is tagged so `PointCloudBuilder`
still applies its noise-aware fusion filters.

## `robot_view_poser.py` (Isaac Sim only)

Inverse kinematics for the perception rig: a Unitree Go1 quadruped base
carrying an OpenManipulator-X arm, with a D435i mounted on the arm's gripper
link (`link5`). Given a requested camera position + look-at target, solves
base `(x, y, yaw)` + arm joints `joint1..joint4` via damped least squares
(Levenberg-Marquardt), keeping the robot body outside the table edge, then
writes the resulting pose to every relevant USD prim. Physics is switched off
on the puppet (`make_puppet`) so nothing falls or fights the commanded pose —
this stage is a stand-in for real locomotion/arm control, not a physics
simulation. The camera rig transform is enforced every call as
`rig = link5_world * MOUNT`, not via USD parenting, so it can't drift out of
sync if some other code touches the arm.

## `voxel_belief.py` — next-best-view support

Log-odds occupancy grid (OctoMap conventions: hit +0.85, miss -0.85, clamped
to [-2.0, 3.5]) over the tabletop volume. `integrate_depth()` ray-casts every
depth pixel of a view from the camera to its surface hit, marking traversed
voxels free and the endpoint occupied. `unknown_near()` / `frontier_near()`
are what a next-best-view planner queries to decide where to look next —
voxels the robot hasn't observed yet, near the target object. Pure numpy, no
new dependencies, so it can run in either process's environment.

## `depth_noise.py` — sim-to-real depth realism

Isaac Sim's depth is an ideal pinhole projection; a real D435i is not. This
module adds, in order: range-squared stereo noise (`sigma_z ∝ z² / (f·b)`),
grazing-angle noise growth (`∝ 1/cos(incidence)`, failing past ~70-80°),
spatially-correlated (blotchy, not per-pixel) noise, disparity quantization
banding, and invalid-pixel dropout (occlusion edges + "flying pixels",
dark/specular surfaces, the disparity-width border strip, too-close range,
random holes). `severity` scales the whole model (0 = off, 1 = nominal D435i,
2 = pessimistic) without touching individual constants — prefer changing
`severity` over hand-tuning `DEFAULT_CFG` unless you have a specific D435i
datasheet number to correct.

Output convention: float32 metres, `0.0` = invalid — matches both the real
RealSense driver and the 16-bit PNGs the pipeline saves to disk, so nothing
downstream needs a special case for simulated vs. real depth.

## `inspect_client.py` — external entry point

The integration surface for a task planner or locomotion node: send one
`inspect` command with a target object name (open-vocabulary phrase) and
optional goal pose, then poll `status` until `state` is terminal
(`done`/`failed`/`idle`) and read the `.pcd` path from the reply. This talks
to a **different** ZeroMQ port (`tcp://<host>:5556`, default in
`DEFAULT_ADDR`) exposed by the Isaac Sim GUI process itself (the orchestrator
that sequences detect → plan views → drive `robot_view_poser` → capture →
call `generate_pointcloud` on Process B) — that orchestrator is not part of
this repo. Only needs `pyzmq`, so it can be imported from the ROS1 container
too.

## Known constraints / gotchas for future changes

- **Don't grow the `SAM2Session.track()` temp-frame window past 2 frames** —
  it's the whole reason this runs near real time; if drift is unacceptable,
  the fix is better re-seeding (e.g. via `detect_all` + `select_mask`), not a
  bigger memory bank.
- **The SAM2 install is pinned to a specific commit** (see
  `requirements_sam2.txt`) because the video-predictor API has changed across
  releases. If you bump it, re-check `SAM2Session` against upstream's current
  `video_predictor_example` notebook.
- **`remap_path()` must stay in sync with the actual bind-mount layout**
  between Process A's container and Process B's host — it's the only thing
  standing between a correct session path and a confusing "file not found."
- **Registration frame conventions differ by pose source** —
  `view0_opencv_optical` (ICP path) vs `world` (ground-truth path). Consumers
  of `fused_pointcloud.pcd` must check `session_meta["fused_pointcloud_frame"]`
  rather than assume one.
- **`venv/` and `checkpoints/` are gitignored** — regenerate the venv from
  `requirements_sam2.txt` and re-download checkpoints per the README rather
  than expecting them in version control.
