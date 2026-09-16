"""
Run in Isaac Sim 4.5 Script Editor (Window > Script Editor) with the stage
already loaded and the ROS1 bridge extension enabled.

Builds one Action Graph publishing everything the perception + locomotion
pipeline needs over ROS1:
  /clock
  /tf, /tf_static                          (Go1 + arm + D435i mount, all descendants of TF_TREE_ROOT)
  /joint_states                            (Go1)
  /arm/joint_states                        (OpenManipulator-X - the second publisher from the roadmap TODO)
  /camera/rgb/image_raw + /camera/camera_info
  /camera/depth/image_raw                  (depth_linear -> metres, required for Open3D)

Uses isaacsim.* node namespaces (Isaac Sim 4.5's pip-based package rename,
not the older omni.isaac.* names) since that's how this project's Isaac Sim
was installed. If a node type isn't found, open Window > Visual Scripting >
Action Graph, search the node by its display name, and read the exact type
string off its Property panel -- attribute names can drift slightly between
point releases.
"""

import omni.usd
import omni.graph.core as og
from pxr import UsdPhysics

# ============================== CONFIG ======================================
GRAPH_PATH = "/World/ROS_ActionGraph"

# Articulation roots -- per lab learning, Go1's is root_joint, NOT base.
GO1_ARTICULATION_ROOT = "/World/go1/root_joint"
ARM_ARTICULATION_ROOT = "/World/open_manipulator_x/root_joint"

# Root prim whose full subtree gets published to /tf. Point this at whatever
# common ancestor contains go1 + arm + D435i mount in your stage.
TF_TREE_ROOT = "/World/go1"

# Update these once the D435i asset is actually parented into the robot.
CAMERA_RGB_PRIM = "/World/go1/d435i_mount/D435i/Camera_RGB"
CAMERA_DEPTH_PRIM = "/World/go1/d435i_mount/D435i/Camera_Depth"
CAMERA_FRAME_ID = "camera_link"

RGB_RESOLUTION = (1920, 1080)
DEPTH_RESOLUTION = (1280, 720)

# ROS1CameraHelper frameSkipCount: 0 = publish every render tick. Raise this
# (e.g. 1 or 2) if the camera publishers start bottlenecking sim FPS.
CAMERA_FRAME_SKIP = 0
# =============================================================================


def warn_if_rigidbody(prim_path: str) -> None:
    """Camera prims nested inside a robot link must NOT carry RigidBodyAPI --
    breaks PhysX transform resolution (known issue from earlier integration work)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[ros1_action_graph] WARNING: prim does not exist yet: {prim_path}")
        return
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"[ros1_action_graph] WARNING: {prim_path} has RigidBodyAPI applied. "
              f"Remove it -- camera prims nested in a robot link should not be rigid bodies.")


for p in (CAMERA_RGB_PRIM, CAMERA_DEPTH_PRIM):
    warn_if_rigidbody(p)


keys = og.Controller.Keys

(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

            ("PublishClock", "isaacsim.ros1.bridge.ROS1PublishClock"),
            ("PublishTF", "isaacsim.ros1.bridge.ROS1PublishTransformTree"),
            ("PublishJointsGo1", "isaacsim.ros1.bridge.ROS1PublishJointState"),
            ("PublishJointsArm", "isaacsim.ros1.bridge.ROS1PublishJointState"),

            ("CreateRenderProductRGB", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CreateRenderProductDepth", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

            ("CameraHelperRGB", "isaacsim.ros1.bridge.ROS1CameraHelper"),
            ("CameraHelperInfoRGB", "isaacsim.ros1.bridge.ROS1CameraHelper"),
            ("CameraHelperDepth", "isaacsim.ros1.bridge.ROS1CameraHelper"),
        ],

        keys.SET_VALUES: [
            ("PublishTF.inputs:targetPrims", [TF_TREE_ROOT]),

            ("PublishJointsGo1.inputs:targetPrim", GO1_ARTICULATION_ROOT),
            ("PublishJointsGo1.inputs:topicName", "/joint_states"),

            ("PublishJointsArm.inputs:targetPrim", ARM_ARTICULATION_ROOT),
            ("PublishJointsArm.inputs:topicName", "/arm/joint_states"),

            ("CreateRenderProductRGB.inputs:cameraPrim", CAMERA_RGB_PRIM),
            ("CreateRenderProductRGB.inputs:width", RGB_RESOLUTION[0]),
            ("CreateRenderProductRGB.inputs:height", RGB_RESOLUTION[1]),

            ("CreateRenderProductDepth.inputs:cameraPrim", CAMERA_DEPTH_PRIM),
            ("CreateRenderProductDepth.inputs:width", DEPTH_RESOLUTION[0]),
            ("CreateRenderProductDepth.inputs:height", DEPTH_RESOLUTION[1]),

            ("CameraHelperRGB.inputs:type", "rgb"),
            ("CameraHelperRGB.inputs:topicName", "/camera/rgb/image_raw"),
            ("CameraHelperRGB.inputs:frameId", CAMERA_FRAME_ID),
            ("CameraHelperRGB.inputs:frameSkipCount", CAMERA_FRAME_SKIP),

            ("CameraHelperInfoRGB.inputs:type", "camera_info"),
            ("CameraHelperInfoRGB.inputs:topicName", "/camera/camera_info"),
            ("CameraHelperInfoRGB.inputs:frameId", CAMERA_FRAME_ID),
            ("CameraHelperInfoRGB.inputs:frameSkipCount", CAMERA_FRAME_SKIP),

            # depth_linear -> metres, required for the Open3D pipeline (known learning)
            ("CameraHelperDepth.inputs:type", "depth"),
            ("CameraHelperDepth.inputs:topicName", "/camera/depth/image_raw"),
            ("CameraHelperDepth.inputs:frameId", CAMERA_FRAME_ID),
            ("CameraHelperDepth.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
        ],

        keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishJointsGo1.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishJointsArm.inputs:execIn"),
            ("OnTick.outputs:tick", "CreateRenderProductRGB.inputs:execIn"),
            ("OnTick.outputs:tick", "CreateRenderProductDepth.inputs:execIn"),

            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointsGo1.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointsArm.inputs:timeStamp"),

            ("CreateRenderProductRGB.outputs:execOut", "CameraHelperRGB.inputs:execIn"),
            ("CreateRenderProductRGB.outputs:renderProductPath", "CameraHelperRGB.inputs:renderProductPath"),
            ("CreateRenderProductRGB.outputs:execOut", "CameraHelperInfoRGB.inputs:execIn"),
            ("CreateRenderProductRGB.outputs:renderProductPath", "CameraHelperInfoRGB.inputs:renderProductPath"),

            ("CreateRenderProductDepth.outputs:execOut", "CameraHelperDepth.inputs:execIn"),
            ("CreateRenderProductDepth.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
        ],
    },
)

# Force one evaluation now so render products / camera helpers finish wiring
# before you hit Play.
og.Controller.evaluate_sync(graph)

print(f"[ros1_action_graph] Built graph at {GRAPH_PATH}")
print("[ros1_action_graph] Topics: /clock /tf /tf_static /joint_states /arm/joint_states "
      "/camera/rgb/image_raw /camera/camera_info /camera/depth/image_raw")
