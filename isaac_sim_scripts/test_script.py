# 1. INITIALIZE ISAAC SIM FIRST (This must happen before other imports)
from isaacsim import SimulationApp

# Run in non-headless mode so we can see the UI
simulation_app = SimulationApp({"headless": False})

# 2. IMPORT EXTENSIONS AND CORE MODULES
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.graph.core as og

# Enable the ROS 1 bridge extension programmatically (Isaac Sim 4.5 naming)
enable_extension("isaacsim.ros1.bridge")

# 3. SETUP THE WORLD AND ROBOT
# Initialize the world with standard physics
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Define paths (UPDATE THIS TO YOUR ACTUAL GO1 USD PATH)
GO1_USD_PATH = r"/home/user/kamarajmagadapallt1/Documents/lab-project-2026/isaac-sim-go1-urdf.usd"
ROBOT_PRIM_PATH = r"/World/Go1"

# Load the Go1 model into the stage
print("Loading Go1 model...")
add_reference_to_stage(usd_path=GO1_USD_PATH, prim_path=ROBOT_PRIM_PATH)

# 4. BUILD THE ACTION GRAPH (OMNIGRAPH) VIA PYTHON
print("Building ROS 1 Action Graph...")
keys = og.Controller.Keys

# This block creates the exact same graph you built manually in the UI
graph_handle, list_of_nodes, _, _ = og.Controller.edit(
    {"graph_path": "/ROS_ActionGraph", "evaluator_name": "execution"},
    {
        # Add the nodes we need
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("PublishJointState", "omni.isaac.ros1_bridge.ROS1PublishJointState"),
        ],
        # Connect the tick output to the publisher execution input
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ],
    },
)

# Assign the Go1 robot as the target for the Joint State Publisher
# (This simulates clicking "Add Targets" in the UI)
og.Controller.set_targets(
    og.Controller.attribute("/ROS_ActionGraph/PublishJointState.inputs:targetPrim"),
    [ROBOT_PRIM_PATH],
)

# 5. RUN THE SIMULATION LOOP
print("Starting simulation loop! Press Stop in the UI or Ctrl+C in terminal to exit.")
world.reset()

# Keep the simulation running and rendering
while simulation_app.is_running():
    world.step(render=True)

# Clean up when closed
simulation_app.close()
