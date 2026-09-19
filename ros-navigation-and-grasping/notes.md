# Start container in background
docker compose up -d

# Open a shell inside the container
docker compose exec ros_dev bash


cd /home/ros_user/ws

# Build the workspace (if you made changes to C++/Python packages)
catkin build

# Source workspace setup
source devel/setup.bash

roslaunch my_navigation slam.launch

docker compose exec ros_dev bash
rosrun my_navigation mpc_ros_node.py

# For visualizing tf tree
rosrun tf view_frames