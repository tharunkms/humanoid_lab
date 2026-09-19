#!/bin/bash
set -e

# Source the main ROS Noetic installation
source "/opt/ros/noetic/setup.bash"

# Source the local workspace if it exists (i.e., if 'catkin build' has been run)
if [ -f "/home/ros_user/ws/devel/setup.bash" ]; then
    source "/home/ros_user/ws/devel/setup.bash"
fi

# Execute the command passed to the container
exec "$@"
