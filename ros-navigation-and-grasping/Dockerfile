FROM ros:noetic-ros-base

# Install essential build tools and ROS tools
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    python3-vcstool \
    ros-noetic-navigation \
    ros-noetic-slam-gmapping \
    ros-noetic-tf2-ros \
    ros-noetic-robot-state-publisher \
    ros-noetic-xacro \
    ros-noetic-joint-state-publisher \
    ros-noetic-rviz \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-pointcloud-to-laserscan \
    ros-noetic-cv-bridge \
    ros-noetic-image-geometry \
    ros-noetic-move-base \
    ros-noetic-global-planner \
    python3-opencv \
    python3-sklearn \
    && rm -rf /var/lib/apt/lists/*

# Install CasADi via pip (useful for Python, and often enough for prototyping)
RUN pip3 install --no-cache-dir casadi

# Setup non-root user (to match host permissions)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ros_user && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ros_user && \
    usermod -aG sudo ros_user && \
    echo "ros_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup workspace directory
USER ros_user
ENV HOME=/home/ros_user
WORKDIR /home/ros_user/ws

# Automatically source ROS setup in bashrc for exec sessions
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/ros_user/.bashrc && \
    echo "if [ -f /home/ros_user/ws/devel/setup.bash ]; then source /home/ros_user/ws/devel/setup.bash; fi" >> /home/ros_user/.bashrc

# Copy entrypoint script
COPY ./entrypoint.sh /
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
