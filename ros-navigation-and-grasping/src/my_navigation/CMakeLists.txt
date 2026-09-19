cmake_minimum_required(VERSION 3.0.2)
project(my_navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf
  cv_bridge
  image_geometry
  message_filters
)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_navigation
#  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/mpc_ros_node.py
  scripts/base_footprint_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
