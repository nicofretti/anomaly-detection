cmake_minimum_required(VERSION 2.8.3)
project(rosbag_manager)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  robotnik_msgs
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS src/rosbag_manager/rosbag_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install( DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

