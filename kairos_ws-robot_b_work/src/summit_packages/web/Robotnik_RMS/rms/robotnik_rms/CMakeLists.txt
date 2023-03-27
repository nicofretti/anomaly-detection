cmake_minimum_required(VERSION 2.8.3)
project(robotnik_rms)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  robotnik_msgs
  actionlib_msgs
  actionlib
)


catkin_python_setup()


catkin_package()


# TODO
#catkin_install_python(PROGRAMS 
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

