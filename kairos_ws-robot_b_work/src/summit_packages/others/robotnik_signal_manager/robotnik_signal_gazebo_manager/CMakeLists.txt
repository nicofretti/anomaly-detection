cmake_minimum_required(VERSION 2.8.3)
project(robotnik_signal_gazebo_manager)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  rcomponent
  gazebo_msgs
  robotnik_signal_msgs
  robotnik_signal_interface
)

catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES accurate_docking
  CATKIN_DEPENDS rospy robotnik_signal_msgs robotnik_signal_interface
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
