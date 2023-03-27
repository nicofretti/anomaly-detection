cmake_minimum_required(VERSION 2.8.3)
project(robotnik_signal_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    robotnik_msgs
    message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
    FILES
    SignalStatus.msg
    SignalStatusArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
    FILES
    SetSignal.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
    robotnik_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_simple_command_manager_msgs
   CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)