cmake_minimum_required(VERSION 2.8.3)
project(robotnik_navigation_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  std_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  PoseStampedArray.msg 
  RobotnikMoveBaseFlexGoalAction.msg
  RobotnikMoveBaseFlexFeedbackAction.msg
  RobotnikMoveBaseFlexResultAction.msg
)

## Generate services in the 'srv' folder
#add_service_files(
#  FILES
#  PickAndPlace.srv
#)

## Generate actions in the 'action' folder
add_action_files(
  FILES
  Dock.action
  Move.action
  BarcodeDock.action
  RobotnikMoveBaseFlex.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robotnik_navigation_msgs
  CATKIN_DEPENDS actionlib actionlib_msgs geometry_msgs std_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

