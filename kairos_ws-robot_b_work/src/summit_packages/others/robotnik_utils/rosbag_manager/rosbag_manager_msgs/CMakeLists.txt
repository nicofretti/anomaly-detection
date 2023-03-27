cmake_minimum_required(VERSION 2.8.3)
project(rosbag_manager_msgs)


find_package(catkin REQUIRED std_msgs robotnik_msgs message_generation)


## Generate messages in the 'msg' folder
add_message_files(
    FILES
    RosbagManagerStatus.msg
)

## Generate services in the 'srv' folder
add_service_files(
   FILES
   Record.srv

)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
   robotnik_msgs  # Or other packages containing msgs
)


catkin_package(CATKIN_DEPENDS
  std_msgs
  robotnik_msgs
)
