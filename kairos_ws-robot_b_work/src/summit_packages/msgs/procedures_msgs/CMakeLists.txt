cmake_minimum_required(VERSION 2.8.3)
project(procedures_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
   FILES
   ProcedureHeader.msg
   ProcedureState.msg
   ProcedureResult.msg
)

add_service_files(
  FILES
  ProcedureQuery.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
# )

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES procedures_msgs
 CATKIN_DEPENDS message_runtime std_msgs 
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


