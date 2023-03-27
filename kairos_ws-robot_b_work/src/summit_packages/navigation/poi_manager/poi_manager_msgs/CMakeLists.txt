cmake_minimum_required(VERSION 3.0.2)
project(poi_manager_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs geometry_msgs robotnik_msgs message_generation)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LabeledPose.msg
  PoiState.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  AddPOI.srv
  AddPOI_params.srv
  GetPOI_params.srv
  GetPOI.srv
  DeletePOI.srv
  DeleteEnvironment.srv
  GetPOIs.srv
  ReadPOIs.srv
  AddPOIs.srv
  GetEnvironments.srv
  GetPoseTrigger.srv
)

generate_messages(
DEPENDENCIES
   std_msgs
   geometry_msgs
   robotnik_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
  robotnik_msgs
)
