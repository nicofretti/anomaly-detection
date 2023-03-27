cmake_minimum_required(VERSION 2.8.3)
project(robot_local_control_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  procedures_msgs
  geometry_msgs
  std_msgs
  robotnik_msgs
  robotnik_signal_msgs
)

add_message_files(
   FILES
   EnterGlovebox.msg
   LeaveShower.msg
   EnterShower.msg
   EnterLift.msg
   LeaveLift.msg
   Uncharge.msg
   Place.msg
   Charge.msg
   Move.msg
   Pick.msg
   Dock.msg
   GoTo.msg
   GoToTag.msg
   GoToNode.msg
   StatusArray.msg
   Status.msg
   Pose2DStamped.msg
   Twist2D.msg
   LocalizationStatus.msg
   MissionStatus.msg
   NavigationStatus.msg
   RobotStatus.msg
   SignalManager.msg
   SensorStatus.msg
   ControllerStatus.msg
#   SetControlState.msg
   Missions.msg
# Mission command & params
  MissionCommand.msg
  MissionParamInt.msg
  MissionParamFloat.msg
  MissionParamBool.msg
  MissionParamString.msg
  MagneticGoTo.msg
  MagneticPick.msg
  MagneticPlace.msg
  MagneticNavigation.msg
  BarcodePick.msg
  LeaveMagneticGuide.msg
  FindMagneticGuide.msg
  PrePick.msg
  PostPick.msg
  PrePlace.msg
  PostPlace.msg
  SwitchMap.msg
  SetElevator.msg
  PointGPS.msg
  GoToGPS.msg
)

add_service_files(
  FILES
  EnterGloveboxPetition.srv
  BatteryExchangePetition.srv
  LeaveShowerPetition.srv
  EnterShowerPetition.srv
  EnterLiftPetition.srv
  LeaveLiftPetition.srv
  UnchargePetition.srv
  PlacePetition.srv
  ChargePetition.srv
  MovePetition.srv
  PickPetition.srv
  DockPetition.srv
  GoToPetition.srv
  GoToTagPetition.srv
  GoToNodePetition.srv
  StatusPetition.srv
  SetControlState.srv
  Cancel.srv
  SimpleGoToWithValidation.srv
  MissionCommandPetition.srv
  MagneticGoToPetition.srv
  MagneticPickPetition.srv
  MagneticPlacePetition.srv
  MagneticNavigationPetition.srv
  BarcodePickPetition.srv
  LeaveMagneticGuidePetition.srv
  FindMagneticGuidePetition.srv
  PrePickPetition.srv
  PostPickPetition.srv
  PrePlacePetition.srv
  PostPlacePetition.srv
  SetGoToPetition.srv
  SwitchMapPetition.srv
  SetElevatorPetition.srv
  MissionPetition.srv
  SetFrameID.srv
  SwitchModule.srv
  SetPose2DStamped.srv
  SetEnvironment.srv
  GetEnvironment.srv
  SaveMap.srv
  GoToGPSPetition.srv

)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
# )

generate_messages(
  DEPENDENCIES
  procedures_msgs
  geometry_msgs
  std_msgs
  robotnik_msgs
  robotnik_signal_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_local_control_msgs
 CATKIN_DEPENDS message_runtime std_msgs geometry_msgs procedures_msgs robotnik_msgs robotnik_signal_msgs
#  DEPENDS system_lib
)
