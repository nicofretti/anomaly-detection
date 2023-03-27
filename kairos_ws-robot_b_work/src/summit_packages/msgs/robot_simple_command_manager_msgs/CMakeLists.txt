cmake_minimum_required(VERSION 2.8.3)
project(robot_simple_command_manager_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED 
    message_generation
    actionlib_msgs
)


## Generate messages in the 'msg' folder
add_message_files(
    FILES
    CommandManagerStatus.msg
    CommandManager.msg
    CommandManagerArray.msg
    CommandString.msg
    CommandStringFeedback.msg
    CommandStringResult.msg
    ReturnMessage.msg
    StatusCodes.msg
    SequencerStatus.msg
    SequenceStatus.msg
    CommandLog.msg
    CommandLogArray.msg
    CommandScheduler.msg
    EventScheduler.msg
    EventSchedulerArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
    DIRECTORY srv
    FILES
    SetCommandString.srv
    GetHandlers.srv
    GetHandlerInfo.srv
    ManageCommandManager.srv
    GetCommandManagerList.srv
    GetCommandSchedulerList.srv
    AddSchedule.srv
)

## Generate actions in the 'action' folder
add_action_files(
    FILES
    RobotSimpleCommand.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    actionlib_msgs
)

catkin_package(
   CATKIN_DEPENDS message_runtime actionlib_msgs
)

