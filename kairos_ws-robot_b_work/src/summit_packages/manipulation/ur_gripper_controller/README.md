# UR GRIPPER CONTROLLER
This package collects a set of fake gripper controllers based on interaction with a UR arm. Typically the UR arm will have a program with the different control cases of the gripper and the node will interact with the arm through user input via ROS.

## XMLRPC CONTROLLER

Controller based in XMLRPC. The node runs a server that the UR will interact with. Requires a program running in the UR that is requesting information and commands to the node (there is an example in the [resources/egh](resources/egh) folder).

### Functions

The XMLRPC server has a list of functions that the client (UR arm) can use to interact with:

- **get_command**: Returns an integer that shows the current command:
  - 0: IDLE. Just waiting
  - 1: MOVE. The controller has received a desired position and the UR has to move the gripper
- **get_desired_pos**: Returns a float with the commanded position in ROS
- **set_idle**: Sets IDLE state. The user has to set_idle after finishing each movement.