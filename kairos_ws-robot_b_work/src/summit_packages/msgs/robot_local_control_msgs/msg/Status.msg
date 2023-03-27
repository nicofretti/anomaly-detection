# GLOBAL ROBOT STATE
# init: the robot is initializing
# standby: the robot is ready to receive orders. Power saving behaviours can be applied
# ready: the robot is ready to work or working
# emergency: the robot cannot work/operate correctly temporary. It can imply a recovery procedure or an external action from the operator
# failure: the robot is not working. It needs the operator intervention and probably restart/reboot the robot
string ROBOT_STATE_INIT=init
string ROBOT_STATE_STANDBY=standby
string ROBOT_STATE_READY=ready
string ROBOT_STATE_EMERGENCY=emergency
string ROBOT_STATE_FAILURE=failure


# CONTROL STATES OF THE ROBOT
# auto: the robot is moving autonomously
# manual: the robot is being tele-operated by an operator
# follow: the robot is following a person
string CONTROL_STATE_UNKNOWN=unknown
string CONTROL_STATE_AUTO=auto
string CONTROL_STATE_MANUAL=manual
string CONTROL_STATE_FOLLOW=follow


# OPERATIONAL STATES
# Defines what's doing the robot
# idle: the robot is doing nothing
# calibrating: the robot is calibrating
# moving: the robot is moving
# raising: the elevator is raising
# lowering: the elevator is lowering
# charging: the robot is charging the battery
string OPERATION_STATE_IDLE=idle
string OPERATION_STATE_CALIBRATION=calibrating
string OPERATION_STATE_MOVING=moving
string OPERATION_STATE_ELEVATION_RAISE=raising_elevator
string OPERATION_STATE_ELEVATION_LOWER=lowering_elevator
string OPERATION_STATE_BATTERY_CHARGE=charging


###############################################################################################

# Human readable ID
string robot_id

# GLOBAL/Overall state of the component or robot
# INIT, STANDBY, READY, EMERGENCY, FAILURE
string robot_state

# ROBOT OPERATION MODE
# AUTO, MANUAL, FOLLOW
string control_state

# idle, calibrating, moving, raising_elevator,...
string operation_state

# Robot position (Define custom pose type)
# RobotPose pose
# type: gps, amcl, gmapping
LocalizationStatus localization_status

# Navigation Status
# move_base, purepursuit, docking
NavigationStatus navigation_status

# Robot status: batt, lights, acoustic signal
# odometry
RobotStatus robot_status

# Signal manager: array with the signal of each sensor
SignalManager signal_manager



# clasificarlo por tipos (action, localization, sensors,controllers...)
# string[] state_procedures
