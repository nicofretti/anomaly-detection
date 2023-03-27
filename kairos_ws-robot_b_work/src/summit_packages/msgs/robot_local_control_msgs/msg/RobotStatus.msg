# robot battery
robotnik_msgs/BatteryStatus battery

# Robot emergency stop
bool emergency_stop

# lights
# acoustic signal
# bool acoustic_signal

# Robot internal odometry
Pose2DStamped pose
Twist2D velocity

# Safety status
robotnik_msgs/SafetyModuleStatus safety_status

# Battery Docking status
robotnik_msgs/BatteryDockingStatus battery_docking_status

ControllerStatus[] controllers

SensorStatus[] sensors
robotnik_msgs/ElevatorStatus elevator
robotnik_msgs/ArmStatus arm