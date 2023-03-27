# summit_xl_bringup

This package contains the launch files needed to run the SUMMIT XL robot.

## Install files

Standard configuration files can be found in the folder **install** of the package.

## Robot parameters

Default params file can be found in the folder **env** of the package.

**These parameters should not be modified by the end user.**

We have to create a file named **robot_params.env**. Besides, we have to include `source ~/robot_params.env` to the *.bashrc* file. This file contains environment variables that establish the actual components of the robot. For example:

```bash
export ROBOT_ID=robot
# summit_xl, rbkairos
export ROBOT_MODEL=summit_xl
# summit_xl.urdf.xacro
export ROBOT_XACRO=summit_xl.urdf.xacro


# true, false
export ROBOT_HAS_FRONT_LASER=false
# sick_tim561, hokuyo, ydlidar_f4
export ROBOT_FRONT_LASER_MODEL=hokuyo
#export ROBOT_FRONT_LASER_PORT=/dev/ttyACM0
#export ROBOT_FRONT_LASER_IP=192.168.0.10

# true, false
export ROBOT_HAS_REAR_LASER=false
# sick_tim561, hokuyo
export ROBOT_REAR_LASER_MODEL=hokuyo
#export ROBOT_REAR_LASER_PORT=/dev/ttyACM1
#export ROBOT_REAR_LASER_IP=192.168.0.11

# true, false
export ROBOT_HAS_BENEWAKE=false
export ROBOT_BENEWAKE_IP=192.168.0.20

# true, false
export ROBOT_HAS_RS_LIDAR_16=false

# true, false
export ROBOT_HAS_FRONT_PTZ_CAMERA=false
export ROBOT_FRONT_PTZ_CAMERA_IP=192.168.0.185
export ROBOT_FRONT_CAMERA_MODEL=axis_m5013

# true, false
export ROBOT_HAS_REAR_PTZ_CAMERA=false
export ROBOT_REAR_PTZ_CAMERA_IP=192.168.0.186
export ROBOT_REAR_CAMERA_MODEL=axis_m5013

# true, false
export ROBOT_HAS_FRONT_RGBD_CAMERA=false
# usb bus
#export ROBOT_FRONT_RGBD_CAMERA_ID=#1
# orbbec, zed
export ROBOT_FRONT_RGBD_CAMERA_MODEL=orbbec
# true, false
export ROBOT_FRONT_RGBD_CAMERA_IS_REMOTE=false
# jetson-robotnik
export ROBOT_FRONT_RGBD_CAMERA_MACHINE_NAME=jetson-robotnik

# true, false
export ROBOT_HAS_REAR_RGBD_CAMERA=false
# usb bus
#export ROBOT_REAR_RGBD_CAMERA_ID=#1
# orbbec, zed
export ROBOT_REAR_RGBD_CAMERA_MODEL=orbbec
# true, false
export ROBOT_REAR_RGBD_CAMERA_IS_REMOTE=false
# jetson-robotnik
export ROBOT_REAR_RGBD_CAMERA_MACHINE_NAME=jetson-robotnik


# pixhawk, vectornav, myahrs
export ROBOT_IMU_MODEL=pixhawk
#export ROBOT_IMU_PORT=/dev/ttyUSB_PX4
# true, false
export ROBOT_HAS_GPS=false


# ps3, ps4 (default)
export ROBOT_PAD_MODEL=ps4
# 24V motors: 12.52, 48V motors: 9.56
export ROBOT_GEARBOX=9.56
# true, false
export ROBOT_HAS_ENCODER=true
# skid, omni
export ROBOT_KINEMATICS=skid
# Flag to set if the robot is differential
export ROBOT_DIFFERENTIAL_KINEMATICS=true
# std diameters: 0.22 rubber wheel, 0.25 mecanuum wheel
export ROBOT_WHEEL_DIAMETER=0.22
# summit_xl: 0.470, summit_xl_steel: 0.477, rbkairos_10: 0.538
export ROBOT_TRACK_WIDTH=0.470
# distance between the centres of the front and rear wheels
# summit_xl: 0.458, summit_xl_steel: 0.430
export ROBOT_WHEEL_BASE=0.430
# true, false
export ROBOT_HAS_SAFETY_MODULE=false
#
# export ROBOT_MODBUS_SERVER_IP=192.168.0.250

# true, false
export ROBOT_HAS_ARM=false
# Robot arm ip
export ROBOT_ARM_IP=192.168.0.210
# For UR arms. Allows just running the driver to read the state or control the arm
export ROBOT_ARM_ENABLE_TRAJECTORY_CONTROL=true
# ARM model ur: (ur5, ur10), kinova: (j2s7s300)
export ROBOT_ARM_MODEL=j2s7s300
# (ur, kinova)
export ROBOT_ARM_MANUFACTURER=kinova

# WRIST RGBD CAMERA
# true, false
export ROBOT_HAS_WRIST_RGBD_CAMERA=true
# usb bus
export ROBOT_WRIST_RGBD_CAMERA_ID=#1
# realsense
export ROBOT_WRIST_RGBD_CAMERA_MODEL=realsense

#GRIPPER
# true, false
export ROBOT_HAS_GRIPPER=true
# egh, wsg50
export ROBOT_GRIPPER_MODEL=egh
# schunk (egh, wsg50)
export ROBOT_GRIPPER_MANUFACTURER=schunk
export ROBOT_GRIPPER_IP=192.168.0.212
export ROBOT_GRIPPER_PORT=1000
# For UR arms. Allows to control the gripper using UR IO. The arm will need a program
# running to manage the gripper control.
export ROBOT_CONTROL_GRIPPER_BY_SIGNALS=true

# WEB VIDEO SERVER
export ROBOT_LAUNCH_WEB_VIDEO_SERVER=true
export ROBOT_WEB_VIDEO_SERVER_ADDRESS=127.0.0.1
export ROBOT_WEB_VIDEO_SERVER_PORT=8081

# RUN SCREENS (true, false)
export ROBOT_RUN_MAP_NAV_MANAGER=true
export ROBOT_RUN_ROBOT_LOCAL_CONTROL=true
export ROBOT_RUN_PERCEPTION=true
export ROBOT_RUN_NAVIGATION=true


# PERCEPTION PARAMETERS
# true, false
export ROBOT_RUN_AR_LOCATOR=false
# true, false
export ROBOT_RUN_REFLECTOR_LOCATOR=true
# default, hokuyo_utm, hokuyo_ust
export ROBOT_LASER_MODEL_INTENSITY_FILTER=hokuyo_utm
# double, distance between tags/reflectors
export ROBOT_DOCKING_STATION_TAG_DISTANCE=0.3
# double, max distance to detect the station
export ROBOT_DOCKING_STATION_TAG_MAX_DISTANCE_DETECTION=3.5
# 2d scan topic used for perception purposes (two tag locator)
export ROBOT_PERCEPTION_SCAN_TOPIC=front_laser/scan
# rgb camera topic used for perception purposes (ar locator)
export ROBOT_PERCEPTION_CAMERA_TOPIC=front_rgbd_camera/rgb/image_raw
# rgb camera info topic used for perception purposes (ar locator)
export ROBOT_PERCEPTION_CAMERA_INFO_TOPIC=front_rgbd_camera/rgb/camera_info

# LOCALIZATION PARAMETERS
# The topic that amcl is using to locate the robot
export ROBOT_LOCALIZATION_SCAN_TOPIC=front_laser/scan
# Type of odometry used for the localization algorithm (mainly amcl). Values: diff, omni
export ROBOT_LOCALIZATION_ODOM_MODEL=diff

# NAVIGATION PARAMETERS
# true, false
export ROBOT_HAS_DOCKER=true
# true, false
export ROBOT_DOCKER_RUN_CAMERA=true
# true, false
export ROBOT_DOCKER_RUN_LASER=true
# summit_xl = -0.641   summit_xls/rbkairos = -0.681
export ROBOT_DOCKER_LASER_BASE_DISTANCE=-0.681
# true, false
export ROBOT_RUN_MOVE=true
# teb, eband
export ROBOT_MOVE_BASE_LOCAL_PLANNER=eband


# BASE_HW
# The real battery voltage (multimeter) minus the one returned by the drive
export ROBOT_BASE_HW_BATTERY_VOLTAGE_OFFSET=0.0
# K multipliers for analog inputs
export ROBOT_K_ANALOG_INPUTS_MULTIPLIERS=[6.978947368,-12.5,1.0,1.0]


# BATTERY ESTIMATION
# Voltage of the battery (24,48)
export ROBOT_BATTERY_VOLTAGE=24
# Read voltage from analog input
export ROBOT_READ_VOLTAGE_FROM_ANALOG_INPUT=true
# Analog input of the driver to read the voltage
export ROBOT_VOLTAGE_ANALOG_INPUT_NUMBER=1
# Analog input of the driver to read the current
export ROBOT_CURRENT_ANALOG_INPUT_NUMBER=2


# CHARGE MANAGER
# disabled, automatic_hw, automatic_sw, manual_sw
export ROBOT_DOCKER_MODE=automatic_sw
# true, false
export ROBOT_BATTERY_INVERTED_CONTACT_RELAY=false
#export ROBOT_BATTERY_CONTACT_RELAY_INPUT_NUMBER=2
#export ROBOT_BATTERY_CHARGER_RELAY_OUTPUT_NUMBER=1

# ROBOT_LOCAL_CONTROL
export ROBOT_RLC_ROSTFUL_SERVER_IP=127.0.0.1
export ROBOT_RLC_ROSTFUL_SERVER_PORT=8080
export ROBOT_RLC_CHARGE_DOCK_OFFSET_X=-0.65
export ROBOT_RLC_CHARGE_DOCKER_NAMESPACE=pp_docker
export ROBOT_RLC_CHARGE_STEP_DISTANCE=0.06
# frame used to identify the docking_station. It can be different depending on the sensors and locators being used
export ROBOT_RLC_CHARGE_FRAME=robot_docking_station_1

# Folder to save the maps
export ROBOT_MAPS_FOLDER=$HOME/maps
```

##### Explanation of each environment variable

- `ROBOT_ID` indicates the name of the robot. This is the name of the namespace under all the nodes will be working. This is also used as the prefix of all the subcomponents.(*summit_xl*)
- `ROBOT_MODEL` indicates the type of robot. This is used to load specific configurations.(*summit_xl/rbkairos*)
- `ROBOT_XACRO` indicates the path where the xacro file is. (inside the robot folder in robot_description)(*summit_xl.urdf.xacro*)
- `ROBOT_FRONT_LASER_MODEL` indicates the model of the laser that the robot is using. The model is the name of the launch file.(*sick_tim561, hokuyo, ydlidar_f4, sick_s300, sick_microscan3*)
- `ROBOT_REAR_LASER_MODEL` indicates the model of the laser that the robot is using. The model is the name of the launch file.(*sick_tim561, hokuyo, ydlidar_f4, sick_s300, sick_microscan3*)
- `ROBOT_HAS_FRONT_LASER` indicates if the robot has a laser in front. (*true/false*)
- `ROBOT_HAS_REAR_LASER` indicates if the robot has a laser in rear. (*true/false*)
- `ROBOT_HAS_FRONT_PTZ_CAMERA` indicates if the robot has the ptz camera in front. (*true/false*)
- `ROBOT_HAS_REAR_PTZ_CAMERA` indicates if the robot has the ptz camera in front. (*true/false*)
- `ROBOT_HAS_GPS` indicates if the robot has gps. (*true/false*)
- `ROBOT_HAS_FRONT_RGBD_CAMERA` indicates if the robot has a front rgbd camera. (*true/false*)
- `ROBOT_FRONT_RGBD_CAMERA_ID` camera id to identify in the bus
- `ROBOT_FRONT_RGBD_CAMERA_MODEL` indicates the model of the front rgbd camera (*orbbec/zed*)
- `ROBOT_FRONT_RGBD_CAMERA_IS_REMOTE` indicates if the front rgbd camera software is run by the robot pc or external one (*true/false*)
- `ROBOT_FRONT_RGBD_CAMERA_MACHINE_NAME` name of the external machine that run the software of the camera. This is not used if ROBOT_FRONT_RGBD_CAMERA_IS_REMOTE is false (jetson-robotnik)
- `ROBOT_HAS_REAR_RGBD_CAMERA` indicates if the robot has a front rgbd camera. (*true/false*)
- `ROBOT_REAR_RGBD_CAMERA_ID` camera id to identify in the bus
- `ROBOT_REAR_RGBD_CAMERA_MODEL` indicates the model of the rear rgbd camera (*orbbec/zed*)
- `ROBOT_REAR_RGBD_CAMERA_IS_REMOTE` indicates if the rear rgbd camera software is run by the robot pc or external one (*true/false*)
- `ROBOT_REAR_RGBD_CAMERA_MACHINE_NAME` name of the external machine that run the software of the rear rgbd camera. This is not used if ROBOT_FRONT_RGBD_CAMERA_IS_REMOTE is false (jetson-robotnik)
- `ROBOT_IMU_MODEL` IMU model used (*pixhawk/vectornav/myahrs*)
- `ROBOT_IMU_PORT` IMU port (*string*)
- `ROBOT_PAD_MODEL` pad model used. (*ps4/ps3/logitechf710/xbox360*)
- `ROBOT_GEARBOX` establishes the motor gearbox value. (*24V: 12.52 | 48V: 9.56*)
- `ROBOT_HAS_ENCODER` indicates if the robot has encoders. (*true/false*)
- `ROBOT_WHEEL_DIAMETER` set the wheel diameter. It can be set for odometry adjustment (*double*)
- `ROBOT_TRACK_WIDTH` set the width distance between wheels. It can be set for odometry adjustment. Default value is  0.462 for summit_xl (*summit_xl: 0.462, summit_xl_steel: 0.477, rbkairos_10: 0.538*)
- `ROBOT_WHEEL_BASE` set the distance between the centres of the front and rear wheels. Default values: (*summit_xl: 0.458, summit_xl_steel: 0.430*)
- `ROBOT_KINEMATICS` kinematic configuration of the robot. (*skid/omni/steel_skid/steel_omni*)
- `ROBOT_DIFFERENTIAL_KINEMATICS` Flag to set if the robot is differential. By detault is set to true. Be careful not to set to false if the robot is differential. It is used for actions like move. (*true/false*)
- `ROBOT_HAS_SAFETY` Flag to set if the robot has a safety plc. (*true/false*)
- `ROBOT_MODBUS_SERVER_IP` ip address of the modbus server used by the robot if it has a safety plc. (*192.168.0.250*)

**ARM**

- `ROBOT_HAS_ARM` indicates if the robot has an arm (*true/false*)
- `ROBOT_ARM_IP` sets the ip of the arm (*192.168.0.210*)
- `ROBOT_ARM_MODEL` sets the model of the arm (ur: *ur5, ur10*, kinova: *j2s7s300*)
- `ROBOT_ARM_MANUFACTURER` sets the manufacturer of the arm (ur, kinova)
- `ROBOT_ARM_ENABLE_TRAJECTORY_CONTROL` For UR arms. Allows just running the driver to read the state or control the arm (*true/false*)

**GRIPPER**

- `ROBOT_HAS_GRIPPER` Indicates if the robot has gripper or not.
- `ROBOT_GRIPPER_MODEL` Shows the gripper's model. This value is used to select the launch file that inits the driver of the gripper (egh, wsg50)
- `ROBOT_GRIPPER_MANUFACTURER` Indicates the manufacturer of the gripper. This value equals to the folder where the gripper launch files are saved (schunk)
- `ROBOT_GRIPPER_IP` Shows the IP of the gripper (192.168.0.212)
- `ROBOT_GRIPPER_PORT` Indicates the port where the gripper server is set (1000)
- `ROBOT_CONTROL_GRIPPER_BY_SIGNALS` For UR arms. Allows to control the gripper using UR IO. The arm will need a program running to manage the gripper control. (true, false)

**WEB VIDEO SERVER**

- `ROBOT_LAUNCH_WEB_VIDEO_SERVER`  Sets whether running the web video server node or not. It converts any ROS topic image into a http image server. ((***true(***/*false*)
- `ROBOT_WEB_VIDEO_SERVER_ADDRESS`  Sets the ip address where the http video server will be available. It should changed into the one the robot will be working. (*127.0.0.1*)
- `ROBOT_WEB_VIDEO_SERVER_PORT`  Sets the port number where the http video server will be available. (*8081*)

**RUN SCREENS**

- `ROBOT_RUN_MAP_NAV_MANAGER`  Sets whether running Map Nav Manager node or not (**true**/false)
- `ROBOT_RUN_ROBOT_LOCAL_CONTROL`  Sets whether running Robot Local Control node or not (**true**/false)
- `ROBOT_RUN_PERCEPTION`  Sets whether running perception nodes or not (**true**/ false)
- `ROBOT_RUN_NAVIGATION`  Sets whether running navigation nodes or not (**true**/false)

**PERCEPTION PARAMETERS**

- `ROBOT_RUN_AR_LOCATOR`  Sets whether running AR Locator or not (*true/false*)
- `ROBOT_RUN_REFLECTOR_LOCATOR`  Sets whether running reflectors Locator or not (*true/false*)
- `ROBOT_LASER_MODEL_INTENSITY_FILTER`  Sets the laser intensity filter config file (*default, hokuyo_utm, hokuyo_ust*)
- `ROBOT_DOCKING_STATION_TAG_DISTANCE`  Sets distance between the two reflective tags used for docking  ((*double, 0.3*)
- `ROBOT_DOCKING_STATION_TAG_MAX_DISTANCE_DETECTION`  max distance to detect the docking station  (*double, 3.5*)
- `ROBOT_PERCEPTION_SCAN_TOPIC`  2d scan topic used for perception purposes (two tag locator)  (*string, front_laser/scan*)
- `ROBOT_PERCEPTION_CAMERA_TOPIC` rgb camera topic used for perception purposes (ar locator) (*string, front_rgbd_camera/rgb/image_raw*)
- `ROBOT_PERCEPTION_CAMERA_INFO_TOPIC` rgb camera topic used for perception purposes (ar locator) (*string, front_rgbd_camera/rgb/camera_info*)

**LOCALIZATION PARAMETERS**

- `ROBOT_LOCALIZATION_SCAN_TOPIC`  Sets the 2d scan used by the localization module (*front_laser/scan*)
- `ROBOT_LOCALIZATION_ODOM_MODEL`  Sets the type of odometry used for the localization module (normally amcl). Values: ((*string, diff / omni *)
- `ROBOT_MAPS_FOLDER` Path to the folder to where all the maps, frames and pois are going to be saved. Values: ((*string, $HOME/maps *)

**NAVIGATION PARAMETERS**

- `ROBOT_HAS_DOCKER`  Sets whether running the node to perform the dock (to docking station) or not  (*true/false*)
- `ROBOT_DOCKER_RUN_CAMERA`  Sets whether running the node to perform the dock by camera or not  (*true/false*)
- `ROBOT_DOCKER_RUN_LASER`  Sets whether running the node to perform the dock by laser or not  (*true/false*)
- `ROBOT_DOCKER_LASER_BASE_DISTANCE` Sets the distance between the laser reflectors and the base_footprint of the robot when it is charging
- `ROBOT_RUN_MOVE`  Sets whether running the node to perform the move action or not  (*true/false*)
- `ROBOT_MOVE_BASE_LOCAL_PLANNER`  Move base local planner loaded  (*eband/teb*)

**BATTERY DOCKING**

- `ROBOT_HAS_DOCKER`  Sets whether the docking servers has to be started or not (*true/false*)
- `ROBOT_DOCKER_RUN_CAMERA`  Sets whether the docking by camera has to be started or not (*true/false*)
- `ROBOT_DOCKER_RUN_CAMERA`  Sets whether the docking by laser has to be started or not (*true/false*)

**BASE_HW**

- `ROBOT_BASE_HW_BATTERY_OFFSET` sets the offset to apply to the battery voltage got from the motordrive. Has to be calculated by using the real battery voltage (multimeter) minus the one returned by the drive (1.0)
- `ROBOT_K_ANALOG_INPUTS_MULTIPLIERS` Constant multiplied to current analog input values

**BATTERY ESTIMATION**

- `ROBOT_BATTERY_VOLTAGE` voltage used for the (dis)charging curve
- `ROBOT_READ_VOLTAGE_FROM_ANOLG_INPUT`sets the mode to read battery voltage from an analog input instead from the motordrive (*true/false*)
- `ROBOT_VOLTAGE_ANALOG_INPUT_NUMBER` analog input of the driver to read the voltage(*unsigned int*).
- `ROBOT_CHARGING_CURRENT_OFFSET` Offset applied to the current measured by the sensor to consider whether or not the robot is charging. (*double*)

**CHARGE MANAGER**

- `ROBOT_DOCKER_MODE` sets the mode of how the hw charges the battery through the docking station contacts. (disabled, automatic_hw, automatic_sw, manual_sw)
- `ROBOT_BATTERY_INVERTED_CONTACT_RELAY`(*true/false*). It inverts the status of the digital input related with the contact relay. By default it should be set to true.
- `ROBOT_BATTERY_CONTACT_RELAY_INPUT_NUMBER`digital input number to detect the contact between the robot and the docking station charger (*unsigned int*). This param should not be modified.
- `ROBOT_BATTERY_CHARGER_RELAY_OUTPUT_NUMBER`digital output number to switch on the internal robot charging contacts (*unsigned int*). This param should not be modified.

**ROBOT_LOCAL_CONTROL**

- `ROBOT_RLC_ROSTFUL_SERVER_IP` IP address for the REST server connected to Robot Local Control (127.0.0.1)
- `ROBOT_RLC_ROSTFUL_SERVER_PORT` Port number for the REST server connected to Robot Local Control (8080)
- `ROBOT_RLC_CHARGE_DOCKER_NAMESPACE` Docker namespace used for charging when using RLC (*string*)
- `ROBOT_RLC_CHARGE_DOCK_OFFSET_X` Offset in x axis used for the docker in RLC. Normally a negative value(*double*)
- `ROBOT_RLC_CHARGE_STEP_DISTANCE` Final distance applied during the charge procedure (*double*)
- `ROBOT_RLC_CHARGE_FRAME` Name of the frame used to perform the docking to the battery charge station (*string*)

### Environment configuration

There is a script located in summit_xl_bringup/scripts to add the enviroment parameters into the system.

It will add the default environment parameters as well as the the autorun configuration.

- Go to summit_xl_bringup/scripts
- run the script
  - ```./configure_autorun.sh```
