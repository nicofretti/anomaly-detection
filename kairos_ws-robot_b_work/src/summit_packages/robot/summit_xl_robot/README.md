# summit_xl_robot
ROS package of the Summit XL, XL-HL mobile robot platforms

![Image of Summit XL](http://www.robotnik.es/web/wp-content/uploads/2014/03/summit-xl-robots-moviles-robotnik_s01.jpg)

<h2>summit_xl_bringup</h2>

This package contains launch files to start the robot. The most relevant file is launch/summit_xl_complete.launch, that is called from the .bashrc at system startup. The file can be configured to start the available devices, usually: lasers, cameras, rgbd devices, imus, gps, etc. Integrates several test launch files to test the components individually. This file defines the robot serial number as parameter.

<h2>summit_xl_controller</h2>

This package defines the summit_xl_controller/SummitXLController as controller_interface::ControllerBase class of ROS Control. The Summit XL Controller controls 4 wheel mobile platforms both in skid and omnidirectional drive. It expects 4 VelocityJointInterface type of hardware interfaces. It contains the robot control functions to operate the skid-steering / omni-steering structure (velocity control of the axes and position control of the robot) and to get accurate odometry estimations from the robot sensors. This node subscribes to cmd_vel messages. Alternatively the robot can also be controlled using the ROS diff_drive_controller, but this controller has less features, e.g. it does not allow the control as omnidirectional platform.

<h2>summit_xl_hw</h2>

Package to manage the Summit XL / Summit XL HL / Summit XL Steel - servo controller set. Hardware interface to the 4 servoamplifiers compliant with ROS Control and associated digital I/O. This node publishes the battery voltage and permits the definition of an alarm voltage. 

<h2>summit_xl_robot</h2>

summit_xl_robot metapackage folder.

<h2>summit_xl_web</h2>

rosbridge server based package to access the robot via web browser. It allows to access the status of the robot, the diagnostics of the servos, the camera, battery level, errors, etc. It allows to control the platform motion, but also the PTZ. The package is intended as a template to be further expanded by the user.







