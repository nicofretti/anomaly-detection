/** \file summit_xl_controller.cpp
 * \author Robotnik Automation S.L.L.
 * \version 3.0
 * \date    2017
 *
 * \brief summit_xl_controller class driver
 * Component to manage the Summit XL servo controller set
 * (C) 2012 Robotnik Automation, SLL
*/

#include <algorithm>
#include <numeric>
#include <sstream>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <summit_xl_controller/summit_xl_controller.h>
#define SKID_STEERING 1
#define MECANUM_STEERING 2
#define SUMMIT_XL_COMMAND_WATCHDOG 0.1
#define DEFAULT_INMOTION_TIMER 2.0
#define DEFAULT_MAX_JOINT_SPEED 27.27

namespace summit_xl_controller
{
double radnorm(double value)
{
  while (value > M_PI)
    value -= M_PI;
  while (value < -M_PI)
    value += M_PI;
  return value;
}

double radnorm2(double value)
{
  while (value > 2.0 * M_PI)
    value -= 2.0 * M_PI;
  while (value < -2.0 * M_PI)
    value += 2.0 * M_PI;
  return value;
}

double radnormHalf(double value)
{  // norms the angle so it is between -M_PI/2 and M_PI/2
  double eps = 1e-5;
  while (value > 0.5 * M_PI + eps)
    value -= M_PI;
  while (value < -0.5 * M_PI - eps)
    value += M_PI;
  return value;
}

// norms a double value so if it is rounded to zero when it's below an epsylon
double normToZero(double value)
{
  double eps = 1e-4;
  if (std::abs(value) < eps)
    return 0;
  return value;
}

// return the sign, as -1 or 1, of the value. 0 is positive
double sign(double value)
{
  return (value < 0) ? -1 : 1;
}

// checks that v and w have the same sign. 0 is positive
bool haveSameSign(double v, double w)
{
  return (v < 0) == (w < 0);
}

SummitXLController::SummitXLController()
{
  // TODO: initialize all variables
}

bool SummitXLController::init(hardware_interface::VelocityJointInterface* hw, ros::NodeHandle& root_nh,
                              ros::NodeHandle& controller_nh)
{
  bool everything_ok = true;

  hard_brake_ = true;
  e_stop_ = false;
  controller_name_ = "summit_xl_controller";

  // the topics are hardcoded, the way to change them is by using the remap
  // option
  command_topic_ = "cmd_vel";
  odom_topic_ = "odom";

  // default values for some variables that can change using the param server
  // related to kinematics

  wheel_base_ = 0.569;
  track_width_ = 0.543;
  wheel_diameter_ = 0.233;

  linear_speed_limit_ = 0.1;
  linear_acceleration_limit_ = 0.1;
  angular_speed_limit_ = 0.1;
  angular_acceleration_limit_ = 0.1;

  //
  kinematic_mode_ = "skid";
  controller_nh.param("wheel_base", wheel_base_, wheel_base_);
  controller_nh.param("track_width", track_width_, track_width_);
  controller_nh.param("wheel_diameter", wheel_diameter_, wheel_diameter_);
  controller_nh.param("cmd_vel", command_topic_, command_topic_);
  controller_nh.param("kinematic_mode", kinematic_mode_, kinematic_mode_);
  controller_nh.param("odom_topic", odom_topic_, odom_topic_);
  controller_nh.param<std::string>("joint/front_right_wheel_joint/name", front_right_traction_joint_,
                                   "summit_xl_front_right_wheel_joint");
  controller_nh.param<std::string>("joint/front_left_wheel_joint/name", front_left_traction_joint_, "summit_xl_front_"
                                                                                                    "left_wheel_joint");
  controller_nh.param<std::string>("joint/back_left_wheel_joint/name", back_left_traction_joint_, "summit_xl_back_left_"
                                                                                                  "wheel_joint");
  controller_nh.param<std::string>("joint/back_right_wheel_joint/name", back_right_traction_joint_,
                                   "summit_xl_back_right_wheel_joint");

  controller_nh.param("joint/front_right_wheel_joint/max_speed", front_right_traction_joint_max_speed_,
                      DEFAULT_MAX_JOINT_SPEED);
  controller_nh.param("joint/front_left_wheel_joint/max_speed", front_left_traction_joint_max_speed_,
                      DEFAULT_MAX_JOINT_SPEED);
  controller_nh.param("joint/back_left_wheel_joint/max_speed", back_left_traction_joint_max_speed_,
                      DEFAULT_MAX_JOINT_SPEED);
  controller_nh.param("joint/back_right_wheel_joint/max_speed", back_right_traction_joint_max_speed_,
                      DEFAULT_MAX_JOINT_SPEED);

  controller_nh.param("joint_speed_noise_threshold", joint_speed_noise_threshold_, 0.002);

  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "::init: joints [" << front_right_traction_joint_ << ","
                                                           << front_left_traction_joint_ << ","
                                                           << back_left_traction_joint_ << ","
                                                           << back_right_traction_joint_);

  if (controller_nh.hasParam("linear_speed_limit") == false)
  {  // limit does not exist!
    ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Cannot find parameter "
                                                                "linear_speed_limit parameter. I "
                                                                "will gently set it to "
                                                             << linear_speed_limit_);
  }
  else
  {
    controller_nh.param("linear_speed_limit", linear_speed_limit_, linear_speed_limit_);
    if (linear_speed_limit_ < 0)
    {
      ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Watch out! You set linear_speed_limit, "
                                                                  "which is the limit of the modulo of the linear "
                                                                  "speed, to a negative value. I will gently set "
                                                                  "it as positive.");
      linear_speed_limit_ = -linear_speed_limit_;
    }
  }

  if (controller_nh.hasParam("linear_acceleration_limit") == false)
  {  // limit does not exist!
    ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Cannot find parameter "
                                                                "linear_acceleration_limit parameter. I will "
                                                                "gently set it to "
                                                             << linear_acceleration_limit_);
  }
  else
  {
    controller_nh.param("linear_acceleration_limit", linear_acceleration_limit_, linear_acceleration_limit_);
    if (linear_acceleration_limit_ < 0)
    {
      ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Watch out! You set "
                                                                  "linear_acceleration_limit, which is the "
                                                                  "limit of the modulo of the linear "
                                                                  "acceleration, to a negative value. I will "
                                                                  "gently set it as positive.");
      linear_acceleration_limit_ = -linear_acceleration_limit_;
    }
  }

  if (controller_nh.hasParam("angular_speed_limit") == false)
  {  // limit does not exist!
    ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Cannot find parameter "
                                                                "angular_speed_limit parameter. "
                                                                "I will gently set it to "
                                                             << angular_speed_limit_);
  }
  else
  {
    controller_nh.param("angular_speed_limit", angular_speed_limit_, angular_speed_limit_);
    if (angular_speed_limit_ < 0)
    {
      ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Watch out! You set angular_speed_limit, "
                                                                  "which is the limit of the modulo of the angular "
                                                                  "speed, to a negative value. I will gently set "
                                                                  "it as positive.");
      angular_speed_limit_ = -angular_speed_limit_;
    }
  }

  if (controller_nh.hasParam("angular_acceleration_limit") == false)
  {  // limit does not exist!
    ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Cannot find parameter "
                                                                "angular_acceleration_limit parameter. I will "
                                                                "gently set it to "
                                                             << angular_acceleration_limit_);
  }
  else
  {
    controller_nh.param("angular_acceleration_limit", angular_acceleration_limit_, angular_acceleration_limit_);
    if (angular_acceleration_limit_ < 0)
    {
      ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::init: Watch out! You set "
                                                                  "angular_acceleration_limit, which is the "
                                                                  "limit of the modulo of the angular "
                                                                  "acceleration, to a negative value. I will "
                                                                  "gently set it as positive.");
      angular_acceleration_limit_ = -angular_acceleration_limit_;
    }
  }

  // related to coordinate frames
  odom_frame_ = "odom";
  robot_base_frame_ = "base_footprint";
  odom_broadcast_tf_ = true;
  controller_nh.param<std::string>("odom_frame", odom_frame_, odom_frame_);
  controller_nh.param<std::string>("robot_base_frame", robot_base_frame_, robot_base_frame_);
  controller_nh.param("odom_broadcast_tf", odom_broadcast_tf_, odom_broadcast_tf_);

  // realted to control mode and odometry orientation source
  controller_nh.param<std::string>("kinematic_mode", kinematic_mode_, "skid");
  motion_odometry_ = false;
  controller_nh.param("motion_odometry", motion_odometry_, motion_odometry_);
  controller_nh.param<std::string>("imu_topic", imu_topic_,
                                   "imu/data");  // Topic published by the imu_complementary_filter
  controller_nh.param<std::string>("e_stop_topic", e_stop_topic_,
                                   "robotnik_base_hw/emergency_stop");  // Topic published by the robotnik_base_hw

  // related to timing
  double cmd_watchdog = 0.25;
  double imu_watchdog = 0.1;
  double odom_frequency = 100;
  in_motion_timer_ = 2.0;
  controller_nh.param("cmd_watchdog_duration", cmd_watchdog, cmd_watchdog);
  controller_nh.param("imu_watchdog_duration", imu_watchdog, imu_watchdog);
  controller_nh.param("odom_publish_frequency", odom_frequency, odom_frequency);
  controller_nh.param("in_motion_timer", in_motion_timer_, in_motion_timer_);
  controller_nh.param("hard_brake", hard_brake_, hard_brake_);

  // begin to register ros stuff
  cmd_watchdog_duration_ = ros::Duration(cmd_watchdog);
  imu_watchdog_duration_ = ros::Duration(imu_watchdog);
  odom_publish_period_ = ros::Duration(1.0 / odom_frequency);

  cmd_vel_subscriber_ = root_nh.subscribe(command_topic_, 1, &SummitXLController::cmdVelCallback, this);
  imu_sub_ = root_nh.subscribe(imu_topic_, 1, &SummitXLController::imuCallback, this);
  e_stop_sub_ = root_nh.subscribe(e_stop_topic_, 1, &SummitXLController::eStopCallback, this);
  odom_publisher_ = root_nh.advertise<nav_msgs::Odometry>(odom_topic_, 1);
  enabled_pub_ = controller_nh.advertise<std_msgs::Bool>("enabled", 1);
  in_motion_publisher_ = controller_nh.advertise<std_msgs::Bool>("in_motion", 1);
  transform_broadcaster_ = new tf::TransformBroadcaster();

  // Advertise services
  srv_GetMode_ = root_nh.advertiseService("get_mode", &SummitXLController::srvCallback_GetMode, this);
  srv_SetOdometry_ = root_nh.advertiseService("set_odometry", &SummitXLController::srvCallback_SetOdometry, this);
  enable_srv_ = controller_nh.advertiseService("enable", &SummitXLController::enableSrvCallback, this);

  if (kinematic_mode_ == "skid")
  {
    active_kinematic_mode_ = SKID_STEERING;
  }
  else if (kinematic_mode_ == "omni")
  {
    active_kinematic_mode_ = MECANUM_STEERING;
  }

  joints_.resize(NUMBER_OF_JOINTS);

  joint_states_.resize(NUMBER_OF_JOINTS);
  joint_states_mean_.resize(NUMBER_OF_JOINTS);
  joint_references_.resize(NUMBER_OF_JOINTS);
  joint_commands_.resize(NUMBER_OF_JOINTS);

  // for now, the controller is for a robot with four wheel, and the joint names
  // are these and only these
  joint_names_.resize(NUMBER_OF_JOINTS);
  joint_names_[FRONT_RIGHT_TRACTION_JOINT] = front_right_traction_joint_;
  joint_names_[FRONT_LEFT_TRACTION_JOINT] = front_left_traction_joint_;
  joint_names_[BACK_RIGHT_TRACTION_JOINT] = back_right_traction_joint_;
  joint_names_[BACK_LEFT_TRACTION_JOINT] = back_left_traction_joint_;

  joint_limits_.resize(NUMBER_OF_JOINTS);

  joint_limits_[FRONT_RIGHT_TRACTION_JOINT] =
      std::make_pair(-front_right_traction_joint_max_speed_, front_right_traction_joint_max_speed_);
  joint_limits_[FRONT_LEFT_TRACTION_JOINT] =
      std::make_pair(-front_left_traction_joint_max_speed_, front_left_traction_joint_max_speed_);
  joint_limits_[BACK_RIGHT_TRACTION_JOINT] =
      std::make_pair(-back_right_traction_joint_max_speed_, back_right_traction_joint_max_speed_);
  joint_limits_[BACK_LEFT_TRACTION_JOINT] =
      std::make_pair(-back_left_traction_joint_max_speed_, back_left_traction_joint_max_speed_);

  joint_states_history_size_ = 10;
  joint_states_history_.resize(NUMBER_OF_JOINTS);
  for (size_t i = 0; i < NUMBER_OF_JOINTS; i++)
  {
    joint_states_history_[i] = boost::circular_buffer<double>(joint_states_history_size_);
  }

  // set velocity limits
  /* for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++) {
       std::string param_limit_name;
       double max_speed = 0;

       param_limit_name = "joint/" + joint_names_[i] + "/max_speed";
       if (controller_nh.hasParam(param_limit_name) == false) { // limit does
   not exist!
           ROS_ERROR_STREAM_NAMED(controller_name_, controller_name_ << "::init:
   cannot find parameter " << param_limit_name << ". It is required!");
           everything_ok = false;
       }
       else {
           controller_nh.param(param_limit_name, max_speed, max_speed);
       }

       joint_limits_[i] = std::make_pair(-max_speed, max_speed);
   }*/

  if (everything_ok)
  {
    ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "::init: everything is OK!");
  }

  // initialize last_yaw_ and delta_yaw_ variable
  init_yaw_ = 0.0;
  last_yaw_ = 0.0;
  delta_yaw_ = 0.0;
  motion_yaw_ = 0.0;
  imu_yaw_ = 0.0;

  // first yaw read
  bFirstYawRead_ = false;
  bYawSensor_ = false;

  joints_[FRONT_RIGHT_TRACTION_JOINT] = hw->getHandle(joint_names_[FRONT_RIGHT_TRACTION_JOINT]);
  joints_[FRONT_LEFT_TRACTION_JOINT] = hw->getHandle(joint_names_[FRONT_LEFT_TRACTION_JOINT]);
  joints_[BACK_RIGHT_TRACTION_JOINT] = hw->getHandle(joint_names_[BACK_RIGHT_TRACTION_JOINT]);
  joints_[BACK_LEFT_TRACTION_JOINT] = hw->getHandle(joint_names_[BACK_LEFT_TRACTION_JOINT]);

  enabled_ = true;

  return everything_ok;
}

std::string SummitXLController::getHardwareInterfaceType() const
{
  // as result of being a Controller which uses different types of
  // JointInterface, return the main interface type
  // in this case, it is a VelocityJointInterface
  return hardware_interface::internal::demangledTypeName<hardware_interface::VelocityJointInterface>();
}

/**
 * \brief Starts controller
 * \param time Current time
 */
void SummitXLController::starting(const ros::Time& time)
{
  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << ": Starting!");
  odom_last_sent_ = ros::Time::now();
  odom_last_update_ = ros::Time::now();  // check if there is some value that invalidates the
                                         // result of a substraction (equals 0). if there isn't
                                         // any, seta flag for first_update_odometry
  cmd_last_stamp_ = ros::Time(0);        // maybe it is better to set it to 0, so if no cmd is received
  e_stop_last_stamp_ = ros::Time(0);
  // TODO: service to restart odometry?
  odom_ = nav_msgs::Odometry();
  received_cmd_ = geometry_msgs::Twist();
  robot_pose_ = geometry_msgs::Pose2D();

  cmd_watchdog_timedout_ = true;
}

/**
 * \brief Stops controller
 * \param time Current time
 */
void SummitXLController::stopping(const ros::Time& time)
{
  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "Stopping!");
}

void SummitXLController::update(const ros::Time& time, const ros::Duration& period)
{
  limitCommand(period.toSec());
  // read joint states:
  //  - convert wheel angular velocity to linear velocity
  //  - normalize motor wheel angular position between [-pi, pi] (only needed
  //  for simulation)
  //  - round to 0 if values are below and epsylon
  //  - update joint_states_history_
  readJointStates();

  // update odometry info:
  //  - from joint values (closed loop)
  //  - from command (open loop)
  updateOdometry();

  // calculate joint velocity and position references, taking into account some
  // constrains
  updateJointReferences();

  // TODO: rate of odom publishing
  if ((time - odom_last_sent_) > odom_publish_period_)
  {
    odom_last_sent_ = time;
    publishOdometry();
  }

  cmd_watchdog_timedout_ = ((time - cmd_last_stamp_) > cmd_watchdog_duration_);
  imu_watchdog_timedout_ = ((time - imu_last_stamp_) > imu_watchdog_duration_);

  writeJointCommands();
  std_msgs::Bool bool_msg;
  bool_msg.data = enabled_;
  enabled_pub_.publish(bool_msg);

  bool_msg.data = InMotion();
  in_motion_publisher_.publish(bool_msg);

  // TODO: soft brake (slow slowing down) and hard brake (hard slowing down)
}

void SummitXLController::limitCommand(double period)
{
  double vx, vy, w;

  static bool must_print_watchdog_message = false;
  // watchdog?
  if (cmd_watchdog_timedout_)
  {
    if (must_print_watchdog_message)
    {
      ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << ":: watchdog timedout!");
      must_print_watchdog_message = false;
    }

    received_cmd_.linear.x = 0.0;
	received_cmd_.linear.y = 0.0;
	received_cmd_.angular.z = 0.0;
  }
  else
  {		
	  must_print_watchdog_message = true;
	  // e-stop?
	  if(e_stop_ == true)
	  {
		received_cmd_.linear.x = 0.0;
		received_cmd_.linear.y = 0.0;
		received_cmd_.angular.z = 0.0;
		ROS_WARN_STREAM_THROTTLE_NAMED(30, controller_name_, controller_name_ << ":: E-STOP active!");
	  }
	  else
	  {
		  if((ros::Time::now() - e_stop_last_stamp_).toSec() < 5.0)
		  {
			ROS_WARN_STREAM_THROTTLE_NAMED(2, controller_name_, controller_name_ << ":: limiting speed ref after a E-STOP event!");
			received_cmd_.linear.x = 0.0;
			received_cmd_.linear.y = 0.0;
			received_cmd_.angular.z = 0.0;
		  }
		// OK!
	  }
  }
  
  double accel_x = (received_cmd_.linear.x - current_cmd_.linear.x) / period;
  double accel_y = (received_cmd_.linear.y - current_cmd_.linear.y) / period;

  double total_accel = std::sqrt(accel_x * accel_x + accel_y * accel_y);
  if (total_accel > linear_acceleration_limit_)
  {
    accel_x = accel_x * linear_acceleration_limit_ / total_accel;
    accel_y = accel_y * linear_acceleration_limit_ / total_accel;
  }

  vx = current_cmd_.linear.x + accel_x * period;
  vy = current_cmd_.linear.y + accel_y * period;

  double total_vel = std::sqrt(vx * vx + vy * vy);
  if (total_vel > linear_speed_limit_)
  {
    vx = vx * linear_speed_limit_ / total_vel;
    vy = vy * linear_speed_limit_ / total_vel;
  }

  // If desired speed is zero, does not apply deceleration
  if (received_cmd_.linear.x == 0.0)
  {
    if (hard_brake_ == true)
        current_cmd_.linear.x = 0.0;
    else
        current_cmd_.linear.x = vx;
  }
  else
  {
    current_cmd_.linear.x = vx;
  }
  if (received_cmd_.linear.y == 0.0)
  {
    if (hard_brake_ == true)
        current_cmd_.linear.y = 0.0;
    else
        current_cmd_.linear.y = vy;
  }
  else
  {
    current_cmd_.linear.y = vy;
  }

  double accel_w = (received_cmd_.angular.z - current_cmd_.angular.z) / period;
  if (std::abs(accel_w) > angular_acceleration_limit_)
    accel_w = sign(accel_w) * angular_acceleration_limit_;

  w = current_cmd_.angular.z + accel_w * period;
  if (std::abs(w) > angular_speed_limit_)
    w = sign(w) * angular_speed_limit_;

  if (received_cmd_.angular.z == 0.0)
  {
    if (hard_brake_ == true)
        current_cmd_.angular.z = 0;
    else
        current_cmd_.angular.z = w;
  }
  else
  {
    current_cmd_.angular.z = w;
  }
}

void SummitXLController::readJointStates()
{
  // read wheel velocity: convert from angular to linear
  for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++)
  {
    joint_states_[i] = normToZero(joints_[i].getVelocity() * (wheel_diameter_ / 2.0));
  }

  for (size_t i = 0; i < NUMBER_OF_JOINTS; i++)
  {
    joint_states_history_[i].push_back(joint_states_[i]);
    double sum = std::accumulate(joint_states_history_[i].begin(), joint_states_history_[i].end(), 0.0);
    joint_states_mean_[i] = sum / joint_states_history_[i].size();
  }
}

void SummitXLController::writeJointCommands()
{
  // set joint_commands_ to the values that must be sent to the actuators.
  // joint_commands_[i] can be, sorted by priority (from more to less):
  //   (1) if watchdog has timedout, vel = 0, pos is not sent
  //   (2) if motorowheels are not in position, vel = 0, pos is sent
  //   (3) default: vel and pos are sent
  for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++)
    joint_commands_[i] = joint_references_[i];

  // send commands to actuators
  for (size_t i = 0; i < NUMBER_OF_JOINTS; i++)
    joints_[i].setCommand(joint_commands_[i]);

  std::ostringstream oss;
  oss << "commands:";
  for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++)
  {
    oss << " wheel " << i << " : " << joint_commands_[i];
  }

  // ROS_INFO_STREAM_THROTTLE(1, oss.str());
}

void SummitXLController::updateJointReferences()
{
  // Speed references for motor control
  std::vector<double> q;
  q.resize(4);
  if (enabled_)
  {
    double vx = current_cmd_.linear.x;
    double vy = 0.0;  // skid by default
    if (active_kinematic_mode_ == MECANUM_STEERING)
      vy = current_cmd_.linear.y;
    double w = current_cmd_.angular.z;

    // Vehicle characteristics
    double L = wheel_base_;
    double W = track_width_;
    double lab = L / 2.0 + W / 2.0;

    q[0] =
        (vx + vy + lab * w) / (wheel_diameter_ / 2.0);  // divide by (wheel_diameter_/2.0) to convert from m/s to rad/s
    q[1] = (vx - vy - lab * w) / (wheel_diameter_ / 2.0);
    q[2] = (vx + vy - lab * w) / (wheel_diameter_ / 2.0);
    q[3] = (vx - vy + lab * w) / (wheel_diameter_ / 2.0);

    // joint velocity references are scaled so each wheel does not exceed it's
    // maximum velocity
    setJointVelocityReferenceBetweenLimits(q);
    // ROS_INFO_THROTTLE(1,"q1234=(%5.2f, %5.2f, %5.2f, %5.2f)",
    // q[0],q[1],q[2],q[3]);
  }
  else
  {
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    ROS_WARN_STREAM_THROTTLE_NAMED(10, controller_name_, controller_name_ << "::updateJointReferences: controller is "
                                                                             "disabled");
  }

  // Motor control actions
  // Axis are not reversed in the omni (swerve) configuration
  joint_references_[FRONT_RIGHT_TRACTION_JOINT] = q[0];
  joint_references_[FRONT_LEFT_TRACTION_JOINT] = q[1];
  joint_references_[BACK_LEFT_TRACTION_JOINT] = q[2];
  joint_references_[BACK_RIGHT_TRACTION_JOINT] = q[3];
}

void SummitXLController::updateOdometry()
{
  // Linear speed of each wheel
  double vx = 0.0, vy = 0.0, w = 0.0;
  double frw1 = joint_states_[FRONT_RIGHT_TRACTION_JOINT];
  double flw2 = joint_states_[FRONT_LEFT_TRACTION_JOINT];
  double blw3 = joint_states_[BACK_LEFT_TRACTION_JOINT];
  double brw4 = joint_states_[BACK_RIGHT_TRACTION_JOINT];

  /*ROS_INFO_THROTTLE(1,"joints vel:      [%.3lf, %.3lf, %.3lf, %.3lf]",
    joint_states_[FRONT_RIGHT_TRACTION_JOINT], joint_states_[FRONT_LEFT_TRACTION_JOINT],
    joint_states_[BACK_LEFT_TRACTION_JOINT],joint_states_[BACK_RIGHT_TRACTION_JOINT]);
  ROS_INFO_THROTTLE(1,"joints mean vel: [%.3lf, %.3lf, %.3lf, %.3lf]", joint_states_mean_[FRONT_RIGHT_TRACTION_JOINT],
    joint_states_mean_[FRONT_LEFT_TRACTION_JOINT], joint_states_mean_[BACK_LEFT_TRACTION_JOINT],
  joint_states_mean_[BACK_RIGHT_TRACTION_JOINT]);*/

  if (fabs(joint_states_mean_[FRONT_RIGHT_TRACTION_JOINT]) < joint_speed_noise_threshold_)
    frw1 = 0.0;
  if (fabs(joint_states_mean_[FRONT_LEFT_TRACTION_JOINT]) < joint_speed_noise_threshold_)
    flw2 = 0.0;
  if (fabs(joint_states_mean_[BACK_LEFT_TRACTION_JOINT]) < joint_speed_noise_threshold_)
    blw3 = 0.0;
  if (fabs(joint_states_mean_[BACK_RIGHT_TRACTION_JOINT]) < joint_speed_noise_threshold_)
    brw4 = 0.0;

  // Get real freq.
  ros::Time current_time = ros::Time::now();
  double seconds_since_last_update = (current_time - odom_last_update_).toSec();
  odom_last_update_ = current_time;

  // Compute Position
  double fSamplePeriod = seconds_since_last_update;

  if (active_kinematic_mode_ == MECANUM_STEERING)
  {
    // Vehicle characteristics
    double L = wheel_base_;
    double W = track_width_;
    double lab = L / 2.0 + W / 2.0;

    double R = 1.0;  // these joint_states are already converted to [m/s]
    vx = (R / 4.0) * (frw1 + flw2 + blw3 + brw4);
    vy = (R / 4.0) * (frw1 - flw2 + blw3 - brw4);
    w = (R / (4.0 * lab)) * (frw1 - flw2 - blw3 + brw4);
  }
  else
  {
    double v_left_mps, v_right_mps;
    v_left_mps = ((flw2 + blw3) / 2.0);
    v_right_mps = ((frw1 + brw4) / 2.0);

    double fDistanceBetweenWheels = 0.92;
    vx = (v_right_mps + v_left_mps) / 2.0;  // m/s
    vy = 0.0;
    w = (v_right_mps - v_left_mps) / fDistanceBetweenWheels;  // rad/s
  }

  // robot_pose_.x += cos(robot_pose_.theta) * vx * fSamplePeriod + cos(M_PI_2 +
  // robot_pose_.theta) * vy * fSamplePeriod;
  // robot_pose_.y += sin(robot_pose_.theta) * vx * fSamplePeriod + sin(M_PI_2 +
  // robot_pose_.theta) * vy * fSamplePeriod;
  robot_pose_.x += cos(robot_pose_.theta) * vx * fSamplePeriod - sin(robot_pose_.theta) * vy * fSamplePeriod;
  robot_pose_.y += sin(robot_pose_.theta) * vx * fSamplePeriod + cos(robot_pose_.theta) * vy * fSamplePeriod;

  // TODO - Check if bYawSensor is necessary.
  if (!imu_watchdog_timedout_)
    robot_pose_.theta = imu_yaw_;
  else
    robot_pose_.theta += w * fSamplePeriod;

  // ROS_INFO("Odom estimated x=%5.2f  y=%5.2f a=%5.2f", robot_pose_px_,
  // robot_pose_py_, robot_pose_pa_);

  tf::Quaternion qt;
  tf::Vector3 vt;
  qt.setRPY(0, 0, robot_pose_.theta);
  vt = tf::Vector3(robot_pose_.x, robot_pose_.y, 0);

  odom_.header.stamp = ros::Time::now();
  odom_.header.frame_id = odom_frame_;
  odom_.child_frame_id = robot_base_frame_;

  odom_.pose.pose.position.x = vt.x();
  odom_.pose.pose.position.y = vt.y();
  odom_.pose.pose.position.z = vt.z();

  odom_.pose.pose.orientation.x = qt.x();
  odom_.pose.pose.orientation.y = qt.y();
  odom_.pose.pose.orientation.z = qt.z();
  odom_.pose.pose.orientation.w = qt.w();

  odom_.twist.twist.linear.x = vx;
  odom_.twist.twist.linear.y = vy;
  odom_.twist.twist.angular.z = w;
}

bool SummitXLController::InMotion()
{
  if (odom_.twist.twist.linear.x == 0.0 and odom_.twist.twist.linear.y == 0.0 and odom_.twist.twist.angular.z == 0.0)
    return false;
  else
    return true;
}

void SummitXLController::cmdVelCallback(const geometry_msgs::Twist::ConstPtr& cmd_msg)
{
  ROS_DEBUG_STREAM_NAMED(controller_name_, "Received command: (" << cmd_msg->linear.x << ")"
                                                                 << ", " << cmd_msg->linear.y << ", "
                                                                 << cmd_msg->angular.z << ")");
  received_cmd_ = *cmd_msg;

  cmd_last_stamp_ = ros::Time::now();
}

void SummitXLController::imuCallback(const sensor_msgs::ImuConstPtr& msg)
{
  double roll_msg, pitch_msg, yaw_msg;
  tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
  tf::Matrix3x3(q).getRPY(roll_msg, pitch_msg, yaw_msg);

  if (bFirstYawRead_)
    delta_yaw_ = yaw_msg - last_yaw_;

  imu_last_stamp_ = ros::Time::now();

  // Store the first value received
  if (!bFirstYawRead_)
  {
    init_yaw_ = yaw_msg;
    bFirstYawRead_ = true;
    motion_yaw_ = init_yaw_;  //
  }

  // Compute angle according to the init value
  last_yaw_ = yaw_msg;

  if (motion_odometry_)
  {
    // Integrate yaw only if robot is in motion (avoids integrating drift when
    // robot is stopped for long periods)
    bool in_motion = this->InMotion();
    if (in_motion)
      motion_yaw_ += delta_yaw_;
    imu_yaw_ = radnorm2(motion_yaw_);
  }
  else
  {
    // Compute yaw just according to imu value (takes into account external
    // actions on the robot)
    imu_yaw_ = radnorm2(last_yaw_ - init_yaw_);
  }

  // Use this value (note that if we use imu, this function is used only as a
  // storage
  bYawSensor_ = true;
}

void SummitXLController::eStopCallback(const std_msgs::BoolConstPtr& msg)
{
  e_stop_ = msg->data;
  if(e_stop_ == true)
  {
	e_stop_last_stamp_ = ros::Time::now();
  }
}

// Service GetMode
bool SummitXLController::srvCallback_GetMode(robotnik_msgs::get_mode::Request& request,
                                             robotnik_msgs::get_mode::Response& response)
{
  response.mode = this->active_kinematic_mode_;
  return true;
}

// Service SetOdometry
bool SummitXLController::srvCallback_SetOdometry(robotnik_msgs::set_odometry::Request& request,
                                                 robotnik_msgs::set_odometry::Response& response)
{
  // ROS_INFO("SummitXLController::srvCallback_SetOdometry: request -> x = %f, y
  // = %f, a = %f", request.x, request.y, request.orientation);
  robot_pose_.x = odom_.pose.pose.position.x = request.x;
  robot_pose_.y = odom_.pose.pose.position.y = request.y;

  tf::Quaternion q;
  // q.setEuler(request.orientation, 0.0, 0.0);
  q.setRPY(0, 0, request.orientation);
  // tf::Quaternion q = tf::createQuaternionMsgFromYaw(request.orientation);

  // odom_.pose.pose.orientation =
  // tf::createQuaternionMsgFromYaw(request.orientation);
  odom_.pose.pose.orientation.x = q.x();
  odom_.pose.pose.orientation.y = q.y();
  odom_.pose.pose.orientation.z = q.z();
  odom_.pose.pose.orientation.w = q.w();

  robot_pose_.theta = request.orientation;

  // store new orientation
  init_yaw_ = last_yaw_;
  motion_yaw_ = request.orientation;

  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "::srvCallback_SetOdometry: q.x=" << q.x() << ", q.y="
                                                           << q.y() << ", q.z=" << q.z() << ", q.w=" << q.w());

  response.ret = true;
  return true;
}

bool SummitXLController::enableSrvCallback(robotnik_msgs::enable_disable::Request& req,
                                           robotnik_msgs::enable_disable::Response& res)
{
  enabled_ = req.value;

  res.ret = true;

  return true;
}

void SummitXLController::publishOdometry()
{
  odom_publisher_.publish(odom_);

  tf::Quaternion qt = tf::Quaternion(odom_.pose.pose.orientation.x, odom_.pose.pose.orientation.y,
                                     odom_.pose.pose.orientation.z, odom_.pose.pose.orientation.w);
  tf::Vector3 vt = tf::Vector3(odom_.pose.pose.position.x, odom_.pose.pose.position.y, odom_.pose.pose.position.z);

  tf::Transform base_footprint_to_odom(qt, vt);
  if (this->odom_broadcast_tf_)
  {
    transform_broadcaster_->sendTransform(
        tf::StampedTransform(base_footprint_to_odom, ros::Time::now(), odom_frame_, robot_base_frame_));
  }
}

void SummitXLController::setJointVelocityReferenceBetweenLimits(std::vector<double>& wheel_speed)
{
  double max_scale_factor = 1.0;

  for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++)
  {
    double lower_limit = joint_limits_[i].first;
    double upper_limit = joint_limits_[i].second;

    double lower_scale_factor, upper_scale_factor;
    lower_scale_factor = upper_scale_factor = 1.0;

    if (wheel_speed[i] < lower_limit)
      lower_scale_factor = std::abs(wheel_speed[i] / lower_limit);
    if (upper_limit < wheel_speed[i])
      upper_scale_factor = std::abs(wheel_speed[i] / upper_limit);

    max_scale_factor = std::max(max_scale_factor, std::max(lower_scale_factor, upper_scale_factor));
  }

  for (size_t i = BEGIN_TRACTION_JOINT; i < END_TRACTION_JOINT; i++)
  {
    wheel_speed[i] /= max_scale_factor;
  }
}
}
