#include <robotnik_pad_plugins/ackermann_movement_plugin.h>

namespace pad_plugins
{
PadPluginAckermannMovement::PadPluginAckermannMovement()
{
}

PadPluginAckermannMovement::~PadPluginAckermannMovement()
{
}

void PadPluginAckermannMovement::initialize(const ros::NodeHandle& nh, const std::string& plugin_ns)
{
  bool required = true;
  pnh_ = ros::NodeHandle(nh, plugin_ns);
  nh_ = ros::NodeHandle();
  button_dead_man_ = 5;
  readParam(pnh_, "config/button_deadman", button_dead_man_, button_dead_man_, required);
  axis_linear_x_ = 1;
  readParam(pnh_, "config/axis_linear_x", axis_linear_x_, axis_linear_x_, required);
  axis_linear_y_ = 0;
  readParam(pnh_, "config/axis_linear_y", axis_linear_y_, axis_linear_y_, required);
  axis_angular_z_ = 2;
  readParam(pnh_, "config/axis_angular_z", axis_angular_z_, axis_angular_z_, required);
  button_speed_up_ = 3;
  readParam(pnh_, "config/button_speed_up", button_speed_up_, button_speed_up_, required);
  button_speed_down_ = 1;
  readParam(pnh_, "config/button_speed_down", button_speed_down_, button_speed_down_, required);
  max_speed_ = 1.5;
  readParam(pnh_, "max_speed", max_speed_, max_speed_, required);
  max_steering_angle_ = 1.57;
  readParam(pnh_, "max_steering_angle", max_steering_angle_, max_steering_angle_, required);
  cmd_topic_vel_ = "cmd_vel";
  readParam(pnh_, "cmd_topic_vel", cmd_topic_vel_, cmd_topic_vel_, required);

  // Publishers
  ackermann_pub_ = nh_.advertise<ackermann_msgs::AckermannDrive>(cmd_topic_vel_, 10);
  pad_status_pub_ = pnh_.advertise<robotnik_pad_msgs::MovementStatus>("status", 10);

  // initialize variables
  current_velocity_level_ = 0.1;
  velocity_level_step_ = 0.1;
  max_velocity_level_ = 1.0;
  min_velocity_level_ = 0.1;
  cmd_ackermann_ = ackermann_msgs::AckermannDrive();
  movement_status_msg_ = robotnik_pad_msgs::MovementStatus();
}

void PadPluginAckermannMovement::execute(const std::vector<Button>& buttons, std::vector<float>& axes)
{
  if (buttons[button_dead_man_].isPressed())
  {
    if (buttons[button_speed_down_].isReleased())
    {
      current_velocity_level_ = std::max(min_velocity_level_, current_velocity_level_ - velocity_level_step_);
      ROS_INFO("PadPluginAckermannMovement::execute: speed down -> velocity level = %.1f%%", current_velocity_level_ * 100.0);
    }
    else if (buttons[button_speed_up_].isReleased())
    {
      current_velocity_level_ = std::min(max_velocity_level_, current_velocity_level_ + velocity_level_step_);
      ROS_INFO("PadPluginAckermannMovement::execute: speed up -> velocity level = %.1f%%", current_velocity_level_ * 100.0);
    }

    cmd_ackermann_.speed = current_velocity_level_ * max_speed_ * axes[axis_linear_x_];
    cmd_ackermann_.steering_angle = max_steering_angle_ * axes[axis_angular_z_];

    ackermann_pub_.publish(cmd_ackermann_);
  }
  /*else if (buttons[button_dead_man_].isReleased())
  {
    cmd_ackermann_.speed = 0.0;
    cmd_ackermann_.steering_angle = 0.0;

    ackermann_pub_.publish(cmd_ackermann_);
  }*/

  movement_status_msg_.velocity_level = current_velocity_level_ * 100;
  pad_status_pub_.publish(movement_status_msg_);
}
}  // namespace pad_plugins
