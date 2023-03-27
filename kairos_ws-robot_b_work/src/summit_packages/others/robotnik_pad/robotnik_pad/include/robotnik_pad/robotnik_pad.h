
#ifndef _ROBOTNIK_PAD_
#define _ROBOTNIK_PAD_

#include <math.h>

#include <ros/console.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Joy.h>

#include <pluginlib/class_loader.h>
#include <robotnik_pad/generic_pad_plugin.h>

#include <rcomponent/rcomponent.h>
#include <robotnik_pad/button.h>

class RobotnikPad : public rcomponent::RComponent
{
public:
  RobotnikPad(ros::NodeHandle h);
  virtual ~RobotnikPad();

protected:
  /* RComponent stuff */

  virtual int setup();
  //! Setups all the ROS' stuff
  virtual int rosSetup();
  //! Shutdowns all the ROS' stuff
  virtual int rosShutdown();
  //! Reads data a publish several info into different topics
  virtual void rosPublish();
  //! Reads params from params server
  virtual void rosReadParams();
  //! Actions performed on init state
  virtual void initState();
  //! Actions performed on standby state
  virtual void standbyState();
  //! Actions performed on ready state
  virtual void readyState();
  //! Actions performed on the emergency state
  virtual void emergencyState();
  //! Actions performed on Failure state
  virtual void failureState();

  //! joy callback
  void joyCb(const sensor_msgs::Joy::ConstPtr &joy);
  void readPluginsFromParams(const ros::NodeHandle &nh, const std::vector<std::string> &names,
                             std::map<std::string, std::string> &plugins_definitions);

public:
protected:
  /* ROS stuff */

  // Subscribers
  ros::Subscriber joy_sub_;

  // JOYSTICK
  //! Current number of buttons of the joystick
  int num_of_buttons_;
  int num_of_axes_;
  std::string pad_type_;
  std::string joy_topic_;

  //! Vector to save the axis values
  std::vector<float> axes_;
  //! Vector to save and control the axis values
  std::vector<Button> buttons_;

  pluginlib::ClassLoader<pad_plugins::GenericPadPlugin> *pad_plugins_loader_;
  std::vector<pad_plugins::GenericPadPlugin::Ptr> plugins_;

  std::map<std::string, std::string> plugins_from_params_;
};

#endif // _ROBOTNIK_PAD_
