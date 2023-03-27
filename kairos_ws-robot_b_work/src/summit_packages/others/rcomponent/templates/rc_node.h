#ifndef _?RC_NODE_
#define _?RC_NODE_

#include <rcomponent/rcomponent.h>

// Insert here general includes:
#include <math.h>

// Insert here msg and srv includes:
#include <std_msgs/String.h>
#include <robotnik_msgs/StringStamped.h>

#include <std_srvs/Trigger.h>

class ?RCNode : public rcomponent::RComponent
{
public:
  ?RCNode(ros::NodeHandle h);
  virtual ~?RCNode();

protected:
  /*** RComponent stuff ***/

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

  /* RComponent stuff !*/

  /* ROS Stuff */

  // Publishers

  //! To publish the basic information
  ros::Publisher status_pub_;
  ros::Publisher status_stamped_pub_;

  //! Subscribers
  ros::Subscriber example_sub_;
  string example_subscriber_name_; // Name of the example_sub_ topic

  //! Services
  ros::ServiceServer example_server_;

  //! Callbacks
  void exampleSubCb(const std_msgs::String::ConstPtr& msg);

  bool exampleServerCb(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response);

  /* ROS stuff !*/

  /* ?RCNode stuff */

  std_msgs::String status_;

  /* ?RCNode stuff !*/


};

#endif  // _?RC_NODE_
