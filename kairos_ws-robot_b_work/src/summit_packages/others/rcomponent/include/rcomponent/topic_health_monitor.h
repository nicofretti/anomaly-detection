#ifndef _RCOMPONENT_TOPIC_HEALTH_MONITOR_
#define _RCOMPONENT_TOPIC_HEALTH_MONITOR_

#include <ros/ros.h>

namespace rcomponent
{
//! Class to check the health of a topic

class TopicHealthMonitor
{
private:
  //! timeout to consider that the topic is not receiving
  double topic_timeout_;
  //! variable to know if stop receiving this topic is critical for the node
  bool required_;
  //! saves the ros time of the last msg received
  ros::Time last_msg_time_;
  //! reference to the subscriber
  ros::Subscriber* sub_;

public:
  ~TopicHealthMonitor()
  {
  }

  TopicHealthMonitor(ros::Subscriber* subscriber = 0, double timeout = 5.0, bool required = true)
  {
    sub_ = subscriber;
    topic_timeout_ = timeout;
    last_msg_time_ = ros::Time(0);
    required_ = required;
  }

  //!
  void tick()
  {
    last_msg_time_ = ros::Time::now();
  }
  //! returns true if the topic is receiving data
  bool isReceiving()
  {
    return isReceiving(topic_timeout_);
  }

  bool isReceiving(double timeout)
  {
    if (sub_ == 0 or sub_->getNumPublishers() == 0)
      return false;

    if ((ros::Time::now() - last_msg_time_).toSec() > timeout)
      return false;

    return true;
  }

  //! returns true if the topic is necessary for node
  bool isRequired()
  {
    return required_;
  }

  void setSubscriber(ros::Subscriber* subscriber)
  {
    sub_ = subscriber;
  }

  ros::Subscriber* getSubscriber()
  {
    return sub_;
  }

  void setTimeout(double timeout)
  {
    topic_timeout_ = timeout;
  }

  double getTimeout()
  {
    return topic_timeout_;
  }
};  // class
}  // namespace
#endif  // _RCOMPONENT_TOPIC_HEALTH_MONITOR_
