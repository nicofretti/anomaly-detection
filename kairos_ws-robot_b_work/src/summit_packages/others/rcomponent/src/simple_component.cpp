/*!
 *  \brief Example to use Publishers and Subscribers
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <rcomponent/rcomponent.h>
#include <std_msgs/String.h>

using namespace std;

//! Test class that inherits RComponent
class SimpleComponent : public rcomponent::RComponent
{
private:
  //! Name of the topic to publish
  string topic_name_;
  //! Name of the topic to subscribe
  string subs_topic_name_;

  //! Publishes the status of the robot
  ros::Publisher status_pub_;

  //! Subscriber
  ros::Subscriber sub_;

public:
  // Constructor
  SimpleComponent(double hz, ros::NodeHandle h) : RComponent(h)
  {
  }

private:
  // Inherits from RComponent
  int rosSetup()
  {
    if (RComponent::rosSetup() == rcomponent::OK)
    {
      pnh_.param<string>("topic_name", topic_name_, "status");
      pnh_.param<string>("subs_topic_name", subs_topic_name_, "any");

      status_pub_ = pnh_.advertise<std_msgs::String>(topic_name_, 1);

      // topic, queue, callback
      sub_ = nh_.subscribe(subs_topic_name_, 10, &SimpleComponent::callback, this);
    }
  }
  // Inherits from RComponent
  int rosShutdown()
  {
    if (RComponent::rosShutdown() == rcomponent::OK)
    {
      RCOMPONENT_INFO("");
    }
  }

  //
  void rosPublish()
  {
    std_msgs::String msg;
    msg.data = "Simple";

    status_pub_.publish(msg);
  }

  // Callback handler associated with the subscriber
  void callback(const std_msgs::StringConstPtr& message)
  {
    RCOMPONENT_INFO("Received msg: %s", message->data.c_str());
  }
};

// MAIN
int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_component");

  ros::NodeHandle n;
  SimpleComponent simple(2.0, n);

  simple.start();

  return (0);
}
