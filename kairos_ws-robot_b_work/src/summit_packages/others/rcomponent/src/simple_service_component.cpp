/*!
 *  \brief Example to use Service Servers and Clients
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
#include <std_srvs/Empty.h>

using namespace std;

//! Test class that inherits RComponent
class SimpleServiceComponent : public rcomponent::RComponent
{
private:
  //! Name of the topic
  string service_server_name_;
  string service_client_name_;

  //! Service server
  ros::ServiceServer service_server_;
  ros::ServiceClient service_client_;

public:
  // Constructor
  SimpleServiceComponent(ros::NodeHandle h) : RComponent(h)
  {
  }

protected:
  // Inherits from RComponent
  int rosSetup()
  {
    if (RComponent::rosSetup() == rcomponent::OK)
    {
      pnh_.param<string>("service_server_name", service_server_name_, "service_server");
      nh_.param<string>("service_client_name", service_client_name_, "sevice_client");

      service_server_ = pnh_.advertiseService(service_server_name_, &SimpleServiceComponent::serviceServerCb, this);
      service_client_ = nh_.serviceClient<std_srvs::Empty>(service_client_name_);
    }
  }

  // Inherits from RComponent
  int rosShutdown()
  {
    if (RComponent::rosShutdown() == rcomponent::OK)
    {
      RCOMPONENT_INFO("rosShutdown");
      service_server_.shutdown();
    }
  }

  // Callback handler for the service server
  bool serviceServerCb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
  {
    RCOMPONENT_INFO("Received server");
    std_srvs::Empty service;
    if (service_client_.call(service))
    {
      RCOMPONENT_INFO("calling service");
    }
    else
    {
      RCOMPONENT_ERROR("Error connecting service %s", service_client_name_.c_str());
    }

    return true;
  }
};

// MAIN
int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_service_component");

  ros::NodeHandle n;

  SimpleServiceComponent simple(n);

  simple.start();

  return (0);
}
