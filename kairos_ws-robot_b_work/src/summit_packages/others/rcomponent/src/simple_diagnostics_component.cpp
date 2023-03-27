/*!
 *  \brief Example to use ros_diagnostics
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

#include "diagnostic_msgs/DiagnosticStatus.h"
#include "diagnostic_updater/diagnostic_updater.h"
#include "diagnostic_updater/update_functions.h"
#include "diagnostic_updater/DiagnosticStatusWrapper.h"
#include "diagnostic_updater/publisher.h"

using namespace std;

//! Test class that inherits RComponent
class SimpleDiagnosticsComponent : public rcomponent::RComponent
{
private:
  //! Name of the topic
  string service_server_name_;

  double min_freq;

  double min_topic_freq, max_topic_freq;
  //! Service server
  ros::ServiceServer service_server_;

  //! General status diagnostic updater
  diagnostic_updater::Updater* diagnostic_;
  //! Component frequency diagnostics
  diagnostic_updater::FrequencyStatus* freq_diag_;

  diagnostic_updater::FunctionDiagnosticTask* command_freq_;

  diagnostic_updater::HeaderlessTopicDiagnostic* subs_command_freq;

  // freq_diag_(diagnostic_updater::FrequencyStatusParam(&desired_freq_, &desired_freq_, 0.05)   ),
public:
  // Constructor
  SimpleDiagnosticsComponent(ros::NodeHandle h) : RComponent(h)
  {
    min_freq = 1.0;
    min_topic_freq = 1.0;
    max_topic_freq = 5.0;
  }

protected:
  // Inherits from RComponent
  int rosSetup()
  {
    if (RComponent::rosSetup() == rcomponent::OK)
    {
      pnh_.param<string>("service_server_name", service_server_name_, "service_server");

      service_server_ = pnh_.advertiseService(service_server_name_, &SimpleDiagnosticsComponent::serviceServerCb, this);

      diagnostic_ = new diagnostic_updater::Updater();
      // Sets the desired frequency
      freq_diag_ = new diagnostic_updater::FrequencyStatus(
          diagnostic_updater::FrequencyStatusParam(&min_freq, &desired_freq_, 0.1));

      command_freq_ = new diagnostic_updater::FunctionDiagnosticTask(
          "Topic frequency check", boost::bind(&SimpleDiagnosticsComponent::checkTopicSubscriber, this, _1));

      subs_command_freq = new diagnostic_updater::HeaderlessTopicDiagnostic(
          "state", *diagnostic_, diagnostic_updater::FrequencyStatusParam(&min_topic_freq, &max_topic_freq, 0.1, 10));

      // Component frequency diagnostics
      diagnostic_->setHardwareID("rcomponent");
      diagnostic_->add("Test", this, &SimpleDiagnosticsComponent::diagnosticUpdate);
      diagnostic_->add(*freq_diag_);
      diagnostic_->add(*command_freq_);
    }
  }

  // Inherits from RComponent
  int rosShutdown()
  {
    if (RComponent::rosShutdown() == rcomponent::OK)
    {
      RCOMPONENT_INFO("");
      service_server_.shutdown();
    }
  }

  // Callback handler for the service server
  bool serviceServerCb(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
  {
    RCOMPONENT_INFO("Received server");

    return true;
  }

  // Callback to update the component diagnostic
  void diagnosticUpdate(diagnostic_updater::DiagnosticStatusWrapper& stat)
  {
    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "Everything OK!");
    // stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "Watch out!");
    // stat.summary(diagnostic_msgs::DiagnosticStatus::ERROR, "Error!");
    stat.add("State", getStateString());
  }

  void allState()
  {
    freq_diag_->tick();
    // subs_command_freq->tick();
    diagnostic_->update();
    rosPublish();
  }

  void checkTopicSubscriber(diagnostic_updater::DiagnosticStatusWrapper& stat)
  {
    stat.summary(diagnostic_msgs::DiagnosticStatus::WARN, "Topic is not receiving commands");
  }
};

// MAIN
int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_diagnostics_component");

  ros::NodeHandle n;

  SimpleDiagnosticsComponent simple(n);

  simple.start();

  return (0);
}
