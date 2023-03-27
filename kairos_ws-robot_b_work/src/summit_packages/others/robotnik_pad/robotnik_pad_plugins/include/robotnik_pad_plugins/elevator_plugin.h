#ifndef PAD_PLUGIN_ELEVATOR_H_
#define PAD_PLUGIN_ELEVATOR_H_

#include <robotnik_msgs/SetElevator.h>
#include <robotnik_pad/generic_pad_plugin.h>

namespace pad_plugins
{
class PadPluginElevator : public GenericPadPlugin
{
public:
  PadPluginElevator();
  ~PadPluginElevator();

  virtual void initialize(const ros::NodeHandle &nh, const std::string &plugin_ns);
  virtual void execute(const std::vector<Button> &buttons, std::vector<float> &axes);

protected:
  int button_dead_man_;
  double axis_elevator_;

  std::string elevator_service_name_;
  ros::ServiceClient set_elevator_client_;
};
}  // namespace pad_plugins
#endif  // PAD_PLUGIN_ELEVATOR_H_