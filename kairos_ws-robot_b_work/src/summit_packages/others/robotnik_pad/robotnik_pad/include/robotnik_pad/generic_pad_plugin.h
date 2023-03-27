#ifndef GENERIC_PAD_PLUGIN_H_
#define GENERIC_PAD_PLUGIN_H_

#include <robotnik_pad/button.h>
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <vector>

namespace pad_plugins
{
class GenericPadPlugin
{
public:
  typedef boost::shared_ptr<GenericPadPlugin> Ptr;

public:
  virtual void initialize(const ros::NodeHandle& nh, const std::string& plugin_ns) = 0;
  virtual void execute(const std::vector<Button>& buttons, std::vector<float>& axes) = 0;
  virtual ~GenericPadPlugin()
  {
  }

  //! Reads a parameter from the param server, and shows a message if parameter is not set
  template <typename T>
  bool readParam(const ros::NodeHandle& h, const std::string& name, T& value, const T& default_value,
                 bool required = false)
  {
    // parameter is read from node handle passed
    // required defines logger lever: if true, will show an error. if false, a warning
    if (h.hasParam(name) == false)
    {
      if (required == false)
      {
        ROS_WARN_STREAM("No parameter \"" << h.resolveName(name) << "\", using default value: " << default_value
                                          << ".");
      }
      else
      {
        ROS_ERROR_STREAM("No parameter \"" << h.resolveName(name) << "\", using default value: " << default_value
                                           << ".");
      }
      value = default_value;
      return false;
    }
    h.param<T>(name, value, default_value);
    return true;
  }

protected:
  GenericPadPlugin()
  {
  }

protected:
  std::vector<Button> buttons_;
  std::vector<float> axes_;
  ros::NodeHandle pnh_;
  ros::NodeHandle nh_;
};
}
#endif  // GENERIC_PAD_PLUGIN_H_