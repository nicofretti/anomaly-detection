#include <robotnik_pad/robotnik_pad.h>

RobotnikPad::RobotnikPad(ros::NodeHandle h) : rcomponent::RComponent(h)
{
  component_name.assign(pnh_.getNamespace());
  rosReadParams();
}

RobotnikPad::~RobotnikPad()
{
}

int RobotnikPad::rosSetup()
{
  RComponent::rosSetup();

/*
  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &RobotnikPad::joyCb, this);
  addTopicsHealth(&joy_sub_, "joy", 5);
*/
  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>(joy_topic_, 10, &RobotnikPad::joyCb, this);
  addTopicsHealth(&joy_sub_, joy_topic_, 5);
}

int RobotnikPad::rosShutdown()
{
  RComponent::rosShutdown();
}

void RobotnikPad::rosReadParams()
{
  bool required = true;
  bool not_required = false;

  num_of_buttons_ = 14;
  num_of_axes_ = 4;
  // JOYSTICK PAD TYPE
  readParam(pnh_, "pad/num_of_buttons", num_of_buttons_, num_of_buttons_, required);
  readParam(pnh_, "pad/num_of_axes", num_of_axes_, num_of_axes_, required);
  readParam(pnh_, "pad/type", pad_type_, "ps4", required);

  joy_topic_ = "joy";
  readParam(pnh_, "pad/joy_topic", joy_topic_,joy_topic_, not_required);

  std::vector<std::string> plugins_names;

  readParam(pnh_, "plugins", plugins_names, plugins_names);
  readPluginsFromParams(pnh_, plugins_names, plugins_from_params_);
}

int RobotnikPad::setup()
{
}

void RobotnikPad::rosPublish()
{
  RComponent::rosPublish();
}

void RobotnikPad::initState()
{
  // Initialize the buttons and axes
  for (int i = 0; i < num_of_buttons_; i++)
  {
    Button b;
    buttons_.push_back(b);
  }

  for (int i = 0; i < num_of_axes_; i++)
  {
    axes_.push_back(0.0);
  }

  pad_plugins_loader_ = new pluginlib::ClassLoader<pad_plugins::GenericPadPlugin>("robotnik_pad", "pad_plugins::"
                                                                                                  "GenericPadPlugin");

  for (auto& param_plugin : plugins_from_params_)
  {
    pad_plugins::GenericPadPlugin::Ptr plugin;
    try
    {
      plugin = pad_plugins_loader_->createInstance(param_plugin.second);
    }
    catch (pluginlib::PluginlibException& ex)
    {
      RCOMPONENT_ERROR_STREAM("Failed to load plugin " << param_plugin.first << "\" of type \"" << param_plugin.second
                                                       << "." << std::endl
                                                       << "Exception: " << ex.what());
      continue;
    }
    plugin->initialize(pnh_, param_plugin.first);
    plugins_.push_back(plugin);
  }

  switchToState(robotnik_msgs::State::STANDBY_STATE);
}

void RobotnikPad::standbyState()
{
  if (checkTopicsHealth() == false)
  {
    switchToState(robotnik_msgs::State::EMERGENCY_STATE);
  }
  else
  {
    switchToState(robotnik_msgs::State::READY_STATE);
  }
}

void RobotnikPad::readyState()
{
  // If joyCb is not publishing then we change the state to emergency
  if (checkTopicsHealth() == false)
  {
    switchToState(robotnik_msgs::State::EMERGENCY_STATE);
  }

  // Call every plugin
  for (auto& plugin : plugins_)
  {
    plugin->execute(buttons_, axes_);
  }

  for (auto& button : buttons_)
  {
    button.resetReleased();
  }
}

void RobotnikPad::emergencyState()
{
  if (checkTopicsHealth() == true)
  {
    switchToState(robotnik_msgs::State::STANDBY_STATE);
  }
}

void RobotnikPad::failureState()
{
}

void RobotnikPad::joyCb(const sensor_msgs::Joy::ConstPtr& joy)
{
  // Fill in the axes and buttons arrays
  for (int i = 0; i < joy->axes.size(); i++)
  {
    axes_[i] = joy->axes[i];
  }
  for (int i = 0; i < joy->buttons.size(); i++)
  {
    buttons_[i].press(joy->buttons[i]);
  }
  //tickTopicsHealth("joy");
  tickTopicsHealth(joy_topic_);
}

void RobotnikPad::readPluginsFromParams(const ros::NodeHandle& nh, const std::vector<std::string>& names,
                                        std::map<std::string, std::string>& plugins_definitions)
{
  plugins_definitions.clear();

  for (const std::string& name : names)
  {
    if (nh.hasParam(name) == false)
    {
      RCOMPONENT_WARN_STREAM("Cannot load component " << name << " because it doesn't exist in the parameter tree");
      continue;
    }

    std::string type = "";
    bool required = true;
    bool ok = false;

    ok = readParam(nh, name + "/type", type, type, required);

    if (ok == false)
    {
      RCOMPONENT_WARN_STREAM("Cannot load component " << name << " because its type is empty");
      continue;
    }
    if (plugins_definitions.count(name) != 0)
    {
      RCOMPONENT_WARN_STREAM("Already loaded component with name " << name << " of type " << type);
    }
    plugins_definitions[name] = type;
  }

  RCOMPONENT_INFO_STREAM("I have read " << plugins_definitions.size() << " components:");
  for (auto& x : plugins_definitions)
    RCOMPONENT_INFO_STREAM(x.first << ": " << x.second);
}
