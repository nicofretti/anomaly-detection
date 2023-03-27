/*! \class RComponent
 *  \file RComponent.cpp
 *	\author Robotnik Automation S.L.L
 *	\version 0.1.0
 *	\date 2014
 *  \brief Class to define a standard and shared structure (attributes & methods) for all the components
 * 			Single thread by default
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
#include <robotnik_msgs/State.h>

/*! \fn RComponent::RComponent()
 *  \brief Constructor by default
 *	\param hz as double, sets the desired frequency of the controlthread
 *	\param h as ros::NodeHandle, ROS node handle
*/

namespace rcomponent
{
RComponent::RComponent()
{
  constructed = false;
}

RComponent::RComponent(ros::NodeHandle h) : RComponent::RComponent(h, ros::NodeHandle("~"))
{
  // XXX: this constructor is left to not break legacy code
}

RComponent::RComponent(ros::NodeHandle h, std::string name) : RComponent::RComponent(h, ros::NodeHandle(h, name))
{
}

RComponent::RComponent(ros::NodeHandle h, ros::NodeHandle ph) : nh_(h), pnh_(ph)
{
  init(nh_, pnh_);
}

int RComponent::init(ros::NodeHandle h)
{
  return init(h, ros::NodeHandle("~"));
}

int RComponent::init(ros::NodeHandle h, std::string name)
{
  return init(h, ros::NodeHandle(h, name));
}

int RComponent::init(ros::NodeHandle h, ros::NodeHandle ph)
{
  // if (ros_initialized or initialized or running)
  // {
  //   RCOMPONENT_INFO("Already initialized");

  //   return INITIALIZED;
  // }
  // desired_freq_ = 0;

  nh_ = h;
  pnh_ = ph;

  // Realizar para cada una de las clases derivadas

  // name assignment
  component_name.assign(constructComponentNameFromHandles());

  // Set main flags to false
  ros_initialized = initialized = running = false;

  // reads params from server
  rosReadParams();

  if (desired_freq_ <= 0.0)
    desired_freq_ = DEFAULT_THREAD_DESIRED_HZ;

  state = robotnik_msgs::State::INIT_STATE;

  threadData.pthreadPar.prio = 25;               // Priority level 0[min]-80[max]
  threadData.pthreadPar.clock = CLOCK_REALTIME;  // 0-CLOCK_MONOTONIC 1-CLOCK_REALTIME
  constructed = true;

  t_state_transition_ = ros::Time::now();

  return OK;
}

/*! \fn RComponent::~RComponent()
 * Destructor by default
*/
RComponent::~RComponent()
{
  running = false;
  constructed = false;
}

/*! \fn int RComponent::setup()
 * Configures and initializes the component
 * \return OK
 * \return INITIALIZED if the component is already intialized
 * \return ERROR
*/
int RComponent::setup()
{
  // Checks if has been created with a fully qualified constructor, or with an empty
  // constructor but then init has been called
  if (constructed == false)
  {
    RCOMPONENT_ERROR("Not properly constructed");
    return ERROR;
  }

  // Checks if has been initialized
  if (initialized)
  {
    RCOMPONENT_INFO("Already initialized");

    return INITIALIZED;
  }

  //
  ///////////////////////////////////////////////////
  // Setups the component or another subcomponents if it's necessary //
  ///////////////////////////////////////////////////

  initialized = true;

  return OK;
}

/*! \fn int RComponent::shutDown()
 * Closes and frees the reserved resources
 * \return OK
 * \return ERROR if fails when closes the devices
 * \return RUNNING if the component is running
 * \return NOT_INITIALIZED if the component is not initialized
*/
int RComponent::shutdown()
{
  if (running)
  {
    RCOMPONENT_INFO("Impossible while thread running, first must be stopped");
    return THREAD_RUNNING;
  }
  if (!initialized)
  {
    RCOMPONENT_INFO("Impossible because of it's not initialized");
    return NOT_INITIALIZED;
  }

  //
  ///////////////////////////////////////////////////////
  // ShutDowns another subcomponents if it's necessary //
  ///////////////////////////////////////////////////////

  initialized = false;

  return OK;
}

/*! \fn int RComponent::start()
 * Starts the control operation of the component and its subcomponents in a blocking way
 * \return OK
 * \return RUNNING if it's already running
 * \return NOT_INITIALIZED if the component is not initialized
*/
int RComponent::start()
{
  if (running)
  {
    RCOMPONENT_INFO("Component's thread is already running");
    return THREAD_RUNNING;
  }

  // Performs ROS setup

  int setup_result = rosSetup();

  if (setup_result == rcomponent::ERROR)
    return rcomponent::ERROR;

  RCOMPONENT_INFO("Started");

  running = true;

  switchToState(robotnik_msgs::State::INIT_STATE);
  // Executes the control loop
  controlLoop();

  return OK;
}

/*! \fn int RComponent::asyncStart()
 * Starts the control thread of the component and its subcomponents
 * \return OK
 * \return RUNNING if it's already running
 * \return NOT_INITIALIZED if the component is not initialized
*/
int RComponent::asyncStart()
{
  if (running)
  {
    RCOMPONENT_INFO("Component's thread is already running");
    return THREAD_RUNNING;
  }

  // Performs ROS setup
  int setup_result = rosSetup();

  if (setup_result == rcomponent::ERROR)
    return rcomponent::ERROR;

  pthread_attr_t attr;  // Thread attributed for the component threads spawned in this function

  RCOMPONENT_DEBUG("Launching the thread");
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  running = true;

  switchToState(robotnik_msgs::State::INIT_STATE);
  if (pthread_create(&threadData.pthreadId, &attr, &RComponent::asyncControlLoop, this) != 0)
  {
    RCOMPONENT_ERROR("Could not create ControlThread");
    pthread_attr_destroy(&attr);
    running = false;
    return ERROR;
  }

  return OK;
}
/*! \fn int RComponent::stop()
 * Stops the control thread of the Motors
 * \return OK
 * \return ERROR if it can't be stopped
 * \return THREAD_NOT_RUNNING if the thread is not running
*/
int RComponent::stop()
{
  if (!running)
  {
    RCOMPONENT_INFO("Thread not running");

    return THREAD_NOT_RUNNING;
  }
  //
  ///////////////////////////////////////////////////
  // Stops another subcomponents, if it's necessary //
  ///////////////////////////////////////////////////
  //
  RCOMPONENT_INFO("Stopping the component");

  running = false;

  usleep(100000);

  return OK;
}

/*! \fn int RComponent::isRunning()
 *  Checks if component is running
 *  \return TRUE component is running
 *  \return FALSE component is not running
*/
bool RComponent::isRunning()
{
  return running;
}

void* RComponent::asyncControlLoop(void* context)
{
  ((RComponent*)context)->controlLoop();
  return NULL;
}

/*!	\fn void RComponent::controlLoop()
 *	\brief All core component functionality is contained in this thread.
*/
void RComponent::controlLoop()
{
  RCOMPONENT_INFO("Init");
  ros::Rate r(desired_freq_);
  ros::Time t1, t2;
  while (running && ros::ok())
  {
    t1 = ros::Time::now();

    switch (state)
    {
      case robotnik_msgs::State::INIT_STATE:
        initState();
        break;

      case robotnik_msgs::State::STANDBY_STATE:
        standbyState();
        break;

      case robotnik_msgs::State::READY_STATE:
        readyState();
        break;

      case robotnik_msgs::State::SHUTDOWN_STATE:
        shutdownState();
        break;

      case robotnik_msgs::State::EMERGENCY_STATE:
        emergencyState();
        break;

      case robotnik_msgs::State::FAILURE_STATE:
        failureState();
        break;
    }

    allState();

    ros::spinOnce();
    r.sleep();

    t2 = ros::Time::now();
    try
    {
      real_freq = 1.0 / (t2 - t1).toSec();
    }
    catch (std::runtime_error& ex)
    {
      RCOMPONENT_ERROR("Exception: [%s]", ex.what());
    }
  }

  shutdownState();
  // Performs ROS Shutdown
  rosShutdown();

  RCOMPONENT_INFO("End");
}

/*!	\fn void RComponent::initState()
 *	\brief Actions performed on initial
 * 	Setups the component
*/
void RComponent::initState()
{
  // If component setup is successful goes to STANDBY (or READY) state
  if (setup() != ERROR)
  {
    switchToState(robotnik_msgs::State::STANDBY_STATE);
  }
}

/*!	\fn void RComponent::shutdownState()
 *	\brief Actions performed on Shutdown state
*/
void RComponent::shutdownState()
{
  if (shutdown() == OK)
  {
    switchToState(robotnik_msgs::State::INIT_STATE);
  }
}

/*!	\fn void RComponent::standbyState()
 *	\brief Actions performed on Standby state
*/
void RComponent::standbyState()
{
}

/*!	\fn void RComponent::readyState()
 *	\brief Actions performed on ready state
*/
void RComponent::readyState()
{
}

/*!	\fn void RComponent::EmergencyState()
 *	\brief Actions performed on emergency state
*/
void RComponent::emergencyState()
{
}

/*!	\fn void RComponent::FailureState()
 *	\brief Actions performed on failure state
*/
void RComponent::failureState()
{
}

/*!	\fn void RComponent::AllState()
 *	\brief Actions performed on all states
*/
void RComponent::allState()
{
  rosPublish();
}

/*!	\fn double RComponent::getUpdateRate()
 * 	\brief Gets current update rate of the thread
 * 	\return real frequency of the thread
*/
double RComponent::getUpdateRate()
{
  return desired_freq_;
}

/*!	\fn int RComponent::getState()
 * 	\brief returns the state of the component
*/
int RComponent::getState()
{
  return state;
}

/*!	\fn char *RComponent::getStateString()
 *	\brief Gets the state of the component as char* string
*/
// char* RComponent::getStateString()
//{
//  return getStateString(state);
//}

/*!	\fn std::string RComponent::getStateString()
 *	\brief Gets the state of the component as std::string
*/
std::string RComponent::getStateString()
{
  return std::string(getStateString(state));
}

/*! \fn char *RComponent::getComponentName()
 *  \brief Returns the name of the component
*/
const char* RComponent::getComponentName()
{
  return component_name.c_str();
}

/*! \fn char *RComponent::setComponentName()
 *  \brief Returns the name of the component
*/
void RComponent::setComponentName(const std::string& name)
{
  component_name.assign(name);
}

/*! \fn std::string RComponent::constructComponentNameFromHandles()
 *  \brief Constructs a proper name for the component, using information from the handles
 *  passed to it
*/
std::string RComponent::constructComponentNameFromHandles()
{
  std::string name = "RComponent";
  if (pnh_.getNamespace() != "" and pnh_.getNamespace() != "~")
  {
    name = pnh_.getNamespace();
  }
  else if (nh_.getNamespace() != "" and nh_.getNamespace() != "~")
  {
    name = nh_.getNamespace();
  }

  size_t i = name.rfind('/', name.length());
  if (i != std::string::npos)
  {
    name = name.substr(i + 1, name.length() - i);
  }
  return name;
}

/*! \fn std::string RComponent::getPrivateNamespace()
 *  \brief Sets the public namespace of the component
*/
const std::string RComponent::getPrivateNamespace()
{
  return pnh_.getNamespace();
}

/*! \fn std::string RComponent::getPublicNamespace()
 *  \brief Returns the public namespace of the component
*/
const std::string RComponent::getPublicNamespace()
{
  return nh_.getNamespace();
}

/*!	\fn char *RComponent::getStateString(int state)
 *	\brief Gets the state as a string
*/
char* RComponent::getStateString(int state)
{
  switch (state)
  {
    case robotnik_msgs::State::INIT_STATE:
      return (char*)"INIT";
      break;
    case robotnik_msgs::State::STANDBY_STATE:
      return (char*)"STANDBY";
      break;
    case robotnik_msgs::State::READY_STATE:
      return (char*)"READY";
      break;
    case robotnik_msgs::State::EMERGENCY_STATE:
      return (char*)"EMERGENCY";
      break;
    case robotnik_msgs::State::FAILURE_STATE:
      return (char*)"FAILURE";
      break;
    case robotnik_msgs::State::SHUTDOWN_STATE:
      return (char*)"SHUTDOWN";
      break;
    default:
      return (char*)"UNKNOWN";
      break;
  }
}

/*!	\fn void RComponent::switchToState(int new_state)
 * 	function that switches the state of the component into the desired state
 * 	\param new_state as an integer, the new state of the component
*/
void RComponent::switchToState(int new_state)
{
  if (new_state == state)
    return;

  // saves the previous state
  previous_state = state;
  RCOMPONENT_INFO("%s -> %s", getStateString(state), getStateString(new_state));
  state = new_state;

  switch (state)
  {
    case robotnik_msgs::State::INIT_STATE:
      switchToInitState();
      break;
    case robotnik_msgs::State::STANDBY_STATE:
      switchToStandbyState();
      break;
    case robotnik_msgs::State::READY_STATE:
      switchToReadyState();
      break;
    case robotnik_msgs::State::EMERGENCY_STATE:
      switchToEmergencyState();
      break;
    case robotnik_msgs::State::FAILURE_STATE:
      switchToFailureState();
      break;
    case robotnik_msgs::State::SHUTDOWN_STATE:
      switchToShutdownState();
      break;
  }

  t_state_transition_ = ros::Time::now();
}

/*!	\fn void RComponent::switchToInitState()
 * 	callback executed when moving to init state
*/
void RComponent::switchToInitState()
{
}

/*!	\fn void RComponent::switchToStandbyState()
 * 	callback executed when moving to standby state
*/
void RComponent::switchToStandbyState()
{
}

/*!	\fn void RComponent::switchToReadyState()
 * 	callback executed when moving to ready state
*/
void RComponent::switchToReadyState()
{
}

/*!	\fn void RComponent::switchToEmergencyState()
 * 	callback executed when moving to emergency state
*/
void RComponent::switchToEmergencyState()
{
}

/*!	\fn void RComponent::switchToFailureState()
 * 	callback executed when moving to failure state
*/
void RComponent::switchToFailureState()
{
}

/*!	\fn void RComponent::switchToShutdownState()
 * 	callback executed when moving to shutdown state
*/
void RComponent::switchToShutdownState()
{
}

/*!	\fn void RComponent::rosSetup()
 * 	\brief Setups all ROS' stuff
*/
int RComponent::rosSetup()
{
  // Checks if has been initialized
  if (ros_initialized)
  {
    RCOMPONENT_INFO("Already initialized");

    return INITIALIZED;
  }

  state_publisher = pnh_.advertise<robotnik_msgs::State>("state", 1);
  // state_publisher = pnh_.advertise<std_msgs::Empty>("state", 1);

  ros_initialized = true;

  return OK;

  /*

  status_pub_ = pnh_.advertise<agvs_controller::DspicStatus>("status", 1);
  odom_pub_ = pnh_.advertise<nav_msgs::Odometry>(odom_frame_id_, 1);
  calibrate_srv_ = pnh_.advertiseService("calibrate",  &dspic_controller_node::CalibrateSrv, this);
  set_odom_service_ = pnh_.advertiseService("set_odometry", &dspic_controller_node::SetOdometry, this);*/
}

/*!	\fn void RComponent::rosReadParams
 * 	\brief Reads the params set in ros param server
*/
void RComponent::rosReadParams()
{
  pnh_.param("desired_freq", desired_freq_, DEFAULT_THREAD_DESIRED_HZ);

  /* Example
  pnh_.param<std::string>("port", port_, DEFAULT_DSPIC_PORT);
  pnh_.param<std::string>("odom_frame_id", odom_frame_id_, "/odom_diff");
  pnh_.param<std::string>("base_frame_id", base_frame_id_, "/base_link");
  pnh_.param("publish_tf", publish_tf_, false);
  pnh_.param("desired_freq", desired_freq_, desired_freq_);*/
}

/*!	\fn int RComponent::rosShutdown()
 * 	\brief Closes all ros stuff
*/
int RComponent::rosShutdown()
{
  if (running)
  {
    RCOMPONENT_INFO("Impossible while thread running, first must be stopped");
    return THREAD_RUNNING;
  }
  if (!ros_initialized)
  {
    RCOMPONENT_INFO("Impossible because of it's not initialized");
    return NOT_INITIALIZED;
  }

  ros_initialized = false;

  return OK;
}

/*!	\fn void RComponent::rosPublish()
 * 	\brief Reads data a publish several info into different topics
*/
void RComponent::rosPublish()
{
  robotnik_msgs::State msg;

  // STATE
  msg.state = this->state;
  msg.desired_freq = this->desired_freq_;
  msg.real_freq = this->real_freq;
  msg.state_description = getStateString();

  state_publisher.publish(msg);
}

/*!	\fn bool RComponent::checkTopicsHealth(std::string topic)
 * 	\brief Checks the topic health of all the subscribed topics or specific ones
 *  \param topic as std::string, topic to check. If empty all the topics are checked as a group
 *  \return true if health is ok, false otherwise
*/
bool RComponent::checkTopicsHealth(std::string topic_id)
{
  std::map<std::string, TopicHealthMonitor>::iterator it;

  if (data_health_monitors_.empty())
  {
    RCOMPONENT_WARN_STREAM_THROTTLE(5, "Topics health monitor is empty");
    return false;
  }

  if (topic_id.empty() == false)
  {
    it = data_health_monitors_.find(topic_id);

    if (it != data_health_monitors_.end())
    {
      return it->second.isReceiving();
    }
    else
    {
      RCOMPONENT_WARN_STREAM_THROTTLE(5, "Topic " << it->first << " not being received");
      return false;
    }
  }
  else  // Only check required topics
  {
    for (it = data_health_monitors_.begin(); it != data_health_monitors_.end(); ++it)
    {
      if (it->second.isReceiving() == false && it->second.isRequired() == true)
      {
        RCOMPONENT_WARN_STREAM_THROTTLE(5, "Topic " << it->first << " not being received");
        return false;
      }
    }

    return true;
  }
}

/*!	\fn int RComponent::addTopicsHealth(ros::Subscriber *subscriber = 0, std::string topic_id, double timeout)
 * 	\brief Adds a topic health for the subscriber
 *  \param subscriber as ros::Subscriber*, pointer to the Subscriber to check health
 *  \param topic_id as std::string, topic id to associate with the subscriber. If empty it will use the full topic name
 *  \param timeout as std::string, topic id to associate with the subscriber. If empty it will use the full topic name
 *  \return 0 if ok, -1 otherwise
*/
int RComponent::addTopicsHealth(ros::Subscriber* subscriber, std::string topic_id, double timeout, bool required)
{
  std::string map_id;
  if (subscriber == NULL)
  {
    RCOMPONENT_ERROR("Subscriber to check is NULL");
    return -1;
  }

  if (topic_id.empty() == true)
  {
    map_id = subscriber->getTopic();
  }
  else
  {
    map_id = topic_id;
  }
  if (timeout <= 0)
  {
    RCOMPONENT_ERROR("timeout (%.lf) has to be >= 0", timeout);
    timeout = 1.0;
  }

  data_health_monitors_[map_id] = TopicHealthMonitor(subscriber, timeout, required);

  return 0;
}

/*!	\fn int RComponent::tickTopicsHealth(std::string topic_id)
 * 	\brief Ticks the selected topic to notify that is being received
 *  \param topic_id as std::string, topic id that identifies the subscriber
 *  \return 0 if ok, -1 otherwise
*/
int RComponent::tickTopicsHealth(std::string topic_id)
{
  if (topic_id.empty() == true)
  {
    RCOMPONENT_ERROR_STREAM_THROTTLE(5, "topic id " << topic_id << " cannot be empty");
    return -1;
  }

  std::map<std::string, TopicHealthMonitor>::iterator it;

  it = data_health_monitors_.find(topic_id);

  if (it != data_health_monitors_.end())
  {
    it->second.tick();
    return 0;
  }

  RCOMPONENT_ERROR_STREAM_THROTTLE(5, "topic id " << topic_id << " does not exist");
  return -1;
}

/*!	\fn ros::Duration RComponent::getElapsedTimeSinceLastStateTransition()
 * 	\brief Returns the elapsed time since the last state transition
*/
ros::Duration RComponent::getElapsedTimeSinceLastStateTransition()
{
  return ros::Time::now() - t_state_transition_;
}

/*!	\fn ros::Time RComponent::getElapsedTimeSinceLastStateTransition()
 * 	\brief Returns the state transition time
*/
ros::Time RComponent::getStateTransitionTime()
{
  return t_state_transition_;
}

}  // namespace rcomponent
