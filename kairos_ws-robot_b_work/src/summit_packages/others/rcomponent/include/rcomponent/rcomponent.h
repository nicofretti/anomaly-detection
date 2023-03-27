/*! \class RComponent
 *  \file RComponent.h
 *	\author Robotnik Automation S.L.L
 *	\version 0.1.0
 *	\date 2014
 *  \brief Class to define a standard and shared structure (attributes & methods) for all the components
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

#ifndef __RCOMPONENT_H
#define __RCOMPONENT_H

#include <ros/ros.h>
#include <pthread.h>
#include <string>
#include <vector>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <robotnik_msgs/State.h>

#include <rcomponent/rcomponent_log_macros.h>
#include <rcomponent/topic_health_monitor.h>
#include <sstream>

//! Size of string for logging
#define DEFAULT_THREAD_DESIRED_HZ 40.0

using namespace std;

//! Defines return values for methods and functions
namespace rcomponent
{
enum ReturnValue
{
  OK = 0,
  INITIALIZED,
  THREAD_RUNNING,
  ERROR = -1,
  NOT_INITIALIZED = -2,
  THREAD_NOT_RUNNING = -3,
  COM_ERROR = -4,
  NOT_ERROR = -5
};

//! Struct used for real time thread
struct thread_param
{
  int prio;   // Thread priority level 0[min]-80[max]
  int clock;  // CLOCK_MONOTONIC or CLOCK_REALTIME
};

//! struct to store main data for a thread
typedef struct thread_data
{
  //! Priority and clock
  struct thread_param pthreadPar;
  //! Contains the desired frequency of the thread
  double dDesiredHz;
  //! Contains the real frequency of the thread
  double dRealHz;
  //! Contains the id of the control thread
  pthread_t pthreadId;
} thread_data;

//! Class Rcomponent
class RComponent
{
protected:
  //! Controls if it has been properly constructed, either by calling
  //! one of the fully qualified constructors, or one of the init funcionts
  bool constructed;
  //! Controls if has been initialized succesfully
  bool initialized, ros_initialized;
  //! Controls the execution of the RComponent's thread
  bool running;

  //! State of the RComponent
  int state;
  //! State before
  int previous_state;
  //!	Saves the name of the component
  string component_name;
  //! ROS node handle
  ros::NodeHandle nh_;
  //! Private ROS node handle
  ros::NodeHandle pnh_;
  //! Desired loop frequency
  double desired_freq_, real_freq;

  //! Publish the component state
  ros::Publisher state_publisher;

  //! Contains data for the main thread
  thread_data threadData;

  //! Contains all the data health monitors
  std::map<std::string, TopicHealthMonitor> data_health_monitors_;

  //! Saves the time of a state transition
  ros::Time t_state_transition_;

public:
  //! Public constructor
  RComponent();
  //! Public constructor
  RComponent(ros::NodeHandle h);
  //! Public constructor, assigning a private namespace
  RComponent(ros::NodeHandle h, std::string name);
  //! Public constructor, assigning a private nodehandle
  RComponent(ros::NodeHandle h, ros::NodeHandle ph);
  //! Public destructor
  virtual ~RComponent();
  //! Acts as a delegate constructor
  virtual int init(ros::NodeHandle h);
  //! Acts as a delegate constructor, assigning a private namespace
  virtual int init(ros::NodeHandle h, std::string name);
  //! Acts as a delegate constructor, assigning a private nodehandle
  virtual int init(ros::NodeHandle h, ros::NodeHandle ph);

  //! Starts the control loop of the component and its subcomponents
  //! @return OK
  //! @return ERROR starting the thread
  //! @return RUNNING if it's already running
  //! @return NOT_INITIALIZED if it's not initialized
  virtual int start();
  //! Starts the control loop of the component and its subcomponents in a separated thread
  //! @return OK
  //! @return ERROR starting the thread
  //! @return RUNNING if it's already running
  //! @return NOT_INITIALIZED if it's not initialized
  virtual int asyncStart();
  //! Stops the main control loop of the component and its subcomponents
  //! @return OK
  //! @return ERROR if any error has been produced
  //! @return NOT_RUNNING if the main thread isn't running
  virtual int stop();
  //! Checks if component is running
  //! @return TRUE component is running
  //! @return FALSE component is not running
  bool isRunning();

  //! Returns the general state of the RComponent
  int getState();
  //! Returns the general state of the RComponent as char* string
  // char* getStateString();
  //! Returns the general state of the RComponent as std::string
  std::string getStateString();
  //! Returns the general state as char* string
  char* getStateString(int state);
  //! Method to get current update rate of the thread
  //! @return pthread_hz
  double getUpdateRate();
  //! Returns the name of the component
  const char* getComponentName();
  //! Sets the name of the component
  void setComponentName(const std::string& name);
  //! Returns the public namespace of the component
  const std::string getPublicNamespace();
  //! Returns the private namespace of the component
  const std::string getPrivateNamespace();
  //! Returns true if the topics healht is
  virtual bool checkTopicsHealth(std::string topic_id = "");
  //! Adds a topic health for the subscriber
  virtual int addTopicsHealth(ros::Subscriber* subscriber = 0, std::string topic_id = "", double timeout = 5.0,
                              bool required = true);
  //! Ticks the selected topic
  virtual int tickTopicsHealth(std::string topic_id);
  //! Returns the elapsed time since the last state transition
  ros::Duration getElapsedTimeSinceLastStateTransition();
  //! Returns the state transition time
  ros::Time getStateTransitionTime();

protected:
  //! Configures and initializes the component
  //! @return OK
  //! @return INITIALIZED if the component is already intialized
  //! @return ERROR
  virtual int setup();
  //! Closes and frees the reserved resources
  //! @return OK
  //! @return ERROR if fails when closes the devices
  //! @return RUNNING if the component is running
  //! @return NOT_INITIALIZED if the component is not initialized
  virtual int shutdown();
  //! All core component functionality is contained in this thread.
  //!	All of the RComponent component state machine code can be found here.
  virtual void controlLoop();
  //! Static helper method that enables the execution of controlLoop inside a thread
  static void* asyncControlLoop(void*);
  //! Actions performed on initial state
  virtual void initState();
  //! Actions performed on standby state
  virtual void standbyState();
  //! Actions performed on ready state
  virtual void readyState();
  //! Actions performed on the emergency state
  virtual void emergencyState();
  //! Actions performed on Failure state
  virtual void failureState();
  //! Actions performed on Shudown state
  virtual void shutdownState();
  //! Actions performed in all states
  virtual void allState();
  //! Switches between states
  virtual void switchToState(int new_state);
  //! callback executed when moving to init state
  virtual void switchToInitState();
  //! callback executed when moving to standby state
  virtual void switchToStandbyState();
  //! callback executed when moving to ready state
  virtual void switchToReadyState();
  //! callback executed when moving to emergency state
  virtual void switchToEmergencyState();
  //! callback executed when moving to failure state
  virtual void switchToFailureState();
  //! callback executed when moving to shutdown state
  virtual void switchToShutdownState();

  //! Setups all the ROS' stuff
  virtual int rosSetup();
  //! Shutdowns all the ROS' stuff
  virtual int rosShutdown();
  //! Reads data a publish several info into different topics
  virtual void rosPublish();
  //! Reads params from params server
  virtual void rosReadParams();
  //! Constructs a name of the component from handles used
  std::string constructComponentNameFromHandles();

  //! Reads a parameter from the param server, and shows a message if parameter is not set
  template <typename T>
  bool readParam(const ros::NodeHandle& h, const std::string& name, T& value, const T& default_value,
                 bool required = false)
  {
    // parameter is read from node handle passed
    // required defines logger lever: if true, will show an error. if false, a warning
    // TODO: improve: return true or false depending on parameter existence and required
    // TODO: maybe would be better to define a log level instead of required
    if (h.hasParam(name) == false)
    {
      if (required == false)
      {
        RCOMPONENT_WARN_STREAM("No parameter \"" << h.resolveName(name) << "\", using default value: " << default_value
                                                 << ".");
      }
      else
      {
        RCOMPONENT_ERROR_STREAM("No parameter \"" << h.resolveName(name) << "\", using default value: " << default_value
                                                  << ".");
      }
      value = default_value;
      return false;
    }
    h.param<T>(name, value, default_value);
    return true;
  }

  // Double template type, used when type of default value does not match type or variable, but can be casted safely.
  // Example: T is double, S is in, or T is std::string, S is "char *"
  template <typename T, typename S>
  bool readParam(const ros::NodeHandle& h, const std::string& name, T& value, const S& default_value,
                 bool required = false)
  {
    return readParam(h, name, value, static_cast<T>(default_value), required);
  }

  template <typename T>
  bool readParam(const ros::NodeHandle& h, const std::string& name, std::vector<T>& value,
                 const std::vector<T>& default_value, bool required = false)
  {
    // parameter is read from node handle passed
    // required defines logger lever: if true, will show an error. if false, a warning
    // TODO: improve: return true or false depending on parameter existence and required
    // TODO: maybe would be better to define a log level instead of required
    if (h.hasParam(name) == false)
    {
      std::stringstream default_value_message;
      default_value_message << "[";
      for (auto& v : default_value)
        default_value_message << v << ",";
      default_value_message << "]";

      if (required == false)
      {
        RCOMPONENT_WARN_STREAM("No parameter \"" << h.resolveName(name)
                                                 << "\", using default value: " << default_value_message.str() << ".");
      }
      else
      {
        RCOMPONENT_WARN_STREAM("No parameter \"" << h.resolveName(name)
                                                 << "\", using default value: " << default_value_message.str() << ".");
      }
      value = default_value;
      return false;
    }
    h.param<std::vector<T>>(name, value, default_value);
    return true;
  }
};
}  // namespace rcomponent

#endif
