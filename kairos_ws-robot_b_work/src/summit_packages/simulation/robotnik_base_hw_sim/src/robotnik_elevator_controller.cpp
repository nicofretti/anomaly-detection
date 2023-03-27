/** \file robotnik_base_hw_sim.cpp
 * \author Robotnik Automation S.L.L.
 * \version 3.0
 * \date    2017
 *
 * \brief robotnik_base_hw_sim class driver
 * Component to manage the RB1 servo controller set
 * (C) 2012 Robotnik Automation, SLL
*/

#include <algorithm>
#include <sstream>
#include <numeric>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <robotnik_base_hw_sim/robotnik_elevator_controller.h>

#define ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_UP 3
#define ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_DOWN 2
#define ROBOTNIK_ELEVATOR_DEFAULT_INPUT_UP 3
#define ROBOTNIK_ELEVATOR_DEFAULT_INPUT_DOWN 2
#define ELEVATOR_ACTION_COMMAND_TIMEOUT_MSG 1.0            // default timeout processing an action received
#define ELEVATOR_MOVING_TIMEOUT 10.0                       // default timeout when raising/lowering the elevator
#define RECEIVED_IO_TIMEOUT 2.0                            // in secs. Timeout without receiving io values
#define ROBOTNIK_ELEVATOR_DEFAULT_ELEVATOR_ACTION_TIME	 5.0  // time in action (seconds)
#define ROBOTNIK_ELEVATOR_DEFAULT_ELEVATOR_ACTION_TIMEOUT 10.0  // Max time in action (seconds)
#define ROBOTNIK_ELEVATOR_DEFAULT_ELEVATOR_IN_TIMEOUT 10.0      // Time in timeout error (auto recover)
#define ROBOTNIK_ELEVATOR_DEFAULT_HZ 10.0                       // controol loop frequency
#define ROBOTNIK_ELEVATOR_DEFAULT_POS_UP 0.03
#define ROBOTNIK_ELEVATOR_DEFAULT_POS_DOWN 0.0
#define ROBOTNIK_DEFAULT_MAX_JOINT_SPEED 27.27

namespace robotnik_elevator_controller
{

RobotnikElevatorController::RobotnikElevatorController()
{
  // TODO: initialize all variables
  elevator_action_.action = robotnik_msgs::ElevatorAction::NO_ACTION;
  elevator_status_.state = robotnik_msgs::ElevatorStatus::IDLE;
  elevator_status_.position = robotnik_msgs::ElevatorStatus::DOWN;
  elevator_current_position_ = elevator_position_down_;
  enabled_ = true; 
  init_ok = false;
}


bool RobotnikElevatorController::init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& root_nh,
                             ros::NodeHandle&	 controller_nh)
{
  

  controller_name_ = "robotnik_elevator_controller";
  elevator_digital_output_up_ = 6;
  elevator_digital_output_down_ = 5;

  set_digital_output_service_hw_ = "robotnik_base_hw/set_digital_output";

  //

  
  controller_nh.param("elevator_digital_output_up", elevator_digital_output_up_, ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_UP);
  controller_nh.param("elevator_digital_output_down", elevator_digital_output_down_, ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_DOWN);
  controller_nh.param("elevator_digital_input_up", elevator_digital_input_up_, ROBOTNIK_ELEVATOR_DEFAULT_INPUT_UP);
  controller_nh.param("elevator_digital_input_down", elevator_digital_input_down_, ROBOTNIK_ELEVATOR_DEFAULT_INPUT_DOWN);
  controller_nh.param<double>("elevator_position_up", elevator_position_up_, ROBOTNIK_ELEVATOR_DEFAULT_POS_UP);
  controller_nh.param<double>("elevator_position_down", elevator_position_down_, ROBOTNIK_ELEVATOR_DEFAULT_POS_DOWN);
  controller_nh.param<double>("elevation_action_time", elevation_action_time_, ROBOTNIK_ELEVATOR_DEFAULT_ELEVATOR_ACTION_TIME);
  controller_nh.param("set_digital_output_service_hw", set_digital_output_service_hw_, set_digital_output_service_hw_);
  controller_nh.param<std::string>("joint/elevator_joint/name", elevator_joint_name_, "robotnik_elevator_platform_joint");
  
  controller_nh.param<bool>("gazebo/disable_gazebo_physics_for_pickup", disable_gazebo_physics_for_pickup_, true);
  controller_nh.param<std::string>("gazebo/pickup_service", gazebo_simple_pick_service_name_, "/elevator_fake_pickup_gazebo/simple_pick");
  controller_nh.param<std::string>("gazebo/place_service", gazebo_simple_place_service_name_, "/elevator_fake_pickup_gazebo/simple_place");
  controller_nh.param<std::string>("gazebo/robot_model", gazebo_robot_model_, "rb2_a");
  controller_nh.param<double>("gazebo/elevation_offset_z", elevation_offset.position.z, 0.05);

  if (elevator_digital_output_up_ < 1)
  {
    ROS_WARN_STREAM_NAMED(controller_name_,
                          controller_name_
                              << "::initController: param elevator_digital_output_up (=" << elevator_digital_output_up_
                              << ") has to be >= 1. Setting default value " << ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_UP);
    elevator_digital_output_up_ = ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_UP;
  }
  if (elevator_digital_output_down_ < 1)
  {
    ROS_WARN_STREAM_NAMED(
        controller_name_,
        controller_name_ << "::initController: param elevator_digital_output_down (=" << elevator_digital_output_down_
                         << ") has to be >= 1. Setting default value " << ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_DOWN);
    elevator_digital_output_down_ = ROBOTNIK_ELEVATOR_DEFAULT_OUTPUT_DOWN;
  }
  if (elevator_digital_input_up_ < 1)
  {
    ROS_WARN_STREAM_NAMED(controller_name_,
                          controller_name_
                              << "::initController: param elevator_digital_input_up (=" << elevator_digital_input_up_
                              << ") has to be >= 1. Setting default value " << ROBOTNIK_ELEVATOR_DEFAULT_INPUT_UP);
    elevator_digital_input_up_ = ROBOTNIK_ELEVATOR_DEFAULT_INPUT_UP;
  }
  if (elevator_digital_input_down_ < 1)
  {
    ROS_WARN_STREAM_NAMED(controller_name_, controller_name_ << "::initController: param elevator_digital_input_down (="
                                                             << elevator_digital_input_down_
                                                             << ") has to be >= 1. Setting default value "
                                                             << ROBOTNIK_ELEVATOR_DEFAULT_INPUT_DOWN);
    elevator_digital_input_down_ = ROBOTNIK_ELEVATOR_DEFAULT_INPUT_DOWN;
  }
  double elevator_action_command_timeout_msg = 0.0;
  controller_nh.param("elevator_action_command_timeout_msg", elevator_action_command_timeout_msg,
                      ELEVATOR_ACTION_COMMAND_TIMEOUT_MSG);

  elevator_status_control_loop_period_ = ros::Duration(1.0 / ROBOTNIK_ELEVATOR_DEFAULT_HZ);
  elevator_action_command_timeout_ = ros::Duration(elevator_action_command_timeout_msg);

  io_sub_ = root_nh.subscribe("robotnik_base_hw/io", 1, &RobotnikElevatorController::ioCallback, this);
  enabled_pub_ = root_nh.advertise<std_msgs::Bool>("robotnik_base_control/enabled", 1);
  elevator_status_pub_ = root_nh.advertise<robotnik_msgs::ElevatorStatus>("robotnik_base_control/elevator_status", 1);

  set_elevator_srv_ = root_nh.advertiseService("robotnik_base_control/set_elevator", &RobotnikElevatorController::setElevatorSrvCallback, this);
  enable_srv_ = root_nh.advertiseService("robotnik_base_control/enable", &RobotnikElevatorController::enableSrvCallback, this);

  set_digital_output_client = root_nh.serviceClient<robotnik_msgs::set_digital_output>(set_digital_output_service_hw_);
  
  if(disable_gazebo_physics_for_pickup_){
	  gazebo_simple_pick_service_client = root_nh.serviceClient<robotnik_base_hw_sim::SimplePick>(gazebo_simple_pick_service_name_);
	  gazebo_simple_place_service_client = root_nh.serviceClient<robotnik_base_hw_sim::SimplePlace>(gazebo_simple_place_service_name_);
  }
  
  elevator_action_server_ = new actionlib::SimpleActionServer<robotnik_msgs::SetElevatorAction>(
      root_nh, "robotnik_base_control/set_elevator", boost::bind(&RobotnikElevatorController::executeElevatorCallback, this, _1), false);

  joint_commands_.resize(1);

  // for now, the controller is for a robot with four wheel, and the joint names are these and only these
  joint_names_.resize(1);
  
  joint_ = hw->getHandle(elevator_joint_name_);
  
  elevator_loop_last_execution_ = ros::Time::now();
  io_last_stamp_ = ros::Time(0);         // maybe it is better to set it to 0, so if no cmd is received
  last_elevator_action_time = ros::Time(0);
  elevator_action_init_time = ros::Time(0);
  elevator_action_server_->start();
  
  init_ok = true;
  
  return init_ok;
}

/**     \fn  RobotnikElevatorController::setElevatorSrvCallback
 *
 */
bool RobotnikElevatorController::setElevatorSrvCallback(robotnik_msgs::SetElevator::Request& req,
                                               robotnik_msgs::SetElevator::Response& res)
{
  ROS_INFO_THROTTLE(5,"%s::setElevatorSrvCallback: action %d. RAISE:%d, LOWER:%d",
           controller_name_.c_str(), req.action.action, robotnik_msgs::ElevatorAction::RAISE,
           robotnik_msgs::ElevatorAction::LOWER);
  res.ret = false;


 /* if (inMotion())
  {
    ROS_WARN_THROTTLE(5,"%s::setElevatorSrvCallback: robot in motion. Action not allowed!", controller_name_.c_str());
    return false;
  }*/

  if (elevator_status_.state != robotnik_msgs::ElevatorStatus::IDLE)
  {
    ROS_WARN_THROTTLE(5,"%s::setElevatorSrvCallback: elevator not idle!", controller_name_.c_str());
    return false;
  }

  if (req.action.action == robotnik_msgs::ElevatorAction::RAISE or
      req.action.action == robotnik_msgs::ElevatorAction::LOWER)
  {
    last_elevator_action_time = ros::Time::now();
  }
  else
  {
    ROS_ERROR_THROTTLE(5,"%s::setElevatorSrvCallback: Action %d not allowed", controller_name_.c_str(), req.action.action);
    return false;
  }

  elevator_action_ = req.action;

  res.ret = true;

  return true;
}

void RobotnikElevatorController::executeElevatorCallback(const robotnik_msgs::SetElevatorGoalConstPtr& goal)
{
  ros::Rate rate(1);
  ros::Time initial_time;
  robotnik_msgs::SetElevator::Request elevator_request;
  robotnik_msgs::SetElevator::Response elevator_response;

  elevator_request.action = goal->action;
  bool callback_result = false;

  ros::Duration wait_time_for_robot_to_stop(5);
  initial_time = ros::Time::now();
  while (callback_result == false and ((ros::Time::now() - initial_time) < wait_time_for_robot_to_stop))
  {
    callback_result = setElevatorSrvCallback(elevator_request, elevator_response);
  }

  if (callback_result == false)
  {
    elevator_action_result_.result = false;
    elevator_action_result_.status = elevator_status_;
    elevator_action_server_->setAborted(elevator_action_result_, "internal callback failed");
    return;
  }
  if (goal->action.action == robotnik_msgs::ElevatorAction::STOP)
  {
    elevator_action_result_.result = true;
    elevator_action_result_.status = elevator_status_;
    elevator_action_server_->setSucceeded(elevator_action_result_, "ok");
    return;
  }
  bool timedout = false;
  ros::Duration timeout(15);
  initial_time = ros::Time::now();
  rate.sleep();
  while (((goal->action.action == robotnik_msgs::ElevatorAction::RAISE and
           elevator_status_.state == robotnik_msgs::ElevatorStatus::RAISING and
           elevator_status_.position != robotnik_msgs::ElevatorStatus::UP) or
          (goal->action.action == robotnik_msgs::ElevatorAction::LOWER and
           elevator_status_.state == robotnik_msgs::ElevatorStatus::LOWERING and
           elevator_status_.position != robotnik_msgs::ElevatorStatus::DOWN)) and
         (not timedout))
  {
    elevator_action_feedback_.status = elevator_status_;
    elevator_action_server_->publishFeedback(elevator_action_feedback_);
    rate.sleep();
    timedout = (ros::Time::now() - initial_time > timeout);
  }
  if (timedout == true)
  {
    elevator_action_result_.result = false;
    elevator_action_result_.status = elevator_status_;
    elevator_action_server_->setAborted(elevator_action_result_, "timedout");
    return;
  }
  else
  {
    elevator_action_result_.result = true;
    elevator_action_result_.status = elevator_status_;
    elevator_action_server_->setSucceeded(elevator_action_result_, "ok");
  }

  return;
}

std::string RobotnikElevatorController::getHardwareInterfaceType() const
{
  // as result of being a Controller which uses different types of JointInterface, return the main interface type
  // in this case, it is a VelocityJointInterface
  return hardware_interface::internal::demangledTypeName<hardware_interface::EffortJointInterface>();
}

/**
 * \brief Starts controller
 * \param time Current time
 */
void RobotnikElevatorController::starting(const ros::Time& time)
{
  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "::starting!");
  
}

/**
 * \brief Stops controller
 * \param time Current time
 */
void RobotnikElevatorController::stopping(const ros::Time& time)
{
  ROS_INFO_STREAM_NAMED(controller_name_, controller_name_ << "Stopping!");
  elevator_action_server_->shutdown();
}

void RobotnikElevatorController::update(const ros::Time& time, const ros::Duration& period)
{

	  
	readJointStates();


	writeJointCommands();

	// limit references according to acceleration and speed limits and hard_brake parameter

	// elevator management
	if ( (time - elevator_loop_last_execution_) > elevator_status_control_loop_period_)
	{
		publishElevatorStatus();
		elevatorControlLoop(time);
		elevator_loop_last_execution_ = time;
	}

	// enabled/disabled status
	std_msgs::Bool enabled_msg;
	enabled_msg.data = enabled_;
	enabled_pub_.publish(enabled_msg);

  
}

/**
 * \fn void RobotnikElevatorController::elevatorControlLoop(const ros::Time& time)
 * \brief Performs the internal state machine to control de elevator system
 */
void RobotnikElevatorController::elevatorControlLoop(const ros::Time& time)
{
  // Check the current status
  
  if (elevator_status_.state == robotnik_msgs::ElevatorStatus::IDLE)
  {
    // Check if a new action has been received
    if ((time - last_elevator_action_time) <= elevator_action_command_timeout_)
    {
      // processing the command
      elevator_ongoing_action_ = elevator_action_;
      
	  if(enabled_){
		  if (elevator_ongoing_action_.action == robotnik_msgs::ElevatorAction::RAISE)
		  {
			switchToElevatorState((string)robotnik_msgs::ElevatorStatus::RAISING);
		  }
		  else if (elevator_ongoing_action_.action == robotnik_msgs::ElevatorAction::LOWER)
		  {
			switchToElevatorState((string)robotnik_msgs::ElevatorStatus::LOWERING);
		  }
      }else
		ROS_WARN_STREAM_THROTTLE_NAMED(10, controller_name_, controller_name_ << "::elevatorControlLoop: controller is disabled");
    }
  }
  
  // RAISING
  else if (elevator_status_.state == robotnik_msgs::ElevatorStatus::RAISING)
  {

    if (elevator_status_.position == robotnik_msgs::ElevatorStatus::UP)
    {
      ROS_INFO("%s::elevatorControlLoop: RAISING - >elevator Up", controller_name_.c_str());
      switchToElevatorState((string)robotnik_msgs::ElevatorStatus::IDLE);
      
    }
    else
    {
      // Checking timeout
      if ((time - elevator_action_init_time).toSec() > elevation_action_time_)
      {
		
         elevator_status_.position = robotnik_msgs::ElevatorStatus::UP;
         // Setting the fake joint of the controller
		 elevator_current_position_ = elevator_position_up_;
      }
    }
  }
  
  // LOWERING
  else if (elevator_status_.state == robotnik_msgs::ElevatorStatus::LOWERING)
  {
    if (elevator_status_.position == robotnik_msgs::ElevatorStatus::DOWN)
    {
      ROS_INFO("%s::elevatorControlLoop: LOWERING - >elevator Down", controller_name_.c_str());
      switchToElevatorState((string)robotnik_msgs::ElevatorStatus::IDLE);
     
    }
    else
    {
      // Checking timeout
      if ((time - elevator_action_init_time).toSec() > elevation_action_time_)
      {
	  
       elevator_status_.position = robotnik_msgs::ElevatorStatus::DOWN;
       // Setting the fake joint of the controller
       elevator_current_position_ = elevator_position_down_;
      }
    }
  }
  
  
  
}

/**
   * \fn void RobotnikElevatorController::switchToElevatorState(string new_state)
   * \brief Transition function between elevator states
   */
void RobotnikElevatorController::switchToElevatorState(string new_state)
{
  if (new_state == elevator_status_.state)
    return;
  ROS_INFO("%s::switchToElevatorState: from %s to %s", controller_name_.c_str(), elevator_status_.state.c_str(),
           new_state.c_str());


  
		 
  if(elevator_status_.state == robotnik_msgs::ElevatorStatus::LOWERING and new_state == robotnik_msgs::ElevatorStatus::IDLE){
     if(disable_gazebo_physics_for_pickup_){
	    placeCart();
	 }
  }else if(elevator_status_.state == robotnik_msgs::ElevatorStatus::RAISING and new_state == robotnik_msgs::ElevatorStatus::IDLE){
     if(disable_gazebo_physics_for_pickup_){
	    pickCart();
	 }
  }
  
  elevator_status_.state = new_state;

  if (elevator_status_.state == robotnik_msgs::ElevatorStatus::ERROR_G_IO)
  {
    elevator_status_.position = robotnik_msgs::ElevatorStatus::UNKNOWN;
  }
  else if (elevator_status_.state == robotnik_msgs::ElevatorStatus::IDLE)
  {
    elevator_action_.action = robotnik_msgs::ElevatorAction::NO_ACTION;
  }
  else if ((elevator_status_.state == robotnik_msgs::ElevatorStatus::LOWERING) or
           (elevator_status_.state == robotnik_msgs::ElevatorStatus::RAISING))
  {
    elevator_action_init_time = ros::Time::now();  // Set the time of the action
  }
  else if (elevator_status_.state == robotnik_msgs::ElevatorStatus::ERROR_TIMEOUT)
  {
    elevator_action_init_time = ros::Time::now();  // Using the same timer to stay in this error state
    elevator_status_.position = robotnik_msgs::ElevatorStatus::UNKNOWN;
  }
}


void RobotnikElevatorController::readJointStates()
{ 
  joint_states_ = joint_.getEffort();
}

void RobotnikElevatorController::writeJointCommands()
{
 
  if (elevator_status_.state != robotnik_msgs::ElevatorStatus::IDLE &&
      elevator_status_.state != robotnik_msgs::ElevatorStatus::ERROR_G_IO)
  {
    ROS_WARN_STREAM_THROTTLE_NAMED(2, controller_name_, controller_name_ << "::writeJointCommands: elevator active!");
  }

 
  // send commands to actuators
  joint_.setCommand(elevator_current_position_);

}


/*bool RobotnikElevatorController::inMotion()
{
  static bool had_ref = false;
  static double starttime = 0.0;
  bool ref_timer = false;
  // bool e_stop = ((motor_velocity_[0]->GetMotorDigitalInputs() & DIGITAL_INPUT_ESTOP)!=0);
  bool e_stop = false;  // could be read from topic.
  bool zero_speed_ref =
      (received_cmd_.linear.x == 0) && (received_cmd_.linear.y == 0) && (received_cmd_.angular.z == 0);
  bool ref = (!e_stop && !zero_speed_ref);
  if (had_ref && !ref)
    starttime = ros::Time::now().toSec();
  double now = ros::Time::now().toSec();
  if ((starttime != 0.0) && (now - starttime < in_motion_timer_))
  {
    ref_timer = true;
  }
  else
  {
    ref_timer = false;
    starttime = 0.0;
  }
  had_ref = ref;

  return ref || ref_timer;
}
*/


void RobotnikElevatorController::ioCallback(const robotnik_msgs::inputs_outputsConstPtr& msg)
{
  // ROS_INFO_STREAM_NAMED(controller_name_, "::ioCallback: Received command");
  received_io_ = *msg;
  io_last_stamp_ = ros::Time::now();
}

void RobotnikElevatorController::publishElevatorStatus()
{
  elevator_status_pub_.publish(elevator_status_);
}


int RobotnikElevatorController::setDigitalOutput(int number, bool value)
{
  robotnik_msgs::set_digital_output io_srv_msg;

  io_srv_msg.request.value = value;
  io_srv_msg.request.output = number;
  set_digital_output_client.call(io_srv_msg);

  return 0;
}

int RobotnikElevatorController::pickCart(){
	robotnik_base_hw_sim::SimplePick pick_srv;
	
	pick_srv.request.robot_model = gazebo_robot_model_;
	
	if(not gazebo_simple_pick_service_client.call(pick_srv)){
		ROS_ERROR_STREAM_NAMED(controller_name_, "::pickCart: error calling service " << gazebo_simple_pick_service_name_);
		return -1;
	}
	
	if(pick_srv.response.success)
		return 0;
	else{
		ROS_ERROR_STREAM_NAMED(controller_name_, "::pickCart: response from service: " << pick_srv.response.msg);
		return -1;
	}
}


int RobotnikElevatorController::placeCart(){
	
	robotnik_base_hw_sim::SimplePlace place_srv;
	
	place_srv.request.robot_model = gazebo_robot_model_;
	
	if(not gazebo_simple_place_service_client.call(place_srv)){
		ROS_ERROR_STREAM_NAMED(controller_name_, "::placeCart: error calling service " << gazebo_simple_place_service_name_);
		return -1;
	}
	
	if(place_srv.response.success)
		return 0;
	else{
		ROS_ERROR_STREAM_NAMED(controller_name_, "::placeCart: response from service: " << place_srv.response.msg);
		return -1;
	}
	
}

/**     \fn  RobotnikElevatorController::enableSrvCallback
 *		Enables/Disables the controller to accept or not velocity and action commands
 */
bool RobotnikElevatorController::enableSrvCallback(robotnik_msgs::enable_disable::Request& req,
                                               robotnik_msgs::enable_disable::Response& res)
{
  enabled_ = req.value;

  res.ret = true;

  return true;
}
}
