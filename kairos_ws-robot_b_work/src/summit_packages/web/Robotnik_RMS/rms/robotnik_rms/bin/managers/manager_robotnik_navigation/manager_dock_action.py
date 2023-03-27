#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


import actionlib

from robotnik_navigation_msgs.msg import DockAction
from robotnik_navigation_msgs.msg import DockGoal
import copy 



class ManagerDockAction(ManagerAstract.ManagerAbstractClass):
    def __init__(self, action_name="/purepursuit", default_speed_linear_x = 0.3, default_speed_linear_y = 0.3, default_speed_angular_z = 0.5, dock_approach_offset_x = -0.5, dock_approach_offset_y = 0, dock_approach_offset_theta = 0.0, dock_offset_x = -0.10, dock_offset_y = 0, dock_offset_theta = 0.0, dock_frame = "docking_station_laser_contact_for_base_frame", robot_dock_frame = "base_footprint", do_approaching = True):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global actionlib
        global DockAction
        global DockGoal
        global copy
	#aproach action configuration	
	self.dock_action_goal_approach = DockGoal()
	self.dock_action_goal_approach.dock_frame = dock_frame
	self.dock_action_goal_approach.robot_dock_frame = robot_dock_frame
	self.dock_action_goal_approach.dock_offset.x = dock_approach_offset_x
	self.dock_action_goal_approach.dock_offset.y = dock_approach_offset_y
	self.dock_action_goal_approach.dock_offset.theta = dock_approach_offset_theta
	self.dock_action_goal_approach.maximum_velocity.linear.x = default_speed_linear_x
	self.dock_action_goal_approach.maximum_velocity.linear.y = default_speed_linear_y
	self.dock_action_goal_approach.maximum_velocity.angular.z = default_speed_angular_z
	#action configuration
	self.dock_action_goal = DockGoal()
	self.dock_action_goal.dock_frame = dock_frame
	self.dock_action_goal.robot_dock_frame = robot_dock_frame
	self.dock_action_goal.dock_offset.x = dock_offset_x
	self.dock_action_goal.dock_offset.y = dock_offset_y
	self.dock_action_goal.dock_offset.theta = dock_offset_theta
	self.dock_action_goal.maximum_velocity.linear.x = default_speed_linear_x
	self.dock_action_goal.maximum_velocity.linear.y = default_speed_linear_y
	self.dock_action_goal.maximum_velocity.angular.z = default_speed_angular_z
	#flag do_approaching
	self.do_approaching = do_approaching
	
	self.action_name = action_name
	#action client
	self.client = actionlib.SimpleActionClient(self.action_name, DockAction)


    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
	print("ManagerMoveAction in action_name = "+ self.action_name+"-->Executing")
	if (self.do_approaching == True):
		print "Executing approaching"
		result = self.action_client(self.dock_action_goal_approach)
		#if(result.success):
		self.changeState(self.STATES_EXECUTING, "Approaching finished, executing the dock action")		
		#else:
			#self.changeState(self.STATES_FAIL, "Failed when approaching to docking")
	else:
        	self.changeState(self.STATES_EXECUTING, "Executing the goal action")

    def stateExecuting(self):
        result = self.action_client(self.dock_action_goal)
	if(result.success):
		self.changeState(self.STATES_FINALIZED, "Finished the dock action")
	else:
		self.changeState(self.STATES_FAIL, "Failed in the docking")


    def stateFinalized(self):
        print("ManagerPoiGoTo--->Finsihed: " + str(self.current_action_msg_))


    def action_client(self, action_goal):
        self.client.wait_for_server()
        print (action_goal)
	# Sends the goal to the action server.
	self.client.send_goal(action_goal)
	# Waits for the server to finish performing the action.
	self.client.wait_for_result()
	# Prints out the result of executing the action
	return self.client.get_result()


