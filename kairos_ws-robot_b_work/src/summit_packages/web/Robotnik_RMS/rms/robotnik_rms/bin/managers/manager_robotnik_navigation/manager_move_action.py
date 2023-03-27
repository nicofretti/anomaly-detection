#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


import actionlib

from robotnik_navigation_msgs.msg import MoveAction
from robotnik_navigation_msgs.msg import MoveGoal
import copy 



class ManagerMoveAction(ManagerAstract.ManagerAbstractClass):
    def __init__(self, action_name="/move", default_speed_linear_x = 0.3, default_speed_linear_y = 0.3, default_speed_angular_z = 0.5, goal_x = -0.5, goal_y= 0.0, goal_theta=0):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global actionlib
        global MoveAction
        global MoveGoal
        global copy
	#action configuration
	self.dock_action_goal = MoveGoal()
	self.dock_action_goal.goal.x = goal_x
	self.dock_action_goal.goal.y = goal_y
	self.dock_action_goal.goal.theta = goal_theta
	self.dock_action_goal.maximum_velocity.linear.x = default_speed_linear_x
	self.dock_action_goal.maximum_velocity.linear.y = default_speed_linear_y
	self.dock_action_goal.maximum_velocity.angular.z = default_speed_angular_z
	
	
	self.action_name = action_name
	#action client
	self.client = actionlib.SimpleActionClient(self.action_name, MoveAction)


    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
	print("ManagerMoveAction in action_name = "+ self.action_name+"-->Executing")
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


