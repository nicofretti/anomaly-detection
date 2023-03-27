#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


import json
import actionlib
from robotnik_navigation_msgs.msg import MoveAction
from robotnik_pp_msg.msg import GoToGoal
from geometry_msgs.msg import Pose2D
from robotnik_pp_msgs.msg import goal



class ManagerMoveAction(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self, action_name = "/robot/move_base", default_speed = 0.3, default_x = 0, default_y = 0, default_theta=0)
        global actionlib
        global MoveAction
	global GoToGoal
	global Pose2D
	global goal
	global json
	self.action_name = action_name
	self.default_speed = default_speed
	self.default_x = default_x
	self.default_y = default_y
	self.default_theta = default_theta
	self.position = {"x":self.default_x, "y":self.default_y, "theta":self.default_theta, "speed": self.default_speed}
	self.goal = goal()
	self.client = actionlib.SimpleActionClient(self.action_name, DockAction)


    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
	print("ManagerMoveAction in action_name = "+ self.action_name+"-->Executing: " + str(self.current_action_msg_))
	reedANewGoal (self, str(self.current_action_msg_))        
	result = self.action_client()
        self.changeState(self.STATES_FINALIZED, "Finalized the task")

    def stateExecuting(self):
        self.changeState(self.STATES_FINALIZED, "Finalized the task")

    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        # None

    def stateError(self):
        print("Error")
        # None

    def stateFinalized(self):
	self.times_called = self.times_called + 1
        print("ManagerPoiGoTo--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")

    def reedANewGoal (self, json_string):
	self.position = {"x":self.default_x, "y":self.default_y, "theta":self.default_theta, "speed": self.default_speed}
	try:
		obj = json.loads(json_string)
		if "x" in obj:
			self.position["x"] = obj["x"]
		if "y" in obj:
			self.position["y"] = obj["y"]
		if "theta" in obj:
			self.position["theta"] = obj["theta"]
		if "speed" in obj:
			self.position["speed"] = obj["speed"]
	except:
		print "RMS - Execption to read json " + json_string
	self.goal = goal()
	self.goal.pose.x = pose.x = self.position["x"]
	self.goal.pose.y = self.position["y"]
	self.goal.pose.theta = self.position["theta"]
	self.goal.speed = self.position["speed"]
	

    def action_client(self):
        self.client.wait_for_server()
	# Creates a goal to send to the action server.
	goal_action_msg = DockGoal()
	goal_action_msg.target.append(self.goal)
	# Sends the goal to the action server.
	client.send_goal(goal_action_msg)
	# Waits for the server to finish performing the action.
	client.wait_for_result()
	# Prints out the result of executing the action
	return client.get_result()


