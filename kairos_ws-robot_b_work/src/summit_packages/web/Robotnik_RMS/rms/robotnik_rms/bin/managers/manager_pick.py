#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


# Brings in the SimpleActionClient
import actionlib 

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.
from robotnik_msgs.msg import SetElevatorAction
from robotnik_msgs.msg import SetElevatorGoal
from robotnik_navigation_msgs.msg import DockAction
from robotnik_navigation_msgs.msg import DockActionGoal


class ManagerPick(ManagerAstract.ManagerAbstractClass):
    def __init__(self, dock_action, elevator_action, dock_frame = "", robot_dock_frame="" ,maximum_velocity_x=0.4, maximum_velocity_y=0.4, maximum_velocity_angular_z = 0.3):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        
        global actionlib
	global SetElevatorAction
	global DockAction
	global SetElevatorGoal
	global DockActionGoal
	self.dock_action = dock_action
        self.elevator_action = elevator_action
	self.dock_frame = dock_frame
	self.robot_dock_frame = robot_dock_frame
	self.maximum_velocity_x = maximum_velocity_x
	self.maximum_velocity_y = maximum_velocity_y
	self.maximum_velocity_angular_z = maximum_velocity_angular_z
	self.current_id = None
	# Creates the SimpleActionClient, passing the type of the action
        # (FibonacciAction) to the constructor.
        self.client_docking = actionlib.SimpleActionClient(self.dock_action, DockAction)
	self.client_elevator = actionlib.SimpleActionClient(self.dock_action, SetElevatorAction)
        
        
    
    
    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
        print ("ManagerPick-> Recive a action")

    def stateIdle(self):
        None

    def stateExecute(self):
	print ("Waiting for docking server: " + self.dock_action)
	self.client_docking.wait_for_server()
	print ("Waiting for docking server: " + self.dock_action)
	self.client_docking.wait_for_server()
	self.changeState(self.STATES_EXECUTING, "Executing the pick action")

    #NECESSARY A DEFINITION
    def stateExecuting(self):
	print("Elevator lower")
	self.elevator_lower_client()
	self.dock_client()
	print("Elevator raise")	
	self.elevator_raise_client()
	self.changeState(self.STATES_FINALIZED, "Finalized")
        
    def stateFail(self):
        self.changeState(self.STATES_FINALIZED, "Fail")

    def stateError(self):
        print("Error")
	self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
	self.current_id = None
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")


    def dock_client():
        # Creates a goal to send to the action server.
        goal_msg = DockActionGoal()
        self.current_id = self.frame_id
        goal_msg.goal_id.id = self.dock_frame + "_" + str(randint(1, 10000))
        goal_msg.goal.dock_frame = self.dock_frame
        goal_msg.goal.robot_dock_frame = self.robot_dock_frame
        goal_msg.goal.maximum_velocity.linear.x = self.maximum_velocity_x
        goal_msg.goal.maximum_velocity.linear.y = self.maximum_velocity_y
        goal_msg.goal.maximum_velocity.angular.z = self.maximum_velocity_angular_z

        # Sends the goal to the action server.
        self.dock_action.send_goal(goal_msg)

        # Waits for the server to finish performing the action.
        self.dock_action.wait_for_result()

        # Prints out the result of executing the action
        return self.dock_action.get_result()  


    def elevator_raise_client():
        # Creates a goal to send to the action server.
        goal_msg = SetElevatorGoal()
        self.current_id = self.frame_id
        goal_msg.action = goal_msg.RAISE

        # Sends the goal to the action server.
        self.dock_action.send_goal(goal_msg)

        # Waits for the server to finish performing the action.
        self.dock_action.wait_for_result()

        # Prints out the result of executing the action
        return self.dock_action.get_result() 



    def elevator_lower_client():
        # Creates a goal to send to the action server.
        goal_msg = SetElevatorGoal()
        self.current_id = self.frame_id
        goal_msg.action = goal_msg.LOWER

        # Sends the goal to the action server.
        self.dock_action.send_goal(goal_msg)

        # Waits for the server to finish performing the action.
        self.dock_action.wait_for_result()

        # Prints out the result of executing the action
        return self.dock_action.get_result() 
