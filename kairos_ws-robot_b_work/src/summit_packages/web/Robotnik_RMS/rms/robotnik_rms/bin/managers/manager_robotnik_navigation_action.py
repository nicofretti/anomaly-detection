#!/usr/bin/env python
import rospy
import abc
# seed the pseudorandom number generator
from random import seed
from random import randint
import manager_astract_base_class as ManagerAstract


from robotnik_navigation_msgs.msg import DockActionGoal
from actionlib_msgs.msg import GoalStatusArray


class ManagerRobotnikNavigationAction(ManagerAstract.ManagerAbstractClass):
    def __init__(self, goal_topic="laser_reflectors_docker/goal", status_topic="laser_reflectors_docker/status", robot_dock_frame="base_footprint", frame_id="", dock_frame="base_dock", maximum_velocity_x=0.5, maximum_velocity_y=0.5, maximum_velocity_angular_z=0.4):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global DockActionGoal
        global GoalStatusArray
	global randint
	global seed
        self.goal_topic_ = goal_topic
        self.status_topic_ = status_topic
        self.status_subscribe_ = rospy.Subscriber(
            self.status_topic_, GoalStatusArray, self.StatusCallBack)
        self.goal_publish_ = rospy.Publisher(
            self.goal_topic_, DockActionGoal, queue_size=1)
        self.current_goal_id = None
        self.robot_dock_frame = robot_dock_frame
        self.frame_id = frame_id
        self.dock_frame = dock_frame
        self.maximum_velocity_x = maximum_velocity_x
        self.maximum_velocity_y = maximum_velocity_y
        self.maximum_velocity_angular_z = maximum_velocity_angular_z
        self.started = False
        self.finsh = False

    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
        if (self.current_goal_id != None):
            return False
        self.started = False
        self.finsh = False
        self.sendGoal()
        self.changeState(self.STATES_EXECUTING,
                         "Accepted with id " + self.current_id)
	return True

    def stateExecuting(self):
        if (self.started == False):
            print "Waiting to accept the action id: " + self.current_goal_id
        else:
            if(self.finsh):
                self.changeState(self.STATES_FINALIZED,
                                 "Finish with id " + self.current_id)
            else:
                print "Waiting to finish the action id: "+self.current_goal_id

    def stateFail(self):
        self.current_id = None
        self.changeState(self.STATES_IDLE, "Fail")
        # None

    def stateError(self):
        print("Error")
        # None

    def stateFinalized(self):
        self.current_goal_id = None
        self.started = False
        self.finsh = False
        print("ManagerRobotnikNavigation--->Finsihed: " + str(self.current_action_msg_))
	self.changeState(self.STATES_IDLE, "Finalized")

    def sendGoal(self):
        goal_msg = DockActionGoal()
        self.current_id = self.frame_id
        goal_msg.goal_id.id = self.dock_frame + "_" + str(randint(1, 10000))
        goal_msg.goal.dock_frame = self.dock_frame
        goal_msg.goal.robot_dock_frame = self.robot_dock_frame
        goal_msg.goal.maximum_velocity.linear.x = self.maximum_velocity_x
        goal_msg.goal.maximum_velocity.linear.y = self.maximum_velocity_y
        goal_msg.goal.maximum_velocity.angular.z = self.maximum_velocity_angular_z
        self.current_goal_id = goal_msg.goal_id.id
        connections = self.goal_publish_.get_num_connections()
        while connections <= 0:
	    print connections
            connections = self.goal_publish_.publish()
            rospy.sleep(0.01)
        self.goal_publish_.publish(goal_msg)

    def StatusCallBack(self, message):
        for status in message.status_list:
            if (status.goal_id.id == self.current_goal_id):
		if(self.started == False):
			self.started = True
		if(status.status == status.SUCCEEDED or status.status == status.ABORTED):
			self.finsh = True

