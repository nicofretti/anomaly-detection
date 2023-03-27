#!/usr/bin/env python

from tf.transformations import quaternion_from_euler, euler_from_quaternion

from ..command_action_interface import *
import math

class MoveBaseWaypointActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        CommandActionInterface.__init__(self, name, parameters)

        self.args_description = ['x', 'y', 'theta', 'distance_threshold']
        self.args_types = [float, float, float, float]
        self.args_void_allowed = [False, False, False, False]
        self.distance_threshold = 0.0
        self.finished_by_threshold = False

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandActionInterface.set_parameters(self, parameters)

        self.frame_id = self.get_parameter('frame_id', 'map')

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''
        if type(args) == list:

            self.distance_threshold = args[3]
            self.finished_by_threshold = False

            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.goal.target_pose.header.frame_id = self.frame_id
            self.goal.target_pose.pose.position.x = args[0]
            self.goal.target_pose.pose.position.y = args[1]
            quaternion = quaternion_from_euler(0, 0, args[2])
            self.goal.target_pose.pose.orientation.x = quaternion[0]
            self.goal.target_pose.pose.orientation.y = quaternion[1]
            self.goal.target_pose.pose.orientation.z = quaternion[2]
            self.goal.target_pose.pose.orientation.w = quaternion[3]

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        position = self.feedback_msg.base_position.pose.position
        quaternion = self.feedback_msg.base_position.pose.orientation
        orientation = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        msg = "x: %2f y: %2f theta: %2f" % (position.x, position.y, orientation[2])
        return msg

    def is_active(self):
        active = CommandActionInterface.is_active(self)

        if active == True:
            position = self.feedback_msg.base_position.pose.position
            remaining_x = self.goal.target_pose.pose.position.x - position.x
            remaining_y = self.goal.target_pose.pose.position.y - position.y
            remaining = math.sqrt(remaining_x * remaining_x + remaining_y * remaining_y)

            if remaining <  self.distance_threshold:
                self.finished_by_threshold = True
                return False
            
        return active
    
    def has_succeeded(self):
        if self.finished_by_threshold == True:
            return True

        state = self.client.get_state()
        return state == GoalStatus.SUCCEEDED
