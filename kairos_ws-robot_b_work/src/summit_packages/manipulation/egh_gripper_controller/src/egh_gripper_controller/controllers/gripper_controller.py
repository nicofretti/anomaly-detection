#!/usr/bin/env python
import actionlib

from control_msgs.msg import GripperCommandAction, GripperCommandResult
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest

from rcomponent.rcomponent import *

IDLE_STATE = 'IDLE'
ACTION_STATE = 'RUNNING'

class GripperController(RComponent):

    def __init__(self):
        RComponent.__init__(self)
        
    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)
        
        self.status_topic_name = rospy.get_param('~status_topic_name', 'status')
        self.close_service_name = rospy.get_param('~close_service_name', 'close')
        self.open_service_name = rospy.get_param('~open_service_name', 'open')

        self.master_joint = rospy.get_param('~master_joint', 'master_joint')
        self.min_opening = rospy.get_param('~min_opening', 0.0)
        self.max_opening = rospy.get_param('~max_opening', 0.11)
        self.master_joint_direction = rospy.get_param('~master_joint_direction', -1)

        self.master_joint_open_position = self.max_opening/2.0
        self.master_joint_close_position = self.min_opening/2.0
        
        self.desired_position = None
        self.master_joint_position = None
        self.master_joint_desired_position = None


    def ros_setup(self):
        """Creates and inits ROS components"""
        RComponent.ros_setup(self)

        self.status_pub = rospy.Publisher('~' + self.status_topic_name, String, queue_size=1)
        self.close_srv = rospy.Service('~' + self.close_service_name, Trigger, self.close_cb)
        self.open_srv = rospy.Service('~' + self.open_service_name, Trigger, self.open_cb)
        self.joint_states_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=10)
        
        self.gripper_command_action_server = actionlib.SimpleActionServer('~gripper_cmd', GripperCommandAction, None, False)
        self.gripper_command_action_server.start()
        self.status = IDLE_STATE

    def ready_state(self):
        self.status_pub.publish(String(self.status))
        if self.status == IDLE_STATE:
            if self.gripper_command_action_server.is_new_goal_available():
                if self.process_new_goal(self.gripper_command_action_server.accept_new_goal()) == 0:
                    self.status = ACTION_STATE
                else:
                    self.gripper_command_action_server.set_aborted()
        elif self.status == ACTION_STATE:
            if self.gripper_command_action_server.is_preempt_requested():
                self.preempt_requested()
            else:
                threshold = (self.max_opening - self.min_opening)/100.0
                state = self.is_goal_reached(threshold)
                if state == 0:	
                    self.goal_reached()
                elif state == -2:
                    self.gripper_command_action_server.set_aborted()
                    self.status = IDLE_STATE
                else:
                    rospy.loginfo('%s::ready_state: executing gripper action'%self._node_name)
    
    def close_cb(self, msg):
        """Close gripper service server callback"""
        raise NotImplementedError("GripperController::close_cb: Should have implemented this" )
    
    def open_cb(self, msg):
        """Open gripper service server callback"""
        raise NotImplementedError("GripperController::close_cb: Should have implemented this" )

    def joint_state_cb(self, msg):
        try:
            joint_index = msg.name.index(self.master_joint)

            self.master_joint_position = msg.position[joint_index]
        except ValueError as identifier:
            pass
        
    def process_new_goal(self, new_goal):
        """Process new action goal"""
        raise NotImplementedError("GripperController::process_new_goal: Should have implemented this" )


    def write_reference(self):
        raise NotImplementedError("GripperController::write_reference: Should have implemented this" )

    def stop_goal(self):
        self.master_joint_desired_position = self.master_joint_position
        self.write_reference()

    def is_goal_reached(self, threshold):
        """
            Returns:
                0  : if the position Goal is reached
                -1 : if the position hasn't been reached yet
        """
        joint_diff = abs(self.master_joint_position - self.master_joint_desired_position)
        joint_reached = joint_diff <= threshold

        if joint_reached == True:
            return 0
        else:
            return -1
    
    def goal_reached(self):
        result = GripperCommandResult()
        result.position = self.desired_position					
        result.reached_goal = True					
        self.gripper_command_action_server.set_succeeded(result)
        self.status = IDLE_STATE

    def preempt_requested(self):
        result = GripperCommandResult()
        result.position = self.master_joint_position*2.0*self.master_joint_direction					
        result.reached_goal = False
        self.stop_goal()
        self.gripper_command_action_server.set_preempted(result)
        self.status = IDLE_STATE
