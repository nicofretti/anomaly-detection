#!/usr/bin/env python

from ur_msgs.srv import SetIO, SetIORequest

from fake_joint_position_publisher.srv import SetJoints, SetJointsRequest
from gripper_controller import *

class URIOGripperController(GripperController):

    def __init__(self):
        GripperController.__init__(self)
    
    def ros_read_params(self):
        """Gets params from param server"""
        GripperController.ros_read_params(self)
        
        self.ur_io_service_name = rospy.get_param('~ur_io_service_name', 'ur_driver/set_io')
        self.update_joints_service_name = rospy.get_param('~update_joints_service_name', 'joints_updater/set_joint')

    def ros_setup(self):
        """Creates and inits ROS components"""
        GripperController.ros_setup(self)

        self.set_joints_request = SetJointsRequest()
        self.set_joints_request.joints_names = [self.master_joint]
        self.open_positions = [self.master_joint_open_position]
        self.close_positions = [self.master_joint_close_position]

        self.update_joints_client = rospy.ServiceProxy(self.update_joints_service_name, SetJoints)
        
        self.set_io_request = SetIORequest()
        self.set_io_request.fun = 1
        self.set_io_request.state = 1

        self.ur_io_client = rospy.ServiceProxy(self.ur_io_service_name, SetIO)
    
    def ready_state(self):
        GripperController.ready_state(self)

    def close_cb(self, msg):
        """Close gripper service server callback"""

        self.master_joint_desired_position = self.master_joint_close_position
        
        response = TriggerResponse()
        response.message = "Gripper correctly closed"
        
        self.set_io_request.pin = 5

        set_io_res = self.ur_io_client.call(self.set_io_request)

        if set_io_res.success == True:
            self.set_joints_request.positions = self.close_positions
            update_joints_res = self.update_joints_client.call(
                self.set_joints_request)
            if update_joints_res.success == True:
                response.success = True
            else:
                response.message = "Failed updating joints positions"
        else:
            response.message = "Failed setting UR output"
        
        return response
    
    def open_cb(self, msg):
        """Open gripper service server callback"""

        self.master_joint_desired_position = self.master_joint_open_position

        response = TriggerResponse()
        response.message = "Gripper correctly opened"
        
        self.set_io_request.pin = 4
        set_io_res = self.ur_io_client.call(self.set_io_request)

        if set_io_res.success == True:
            self.set_joints_request.positions = self.open_positions
            update_joints_res = self.update_joints_client.call(
                self.set_joints_request)
            if update_joints_res.success == True:
                response.success = True
            else:
                response.message = "Failed updating joints positions"
        else:
            response.message = "Failed setting UR output"
        
        return response

    def process_new_goal(self, new_goal):
        """Process new action goal"""
        rospy.loginfo('%s::process_new_goal: New goal -> position =  %.2f, max_effort = %.2f'%(self._node_name, new_goal.command.position, new_goal.command.max_effort))	
        
        desired_position = new_goal.command.position
        
        if desired_position != 0.0 and desired_position != 1.0:
            rospy.logerr('%s::write_reference: For the moment 0.0 (close) and 1.0 (open) values are accepted'\
                        %self._node_name)
            return -1

        if desired_position == 0.0:
            self.close_cb(TriggerRequest())
        elif desired_position == 1.0:
            self.open_cb(TriggerRequest())

        return 0  
        
