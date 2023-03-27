#!/usr/bin/env python
from robot_simple_command_manager.handlers.command_action_interface import *

# DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
# YAML CONFIGURATION
# command_handlers:
#  - TEMPLATE
# 
# TEMPLATE:
#   type: template_action_interface/TemplateActionInterface
#   namespace: action_ns_you_want_to_use

class TemplateActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        CommandActionInterface.__init__(self, name, parameters)

        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
        # ADD THE TYPES OF THE ARGUMENTS THE COMMAND WILL USE
        # DESCRIPTION SHOULD CONTAIN ONE STRING PER ARG
        self.args_description = []
        self.args_types = []
        self.args_void_allowed = []

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandActionInterface.set_parameters(self, parameters)
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
        # ADD NECESSARY PARAMETERS DEFINED IN THE YAML FILE 
        # EXAMPLE FOR A CUSTOM STRING PARAMETER
        # self.custom_param = self.get_parameter('custom_param', 'default_value')

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list:
            # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER
            # GOAL HAS THE SAME STRUCTURE AS THE GOAL ACTION MSG
            # THAT USES THE ACTION SERVER DEFINED BY THE NAMESPACE OF THE HANDLER
            # FILL THE GOAL MESSAGE WITH THE ARGUMENTS PASSED TO THE COMMAND
            # MOVE BASE GOAL EXAMPLE

            #self.goal.target_pose.header.stamp = rospy.Time.now()
            #self.goal.target_pose.header.frame_id = self.frame_id
            #self.goal.target_pose.pose.position.x = args[0]
            #self.goal.target_pose.pose.position.y = args[1]
            #quaternion = quaternion_from_euler(0, 0, args[2])
            #self.goal.target_pose.pose.orientation.x = quaternion[0]
            #self.goal.target_pose.pose.orientation.y = quaternion[1]
            #self.goal.target_pose.pose.orientation.z = quaternion[2]
            #self.goal.target_pose.pose.orientation.w = quaternion[3]

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER
        # FEEDBACK_MSG HAS THE SAME STRUCTURE AS THE ACTION FEEDBACK MSG
        # THAT USES THE ACTION SERVER DEFINED BY THE NAMESPACE OF THE HANDLER
        # DEFINE THE RETURN VALUE BASED ON THE ACTION FEEDBACK
        return str(self.feedback_msg)