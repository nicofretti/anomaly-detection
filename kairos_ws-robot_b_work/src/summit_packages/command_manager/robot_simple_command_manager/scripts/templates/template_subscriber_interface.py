#!/usr/bin/env python

from robot_simple_command_manager.handlers.command_subscriber_interface import *

# DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
# YAML CONFIGURATION
# command_handlers:
#  - TEMPLATE
# 
# TEMPLATE:
#   type: template_subscriber_interface/TemplateSubscriberInterface
#   namespace: topic_ns_you_want_to_use

class TemplateSubscriberInterface(CommandSubscriberInterface):
    def __init__(self, name, parameters):
        CommandSubscriberInterface.__init__(self, name, parameters)
        
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
        # ADD THE TYPES OF THE ARGUMENTS THE COMMAND WILL USE
        # DESCRIPTION SHOULD CONTAIN ONE STRING PER ARG
        self.args_description = ['expected_value']
        self.args_types = [bool]
        self.args_void_allowed = [False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandSubscriberInterface.set_parameters(self, parameters)
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
        # ADD NECESSARY PARAMETERS DEFINED IN THE YAML FILE 
        # EXAMPLE FOR A CUSTOM STRING PARAMETER
        # self.custom_param = self.get_parameter('custom_param', 'default_value')
    
    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''
        if type(args) == list:
            self.desired_value.data = args[0]