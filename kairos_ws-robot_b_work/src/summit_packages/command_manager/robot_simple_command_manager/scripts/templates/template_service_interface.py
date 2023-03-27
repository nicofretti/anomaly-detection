#!/usr/bin/env python

from robot_simple_command_manager.handlers.command_service_interface import *

# DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
# YAML CONFIGURATION
# command_handlers:
#  - TEMPLATE
# 
# TEMPLATE:
#   type: template_service_interface/TemplateServiceInterface
#   namespace: service_ns_you_want_to_use

class TemplateServiceInterface(CommandServiceInterface):
    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

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
        CommandServiceInterface.set_parameters(self, parameters)
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
            # REQUEST HAS THE SAME STRUCTURE AS THE SRV REQ MSG
            # THAT USES THE SERVICE DEFINED BY THE NAMESPACE OF THE HANDLER
            # FILL THE REQUEST MESSAGE WITH THE ARGUMENTS PASSED TO THE COMMAND        
            self.request.data = args[0]


    def parse_feedback(self):
        '''
            Parse from feedback object to string
        '''
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER
        # FEEDBACK_MSG HAS THE SAME STRUCTURE AS THE SRV RESPONSE MSG
        # THAT USES THE SERVICE DEFINED BY THE NAMESPACE OF THE HANDLER
        # DEFINE THE RETURN VALUE BASED ON THE RESPONSE OF THE SERVICE        
            
        return self.feedback_msg.message
    
    def has_succeeded(self):
        # DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER
        # FEEDBACK_MSG HAS THE SAME STRUCTURE AS THE SRV RESPONSE MSG
        # THAT USES THE SERVICE DEFINED BY THE NAMESPACE OF THE HANDLER
        # DEFINE THE SUCCESS VALUE BASED ON THE RESPONSE OF THE SERVICE  
        return self.feedback_msg.success
