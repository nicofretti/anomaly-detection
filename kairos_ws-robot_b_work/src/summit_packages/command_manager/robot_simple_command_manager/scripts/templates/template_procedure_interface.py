#!/usr/bin/env python

from robot_simple_command_manager.handlers.command_procedure_interface import *

# DELETE THIS BLOCK AFTER IMPLEMENTING YOUR HANDLER 
# YAML CONFIGURATION
# command_handlers:
#  - TEMPLATE
# 
# TEMPLATE:
#   type: template_procedure_interface/TemplateProcedureInterface
#   namespace: procedure_ns_you_want_to_use

class TemplateProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)

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
        CommandProcedureInterface.set_parameters(self, parameters)
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
            # THAT USES THE PROCEDURE DEFINED BY THE NAMESPACE OF THE HANDLER
            # FILL THE REQUEST MESSAGE WITH THE ARGUMENTS PASSED TO THE COMMAND

            # EXAMPLE
            #self.request.procedure.goal.x = 0.0
            #self.request.procedure.goal.y = 0.0
            #self.request.procedure.goal.theta = args[0]

