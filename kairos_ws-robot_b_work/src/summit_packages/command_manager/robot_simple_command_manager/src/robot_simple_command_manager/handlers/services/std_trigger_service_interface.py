#!/usr/bin/env python

from ..command_service_interface import *


class StdTriggerServiceInterface(CommandServiceInterface):
    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

        self.args_description = []
        self.args_types = []
        self.args_void_allowed = []

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandServiceInterface.set_parameters(self, parameters)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''


    def parse_feedback(self):
        '''
            Parse from feedback object to string
        '''
        return self.feedback_msg.message
    
    def has_succeeded(self):
        return self.feedback_msg.success
