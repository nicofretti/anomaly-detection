#!/usr/bin/env python

from ..command_service_interface import *


class SaveFrameServiceInterface(CommandServiceInterface):
    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

        self.args_description = ["frame_id"]
        self.args_types = [str]
        self.args_void_allowed = [False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandServiceInterface.set_parameters(self, parameters)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''
        if type(args) == list:
            self.request.frame_id = args[0]

    def parse_feedback(self):
        '''
            Parse from feedback object to string
        '''
        return self.feedback_msg.message
    
    def has_succeeded(self):
        return self.feedback_msg.success
