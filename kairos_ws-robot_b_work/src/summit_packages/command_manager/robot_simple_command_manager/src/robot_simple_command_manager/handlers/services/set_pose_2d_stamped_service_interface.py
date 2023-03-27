#!/usr/bin/env python

from ..command_service_interface import *


class SetPose2dStampedServiceInterface(CommandServiceInterface):
    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

        self.args_description = ["frame_id", "x", "y", "theta"]
        self.args_types = [str, float, float, float]
        self.args_void_allowed = [False, False, False, False, False]

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
            self.request.pose.header.frame_id = args[0]
            self.request.pose.pose.x = args[1]
            self.request.pose.pose.y = args[2]
            self.request.pose.pose.theta = args[3]



    def parse_feedback(self):
        '''
            Parse from feedback object to string
        '''
        return self.feedback_msg.message
    
    def has_succeeded(self):
        return self.feedback_msg.success
