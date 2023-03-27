#!/usr/bin/env python

from tf.transformations import quaternion_from_euler, euler_from_quaternion

from ..command_service_interface import *


class ElevatorServiceInterface(CommandServiceInterface):
    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

        self.args_description = ['position']
        self.args_types = [int]
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
            self.request.action.action = args[0]

    def parse_feedback(self):
        '''
            Parse from feedback object to string
        '''
        return ''
