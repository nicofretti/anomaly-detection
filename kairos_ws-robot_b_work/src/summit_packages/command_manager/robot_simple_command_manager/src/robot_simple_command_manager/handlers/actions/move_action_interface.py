#!/usr/bin/env python

from ..command_action_interface import *


class MoveActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        CommandActionInterface.__init__(self, name, parameters)

        self.args_description = ['x', 'y']
        self.args_types = [float, float]
        self.args_void_allowed = [False, False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandActionInterface.set_parameters(self, parameters)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list:
            self.goal.goal.x = args[0]
            self.goal.goal.y = args[1]
            self.goal.goal.theta = 0.0

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        remaining = self.feedback_msg.remaining
        msg = "x: %2f y: %2f theta: %2f" % (remaining.x, remaining.y, remaining.theta)
        return msg

class TurnActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        CommandActionInterface.__init__(self, name, parameters)

        self.args_description = ['theta']
        self.args_types = [float]
        self.args_void_allowed = [False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandActionInterface.set_parameters(self, parameters)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list:
            self.goal.goal.x = 0.0
            self.goal.goal.y = 0.0
            self.goal.goal.theta = args[0]

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        remaining = self.feedback_msg.remaining
        msg = "x: %2f y: %2f theta: %2f" % (remaining.x, remaining.y, remaining.theta)
        return msg

