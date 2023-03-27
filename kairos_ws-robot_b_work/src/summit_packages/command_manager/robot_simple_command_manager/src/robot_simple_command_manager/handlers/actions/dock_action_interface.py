#!/usr/bin/env python

from ..command_action_interface import *


class DockActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        CommandActionInterface.__init__(self, name, parameters)
        
        self.args_description = ['dock_frame', 'robot_docke_frame', 'offset_x', 'offset_y', 'offset_theta']
        self.args_types = [str, str, float, float, float]
        self.args_void_allowed = [False, False, False, False, False]

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
            self.goal.dock_frame = args[0]
            self.goal.robot_dock_frame = args[1]
            self.goal.dock_offset.x = args[2]
            self.goal.dock_offset.y = args[3]
            self.goal.dock_offset.theta = args[4]

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        remaining = self.feedback_msg.remaining
        msg = "x: %2f y: %2f theta: %2f" % (remaining.x, remaining.y, remaining.theta)
        return msg