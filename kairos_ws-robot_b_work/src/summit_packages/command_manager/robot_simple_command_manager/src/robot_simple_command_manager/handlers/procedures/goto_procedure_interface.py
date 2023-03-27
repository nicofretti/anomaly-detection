#!/usr/bin/env python


from robot_local_control_msgs.msg import Pose2DStamped
from ..command_procedure_interface import *


class GoToProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)

        self.args_description = ['x', 'y', 'theta']
        self.args_types = [float, float, float]
        self.args_void_allowed = [False, False, False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface 
        '''
        CommandProcedureInterface.set_parameters(self, parameters)

        self.frame_id = self.get_parameter('frame_id', 'map')

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list:
            self.request.procedure.goals = [Pose2DStamped()]
            self.request.procedure.goals[0].header.frame_id = self.frame_id
            self.request.procedure.goals[0].pose.x = args[0]
            self.request.procedure.goals[0].pose.y = args[1]
            self.request.procedure.goals[0].pose.theta = args[2]
