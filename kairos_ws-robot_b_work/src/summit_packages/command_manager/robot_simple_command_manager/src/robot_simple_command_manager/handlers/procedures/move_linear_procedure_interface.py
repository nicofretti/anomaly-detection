#!/usr/bin/env python


from ..command_procedure_interface import *


class MoveLinearProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)

        self.args_description = ['x', 'y']
        self.args_types = [float, float]
        self.args_void_allowed = [False, False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface 
        '''
        CommandProcedureInterface.set_parameters(self, parameters)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list: 
            self.request.procedure.goal.x = args[0]
            self.request.procedure.goal.y = args[1]
            self.request.procedure.goal.theta = 0.0


