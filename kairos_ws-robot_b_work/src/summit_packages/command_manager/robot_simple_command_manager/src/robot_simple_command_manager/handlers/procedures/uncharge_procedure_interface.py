#!/usr/bin/env python


from ..command_procedure_interface import *


class UnchargeProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)

        self.args_types = []
        self.args_void_allowed = []

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
            self.request.procedure.charge_station = '' #args[0]

