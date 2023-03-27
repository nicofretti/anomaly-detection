#!/usr/bin/env python

from ..command_action_interface import *


class WaitActionInterface(CommandActionInterface):
    def __init__(self, name, parameters):
        self.client = None
        self.feedback_msg = None
        self.feedback_str = ''
        self.init_time = rospy.Time(0)
        self.time_to_wait = 0.0

        self.name = name

        self.args_description = ['seconds']
        self.args_types = [float]
        self.args_void_allowed = [False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        self.check_parameters(parameters)

        self.parameters = parameters
        self.namespace = parameters['namespace']

    def check_parameters(self, parameters):
        '''
            Checks that the handler has all required parameters set
        '''
        if 'type' not in parameters.keys():
            self.raise_exception('check_parameters: Handler has not "type" parameter defined.')
        else:
            if not self.is_parameter_type_correct(parameters['type'], str):
                self.raise_exception('check_parameters: Handler has "type" parameter defined but is not a string.')

        return

    def send_command(self, args):
        '''
            Return True if the command has been sent
        '''
        parsed_arguments = self.parse_args(args)
        self.time_to_wait = parsed_arguments[0]
        if self.time_to_wait < 0.0:
          return False
        self.init_time = rospy.Time.now()

        return True

    def cancel_cmd(self):
        return

    def get_feedback(self):
        '''
            Return feedback string
        '''
        elapsed_time = (rospy.Time.now() - self.init_time).to_sec()
        return "Elapsed %.1f of %.1f seconds"%(elapsed_time, self.time_to_wait)

    def is_active(self):
        elapsed_time = (rospy.Time.now() - self.init_time).to_sec()
        
        return (elapsed_time < self.time_to_wait)

    def has_succeeded(self):
        
        return True

    def get_result(self):
        '''
            Gets the result as string of the action
        '''
        
        return "OK"

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        return

    def parse_feedback(self):
        '''
            Parse from feedback object to string 
        '''
        return
