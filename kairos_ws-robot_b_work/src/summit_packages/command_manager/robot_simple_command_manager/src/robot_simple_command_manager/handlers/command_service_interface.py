#!/usr/bin/env python

from importlib import import_module

import rosservice

from command_interface import *


class CommandServiceInterface(CommandInterface):

    def __init__(self, name, parameters):
        CommandInterface.__init__(self, name, parameters)

        self.import_messages()

        self.request = self.type_request()
        self.client = rospy.ServiceProxy(self.namespace, self.service_type)

    def import_messages(self):
        resolved_service_name = rospy.resolve_name(self.namespace)
        service_type = rosservice.get_service_type(resolved_service_name)

        if service_type is None:
            msg = 'Could not find \"' + self.namespace + '\" service server.'
            CommandInterface.raise_exception(self, msg)

        service_type_module, service_type_name = tuple(service_type.split('/'))
        srv_module = import_module(service_type_module + '.srv')

        self.type_name = service_type
        self.service_type = getattr(srv_module, service_type_name)
        self.type_request = getattr(srv_module, service_type_name + 'Request')
        self.type_response = getattr(srv_module, service_type_name + 'Response')

    def send_command(self, args):
        '''
            Return True if the command has been sent
        '''
        parsed_arguments = self.parse_args(args)
        self.build_msg(parsed_arguments)
        tmp = self.client.call(self.request)
        self.feedback_cb(tmp)
        return True,""

    def cancel_cmd(self):
        print 'THIS SHOULD NEVER APPEAR.'

    def feedback_cb(self, feedback):
        self.feedback_msg = feedback
        self.feedback_str = self.parse_feedback()

    def get_feedback(self):
        '''
            Return feedback string
        '''
        return self.feedback_str

    def is_active(self):
        return False

    def has_succeeded(self):
        # TODO: This should return the result of the service as bool
        # will this depend on the type of the service?
        return True

    def get_result(self):
        '''
            Gets the result  as string of the action
        '''
        return self.feedback_str

    @abstractmethod
    def build_msg(self, args):
        '''
            Return True if the command has been sent
        '''
        pass
    
    @abstractmethod
    def parse_feedback(self):
        pass
