#!/usr/bin/env python

from importlib import import_module

import rosservice
from procedures_msgs.srv import ProcedureQuery, ProcedureQueryRequest

from command_service_interface import *


class CommandProcedureInterface(CommandServiceInterface):

    def __init__(self, name, parameters):
        CommandServiceInterface.__init__(self, name, parameters)

        self.query_state_request = self.type_query_request()
        self.query_state_request.header.id = -1
        self.client = rospy.ServiceProxy(self.namespace + '/add', self.service_type)
        self.query_state_client = rospy.ServiceProxy(self.namespace + '/query_state', self.service_query_type)
        self.cancel_client = rospy.ServiceProxy(self.namespace + '/cancel', self.service_query_type)

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface 
        '''
        CommandServiceInterface.set_parameters(self, parameters)

        self.namespace = 'robot_local_control/NavigationComponent/' + parameters['namespace']

    def import_messages(self):
        namespace_add = self.namespace + '/add'
        resolved_service_name = rospy.resolve_name(namespace_add)
        service_type = rosservice.get_service_type(resolved_service_name)

        if service_type is None:
            msg = 'Could not find \"' + namespace_add + '\" service server.'
            CommandInterface.raise_exception(self, msg)

        service_type_module, service_type_name = tuple(service_type.split('/'))
        srv_module = import_module(service_type_module + '.srv')

        self.type_name = service_type
        self.service_type = getattr(srv_module, service_type_name)
        self.type_request = getattr(srv_module, service_type_name + 'Request')
        self.type_response = getattr(srv_module, service_type_name + 'Response')

        namespace_query = self.namespace + '/query_state'
        resolved_service_name = rospy.resolve_name(namespace_query)
        service_type = rosservice.get_service_type(resolved_service_name)

        if service_type is None:
            msg = 'Could not find \"' + namespace_add + '\" service server.'
            CommandInterface.raise_exception(self, msg)

        service_type_module, service_type_name = tuple(service_type.split('/'))
        srv_module = import_module(service_type_module + '.srv')

        self.type_query_name = service_type
        self.service_query_type = getattr(srv_module, service_type_name)
        self.type_query_request = getattr(srv_module, service_type_name + 'Request')
        self.type_query_response = getattr(srv_module, service_type_name + 'Response')

    def send_command(self, args):
        '''
            Return True if the command has been sent
        '''
        parsed_arguments = self.parse_args(args)
        self.build_msg(parsed_arguments)
        res = self.client.call(self.request)
        
        if res.result.result == 'error':
            return False,res.result.message

        return True,""

    def cancel_cmd(self):
        res = self.cancel_client.call(self.query_state_request)
        current_state = res.state.current_state
        last_event = res.state.last_event

        return current_state == 'finished' and last_event == 'cancel'

    def feedback_cb(self, feedback):
        self.feedback_msg = feedback
        self.parse_feedback()
    
    def parse_feedback(self):
        current_state = self.feedback_msg.state.current_state
        last_message = self.feedback_msg.last_message.data
        self.feedback_str = 'current_state: %s , last_message: %s' %(current_state, last_message)

    def get_feedback(self):
        '''
            Return feedback string
        '''
        return self.feedback_str

    def is_active(self):
        res = self.query_state_client.call(self.query_state_request)
        current_state = res.state.current_state
        self.feedback_cb(res)
        return current_state != 'finished'

    def has_succeeded(self):
        current_state = self.feedback_msg.state.current_state
        last_event = self.feedback_msg.state.last_event
        return current_state == 'finished' and last_event == 'finish'

    def get_result(self):
        '''
            Gets the result  as string of the action
        '''
        self.feedback_msg
        return self.feedback_str
        
    @abstractmethod
    def build_msg(self, args):
        '''
            Return True if the command has been sent
        '''
        pass
