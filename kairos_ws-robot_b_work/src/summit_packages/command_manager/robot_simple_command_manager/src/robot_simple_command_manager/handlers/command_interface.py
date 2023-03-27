#!/usr/bin/env python
from abc import ABCMeta, abstractmethod
import rospy

class CommandInterface():
    __metaclass__ = ABCMeta
    
    def __init__(self, name, parameters):
        self.client = None
        self.feedback_msg = None
        self.feedback_str = ''

        self.name = name
        self.args_description = []
        self.args_types = []
        self.args_void_allowed = []  # Not used yet

        self.set_parameters(parameters)

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

        if 'namespace' not in parameters.keys():
            self.raise_exception('check_parameters: Handler has not "namespace" parameter defined.')
        else:
            if not self.is_parameter_type_correct(parameters['namespace'], str):
                self.raise_exception('check_parameters: Handler has "namespace" parameter defined but is not a str.')

    def get_parameter(self, param_key, default_value):
        '''
            Return the desired paremter or the default_value
        '''
        parameter = None
        if self.parameters.has_key(param_key):
            parameter = self.parameters[param_key]
        else:
            parameter = default_value
            rospy.logwarn('Setting the parameter "%s" to the default value "%s"'
                          % (param_key, str(default_value)))

        return parameter

    def raise_exception(self, msg):
        '''
            Raise an exception related to the parameters check
        '''
        raise Exception(msg)

    def is_parameter_type_correct(self, parameter, desired_type):
        '''
            Returns True if the type of the parameter equals to the desired_type
        '''
        return type(parameter) == desired_type

    def parse_args(self, args):
        '''
            Return the list of arguments correctly typed.
            If any argument is wrong, raises an exception.
        '''
        # TODO: check if an argument can be void/null/none
        parsed_arguments = []
        if len(self.args_types) != len(args):
            self.raise_exception("Incorrect number of arguments to parse. Should be %d vs %d" %
                                 (len(self.args_types), len(args)))

        for index in range(0, len(args)):
            parsed_argument = self.parse_arg(args[index], self.args_types[index])
            parsed_arguments.append(parsed_argument)

        return parsed_arguments

    def parse_arg(self, argument, desired_type):
        '''
            Converts an string into a specific type, as long as it is feasible

            Args:
                argument: string to be parsed
                desired_type: type to parse the param
            Return:
                - The parsed argument as the desired_type if OK
                - Raises an exception if ERROR
        '''

        if desired_type == bool:
            parsed_argument = self.parse_boolean(argument)
        elif desired_type == int:
            parsed_argument = self.parse_int(argument)
        elif desired_type == float:
            parsed_argument = self.parse_float(argument)
        elif desired_type == str:
            parsed_argument = argument
        else:
            self.raise_exception("Unknown type: the type " + desired_type + " to parse is not supported")

        return parsed_argument

    def parse_boolean(self, param):
        '''
            Return a tuple with a 1 and the boolean value provided as param or
            a -1 with the error message
        '''
        param_lower = param.lower()
        if param_lower == 'true':
            return True
        elif param_lower == 'false':
            return False
        else:
            self.raise_exception('Expected input: true or false. Error parsing argument: ' + param)

    def parse_int(self, param):
        '''
            Return a tuple with a 1 and the int value provided as param or
            a -1 with the error message
        '''
        tmp = param
        if tmp.startswith("-"):
            tmp = param[1:]

        if tmp.isdigit():
            return int(param)
        else:
            self.raise_exception('Expected input type: int. Error parsing argument: ' + param)

    def parse_float(self, param):
        '''
            Return a tuple with a 1 and the float value provided as param or
            a -1 with the error message
        '''
        try:
            return float(param)
        except ValueError:
            self.raise_exception('Expected input type: float. Error parsing argument: ' + param)

    def get_info(self):
        types = [args_type.__name__ for args_type in self.args_types]
        descriptions = [arg_description for arg_description in self.args_description]
        return (types, descriptions)

    @abstractmethod
    def import_messages(self):
        pass
    
    @abstractmethod
    def send_command(self, args):
        '''
            Return True if the command has been sent
        '''
        pass
    
    @abstractmethod
    def cancel_cmd(self):
        pass

    @abstractmethod
    def feedback_cb(self, feedback):
        pass

    @abstractmethod
    def parse_feedback(self):
        pass

    @abstractmethod
    def get_feedback(self):
        '''
            Return feedback string
        '''
        pass
    
    @abstractmethod
    def is_active(self):
        pass
    
    @abstractmethod
    def has_succeeded(self):
        pass
    
    @abstractmethod
    def get_result(self):
        '''
            Gets the result  as string of the action
        '''
        pass

    @abstractmethod
    def build_msg(self, args):
        '''
            Return True if the command has been sent
        '''
        pass
