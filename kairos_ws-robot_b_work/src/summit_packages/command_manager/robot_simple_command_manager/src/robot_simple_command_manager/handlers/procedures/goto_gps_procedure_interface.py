#!/usr/bin/env python


from ..command_procedure_interface import *
from robot_local_control_msgs.msg import PointGPS


class GoToGPSProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)
        # frame,  max_vel, [lat,lon lat,lon ...],
        self.args_types = [str, float, list]
        self.args_void_allowed = [False, False, False]

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface
        '''
        CommandProcedureInterface.set_parameters(self, parameters)

    # Add a custom parser
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
        elif desired_type == list:
            # Custom parser
            parsed_argument = self.parse_list(argument)
        else:
            self.raise_exception("Unknown type: the type " + desired_type + " to parse is not supported")

        return parsed_argument

    def parse_list(self, param):
        '''
            Return a list of [float,float] (lat,lon) -> [[3,4],[123.1,12312.4]]
            a -1 with the error message
        '''
        try:
            coordinates = param.replace('[', '').replace(']', '').split(';')
            coordinates_array = []
            for c in coordinates:
                formatted_coordinate = c.split(',')
                # Convert to float
                formatted_coordinate = [float(formatted_coordinate[0]), float(formatted_coordinate[1])]
                coordinates_array.append(formatted_coordinate)
            return coordinates_array
        except ValueError as e:
            rospy.logerr(e)
            self.raise_exception('Expected input type: [float,float ... float,float]. Error parsing argument: ' + param)

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''

        if type(args) == list:
            # self.request.procedure.goals = [PointGPS(), ...]
            goals = []
            for i in args[2]:
                p = PointGPS()
                p.latitude = i[0]
                p.longitude = i[1]
                goals.append(p)
            self.request.procedure.frame_id = args[0]
            self.request.procedure.max_velocity = args[1]
            self.request.procedure.goals = goals
