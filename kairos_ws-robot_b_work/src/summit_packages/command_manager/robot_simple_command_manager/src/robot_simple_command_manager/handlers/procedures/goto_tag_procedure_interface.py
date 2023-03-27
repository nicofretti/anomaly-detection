#!/usr/bin/env python

# This handler structure is temporary. Not standard.

from robot_local_control_msgs.msg import Pose2DStamped, LocalizationStatus
from poi_manager_msgs.srv import GetPOI, GetPOIRequest
from tf.transformations import euler_from_quaternion
from ..command_procedure_interface import *


class GoToTagProcedureInterface(CommandProcedureInterface):
    def __init__(self, name, parameters):
        CommandProcedureInterface.__init__(self, name, parameters)

        self.args_description = ['tag_name']
        self.args_types = [str]
        self.args_void_allowed = [False]
        self.localization_sub = rospy.Subscriber("robot_local_control/LocalizationComponent/status", LocalizationStatus, self.localization_cb)

    def set_parameters(self, parameters):
        '''
            Set all the required parameters of the interface 
        '''
        CommandProcedureInterface.set_parameters(self, parameters)
        self.get_poi_ns = self.get_parameter('get_poi_ns', 'poi_manager/get_poi')
        self.frame_id = self.get_parameter('frame_id', 'robot_map')
        self.environment = self.get_parameter('default_environment', 'unknown')

    def build_msg(self, args):
        '''
            Return the desired goal or None
        '''
        client = rospy.ServiceProxy(self.get_poi_ns, GetPOI)
        request = GetPOIRequest()
        request.name = args[0]
        request.environment = self.environment
        response = client.call(request)
        if not response.success:
            self.raise_exception("Could not get pose from POI "+args[0]+" ns = "+self.get_poi_ns+", environment = "+self.environment)

        if type(args) == list: 

            self.request.procedure.goals = [Pose2DStamped()]
            self.request.procedure.goals[0].header.frame_id = self.frame_id
            self.request.procedure.goals[0].pose.x = response.p.pose.position.x
            self.request.procedure.goals[0].pose.y = response.p.pose.position.y
            quat = [response.p.pose.orientation.x, response.p.pose.orientation.y, response.p.pose.orientation.z, response.p.pose.orientation.w]
            (_, _, yaw) = euler_from_quaternion (quat)
            self.request.procedure.goals[0].pose.theta = yaw
    
    def localization_cb(self, msg):
        self.environment = msg.environment
