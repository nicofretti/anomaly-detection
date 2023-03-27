#!/usr/bin/env python
import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandResult

from xmlrpc_server.rc_server import XMLRPCServer
from robotnik_msgs.srv import set_float_value, set_float_valueResponse, set_float_valueRequest

COMMAND_IDLE = 1
COMMAND_MOVE = 2

class XMLRPCGripperController(XMLRPCServer):
    
    def __init__(self):
        XMLRPCServer.__init__(self)

        self.__command = COMMAND_IDLE
        self.__desired_pos = 0.0
        self.__real_pose = 0.0


    def ros_read_params(self):
        XMLRPCServer.ros_read_params(self)

        # Joint limits in millimeters
        self.__min_joint_pos = rospy.get_param("~min_pos", 0.0)
        self.__max_joint_pos = rospy.get_param("~max_pos", 30.0)
        

        return 0

    
    def setup(self):
        self.server.register_function(self.get_command, 'get_command')
        self.server.register_function(self.get_desired_pos, 'get_desired_pos')
        self.server.register_function(self.set_idle, 'set_idle')
        self.server.register_function(self.set_real_pos, 'set_real_pos')
        XMLRPCServer.setup(self)

        return 0

    def ros_setup(self):
        XMLRPCServer.ros_setup(self)

        self.__service_server = rospy.Service('~set_pos', set_float_value, self.set_pos)
        self.__action_server = actionlib.SimpleActionServer('~cmd', GripperCommandAction, None, False)
        self.__action_server.start()

    def ready_state(self):
        if self.__command == COMMAND_IDLE:
            if self.__action_server.is_new_goal_available():
                goal = self.__action_server.accept_new_goal()
                request = set_float_valueRequest()
                request.value = goal.command.position
                self.set_pos(request)

        elif self.__command == COMMAND_MOVE:
            if self.__action_server.is_preempt_requested():
                self.preempt_requested()

    def get_command(self):
        rospy.loginfo("Command requested: %i", self.__command)
        return self.__command
    
    def get_desired_pos(self):
        rospy.loginfo("Desired pos requested: %d", self.__desired_pos)
        return self.__desired_pos
    
    def set_idle(self):
        rospy.loginfo("Set idle requested")
        self.__command = COMMAND_IDLE
    
    def set_real_pos(self, millimeters):
        rospy.logwarn("I need to update joint position -> %d" %(millimeters))
        return 0
    
    def set_pos(self, request):
        response = set_float_valueResponse()
        response.ret = True
        if request.value > self.__max_joint_pos:
            self.__desired_pos = self.__max_joint_pos
            response.errorMessage.data = "Requested gripper position is higher than the max allowed. Desired position set to " + str(self.__max_joint_pos) + "mm"
        elif request.value < self.__min_joint_pos:
            self.__desired_pos = self.__min_joint_pos
            response.errorMessage.data = "Requested gripper position is lower than the min allowed. Desired position set to " + str(self.__min_joint_pos) + "mm"
        else:
            self.__desired_pos = request.value
            response.errorMessage.data = "No error"

        
        self.__command = COMMAND_MOVE
        rospy.loginfo("%s::set_pos: Desired pos set to %dmm" %(self._node_name, self.__desired_pos))

        return response