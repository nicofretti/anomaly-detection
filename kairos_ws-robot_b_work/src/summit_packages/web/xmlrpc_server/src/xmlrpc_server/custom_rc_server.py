#!/usr/bin/env python
import rospy
from rc_server import XMLRPCServer

class CustomServer(XMLRPCServer):
    
    def __init__(self):
        XMLRPCServer.__init__(self)

    def ros_read_params(self):
        XMLRPCServer.ros_read_params(self)
        self.custom_server_parameter = rospy.get_param('~custom_server_parameter', "CUSTOM PARAMETER")

        return 0

    
    def setup(self):
        self.server.register_function(self.custom_server_function, 'custom_server_function')
        XMLRPCServer.setup(self)

        return 0
    
    def custom_server_function(self):
        return self.custom_server_parameter


rospy.init_node("rc_xmlrpc_server_node")

rc_node = CustomServer()

rospy.loginfo('%s: starting' % (rospy.get_name()))

rc_node.start()
