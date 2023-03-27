#!/usr/bin/env python
import rospy

from rc_server import XMLRPCServer 

rospy.init_node("rc_xmlrpc_server_node")

rc_node = XMLRPCServer()

rospy.loginfo('%s: starting' % (rospy.get_name()))

rc_node.start()
