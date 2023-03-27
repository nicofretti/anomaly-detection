#!/usr/bin/env python
import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

s = xmlrpclib.ServerProxy('http://0.0.0.0:50000')
print s.get_state()
print s.get_desired_freq()
print s.get_real_freq()
print s.get_state_description()
print s.custom_server_function()


# gripper
#print("COMMAND", s.get_command())
#print("DESIRED POS", s.get_desired_pos())
#s.set_real_pos(100)
