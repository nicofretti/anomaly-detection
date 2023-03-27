#!/usr/bin/env python

import rospy
from robotnik_signal_msgs.srv import SetSignal, SetSignalRequest

command = SetSignalRequest()
rospy.wait_for_service('leds_driver_node/set_signal')
leds_driver_client = rospy.ServiceProxy('leds_driver_node/set_signal', SetSignal)

rospy.sleep(0.5)
command.id = "EMERGENCY"
command.enable = True
leds_driver_client(command)