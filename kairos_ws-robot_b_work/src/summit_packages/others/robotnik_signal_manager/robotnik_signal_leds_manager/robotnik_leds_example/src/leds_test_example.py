#!/usr/bin/env python

import rospy
from robotnik_signal_msgs.srv import SetSignal, SetSignalRequest


class LedsControl:

	def __init__(self):

		# Init node
		rospy.init_node('leds_test_example_node')

		rospy.loginfo("Starting example...")		

        # Check if ALS driver is already
		rospy.wait_for_service('leds_driver_node/set_signal')

		rospy.loginfo("Running example!")


	def setEffect(self):

		command = SetSignalRequest()
		rospy.wait_for_service('leds_driver_node/set_signal')
		leds_driver_client = rospy.ServiceProxy('leds_driver_node/set_signal', SetSignal)

		rospy.sleep(2)
		command.id = "EMERGENCY"
		command.enable = True
		leds_driver_client(command)

		rospy.sleep(2)
		command.id = "TURN_LEFT"
		command.enable = True
		leds_driver_client(command)

		rospy.sleep(2)
		command.id = "TURN_RIGHT"
		command.enable = True
		leds_driver_client(command)

		rospy.sleep(2)	
		command.id = "FORWARD"
		command.enable = True
		leds_driver_client(command)


	def clearEffect(self):

		command = SetSignalRequest()
		rospy.wait_for_service('leds_driver_node/set_signal')
		leds_driver_client = rospy.ServiceProxy('leds_driver_node/set_signal', SetSignal)

		rospy.sleep(2)
		command.id = "EMERGENCY"
		command.enable = False
		leds_driver_client(command)

		rospy.sleep(2)
		command.id = "TURN_LEFT"
		command.enable = False
		leds_driver_client(command)

		rospy.sleep(2)
		command.id = "TURN_RIGHT"
		command.enable = False
		leds_driver_client(command)

		rospy.sleep(2)	
		command.id = "FORWARD"
		command.enable = False
		leds_driver_client(command)
		

def main():

	leds_control = LedsControl()

	while not rospy.is_shutdown():

		leds_control.setEffect()
		rospy.sleep(10)
		leds_control.clearEffect()
		rospy.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
