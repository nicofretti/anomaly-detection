#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from robotnik_signal_msgs.srv import SetSignal, SetSignalRequest


class LedsControl:

	def __init__(self):

		# Init node
		rospy.init_node('leds_robot_example_node')

		#Get name of this node
		self.node_name = rospy.get_name()

		#Get cmd_vel topic
		cmd_vel_topic_name = rospy.get_param(self.node_name + "/cmd_vel")

		# Subscribe to cmd_vel robot 
		rospy.Subscriber(cmd_vel_topic_name, Twist, self.cmd_vel_callaback, queue_size=1)

		rospy.loginfo("Starting example...")		
		
        # Check if ALS module is already
		rospy.wait_for_service('leds_driver/set_signal')

		self.threshold_vel = 0.2
		
		rospy.loginfo("Running example!")


	def cmd_vel_callaback(self, msg):

		command = SetSignalRequest()
		rospy.wait_for_service('leds_driver/set_signal')
		leds_driver_client = rospy.ServiceProxy('leds_driver/set_signal', SetSignal)

		vel_x = msg.linear.x
		vel_y = msg.linear.y
		vel_z = msg.angular.z

		vel = self.threshold_vel

		if vel_x <= vel and vel_x >= -vel and vel_y <= vel and vel_y >= -vel and vel_z <= vel and vel_z >= -vel:

			#Important: First disable effects, then activate effects
			
			#print("Robot stops!")

			command.signal_id = "MOVING"
			command.enable = False
			leds_driver_client(command)

			command.signal_id = "STOP"
			command.enable = True
			leds_driver_client(command)

		else:

			#Important: First disable effects, then activate effects
			
			#print("Robot moves!")

			command.signal_id = "STOP"
			command.enable = False
			leds_driver_client(command)

			command.signal_id = "MOVING"
			command.enable = True
			leds_driver_client(command)
			


		
def main():

	leds_control = LedsControl()

	while not rospy.is_shutdown():

		dummy = 1        




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
