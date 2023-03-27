#!/usr/bin/env python

''' ---------------------- LEDS ROBOT RBSHERPA ------------------------ '''

import rospy
from geometry_msgs.msg import Twist
from robotnik_signal_msgs.srv import SetSignal, SetSignalRequest, SetSignalResponse
from robotnik_msgs.msg import State
from std_msgs.msg import Bool

class RobotLeds:

	def __init__(self):

		# Init node
		rospy.init_node('leds_robot_example_node')
		rospy.loginfo("Starting example...")

		# Get name of this node
		self.node_name = rospy.get_name()
		# Get vel topic
		vel_topic_name = rospy.get_param("~cmd_vel_topic", "robotnik_base_control/cmd_vel")
		# Verbose for debug
		self.verbose = rospy.get_param("~verbose", False)

		# Variables for current and last state
		self.state = ""
		self.last_state = ""
		
		# Variable for velocity threshold
		self.threshold_vel = 0.02
		
		# Variable to save the emergency stop state
		self.emergency_stop = False

		# Variables to save the current velocity
		self.vel_x = 0
		self.vel_y = 0
		self.vel_z = 0

		# Check if ALS driver service is 
		rospy.loginfo("Waiting driver...")
		rospy.wait_for_service('leds_driver/set_signal')
		self.send_led_command = rospy.ServiceProxy('leds_driver/set_signal', SetSignal)

		# Wait for rcomponent ready driver
		rospy.loginfo("Waiting rcomponent...")
		rospy.wait_for_message('leds_driver/state', State)

		# Subscribe to vel robot
		rospy.Subscriber(vel_topic_name, Twist, self.vel_callaback, queue_size=1)
		# Subscribe to emergency_stop
		rospy.Subscriber("robotnik_base_hw/emergency_stop", Bool, self.emergency_stop_callaback, queue_size=1)

		# Timeout for first emergency stop msg
		rospy.sleep(1)

		rospy.loginfo("Running example!")


	def vel_callaback(self, msg):

		self.vel_x = msg.linear.x
		self.vel_y = msg.linear.y
		self.vel_z = msg.angular.z


	def emergency_stop_callaback (self, msg):

		self.emergency_stop = msg.data


	def getRobotState(self):

		vel_x = self.vel_x
		vel_y = self.vel_y
		vel_z = self.vel_z

		emergency_stop = self.emergency_stop
		vel = self.threshold_vel

		if emergency_stop == False:

			# Check direction
			if vel_x >= 0:
				direction = True
			else:
				direction = False


			if abs(vel_y) > vel:

				# Right shift
				if vel_y > vel:

					state = "omni_left"

				# Left shift
				elif vel_y < -vel:

					state = "omni_right"

			elif abs(vel_z) > vel:

				# Forward turning
				if direction:

					if vel_z > vel:

						state = "turn_left"

					elif vel_z < -vel:

						state = "turn_right"

				# Backward turning
				else:

					if vel_z > vel:

						state = "turn_left_inverse"

					elif vel_z < -vel:

						state = "turn_right_inverse"

			else:

				# Only forward
				if vel_x > vel:

					state = "forward"

				# Only backward
				elif vel_x < -vel:

					state = "backward"

				else:

					state = "stop_robot"

		else:

			state = "emergency_stop"


		return state


	def setLedState(self, signal):

		res = SetSignalResponse()
		command = SetSignalRequest()

		command.signal_id =	signal
		command.enable = True
		res = self.send_led_command(command)
		if self.verbose == True:
			print("Sending driver says: " + res.ret.message)

	def clearLedState(self, signal):

		res = SetSignalResponse()
		command = SetSignalRequest()

		command.signal_id =	signal
		command.enable = False
		res = self.send_led_command(command)
		
		if self.verbose == True:
			print("Deactivating driver says: " + res.ret.message)


	def run(self):

		self.last_state = self.state
		self.state = self.getRobotState()

		if self.state != self.last_state:
	
			self.clearLedState(self.last_state)
			self.setLedState(self.state)

			if self.verbose == True:
				print("I send: " + self.state)
				print("I deactivate: " + self.last_state)
				print("---------")


def main():
	
	robot_leds = RobotLeds()
	r = rospy.Rate(100)

	while not rospy.is_shutdown():

		robot_leds.run()
		r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
