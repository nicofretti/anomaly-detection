#!/usr/bin/env python

''' ---------------------- LEDS ROBOT DEBUG ------------------------ '''

import rospy
from geometry_msgs.msg import Twist
from robotnik_signal_msgs.srv import SetSignal, SetSignalRequest, SetSignalResponse
from std_msgs.msg import Bool

class RobotLeds:

	def __init__(self):

		# Init node
		rospy.init_node('leds_robot_example_node')
		rospy.loginfo("Starting example...")

		# Get name of this node
		self.node_name = rospy.get_name()
		# Get cmd_vel topic
		cmd_vel_topic_name = rospy.get_param(self.node_name + "/cmd_vel")


		# Variables for current and last state
		self.state = ""
		self.last_state = ""
		
		# Variable for velocity threshold
		self.threshold_vel = 0.02
		
		# Variable to save the emergency stop state
		self.emergency_stop = True

		# Variables to save the current velocity
		self.vel_x = 0
		self.vel_y = 0
		self.vel_z = 0

		# Variable to execute run()
		self.execute_run = True

		
		# Check if ALS driver service is 
		rospy.loginfo("Waiting driver...")
		rospy.wait_for_service('leds_driver/set_signal')
		self.send_led_command = rospy.ServiceProxy('leds_driver/set_signal', SetSignal)


		# Subscribe to cmd_vel robot
		rospy.Subscriber(cmd_vel_topic_name, Twist, self.cmd_vel_callaback, queue_size=1)
		# Subscribe to emergency_stop
		rospy.Subscriber("/robot/robotnik_base_hw/emergency_stop", Bool, self.emergency_stop_callaback, queue_size=1)
		
		# Create timer
		rospy.Timer(rospy.Duration(0.2), self.timer_callback)
	
		rospy.loginfo("Running example!")


	def cmd_vel_callaback(self, msg):

		self.vel_x = msg.linear.x
		self.vel_y = msg.linear.y
		self.vel_z = msg.angular.z


	def emergency_stop_callaback (self, msg):

		self.emergency_stop = msg.data


	def timer_callback(self, event):

		self.execute_run = True


	def executeRun(self):
		
		execute = self.execute_run
		# Clear pending flag
		self.execute_run = False 

		return execute



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
		print("Sending driver says: " + res.ret.message)

	def clearLedState(self, signal):

		res = SetSignalResponse()
		command = SetSignalRequest()

		command.signal_id =	signal
		command.enable = False
		res = self.send_led_command(command)
		print("Deactivating driver says: " + res.ret.message)


	def robotMissionMode(self):
		
		command = SetSignalRequest()
		command.signal_id =	"mission"
		command.enable = True
		self.send_led_command(command)
		print("Mission mode!")


	def run(self):

		if self.executeRun() == True:

			self.last_state = self.state
			self.state = self.getRobotState()

			if self.state != self.last_state:
				
				self.clearLedState(self.last_state)
				self.setLedState(self.state)

				print("I send: " + self.state)
				print("I deactivate: " + self.last_state)
				print("---------")


def main():
	
	robot_leds = RobotLeds()
	rospy.sleep(2) # Wait for rcomponent ready driver

	#robot_leds.robotMissionMode() # Put leds in mission mode

	while not rospy.is_shutdown():

		robot_leds.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
