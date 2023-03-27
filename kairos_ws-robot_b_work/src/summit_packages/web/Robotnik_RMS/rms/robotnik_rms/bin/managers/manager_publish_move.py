#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract
from geometry_msgs.msg import Twist



class ManagerPublishMove(ManagerAstract.ManagerAbstractClass):
    def __init__(self, publish_topic = "/robot/move/cmd_vel" ,linear_x=0, linear_y=0, angular_z=0):
        global Twist
        ManagerAstract.ManagerAbstractClass.__init__(self)
        self.publish_topic_ = publish_topic
	self.linear_x_ = linear_x
	self.linear_y_ = linear_y
	self.angular_z_ = angular_z
        self.pub_ = rospy.Publisher(self.publish_topic_, Twist, queue_size=1)
        
        
    def recivedAction(self):
        
        print ("ManagerPublishMove-> Recive a action: " + str(self.current_action_msg_))

    #NECESSARY A DEFINITION
    def stateExecute(self):
        print ("ManagerPublishMove-->Execute: " + str(self.current_action_msg_))
        self.publish()
	self.changeState(self.STATES_EXECUTING)
        

    def stateExecuting(self):
	print ("ManagerPublishMove-->Executing");
	rospy.sleep(0.5)
	self.changeState(self.STATES_FINALIZED)


    def publish(self):
	twist_message = Twist()
	twist_message.linear.x = self.linear_x_
	twist_message.linear.y = self.linear_y_
	twist_message.angular.z = self.angular_z_
	connections = self.pub_.get_num_connections()
        while connections <= 0:
		connections = self.pub_.get_num_connections()
	self.pub_.publish(twist_message)
