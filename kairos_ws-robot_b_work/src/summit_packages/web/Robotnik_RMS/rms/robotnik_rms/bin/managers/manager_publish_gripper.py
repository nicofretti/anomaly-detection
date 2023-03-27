#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract
from std_msgs.msg import Float64 as StdMsgsFloat64


class ManagerPublishGripper(ManagerAstract.ManagerAbstractClass):
    def __init__(self, publish_topic, open_value = 0.03, close_value = 0.0):

        ManagerAstract.ManagerAbstractClass.__init__(self)
        global StdMsgsFloat64
        self.open_value_ = open_value
        self.close_value_ = close_value
        self.publish_topic_ = publish_topic
        self.pub = rospy.Publisher(self.publish_topic_, StdMsgsFloat64, queue_size=10)
        
    #NECESSARY A DEFINITION
    def stateExecute(self):
	
	global StdMsgsFloat64
        msg = StdMsgsFloat64()
        msg.data = 0.0
        if(self.current_action_msg_=="0"): #Open
            	msg.data = self.open_value_
        	self.pub.publish(msg)		
        else:
	    msg.data = self.close_value_
	    self.pub.publish(msg)
        self.changeState(self.STATES_EXECUTING, "Finalize") 

    def stateExecuting(self):
	wait_to_finsih = 5
        rate = rospy.Rate(4) # 10hz
	for x in range(0, wait_to_finsih*4):
		print ("waiting for grepper")
		rate.sleep()
	self.changeState(self.STATES_FINALIZED, "Finalize") 
		

    def stateFinalized(self):
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")
