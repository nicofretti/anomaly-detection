#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract
from std_msgs.msg import String


class ManagerPublishExample(ManagerAstract.ManagerAbstractClass):
    def __init__(self, publish_topic):
        global String
        ManagerAstract.ManagerAbstractClass.__init__(self)
        self.publish_topic_ = publish_topic
        self.pub = rospy.Publisher(self.publish_topic_, String, queue_size=10)
        
        
    def recivedAction(self):
        self.current_action_msg_ = str(self.current_action_msg_)
        print ("ManagerPublishExample-> Recive a action: " + str(self.current_action_msg_))

    #NECESSARY A DEFINITION
    def stateExecuting(self):
        print ("ManagerPublishExample-->Executing: " + str(self.current_action_msg_))
        self.pub.publish(str(self.current_action_msg_))
        self.changeState(self.STATES_FINALIZED)
