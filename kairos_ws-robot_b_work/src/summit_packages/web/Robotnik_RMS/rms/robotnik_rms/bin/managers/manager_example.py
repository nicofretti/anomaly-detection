#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract

class ManagerExample(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self)
    
    
    
    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
        print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    #NECESSARY A DEFINITION
    def stateExecuting(self):
        print ("ManagerExample-->Executing: " + str(self.current_action_msg_))
        self.changeState(self.STATES_FINALIZED, "Not defined stateExecuting metoth")

    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        #None

    def stateError(self):
        print("Error")
        #None

    def stateFinalized(self):
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")

