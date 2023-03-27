#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from ur_msgs.srv import SetIO
from ur_msgs.srv import SetIORequest
from ur_msgs.msg import IOStates

class ManagerUniversalRobots(ManagerAstract.ManagerAbstractClass):
    def __init__(self, topic_arm_io_state = "/robot/arm/ur_hardware_interface/io_states", service_arm_set_io = "/robot/arm/ur_hardware_interface/set_io", digital_output_start = 2, digital_output_state = 3):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global SetIO
        global SetIORequest
        global IOStates
        self.topic_arm_io_state = topic_arm_io_state
        self.service_arm_set_io = service_arm_set_io
        self.digital_output_start = digital_output_start
	self.digital_output_state = digital_output_state
        self.finish_move = False
        self.arm_state_change = False
	self.last_state_arm = None
	self.send_to_arm = False
	self.finish = False
	self.first_execution = False
	self.subscribe_io_state = rospy.Subscriber(self.topic_arm_io_state, IOStates,
                         self.io_state_callback)

    def callStartMoveService(self, active_move = True):
	rospy.wait_for_service(self.service_arm_set_io)
	try:
		service_arm_set_io_petition = rospy.ServiceProxy(self.service_arm_set_io, SetIO)
		set_io_petition = SetIORequest()
		set_io_petition.fun = set_io_petition.FUN_SET_DIGITAL_OUT
		set_io_petition.pin = self.digital_output_start
		if (active_move):	
			set_io_petition.state = 1.0
		else:
			set_io_petition.state = 0.0
		response = service_arm_set_io_petition(set_io_petition)
		print ("Send to "+ str(self.digital_output_start)+ ": " + str(active_move))
		return True
	except rospy.ServiceException, e:
		self.id_pick = -1
		print "Service call failed"
		return False

    def io_state_callback(self, data):
	if(self.first_execution == False):
		self.last_state_arm = data
		self.first_execution = True
		#print "First change"
	elif(self.finish == False):
		#print("state_callbakc:\n digital_output_start " + str(self.digital_output_start) + ":" + str(data.digital_out_states[self.digital_output_start].state) + "\n digital_output_state " + str(self.digital_output_state) + ":" + str(data.digital_out_states[self.digital_output_state].state)) 
		if(data.digital_out_states[self.digital_output_start].state==self.send_to_arm):
			self.arm_state_change = True
			self.callStartMoveService(False)
		elif self.arm_state_change == True and data.digital_out_states[self.digital_output_state].state==False:
			print(str(data.digital_out_states[self.digital_output_state].state))
			self.finish = True

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):	
	self.first_execution = False	
	self.finish_move = False
	self.arm_state_change = False
	self.finish = False
	self.send_to_arm = True	
	if (self.callStartMoveService()):
		print ("ManagerArm-->Executing ")
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FINALIZED, "Finalize, the action not exist")


    #NECESSARY A DEFINITION
    def stateExecuting(self):
	if self.finish:
		self.changeState(self.STATES_FINALIZED, "Finalized")
	else:
		print ("Waiting to finish arm performance")

    def stateFail(self):
        self.changeState(self.STATES_FINALIZED, "Fail")

    def stateError(self):
        print("Error")
	self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")


