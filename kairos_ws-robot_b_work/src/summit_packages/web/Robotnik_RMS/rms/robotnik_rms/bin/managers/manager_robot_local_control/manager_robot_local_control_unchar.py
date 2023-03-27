#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robot_local_control_msgs.srv import UnchargePetition
from robot_local_control_msgs.srv import UnchargePetitionRequest
from procedures_msgs.srv import ProcedureQuery
from procedures_msgs.srv import ProcedureQueryRequest

class ManagerRobotLocalControlUnchar(ManagerAstract.ManagerAbstractClass):
    def __init__(self, service_unchar_petition_add = "/robot/robot_local_control/NavigationComponent/UnchargeComponent/add", service_unchar_petition_query_state = "/robot/robot_local_control/NavigationComponent/UnchargeComponent/query_state", service_unchar_petition_cancel="/robot/robot_local_control/NavigationComponent/UnchargeComponent/cancel", Uncharge_frame_id = ""):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global UnchargePetition
        global UnchargePetitionRequest
        global ProcedureQuery
        global ProcedureQueryRequest
        self.service_unchar_petition_add = service_unchar_petition_add
        self.service_unchar_petition_query_state = service_unchar_petition_query_state
        self.service_unchar_petition_cancel = service_unchar_petition_cancel
        self.Uncharge_frame_id = Uncharge_frame_id
        self.finish_action = False
        self.id_uncharge = -1
	self.finalized_with_error = ""

    def callUnchargePetitionAddService(self):
	self.finish_action = False
	self.id_uncharge = -1
	rospy.wait_for_service(self.service_unchar_petition_add)
	try:
		service_robot_local_control_Uncharge_add_petition = rospy.ServiceProxy(self.service_unchar_petition_add, UnchargePetition)
		robot_local_control_Uncharge_petition = UnchargePetitionRequest()
		response = service_robot_local_control_Uncharge_add_petition(robot_local_control_Uncharge_petition)
		if(response.result.result == "ok"):
			self.id_uncharge = response.state.header.id
			print "Added Uncharge id=" + str(self.id_uncharge)
			return True
		else:
			self.id_uncharge = -1
			print "Not added Uncharge"
			return False
	except rospy.ServiceException, e:
		self.id_uncharge = -1
		print "Service call failed"
		return False

    def callProcedureQueryIsThePetitionFinish(self):
		rospy.wait_for_service(self.service_unchar_petition_query_state)
		try:
			service_unchar_petition_query_state_petition = rospy.ServiceProxy(self.service_unchar_petition_query_state, ProcedureQuery)
			robot_local_control_query_petition = ProcedureQueryRequest()
			robot_local_control_query_petition.header.id = self.id_uncharge
			response = service_unchar_petition_query_state_petition(robot_local_control_query_petition)
			if(response.state.last_event == response.state.FINISH):
				print "Finish uncharge id=" + str(self.id_uncharge)
				self.finish_action = True
				return True
			elif(response.state.last_event == response.state.CANCEL):
				print "Cancel uncharge id=" + str(self.id_uncharge)
				self.finish_action = True
				return False
			elif(response.state.last_event == response.state.ABORT):
				print "Abort uncharge id=" + str(self.id_uncharge)
				self.finish_action = True
				return False
			else:
				print "Wait for cancel charge id=" + str(self.id_uncharge)
				return False
		except rospy.ServiceException, e:
			print "Service call failed"
			return False

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callUnchargePetitionAddService()):
		print ("ManagerUncharge-->Executing ")
		rospy.sleep(1.)
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FINALIZED, "Finalize, the action not exist")


    #NECESSARY A DEFINITION
    def stateExecuting(self):
	rospy.sleep(1.)
	query_finish = self.callProcedureQueryIsThePetitionFinish()
	if (query_finish and self.finish_action):
		self.changeState(self.STATES_FINALIZED, "Finalized")
	elif (not query_finish and self.finish_action):
		self.changeState(self.STATES_FAIL, "Fail status : "+self.finalized_with_error+"  .the manager uncharge is in fail")

    def stateError(self):
        print("Error")

    def stateFinalized(self):
        print("Manager Uncharge--->Finsihed.")

    def cancel(self):
	rospy.wait_for_service(self.service_unchar_petition_cancel)
	if(self.id_uncharge == -1):
		return True
	try:
		service_uncharge_petition_query_state_petition = rospy.ServiceProxy(self.service_unchar_petition_cancel, ProcedureQuery)
		robot_local_control_query_petition = ProcedureQueryRequest()
		robot_local_control_query_petition.header.id = self.id_uncharge
		response = service_uncharge_petition_query_state_petition(robot_local_control_query_petition)
		if(response.state.last_event == response.state.CANCEL):
			print "Cancel uncharge id=" + str(self.id_uncharge)
			self.finish_action = True
			return True
		elif(response.state.last_event == response.state.ABORT):
			print "Abort uncharge id=" + str(self.id_uncharge)
			self.finish_action = False
			return True
		else:
			print "Wait for cancel uncharge id=" + str(self.id_uncharge)
			return False
	except rospy.ServiceException, e:
		self.finalized_with_error = "Exception - " + str(e)
		self.finish_action = True
		print "Service get uncharge cancel call failed"
		return False

