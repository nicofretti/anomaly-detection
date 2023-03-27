#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robot_local_control_msgs.srv import ChargePetition
from robot_local_control_msgs.srv import ChargePetitionRequest
from procedures_msgs.srv import ProcedureQuery
from procedures_msgs.srv import ProcedureQueryRequest
from poi_manager.srv import ReadPOIs
from poi_manager.srv import ReadPOIsRequest

class ManagerRobotLocalControlCharge(ManagerAstract.ManagerAbstractClass):
    def __init__(self, service_charge_petition_add = "/robot/robot_local_control/NavigationComponent/ChargeComponent/add", service_charge_petition_query_state = "/robot/robot_local_control/NavigationComponent/ChargeComponent/query_state", service_charge_petition_cancel = "/robot/robot_local_control/NavigationComponent/ChargeComponent/cancel"):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global ChargePetition
        global ChargePetitionRequest
        global ProcedureQuery
        global ProcedureQueryRequest
        global ReadPOIs
        global ReadPOIsRequest
        self.service_charge_petition_add = service_charge_petition_add
        self.service_charge_petition_query_state = service_charge_petition_query_state
	self.service_charge_petition_cancel = service_charge_petition_cancel
        self.finish_action = False
        self.id_charge = -1
	self.finalized_with_error = ""

    def callChargePetitionAddService(self):
	self.finish_action = False
	self.id_charge = -1
        #add in robot local control charge
	rospy.wait_for_service(self.service_charge_petition_add)
	try:
		service_robot_local_control_charge_add_petition = rospy.ServiceProxy(self.service_charge_petition_add, ChargePetition)
		robot_local_control_charge_petition = ChargePetitionRequest()
		response = service_robot_local_control_charge_add_petition(robot_local_control_charge_petition)
		if(response.result.result == "ok"):
			self.id_charge = response.state.header.id
			print "Added charge id=" + str(self.id_charge)
			return True
		else:
			self.id_charge = -1
			print "Not added charge"
			return False
	except rospy.ServiceException, e:
		self.id_charge = -1
		print "Service add charge call failed " + str(e)
		return False

    def callProcedureQueryIsThePetitionFinish(self):
		rospy.wait_for_service(self.service_charge_petition_query_state)
		try:
			service_charge_petition_query_state_petition = rospy.ServiceProxy(self.service_charge_petition_query_state, ProcedureQuery)
			robot_local_control_query_petition = ProcedureQueryRequest()
			robot_local_control_query_petition.header.id = self.id_charge
			response = service_charge_petition_query_state_petition(robot_local_control_query_petition)
			if(response.state.current_state == response.state.FINISHED):
				print "Finish charge id=" + str(self.id_charge)
				self.finish_action = True
				return True
			elif(response.state.last_event == response.state.CANCEL):
				print "Cancel charge id=" + str(self.id_charge)
				self.finish_action = True
				return False
			elif(response.state.last_event == response.state.ABORT):
				print "Abort charge id=" + str(self.id_charge)
				self.finish_action = True
				return False
			else:
				print "Wait for cancel charge id=" + str(self.id_charge)
				return False
		except rospy.ServiceException, e:
			self.finalized_with_error = "Exception - " + str(e)
			self.finish_action = True
			print "Service get charge state call failed"
			return False

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	self.finalized_with_error = ""
	print ("ManagerRobotLocalControlCharge-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callChargePetitionAddService()):
		print ("ManagerRobotLocalControlCharge-->Executing ")
		rospy.sleep(1.)
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FAIL, "Fail when try send the point, the manager charge is in fail")


    def stateExecuting(self):
	rospy.sleep(1.)
	query_finish = self.callProcedureQueryIsThePetitionFinish()
	if (query_finish and self.finish_action):
		self.changeState(self.STATES_FINALIZED, "Finalized")
	elif (not query_finish and self.finish_action):
		self.changeState(self.STATES_FAIL, "Fail status : "+self.finalized_with_error+"  .the manager charge is in fail")
	
    def stateError(self):
        print("Error")
	#self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
        print("Manager charge--->Finsihed.")

    def cancel(self):
	rospy.wait_for_service(self.service_charge_petition_cancel)
	if(self.id_charge == -1):
		return True
	try:
		service_charge_petition_query_state_petition = rospy.ServiceProxy(self.service_charge_petition_cancel, ProcedureQuery)
		robot_local_control_query_petition = ProcedureQueryRequest()
		robot_local_control_query_petition.header.id = self.id_charge
		response = service_charge_petition_query_state_petition(robot_local_control_query_petition)
		if(response.state.last_event == response.state.CANCEL):
			print "Cancel charge id=" + str(self.id_charge)
			self.finish_action = True
			return True
                elif(response.state.last_event == response.state.ABORT):
			print "Abort uncharge id=" + str(self.id_charge)
			self.finish_action = False
			return True
		else:
			print "Wait for cancel charge id=" + str(self.id_charge)
			return False
	except rospy.ServiceException, e:
		self.finalized_with_error = "Exception - " + str(e)
		self.finish_action = True
		print "Service get charge cancel call failed"
		return False
	

