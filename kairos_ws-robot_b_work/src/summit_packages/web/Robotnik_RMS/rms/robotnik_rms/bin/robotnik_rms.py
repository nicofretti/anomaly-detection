#!/usr/bin/env python

import rospy
import yaml
import io
import os
from std_msgs.msg import String as StdMsgsString
from robotnik_rms_msgs.msg import status as StatusMsg
from robotnik_rms_msgs.msg import action as RmsActionMsgs
from robotnik_rms_msgs.msg import mission as RmsMissionMsgs

from robotnik_rms_msgs.srv import missionManager as MissionManager
from robotnik_rms_msgs.srv import missionManagerResponse as MissionManagerResponse

from robotnik_rms_msgs.srv import idMission as IdMission
from robotnik_rms_msgs.srv import idMissionResponse as IdMissionResponse

from robotnik_rms_msgs.srv import getMissions as GetMissions
from robotnik_rms_msgs.srv import getMissionsResponse as GetMissionsResponse

from std_srvs.srv import Trigger as Trigger
from std_srvs.srv import TriggerResponse as TriggerResponse

import manager_astract_base_class as ManagerAstract


# Status state machine
class StateMachine:
    def __init__(self):
        self.last_status_ = self.current_status_ = 0
        self.message_ = "Initializing"

        self.descriptions_ = {
            StatusMsg.CURRENT_STATUS_ERROR: "Error",
            StatusMsg.CURRENT_STATUS_PREIDLE: "Pre-Idle",
            StatusMsg.CURRENT_STATUS_IDLE: "Idle",
            StatusMsg.CURRENT_STATUS_CHARGING_MISSION: "Charging mission",
            StatusMsg.CURRENT_STATUS_EXECUTING_MISSION: "Executing mission",
            StatusMsg.CURRENT_STATUS_PAUSE_MISSION: "Pause mission",
	    StatusMsg.CURRENT_STATUS_CANCEL_MISSION: "Cancel mission"
        }
        self.description_ = self.descriptions_.get(
            self.current_status_, "Unknown")
        self.status_topic_ = rospy.get_param('status_topic', '~status')
        self.status_publish_ = rospy.Publisher(
            self.status_topic_, StatusMsg, queue_size=0)
        self.execution_queue_ = []
        self.action_in_execution_index_ = -1
        self.action_list_execution_ = []
        self.goal_message_ = []
        self.pause_ = False
	self.cancel_ = False
        self.managers_ = {}
        self.managers_configuration_ = []
	self.last_status_executing_ = None

    # Get the status description
    def statusDescription(self, status):
        return self.descriptions_.get(status, "Unknown " + str(status))

    # Change the state machine
    def changeStatus(self, status, message="", publish=True):
        if(status != self.current_status_):
            self.last_status_ = self.current_status_
            self.current_status_ = status
            self.description_ = self.statusDescription(self.current_status_)
        self.message_ = "Changed from " + self.statusDescription(
            self.last_status_) + " to " + self.statusDescription(self.current_status_)
        if(message != ""):
            self.message_ = self.message_ + ".Aditional information: " + message
        if(publish):
            self.publishStatus()

    # Publish the stete machine information
    def publishStatus(self):
        message = StatusMsg()
        message.current_status = self.current_status_
        message.description = self.description_
        message.message = self.message_
	try:
		for execution_queue_foreach in self.execution_queue_:
			print execution_queue_foreach["name_mission"]
			message.mission_queue.append(execution_queue_foreach["name_mission"])
	except:
		print "Error to create message"
        self.status_publish_.publish(message)

    # Get the current status
    def getCurrentStatus(self):
        return self.current_status_

    # Get the last status
    def getLastStatus(self):
        return self.last_status_

    # Get the descrition status
    def getDescription(self):
        return self.description_

    # Get the message status
    def getMessage(self):
        return self.message_

    def stateMachineLogic(self, event):
        self.checkStatus()
        if (StatusMsg.CURRENT_STATUS_ERROR == self.current_status_):
            self.errorState()
        elif(StatusMsg.CURRENT_STATUS_PREIDLE == self.current_status_):
            self.preidleState()
        elif(StatusMsg.CURRENT_STATUS_IDLE == self.current_status_):
            self.idleState()
        elif(StatusMsg.CURRENT_STATUS_CHARGING_MISSION == self.current_status_):
            self.chargingMissionState()
        elif(StatusMsg.CURRENT_STATUS_EXECUTING_MISSION == self.current_status_):
            self.executingMissionState()
        elif(StatusMsg.CURRENT_STATUS_PAUSE_MISSION == self.current_status_):
            self.executingPauseState()
	elif(StatusMsg.CURRENT_STATUS_CANCEL_MISSION == self.current_status_):
            self.cancelState()
        else:
            self.UnknownState()

    def checkStatus(self):
        # Function to check if something is bad. Use: self.changeStatus(StatusMsg.CURRENT_STATUS_ERROR, "error description", True)
        None

    # Error state
    def errorState(self):
	self.execution_queue_ = []
        self.action_in_execution_index_ = -1
        self.action_list_execution_ = []
        self.goal_message_ = []
        self.pause_ = False
	self.cancel_ = False
	print ("RMS in Error state")
        if(len(self.managers_) != 0):
            self.changeStatus(StatusMsg.CURRENT_STATUS_IDLE,
                              "Solved error", True)
    def cancelState(self):
	self.execution_queue_ = []
        self.action_in_execution_index_ = -1
        self.action_list_execution_ = []
        self.goal_message_ = []
        self.pause_ = False
	self.cancel_ = False
	print ("RMS in Cancel state")
        if(len(self.managers_) != 0):
            self.changeStatus(StatusMsg.CURRENT_STATUS_IDLE,
                              "Cancel", True)

    # Preidle state

    def preidleState(self):
        if(self.addManagers()):
            self.changeStatus(StatusMsg.CURRENT_STATUS_IDLE, "", True)
        else:
            self.changeStatus(StatusMsg.CURRENT_STATUS_ERROR,
                              "The managers are not chargin. Do you not send from params?", True)

    # Idle state
    def idleState(self):
        if(len(self.execution_queue_) > 0):
            message = "Executing mission " + \
                self.execution_queue_[0]["name_mission"] + ". "
            if(len(self.execution_queue_) > 1):
                message = message + \
                    str(len(self.execution_queue_)-1) + " in queue."
            self.changeStatus(
                StatusMsg.CURRENT_STATUS_CHARGING_MISSION, message, True)

    # Charging mission State

    def chargingMissionState(self):
        # Todo: Get all message of id_mission, get from the diferents services
        self.action_in_execution_index_ = 0
        self.action_list_execution_ = []
        print("Charging mission: " + self.execution_queue_[0]["name_mission"])
        for actions_forech in self.execution_queue_[0]["actions"]:
            self.action_list_execution_.append(actions_forech)
            # self.goal_message_ = call the service to get the goal_message_
        self.changeStatus(StatusMsg.CURRENT_STATUS_EXECUTING_MISSION,
                          "Executing mission " + self.execution_queue_[0]["name_mission"], True)

    # Executing mission state

    def executingMissionState(self):
        print("Executing mission " + self.execution_queue_[0]["name_mission"] + " action " +
              self.action_list_execution_[self.action_in_execution_index_]["name_action"])
        type_action = self.action_list_execution_[
            self.action_in_execution_index_]["type_action"]
        id_action = self.action_list_execution_[
            self.action_in_execution_index_]["id_action"]
	self.changeStatus(StatusMsg.CURRENT_STATUS_EXECUTING_MISSION, "Executing action " +
                              self.action_list_execution_[self.action_in_execution_index_] ["name_action"] + " from " + self.execution_queue_[0]["name_mission"], True)
        if(not self.executeActionInManager(type_action, id_action)):
	    if self.cancel_ == True:
		message = "Cancel mission"
		self.changeStatus(StatusMsg.CURRENT_STATUS_ERROR, message, True)
	    else:
            	message = "Doesn't finish action " + str(type_action) + ". RMS changed to Status Error"
            	print(message)
            	self.changeStatus(StatusMsg.CURRENT_STATUS_ERROR, message, True)
	    return
        if(self.action_in_execution_index_ < len(self.action_list_execution_)-1):

            
            # Call manager
            self.action_in_execution_index_ += 1
        else:
            self.action_in_execution_index_ = -1
            self.action_list_execution_ = []
	    try:
		    self.changeStatus(StatusMsg.CURRENT_STATUS_IDLE, "Finish " +
		                      self.execution_queue_[0]["name_mission"], True)
		    del self.execution_queue_[0]
	    except:
		    self.changeStatus(StatusMsg.CURRENT_STATUS_IDLE, "Finish ", True)

    # Pause action state

    def executingPauseState(self):
        if (not self.pause_):
            if(self.action_in_execution_index_ < len(self.action_list_execution_)):
                self.changeStatus(StatusMsg.CURRENT_STATUS_EXECUTING_MISSION, "Executing action " +
                                  self.action_list_execution_[self.action_in_execution_index_]["name_action"], True)

    # Unknown state

    def UnknownState(self):
        print("unknown state")
        None

    def addMissionInQueue(self, mission):
        self.execution_queue_.append(mission)

    def cancelAllMissions(self):
	self.execution_queue_ = []
        self.action_in_execution_index_ = -1
        self.action_list_execution_ = []
        self.goal_message_ = []
        self.pause_ = False
	self.cancel_ = True
        for key, value in self.managers_.iteritems():
		value.callCancel()
	self.changeStatus(StatusMsg.CURRENT_STATUS_CANCEL_MISSION,
                              "Cancel all the missions", True)
	return True
	
    def cancelCurrentMission(self):
        None

    def pauseMission(self):
        None

    def resumeCurrentMission(self):
        None

    def cancelManagerAction(self):
        None

    def addManagers(self):
        if rospy.has_param("~managers"):
            self.managers_configuration_ = rospy.get_param("~managers")
        print "Manager params:" + str(len(self.managers_configuration_))
        if(self.managers_configuration_ == None):
            print("Not params")
            return False
        try:

            manager_path = os.path.dirname(os.path.realpath(__file__))
            manager_path = manager_path + "/managers/"
            print("Manager list:")
            print("Type_manager\tName_manager")
            for manager_configuration_forech in self.managers_configuration_:
                type_action = manager = name_class = None
                other_configurations = []
                for key, value in manager_configuration_forech.items():
                    if(key == "type_action"):
                        type_action = value
                    elif(key == "manager"):
                        manager = value
                    elif(key == "name_class"):
                        name_class = value
                    else:
                        aux_other_configurations = {}
                        aux_other_configurations[str(key)] = str(value)
                        other_configurations.append(aux_other_configurations)
                if(type_action == None):
                    print("Needed a type_action in param")
                    return False
                if(manager == None):
                    print("Needed a manager in param")
                    return False
                if(name_class == None):
                    print("Needed a name_class in param")
                    return False
                code_manager = open(manager_path + manager + ".py", "r")
                exec(code_manager)
                class_configuration = ""
                for other_configuration_forech in other_configurations:
                    for key, value in other_configuration_forech.items():
                        class_configuration = class_configuration + key + "=" + value + ", "
                if(len(other_configurations) > 0):
                    class_configuration = class_configuration[:-2]
                aux_class = None
                get_class = "aux_class = " + name_class + \
                    "(" + class_configuration + ")"
                exec(get_class)
                self.managers_[type_action] = aux_class
                print(str(type_action) + "\t\t" + manager)
                #self.managers_.append({type_action: aux_class})

        except Exception, e:
            print("Exception")
            print(str(e))
            return False
        return True

    def executeActionInManager(self, type_action, id_action):
        try:
            try_to_execute_action = 0
	    last_status_executing_ = None
            if(self.managers_[type_action] != None):
                print("Executing type_action" + str(type_action))
                manager_in_idle_state = False
		rate = rospy.Rate(15)
		#wait for Idle state before launch the action                
		while(last_status_executing_ != self.managers_[type_action].STATES_IDLE):
                    if(self.managers_[type_action].getState() == self.managers_[type_action].STATES_FAIL or self.managers_[type_action].getState == self.managers_[type_action].STATES_ERROR):
                        self.changeStatus(
                            StatusMsg.CURRENT_STATUS_ERROR, "Error while executing actiong", True)
			return False
                    elif (self.managers_[type_action].getState() == self.managers_[type_action].STATES_IDLE):
			last_status_executing_ = self.managers_[type_action].getState()
			self.managers_[type_action].startAction(str(id_action))
                    else:
			None
			#print(str(type_action)+":Waiting for the action change to Idle, now the state is " + self.managers_[type_action].getDescriptionCurrentState())
		    rate.sleep()
		#wait for a finish state before finish the action 
		while (last_status_executing_ != self.managers_[type_action].STATES_FAIL and
                       last_status_executing_ != self.managers_[type_action].STATES_ERROR and
                       last_status_executing_ != self.managers_[type_action].STATES_FINALIZED):
		        last_status_executing_ = self.managers_[type_action].getState()	
			print(str(type_action)+":The state is in " + self.managers_[type_action].getDescriptionCurrentState()) + " state"
			rate.sleep()
		print("Finish action, checking the state:" + self.managers_[type_action].getDescriptionCurrentState())
		if (last_status_executing_ == self.managers_[type_action].STATES_FAIL or last_status_executing_ == self.managers_[type_action].STATES_ERROR):
			print ("Fail or Error:" + self.managers_[type_action].getDescriptionCurrentState())
			return False
		elif (last_status_executing_ == self.managers_[type_action].STATES_FINALIZED):
			print ("Finalized:" + self.managers_[type_action].getDescriptionCurrentState())			
			if (self.managers_[type_action].confirmationFinished()):
				print ("Confirmed:" + self.managers_[type_action].getDescriptionCurrentState())
				return True
			else:
				print ("Not confirmed:" + self.managers_[type_action].getDescriptionCurrentState())
				return False
		print ("WTF???" + self.managers_[type_action].getDescriptionCurrentState())
            else:
		
                return False
        except Exception, e:
            rospy.logerr('%s:executeActionInManager: %s', rospy.get_name, e)
            return False


class RMS:

    def getListOfMissionFromYaml(self):
        with open(os.path.join(self.path_of_action_file_), 'r') as stream:
            self.available_mission_list = yaml.safe_load(stream)
            try:
                # print "Available_mission:"
                # for available_mission_list_forech in self.available_mission_list:
                #     print available_mission_list_forech['name_mission']
                #     print available_mission_list_forech['id_mission']
                #     print available_mission_list_forech['description']
                #     print "mission:"
                #     for action_forech in available_mission_list_forech['mission']:
                #         print "\t"+str(actions_forech['type_action'])
                #         print "\t"+str(actions_forech['id_action'])
                #         print "\t"+actions_forech['name_action']
                #         print "\t"+actions_forech['description']
                #         print "\t----"
                if self.available_mission_list is None:
                    self.available_mission_list = []
                return True
            except:
                self.status_.changeStatus(
                    StatusMsg.CURRENT_STATUS_ERROR, "Error to read action yaml file", True)
                return False

    def addAvailableMission(self, mission):
        try:
            # Create the object
            new_mission = {
                "name_mission": mission.name_mission,
                "id_mission": 0,
                "description": mission.description,
                "actions": []
            }
            for action in (mission.actions):
                aux_actions_list = {
                    "type_action": action.type_action,
                    "id_action": action.id_action,
                    "name_action": action.name_action,
                    "description": action.description
                }
                new_mission["actions"].append(aux_actions_list)
            # Get the last free id
            id_mission = len(self.available_mission_list)
            last = -1
            for mission in self.available_mission_list:
                if(last < mission["id_mission"]):
                    last = mission["id_mission"]
            if(last != -1):
                id_mission = last + 1
            new_mission["id_mission"] = id_mission
            # Save
            self.available_mission_list.append(new_mission)
            return self.saveListOfAvailableMissionsInYaml()

        except Exception, e:
            print(str(e))
            return False

    def deleteAvailableMission(self, id_mission):
        index = 0
        for mission in self.available_mission_list:
            if(mission["id_mission"] == id_mission):
                del self.available_mission_list[index]
                return self.saveListOfAvailableMissionsInYaml()
            index = index + 1
        return False

    def modifyAvailableMission(self, mission):
        # Exist action?
        exist = False
        for mission_forech in self.available_mission_list:
            if(mission_forech["id_mission"] == mission.id_mission and not exist):
                exist = True
        if(not exist):
            return False
        else:
            try:
                # Create the object
                new_mission = {
                    "name_mission": mission.name_mission,
                    "id_mission": mission.id_mission,
                    "description": mission.description,
                    "actions": []
                }
                for actions_forech in mission.actions:
                    aux_actions_list = {
                        "type_action": actions_forech.type_action,
                        "id_action": actions_forech.id_action,
                        "name_action": actions_forech.name_action,
                        "description": actions_forech.description
                    }
                    new_mission["actions"].append(aux_actions_list)
                # Delete the old mission
                self.deleteAvailableMission(mission.id_mission)
                # Save
                self.available_mission_list.append(new_mission)
                return self.saveListOfAvailableMissionsInYaml()
            except Exception, e:
                print(str(e))
                return False

    def saveListOfAvailableMissionsInYaml(self):
        try:
            with io.open(self.path_of_action_file_, 'w', encoding='utf8') as outfile:
                yaml.dump(self.available_mission_list, outfile,
                          default_flow_style=False, allow_unicode=True)
                return True
        except:
            return False

    def getMissionsService(self, req):
        try:
            mission_list = []
            for mission_forech in self.available_mission_list:
                aux_mission = RmsMissionMsgs()
                aux_mission.name_mission = mission_forech["name_mission"]
                aux_mission.id_mission = mission_forech["id_mission"]
                aux_mission.description = mission_forech["description"]
                aux_array_action = []
                for action in mission_forech["actions"]:
                    aux_action = RmsActionMsgs()
                    aux_action.type_action = action["type_action"]
                    aux_action.id_action = action["id_action"]
                    aux_action.name_action = action["name_action"]
                    aux_action.description = action["description"]
                    aux_array_action.append(aux_action)
                aux_mission.actions = aux_array_action
                mission_list.append(aux_mission)
            return GetMissionsResponse(mission_list, True,  "The number of mission is " + str(len(mission_list)))
        except Exception, e:
            return GetMissionsResponse(RmsMissionMsgs(), False,  "An error happened: " + str(e))

    def addMissionService(self, req):
        try:
            if (not self.addAvailableMission(req.mission)):
                return MissionManagerResponse(False, "An error happened when try to add the mission")
            else:
                return MissionManagerResponse(True, "Added")
        except Exception, e:
            return MissionManagerResponse(False, "An error happened: " + str(e))

    def deleteMissionService(self, req):
        try:
            if self.deleteAvailableMission(req.id_mission):
                return IdMissionResponse(True, "Deleted")
            else:
                return IdMissionResponse(False, "The id not exist")
        except Exception, e:
            return IdMissionResponse(False, "An error happened: " + str(e))

    def modifyMissionService(self, req):
        try:
            if (not self.modifyAvailableMission(req.mission)):
                return MissionManagerResponse(False, "An error happened when try to modify the mission")
            else:
                return MissionManagerResponse(True, "Modify")
        except Exception, e:
            return MissionManagerResponse(False, "An error happened: " + str(e))

    def addMissionInQueueService(self, req):
        try:
            for mission in self.available_mission_list:
                if(mission["id_mission"] == req.id_mission):
                    self.status_.addMissionInQueue(mission)
                    return IdMissionResponse(True, "Added in queue")
            return IdMissionResponse(False, "The mission not exist")
        except Exception, e:
            return IdMissionResponse(False, "An error happened: " + str(e))

    def cancelAllMissionsService(self, req):
        try:
	    self.status_.cancelAllMissions()
            return TriggerResponse(True, "Cancel")
        except Exception, e:
            return TriggerResponse(False, "Error to cancel, exception: " + str(e))

    def __init__(self):
        self.action_list_ = []
        self.available_mission_list = []
        self.status_ = StateMachine()
        rospy.Timer(rospy.Duration(0.05), self.status_.stateMachineLogic)
        self.path_of_action_file_ = rospy.get_param(
            'path_of_action_file', '~/mission.yaml')
        self.path_of_action_file_ = os.path.expanduser(
            self.path_of_action_file_)
        if(os.path.isfile(self.path_of_action_file_) == False):
            file = open(self.path_of_action_file_, "a")
            file.close()
        if self.getListOfMissionFromYaml():
            self.get_list_of_mission_service_ = rospy.Service(
                '~get_list_of_mission', GetMissions, self.getMissionsService)
            self.add_mission_in_list_service_ = rospy.Service(
                '~add_mission_in_list', MissionManager, self.addMissionService)
            self.delete_mission_service_ = rospy.Service(
                '~delete_mission', IdMission, self.deleteMissionService)
            self.modify_mission_service_ = rospy.Service(
                '~modify_mission', MissionManager, self.modifyMissionService)
            self.addMissionInQueue = rospy.Service(
                '~add_mission_in_queue', IdMission, self.addMissionInQueueService)
	    self.cancelAllMissions = rospy.Service(
                '~cancel_all_missions', Trigger, self.cancelAllMissionsService)

        else:
            self.status_.changeStatus(
                StatusMsg.CURRENT_STATUS_ERROR, "ERROR to try open " + self.path_of_action_file_, True)

        self.rate_ = float(rospy.get_param('~rate', '1.0'))
        while not rospy.is_shutdown():
            self.status_.publishStatus()
            if(self.rate_):
                rospy.sleep(1/self.rate_)
            else:
                rospy.sleep(1.0)


# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('robotnik_rms')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        ne = RMS()
    except rospy.ROSInterruptException:
        pass
