#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from map_nav_manager.msg import PoiState
from map_nav_manager.srv import GoToTag
from map_nav_manager.srv import GoToTagRequest


class ManagerGoToGPS(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self)

        global PoiState
        global GoToTag
        global GoToTagRequest

        self.state_poi_topic = "/poi_interactive_marker/state"
        self.send_poi_service = "/poi_interactive_marker/goto_tag"
        self.state_feedback_message = None

        rospy.Subscriber(self.state_poi_topic, PoiState,
                         self.state_poi_callback)

    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
        print("ManagerPoiGoTo-->0Executing: " + str(self.current_action_msg_))
        if (self.callPoiManagerService(self.current_action_msg_)):
            print("ManagerPoiGoTo-->Executing: " +
                  str(self.current_action_msg_))
            self.changeState(self.STATES_FINALIZED, "Finalize")
        else:
            self.changeState(self.STATES_FAIL, "Fail to send a point")

    

    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        # None

    def stateError(self):
        print("Error")
        # None

    def stateFinalized(self):
        print("ManagerPoiGoTo--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")

    def state_poi_callback(self, data):
        self.state_feedback_message = data

    def callPoiManagerService(self, id):

        rospy.wait_for_service(self.send_poi_service)
        try:
            service_go_to_tag = rospy.ServiceProxy(
                self.send_poi_service, GoToTag)
            request_pose_list = GoToTagRequest()
            request_pose_list.name = id
            response = service_go_to_tag(request_pose_list)
            if(response.ret == False):
                return False
            else:
                rate = rospy.Rate(10)  # 10hz
                '''if(self.state_feedback_message.action!=PoiState.IDLE):
                    while(self.state_feedback_message.action!=PoiState.IDLE):
                        print("waiting for init state")
                        rate.sleep()'''
                while(self.state_feedback_message.action != PoiState.IDLE):
                    print("waiting for finish action")
                    rate.sleep()
                return True

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False
