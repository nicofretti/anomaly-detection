#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from actionlib_msgs.msg import GoalStatusArray as StateDockingQR
from std_srvs.srv import SetBool
from std_srvs.srv import SetBoolRequest


class ManagerQr(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self)

        global StateDockingQR
        global SetBool
        global SetBoolRequest
        self.state_qr_topic = "/docker/status"
        self.send_qr_service = "/poi_interactive_marker/docking_qr"
        self.state_feedback_message = None
        rospy.Subscriber(self.state_qr_topic, StateDockingQR,
                         self.state_qr_callback)

    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
        print("ManagerExample-> Recive a action: " +
              str(self.current_action_msg_))

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
        if (self.callPoiManagerService(self.current_action_msg_)):
            print("ManagerExample-->Executing: " +
                  str(self.current_action_msg_))
            self.changeState(self.STATES_EXECUTING, "Executing")
        else:
            self.changeState(self.STATES_FAIL, "Fail to send a point")

    def stateExecuting(self):
        rate = rospy.Rate(2)  # 10hz
        for statu in self.state_feedback_message["status_list"].status_list:
            if (statu["status"] == statu["SUCCEEDED"] or statu["status"] == statu["ABORTED"]):
                self.changeState(self.STATES_FINALIZED, "Finalize")
            else:
                print("waiting for finish action")
                rate.sleep()

    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        # None

    def stateError(self):
        print("Error")
        # None

    def stateFinalized(self):
        print("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")

    def state_qr_callback(self, data):
        self.state_feedback_message = data

    def callPoiManagerService(self, id):
        rospy.wait_for_service(self.send_qr_service)
        try:
            service_qr = rospy.ServiceProxy(self.send_qr_service, SetBool)
            request_qr = SetBoolRequest()
            request_qr.data = True
            response = service_qr(request_qr)
            if(response.success == False):
                return False
            else:

                return True

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False
