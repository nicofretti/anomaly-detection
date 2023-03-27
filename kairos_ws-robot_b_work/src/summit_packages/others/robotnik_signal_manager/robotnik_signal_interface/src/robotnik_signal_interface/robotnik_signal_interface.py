#!/usr/bin/env python

import rospy

from robotnik_signal_msgs.msg import SignalStatus
from robotnik_signal_msgs.srv import SetSignal, SetSignalResponse

from rcomponent.rcomponent import *
import exceptions


class SignalInterface(RComponent):
    """
    An interface used for the signal management nodes
    """

    def __init__(self):
        self.signal_modified = []  # list with the ID's

        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""

        self.exclusive_signals = rospy.get_param('~exclusive_signals', True)

        signals_raw = rospy.get_param('signals')
        self.signals = {}
        try:
            enabled_signals = 0
            for signal in signals_raw:
                self.signals[signal['id']] = signal
                if(not isinstance(signal['enabled'], bool)):
                    raise TypeError(signal['id'])
                elif signal['enabled'] == True:
                    enabled_signals += 1
            if self.exclusive_signals == True and enabled_signals > 1:
                raise ValueError()
        except KeyError:
            rospy.logerr(
                "%s::ros_read_params: Any configuration YAML \"id\" or" +
                " \"enabled\" value has not been defined.", self._node_name)
            exit()
        except TypeError as error_id:
            rospy.logerr("%s::ros_read_params: The \"enabled\" value of " +
                         str(error_id)+" signal must be boolean type.", self._node_name)
            exit()
        except ValueError:
            rospy.logerr("%s::ros_read_params: If exclusive_signals == True, " +
                         "the number of enabled signals can be 0 or 1.", self._node_name)
            exit()

        RComponent.ros_read_params(self)

    def ros_setup(self):
        """Creates and inits ROS components"""

        # Topics
        self.signal_status_pub = rospy.Publisher(
            '~status', SignalStatus, queue_size=10)

        # Services
        self.set_signal_service = rospy.Service(
            '~set_signal', SetSignal, self.set_signal_callback)

        RComponent.ros_setup(self)

    def set_signal_callback(self, req):
        res = SetSignalResponse()
        signal_id = req.signal_id.lower()

        if self._state != State.READY_STATE:
            res.ret.success = False
            res.ret.message = self._node_name+" not in ready_state"

        elif signal_id in self.signals:  # Signal name found
            if self.signals[signal_id]['enabled'] != req.enable:
                self.signal_modified.append(signal_id)
                self.signals[signal_id]['enabled'] = req.enable
                res.ret.success = True
                res.ret.message = "Signal " + \
                    str(signal_id) + " set to " + str(req.enable) + "."

                # If exclusive_signals, disable the old signal
                if self.exclusive_signals and req.enable == True:
                    for signal in self.signals.values():
                        if signal['id'] != signal_id and signal['enabled']:
                            signal['enabled'] = False
                            res.ret.message += " Signal " + \
                                str(signal['id']) + " set to " + \
                                "False."

            else:  # Signal already on this state
                res.ret.success = True
                res.ret.message = "Signal " + \
                    str(signal_id) + " is already set to " + \
                    str(req.enable) + "."

        else:  # Signal name not found
            res.ret.success = False
            res.ret.message = "Signal " + \
                str(signal_id) + " does not exist."

        return res

    def ros_publish(self):
        '''
                Publish topics at standard frequency
        '''
        # Publish status
        signal_status_msg = SignalStatus()
        signal_status_msg.node_name = self._node_name[len(
            rospy.get_namespace()):]
        for i in self.signals.values():
            if i['enabled'] == True:
                signal_status_msg.active_signals.append(i['id'])
        self.signal_status_pub.publish(signal_status_msg)

        RComponent.ros_publish(self)

    def ready_state(self):
        """Actions performed in ready state"""

    def shutdown(self):

        return RComponent.shutdown(self)

    def switch_to_state(self, new_state):
        """Performs the change of state"""

        return RComponent.switch_to_state(self, new_state)
