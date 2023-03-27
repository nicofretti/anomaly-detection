#!/usr/bin/env python

import rospy

from std_srvs.srv import Trigger
from robotnik_msgs.msg import State
from gazebo_msgs.srv import SetLightProperties, SetLightPropertiesRequest

from robotnik_signal_interface.robotnik_signal_interface import SignalInterface


class LedsManager(SignalInterface):
    """
    A class used to control the leds of the robot simulation
    """

    def __init__(self):
        SignalInterface.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""

        self.set_light_properties_service_name = rospy.get_param(
            '~set_light_properties_service_name', '/gazebo/set_light_properties')

        SignalInterface.ros_read_params(self)

    def ros_setup(self):
        """Creates and inits ROS components"""

        # Connect to the service offered by the gazebo leds plugin
        self.set_light_properties_service = rospy.ServiceProxy(
            self.set_light_properties_service_name, SetLightProperties)

        SignalInterface.ros_setup(self)

    def ros_publish(self):
        """Publish topics at standard frequency"""

        SignalInterface.ros_publish(self)

    def ready_state(self):
        """Actions performed in ready state"""

        try:

            # Wait for the service offered by the gazebo leds plugin
            rospy.wait_for_service(
                self.set_light_properties_service_name, 1 / self._desired_freq)

            if len(self.signal_modified) > 0:
                signal_id = self.signal_modified[0]
                led_config = SetLightPropertiesRequest()
                led_config.diffuse.a = 1.0
                led_config.attenuation_constant = 1.0

                if signal_id == "ready":
                    led_config.diffuse.g = 1.0

                elif signal_id == "warning":
                    led_config.diffuse.r = 0.5
                    led_config.diffuse.g = 0.5

                elif signal_id == "emergency":
                    led_config.diffuse.r = 1.0

                else:
                    led_config.diffuse.r = 1.0
                    led_config.diffuse.g = 1.0
                    led_config.diffuse.b = 1.0

                response = True
                for led_name in ['front_led', 'left_led', 'right_led']:
                    led_config.light_name = led_name
                    response = response and self.set_light_properties_service(
                        led_config)
                if response.success == True:
                    rospy.loginfo(
                        "%s::ready_state: The led state has been changed successfully.", self._node_name)
                    self.signal_modified.pop(0)
                else:
                    rospy.logerr(
                        "%s::ready_state: Error on changing led state.", self._node_name)

        except rospy.ServiceException as exc:
            rospy.logwarn("%s::ready_state: "+str(exc)+".", self._node_name)
            self.switch_to_state(State.EMERGENCY_STATE)
        except rospy.ROSException as exc:
            rospy.logwarn("%s::ready_state: "+str(exc)+".", self._node_name)
            self.switch_to_state(State.EMERGENCY_STATE)

    def emergency_state(self):
        try:
            # Wait for the service offered by the gazebo leds plugin
            rospy.wait_for_service(
                self.set_light_properties_service_name, 1 / self._desired_freq)
            self.switch_to_state(State.READY_STATE)
        except rospy.ROSException as exc:
            rospy.logwarn("%s::emergency_state: " +
                          str(exc)+".", self._node_name)

    def shutdown(self):

        return SignalInterface.shutdown(self)
