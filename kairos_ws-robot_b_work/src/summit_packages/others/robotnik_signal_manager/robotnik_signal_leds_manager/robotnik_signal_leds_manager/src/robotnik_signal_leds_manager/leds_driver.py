#!/usr/bin/env python

import rospy

from std_srvs.srv import Trigger
from robotnik_signal_msgs.srv import SetSignal, SetSignalResponse
from robotnik_leds_msgs.srv import LedEffects, LedEffectsRequest, LedEffectsResponse
from std_srvs.srv import Trigger, TriggerResponse

from robotnik_signal_interface.robotnik_signal_interface import SignalInterface


class LedsDriver(SignalInterface):
    """
    A class used to control the leds of the robot
    """

    def __init__(self):
        SignalInterface.__init__(self)


    def ros_read_params(self):
        """Gets params from param server"""

        leds_config_raw = rospy.get_param('leds_config')
        self.leds_config = {}
        for led_config in leds_config_raw:
            self.leds_config[led_config['led_label']] = led_config

        SignalInterface.ros_read_params(self)


    def ros_setup(self):
        """Creates and inits ROS components"""

        rospy.loginfo("Loading driver, waiting for Arduino Lighting Signaling (ALS module)...")

        # Wait for the service offered by the Arduino hardware
        rospy.wait_for_service('arduino_led_signaling/ack')
        rospy.wait_for_service('arduino_led_signaling/set_led_properties')
        rospy.wait_for_service('arduino_led_signaling/update_led_properties')
        rospy.wait_for_service('arduino_led_signaling/clear_effects')

        # Connect to the service offered by the Arduino hardware
        self.leds_driver_ack_service   = rospy.ServiceProxy('arduino_led_signaling/ack', Trigger)
        self.leds_driver_effect_service = rospy.ServiceProxy('arduino_led_signaling/set_led_properties', LedEffects)
        self.leds_driver_clear_effect_service = rospy.ServiceProxy('arduino_led_signaling/clear_effects', Trigger)
        self.leds_driver_update_effect_service = rospy.ServiceProxy('arduino_led_signaling/update_led_properties', Trigger)

        # Create service for clear signals
        self.set_signal_service = rospy.Service('~clear_signals', Trigger, self.clear_signals_callback)

        rospy.loginfo("ALS module activated, node already! ")

        SignalInterface.ros_setup(self)


    def ros_publish(self):
    	'''
    		Publish topics at standard frequency
    	'''
        
        SignalInterface.ros_publish(self)


    def effect_arduino_signaling_led(self, led_config, state_config):

        led_effect_config = LedEffectsRequest()
        response = LedEffectsResponse()

        # Basic parameters

        led_effect_config.id        =  state_config.get("id")
        led_effect_config.enabled   =  state_config.get("enabled")
        led_effect_config.mode      =  state_config.get("mode")
        led_effect_config.channel   =  led_config.get("channel")
        led_effect_config.type      =  led_config.get("type")
        led_effect_config.color_R   =  state_config.get("color")[0]
        led_effect_config.color_G   =  state_config.get("color")[1]
        led_effect_config.color_B   =  state_config.get("color")[2]
        led_effect_config.color_W   =  state_config.get("color")[3]
        led_effect_config.start_led =  led_config.get("leds_zone")[0]
        led_effect_config.end_led   =  led_config.get("leds_zone")[1]

        # Optional parameters

        if state_config.get("layer") is not None:

            led_effect_config.layer = state_config.get("layer")
        else:
            led_effect_config.layer = 0


        if state_config.get("ms_on") is not None:

            led_effect_config.ms_on = state_config.get("ms_on")


        if state_config.get("ms_off") is not None:

            led_effect_config.ms_off = state_config.get("ms_off")


        if state_config.get("fade_in") is not None:

            led_effect_config.fade_in = state_config.get("fade_in")


        if state_config.get("fade_out") is not None:

            led_effect_config.fade_out = state_config.get("fade_out")



        if state_config.get("background_R") is not None:

            led_effect_config.background_R = state_config.get("background_R")

        if state_config.get("background_G") is not None:

            led_effect_config.background_G = state_config.get("background_G")

        if state_config.get("background_B") is not None:

            led_effect_config.background_B = state_config.get("background_B")

        if state_config.get("background_W") is not None:

            led_effect_config.background_W = state_config.get("background_W")




        if state_config.get("direction") is not None:  

            led_effect_config.direction = state_config.get("direction")


        if state_config.get("speed") is not None: 

            led_effect_config.speed = state_config.get("speed")


        if state_config.get("sleep") is not None:

            led_effect_config.sleep = state_config.get("sleep")


        if state_config.get("led_increment") is not None:

            led_effect_config.led_increment = state_config.get("led_increment")
        else:
            led_effect_config.led_increment = 1



        response = self.leds_driver_effect_service (led_effect_config)

        return response


    def clear_signals_callback(self, req):

        res = TriggerResponse()

        # Clear driver signal array
        for signal in self.signals.values():
            signal['enabled'] = False

        # Clear effects in the ALS module
        self.leds_driver_clear_effect_service()

        res.message = "All signals have been set to False"
        res.success = True

        return res


    def hold_connection(self):
        
        res = TriggerResponse
        res = self.leds_driver_ack_service()
        #print(res.message)
        #print(res.success)

        #if res.success == True:
        #    print("---------------------------")

    def ready_state(self):
        """Actions performed in ready state"""

        # ACK bit must be sent first to avoid sending first command before ready mode in ALS module
        self.hold_connection()

        if len(self.signal_modified) > 0:

            if self.exclusive_signals:
                 self.leds_driver_clear_effect_service()

            signal_id = self.signal_modified.pop(0)
            signal_labels = self.signals[signal_id]['led_label']

            i = 0
            for signal_label in signal_labels:
                i += 1

                multi_state_config = self.signals[signal_id].copy()
                multi_state_config['led_label'] = signal_label

                #If multiple led labels, add _PART_$(arg number)
                if len(signal_labels) > 1:
                    multi_state_config['id'] += "_PART_" + str(i)
                
                led_config = self.leds_config[signal_label].copy()

                #Send the configuration to ALS device
                self.effect_arduino_signaling_led(led_config, multi_state_config)

            # Update the configuration in the ALS device
            self.leds_driver_update_effect_service()

        
        SignalInterface.ready_state(self)


    def shutdown(self):

        return SignalInterface.shutdown(self)
