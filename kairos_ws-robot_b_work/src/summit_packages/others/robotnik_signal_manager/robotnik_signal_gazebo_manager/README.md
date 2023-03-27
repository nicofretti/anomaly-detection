# robotnik_signal_gazebo_manager

Package with nodes for the control of gazebo signals.

## 1 gazebo_leds_manager

Node to control the leds of the robot on a gazebo simulation.

### 1.1 Parameters

* ~signals (config/signals.yaml)
   The different available signals on the node, and the initial state of each of them.

* ~exclusive_signals (bool, default: true)
   This param allows to limit the number of concurrent states to one.

* ~set_light_properties_service_name (string, default: /gazebo/set_light_properties)
  Name of the service used to set the colors of the leds on gazebo.
   
### 1.2 Subscribed Topics

None

### 1.3 Published Topics

* ~status (robotnik_signal_msgs/SignalStatus)
  This topic publishes the actual enabled signals

### 1.4 Services
* ~set_signal (robotnik_signal_msgs/SetSignal)
  This service allows to modify the state of the signals

### 1.5 Services Called

* ~set_light_properties_service (gazebo_msgs/SetLightProperties)
  This service is used to set the colors of the leds on gazebo.

### 1.6 Action server

None

### 1.7 Action clients called

None

### 1.8 Required tf Transforms

None

### 1.9 Provided tf Transforms

None

### 1.10 Bringup

Launch for starting the gazebo leds manager
```
roslaunch robotnik_signal_gazebo_manager gazebo_leds_manager.launch
```
