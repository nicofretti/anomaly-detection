# robotnik_signal_interface

Interface that the other signal management nodes uses. This interface is based on Robotnik [RComponent](https://github.com/RobotnikAutomation/rcomponent).


## 1 robotnik_signal_interface

### 1.1 Parameters

* ~signals (config/signals.yaml)
   The different available signals on the node, and the initial state of each of them.

* ~exclusive_signals (bool, default: true)
   This param allows to limit the number of concurrent states to one.
   
### 1.2 Subscribed Topics

None

### 1.3 Published Topics

* ~status (robotnik_signal_msgs/SignalStatus)
  This topic publishes the actual enabled signals

### 1.4 Services
* ~set_signal (robotnik_signal_msgs/SetSignal)
  This service allows to modify the state of the signals

### 1.5 Services Called

None

### 1.6 Action server

None

### 1.7 Action clients called

None

### 1.8 Required tf Transforms

None

### 1.9 Provided tf Transforms

None

### 1.10 Bringup

Launch for testing the interface
```
roslaunch robotnik_signal_interface signal_interface.launch
```
