# Safety_module

Node that interacts with the Flexisoft Safety Module via Modbus protocl

## 1.Dependencies

### 1.1 ROS dependencies

* [robotnik_msgs](https://github.com/RobotnikAutomation/robotnik_msgs)
* [robotnik_modbus_io](https://github.com/RobotnikAutomation/robotnik_modbus_io): The communication with modbus id done through this node

## 2. Start-up

First, you need robotnik_modbus_io node running (check that the IP is correct)

`roslaunch robotnik_modbus_io robotnik_modbus_io.launch`

After that, run the node:

`roslaunch safety_module safety_module_node.launch`

## 3. Params
Most params are to configure modbus addresses and I/O numbers.

* **desired_freq**(double)

* **address_registers**: list of all the addresses to work with registers
    * **laser_mode_output**(unsigned int): writes the desired laser mode
     * **current_speed**(unsigned int): writes the current robot speed


* **outputs**: list of the standard outputs to interact with module
    * **standby**(unsigned int): sets the lasers and safety module into standby, disabling power in motors and scans
    * **watchdog_signals**(array of two unsigned int): sets a quadratic signal that allows the Flexisoft module to know that the controller is alive. Otherwise it disables power. Setting an empty array [] means to disable the publication of the signal.
    * **emergency_stop_sw**(unsigned int): causes E-Stop hardware equivalent

* **lasers_mode**: available configuration for the lasers safety mode. By default "standard". Example:
``` 
  standard:     #  custom name of the mode
    input: 237  # input to read the mode
    output: 0   # value of the register when writing in the modbus address for the lasers mode
```

* **custom_outputs**: list of custom outputs mapped and tagged for specific applications


* **inputs**: list of inputs read from the module
    * **emergency_stop** 
    * **auto_mode** 
    * **emergency_mode** 
    * **manual_mode** 
    * **safety_overrided** 
    * **safety_stop** 
    * **standby** 
    * **wheels_power_enabled** 
    * **laser_ok** 
    * **edm_fault** 
    * **emergency_stop_fault** 
    * **motion_enabled** 
    * **emergency_stop_sw** 
    * **watchdog_ok** 
    * **lasers**
        * **front**:
            * detecting_obstacles: True if there's an obstacle in the stop area
            * contamination_led: 
            * reset_pressed: 
            * free_warning: 
        * **rear**:
            * detecting_obstacles: 
            * contamination_led: 
            * reset_pressed: 
            * free_warning: 

* **custom_inputs**:


**watchdog_signals_frequency**(double): frequency to write the quadratic signal to keep the module
**set_speed_feedback_to_safety_module**(bool): sets the current speed to the safety module

## 4. Topics

### 4.1 Publishers
 * **~/safety_module/emergency_stop [std_msgs/Bool]**
 * **~/safety_module/named_io [robotnik_msgs/named_inputs_outputs]**
It publishes the current state of all the signals, the standard and custom ones. 
```
digital_inputs: 
  - 
    name: "wheels_power_enabled"
    value: True
  - 
    name: "watchdog_ok"
    value: False
  - 
    name: "edm_fault"
    value: False
  - 
    name: "battery_ok"
    value: True
  - 
    name: "elevator_down"
    value: False
  - 
    name: "elevator_up"
    value: False
  - 
    name: "charge_photocell_fault"
    value: True
  - 
    name: "selector_fault"
    value: False
  - 
    name: "laser_ok"
    value: True
  - 
    name: "emergency_stop_fault"
    value: False
  - 
    name: "emergency_stop_sw"
    value: False
  - 
    name: "motion_enabled"
    value: True
  - 
    name: "ready_to_swap_batteries"
    value: False
digital_outputs: 
  - 
    name: "release_battery"
    value: False
  - 
    name: "block_battery"
    value: False
  - 
    name: "emergency_stop_sw"
    value: False
```

 * **~/safety_module/safety_stop [std_msgs/Bool]**
 It publishes true if the safety is triggered.
 
 * **~/safety_module/speed_feedback [std_msgs/Int32]**
 Current speed being sent to the module.
 
 * **~/safety_module/state [robotnik_msgs/State]**
 Current component state
 
``` 
state: 300
desired_freq: 5.0
real_freq: 4.99675226212
state_description: "READY_STATE"
```

 * **~/safety_module/status [robotnik_msgs/SafetyModuleStatus]**
    This one gives you the status of the module

```
safety_mode: "safe"
charging: False
emergency_stop: False
safety_stop: True
safety_overrided: False
lasers_on_standby: False
lasers_mode: 
  name: "standard"
lasers_status: 
  - 
    name: "front"
    detecting_obstacles: False
    contaminated: False
    free_warning: False
  - 
    name: "rear"
    detecting_obstacles: False
    contaminated: False
    free_warning: False
``` 
 
 * **~/safety_module/watchdog_signals [robotnik_msgs/BoolArray]**
 Current value of the signals sent to the module. For debugging

### 4.2 Subscribers

 * **/base/odom [nav_msgs/Odometry]**
    * Receives the odometry from the robot
 * **/base/robotnik_modbus_io/input_output [robotnik_msgs/inputs_outputs]**
    * Receives the current IO modbus state


### Services

 * **~/safety_module/set_laser_mode [robotnik_msgs/SetLaserMode]**
 Sets the current laser mode based on internal configuration.
 Modes available: standard, docking_station
 This modes will change the laser detection ranges
 
 * **~/safety_module/set_named_output [robotnik_msgs/SetNamedDigitalOutput]**
 Sets any of the named digital outputs set in configuration.
 
 * **~/safety_module/set_to_standby [std_srvs/SetBool]** 
 Puts the module on standby, disabling power and the lasers.

