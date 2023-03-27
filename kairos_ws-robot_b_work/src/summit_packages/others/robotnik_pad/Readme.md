# robotnik_pad

This package is intended to be used as a standard pad for all Robotnik robots.

## Installation

The package depends on some Robotnik packages:

- rcomponent [🔗](https://github.com/RobotnikAutomation/rcomponent/)
```bash
git clone https://github.com/RobotnikAutomation/rcomponent
```
- robotnik_msgs [🔗](https://github.com/RobotnikAutomation/robotnik_msgs/)
```bash
git clone https://github.com/RobotnikAutomation/robotnik_msgs
```

This package may depend on other Robotnik or ROS standard packages in function of the handlers that are developed. The standard ROS packages can be installed using the rosdep install command:

```bash
# located in the workspace folder
rosdep install --from-path src --ignore-src -y -r
```
### ds4drv automatic install
in order to install the ds4drv and its components you can use the installer:
```bash
sudo ./ds4drv-install.sh
```
Now you your system should be ready

### ds4drv manual install
You need to install the ds4drv pip script:

```bash
sudo pip install ds4drv
```

Install PS4 controller config for ds4drv:
```bash
cd /etc && sudo wget https://raw.githubusercontent.com/RobotnikAutomation/robotnik_pad/master/ds4drv.conf
```
```bash
cd /etc/systemd/system && sudo wget https://raw.githubusercontent.com/RobotnikAutomation/robotnik_pad/master/ds4drv.service
```
Add the udev rules for PS4 controller:

```bash
cd /etc/udev/rules.d/ && sudo wget https://raw.githubusercontent.com/RobotnikAutomation/robotnik_pad/master/55-ds4drv.rules
```
And paste the following text:
```
KERNEL=="js[0-9]*", SUBSYSTEM=="input", SYMLINK+="input/js_base", ATTRS{name}=="Sony Computer Entertainment Wireless Controller"
```

Enable the execution of the ds4drv service on boot:
```bash
sudo systemctl daemon-reload
```
```bash
sudo systemctl enable ds4drv.service
```
Now the ds4drv is loaded on boot.

To enable the joystick without rebooting:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo systemctl start ds4drv.service
```

---
## 1. robotnik_pad_node

The `robotnik_pad_node` loads plugins that specify the desired behaviour. This allows you to load different plugins depending on your needs. Besides, it is possible to create your own plugins.

Available plugins:
  * **Movement**
    * Intended to command the mobile base via velocity references
  * **Elevator**
    * Intended to command a elevator actuator of the robot

### 1.1 Parameters

#### 1.1.1 Common Parameters

* ~pad/type (string, default: ps4)
   Type/model of the gamepad used
* ~pad/num_of_buttons (int, default: 14)
   Number of buttons of the gamepad
* ~pad/num_of_axes (int, default: 4)
   Number of axes of the gamepad
* ~plugins (list of strings, default: [Movement])
   Declares the available plugins to use


#### 1.1.2 Plugin Parameters

*  ~*Plugin*/type (string, default: *depends on the plugin*)
   Type of plugin

#### 1.1.2.1 Movement

* ~max_linear_speed (double, default: 1.5)
  Maximum linear speed that can be sent to the controller based on the current velocity level (0.1->1) and the current axis_linear_x value (0->1)  
* ~max_angular_speed (double, default: 3.0)
  Maximum angular speed that can be sent to the controller based on the current velocity level (0.1->1) and the current axis_linear_x value (0->1)  
* ~cmd_topic_vel (string, default: cmd_vel)
  Name of topic where the command vel is being published
* ~config/button_deadman (int, default: 5)
  Button number to enable any command sent to the controller.
* ~config/button_speed_up (int, default: 3)
  Button number to increase the current velocity level applied to the max_speed params
* ~config/button_speed_down (int, default: 1)
  Button number to decrease the current velocity level applied to the max_speed params
* ~config/button_kinematic_mode (int, default: 7)
  Button number to switch between two kinematic modes: diff and omni
* ~config/axis_linear_x (int, default: 1)
  Axis number to set the linear x speed
* ~config/axis_linear_y (int, default: 0)
  Axis number to set the linear y speed
* ~config/axis_angular_z (int, default: 2)
  Axis number to set the angular speed

#### 1.1.3 Configuration file example

This an example of a config file loading a single plugin:

```yaml
---
plugins:
  - Movement

pad:
  type: ps4
  num_of_buttons: 14
  num_of_axes: 14
  joy_topic: joy

Movement:
  type: robotnik_pad_plugins/Movement
  max_linear_speed: 1.5
  max_angular_speed: 3.0
  cmd_topic_vel: pad_teleop/cmd_vel
  config:
    button_deadman: 5
    axis_linear_x: 1
    axis_linear_y: 0
    axis_angular_z: 2
    button_speed_up: 3
    button_speed_down: 1
    button_kinematic_mode: 7
```

First of all you need to define a list containing the different plugins you want to load. Then, for each of the plugins you want to load, you should specify its parameters.

### 1.2 Subscribed Topics

* joy (sensor_msgs/Joy)
  Gets the buttons and axis status

### 1.3 Published Topics

* *cmd_topic_vel* (geometry_msgs/Twist)
  Sends the velocity references to define topic
* ~state (robotnik_msgs/State)
  Sends current state based on Robotnik RComponent architecture

#### 1.3.1 Plugin Published Topics

##### 1.3.1.1 Movement

* ~status (robotnik_pad_msgs/MovementStatus)
  Status of the component
  * Example:
    ```
    velocity_level: 100.0
    kinematic_mode: "omni"
    ```

### 1.4 Services

None

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

```bash
roslaunch robotnik_pad robotnik_pad.launch
```

This will launch two nodes:
- joy: This node is in charge of reading from the joystick and publish the information (sensor_msgs/Joy) through a topic
- robotnik_pad_node: This node will load the different plugins included in the config file
