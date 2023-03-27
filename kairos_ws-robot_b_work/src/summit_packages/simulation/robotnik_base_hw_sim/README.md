# robotnik_base_hw_sim

This package is intended to simulate the Robotnik base hw common in most of the platforms

## robotnik_base_hw_sim_node

### Bringup

Run the following launch inside the desired namespace.

**Arguments**

* prefix: prefix for the default joints (for multirobots envs)
* config_yaml: default config yaml. Every robot should have one.

For RB2:
```
roslaunch robotnik_base_hw_sim rb2_hw_sim.launch prefix:=rb2_a config_yaml:=rb2 
```

For SummitXLS
```
roslaunch robotnik_base_hw_sim summit_xl_hw_sim.launch prefix:=summit_xl_a config_yaml:=summit_xls
```

## robotnik_elevator_controller plugin

Plugin to simulate the Elevator interface available in some robots.

It requires a transmission of **<hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>**.

Example of control configuration:

```
  robotnik_elevator_controller:
    type        : "robotnik_elevator_controller/RobotnikElevatorController"
    joint:    
      elevator_joint:
        name: rb2_a_elevator_platform_joint
    elevator_position_up: 0.05
    elevator_position_down: 0.0
    # delay time to perform the elevation
    elevation_action_time: 1.0
    # params for the pickup avoiding Gazebo physics (see elevator_fake_pickup_gazebo)
    gazebo:
      # if true, the carts will be raised avoiding contact
      disable_gazebo_physics_for_pickup: true
      pickup_service: "/elevator_fake_pickup_gazebo/simple_pick"
      place_service: "/elevator_fake_pickup_gazebo/simple_place"
      # model id used in Gazebo
      robot_model: "rb2_a"
      # distance in z to move the picked cart
      elevation_offset_z: 0.07  
```

## elevator_fake_pickup_gazebo node

ROS node to perform pick and place of object avoiding the physics. It uses the available gazebo ROS topics and services.
It is intended to pick & place carts/trolleys with a mobile base.

The way it works:

1. Pick: removes gravity of the object to pick
2. Sets the position of the object in the same one than the robot. This is done continuosly.
3. Place: sets the gravity of the picked object and stops setting the position of the robot

### bringup

```
roslaunch robotnik_base_hw_sim elevator_fake_pickup_gazebo.launch
```

*Arguments:*
 * config_yaml: path to the yaml containing the configuration


### params

*YAML Example:*
   ```
   objects:
 -
   model: rb2cart
   default_link: link_0
 -
   model: rb2cart_0
   default_link: link_0

robots:
- 
   model: rb2_a
   default_link: rb2_a_base_footprint
- 
   model: rb2_b
   default_link: rb2_b_base_footprint

config:
  # min distance for the simple pick
  min_picking_distance: 0.1
   ```
### topics
#### publishers

 * gazebo_picking_links [gazebo_msgs/LinkStates]
   * gazebo links of the available objects to pick, based on configuration
 * gazebo_picking_models [gazebo_msgs/ModelStates]
   * gazebo models of the available objects to pick, based on configuration
 * gazebo_robot_links [gazebo_msgs/LinkStates]
   * gazebo links of the available robots/links able to pick, based on configuration
 * gazebo_robot_models [gazebo_msgs/ModelStates]
   * gazebo models of the available robots able to pick, based on configuration
 * pick_states [robotnik_base_hw_sim/PickStates]
   * List of the current elements that are attached
 * state [robotnik_msgs/State]
   * Component state machine

#### subscribers

 * /gazebo/link_states [gazebo_msgs/LinkStates]
   * state of all the links of Gazebo 
 * /gazebo/model_states [gazebo_msgs/ModelStates]
   * state of all the models of Gazebo

### services
#### clients
 * /gazebo/get_link_properties [gazebo_msgs/GetLinkProperties]
 * /gazebo/set_link_properties [gazebo_msgs/SetLinkProperties]
 * /gazebo/set_link_state [gazebo_msgs/SetLinkState]

#### servers
 * pick [robotnik_base_hw_sim/Pick]
   * Pick an object and link it to a robot
   * Example:

```
rosservice call /elevator_fake_pickup_gazebo/pick "object_model: 'rb2cart'
object_link: 'link_0'
robot_model: 'rb2_a'
robot_link: 'rb2_a_base_footprint'
pose:
  position: {x: 0.0, y: 0.0, z: 0.2}
  orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}" 
success: True
msg: "OK"
```

 * place [robotnik_base_hw_sim/Place]
   * Place a picked object    
   * Example:

```
rosservice call /elevator_fake_pickup_gazebo/place "object_model: 'rb2cart'
robot_model: 'rb2_a'" 
```
* simple_pick [robotnik_base_hw_sim/SimplePick]
   * Pick the closest object to a robot (within a min distance)
   * Example:

```
rosservice call /elevator_fake_pickup_gazebo/simple_pick "robot_model: 'rb2_a'                     
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.1
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0" 
success: True
msg: "OK"
```
* simple_place[robotnik_base_hw_sim/SimplePlace]
   * Places the object(s) the robot has picked
   * Example:

```
rosservice call /elevator_fake_pickup_gazebo/simple_place "robot_model: 'rb2_a'" 
success: True
msg: "OK"
```
