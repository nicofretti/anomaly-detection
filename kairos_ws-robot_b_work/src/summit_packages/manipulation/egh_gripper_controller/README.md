# egh_gripper_controller

Package to control the Schunk EGH gripper.

At the moment the gripper can only be controlled via the UR's digital input/output interface.

## Index

- [Dependencies](#Dependencies)
- [Gripper Controller](#Gripper-Controller)
- [UR IO Gripper Controller](#UR-IO-Gripper-Controller)

---

## Dependencies

The package depends on some Robotnik packages:

- rcomponent [🔗](https://github.com/RobotnikAutomation/rcomponent/)

```bash
git clone https://github.com/RobotnikAutomation/rcomponent/
```

The standard ROS packages can be installed using the rosdep install command:

```bash
# located in the workspace folder
rosdep install --from-path src --ignore-src -y -r
```

---

## Gripper controller

The GripperController class defines the generic structure of the controller. It defines the parameters, actions and services common to all possible gripper controllers.

### Parameters

| Name                   | Type   | Description                                                        | Default value |
|------------------------|--------|--------------------------------------------------------------------|---------------|
| **status_topic_name**      | string | The name of the topic where the status of the gripper is published | status        |
| **close_service_name**     | string | The name of the service to close the gripper                       | close         |
| **open_service_name**      | string | The name of the service to open the gripper                        | open          |
| **master_joint**           | string | The master joint of the mimic gripper                              | master_joint  |
| **min_opening**            | float  | Minimum separation between fingers                                 | 0.0           |
| **max_opening**            | float  | Maximum separation between fingers                                 | 0.11          |
| **master_joint_direction** | int    | Axis direction of the master joint                                 | -1            |

### Subscriptions

| Name         | Type                   | Description                                                                               |
|--------------|------------------------|-------------------------------------------------------------------------------------------|
| **joint_states** | sensor_msgs/JointState | The controller reads the joint position to know if the gripper is in the desired position |

### Publications

| Name   | Type                | Description                                                 |
|--------|---------------------|-------------------------------------------------------------|
| **status** | robotnik_msgs/State | Shows the global status of the node                         |
| **state**  | string              | Shows if the gripper is currently running an action or idle |

### Services

| Name  | Type             | Description                  |
|-------|------------------|------------------------------|
| **open**  | std_srvs/Trigger | Service to open the gripper  |
| **close** | std_srvs/Trigger | Service to close the gripper |

### Action Server

| Namespace   | Type                              | Descrition                                                                                      |
|-------------|-----------------------------------|-------------------------------------------------------------------------------------------------|
| gripper_cmd | control_msgs/GripperCommandAction | Manages the position of the gripper with the position provided by the goals that the user sends |

---

## UR IO Gripper Controller

This controller extends GripperController by managing the actual movement of the gripper through digital signals to the UR arm.

### Parameters

| Namespace                  | Type   | Descrition                                                      | Default value            |
|----------------------------|--------|-----------------------------------------------------------------|--------------------------|
| ur_io_service_name         | String | Name of the service to set UR arm IO                            | ur_driver/set_io         |
| update_joints_service_name | String | Name of the service to update the position of the desired joint | joints_updater/set_joint |
