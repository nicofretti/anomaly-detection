# robot_command_scheduler

This package contains a node (**robot_command_scheduler**) that provides a way to program the launch of sequences of commands . This commands are supported by the packages [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_simple_command_manager/) and [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_simple_command_sequencer/) and [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_complex_command_sequencer/).

## Index

- [Installation](#Installation)

## Installation

The package depends on some Robotnik packages:

- rcomponent [🔗](https://github.com/RobotnikAutomation/rcomponent/)

```bash
git clone https://github.com/RobotnikAutomation/rcomponent/
```

- robot_simple_command_manager_msgs[🔗](https://github.com/RobotnikAutomation/robot_simple_command_manager_msgs)

```bash
git clone https://github.com/RobotnikAutomation/robot_simple_command_manager_msgs
```

- robotnik_msgs[🔗](https://github.com/RobotnikAutomation/robotnik_msgs)

```bash
git clone https://github.com/RobotnikAutomation/robotnik_msgs
```


This package may depend on other Robotnik or ROS standard packages in function of the handlers that are developed. The standard ROS packages can be installed using the rosdep install command:

```bash
# located in the workspace folder
rosdep install --from-path src --ignore-src -y -r
```

---

## 1. command_scheduler

The main features:

* Program the execution of sequences (of commands)

It can be used to send commands to the simple command manager, the simple command sequencer and the complex command sequencer changing the action client namespace
### 1.1 Parameters
The scheduled sequences are defined in the yaml sequences file

### 1.1 Parameters

- **action_client_namespace** action client namespace to send the commands
- **schedules_file** yaml file to define the sequences or commands to schedule

schedules:
id: Day of Week(A or 0..6) Hour (A or 0..23) Minute (0..59) Sequence

 Example:
   
 Schedule S1: Run sequence M1 on Wednesday at 16:50
  S1: 2 16 50 M1

 Schedule S2: Run sequence M2 Every Day at 16:51
  S2: A 16 51 M2


This node also has all the parameters available from its parent node, [RComponent](https://github.com/RobotnikAutomation/rcomponent/blob/master/src/rcomponent/rcomponent.py).

