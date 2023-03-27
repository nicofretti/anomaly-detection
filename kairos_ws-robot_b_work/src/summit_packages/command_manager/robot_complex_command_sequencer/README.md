# robot_simple_command_sequencer

This package contains a node (**robot_complex_command_sequencer**) that provides a way to create sequences of commands. This commands are supported by the packages [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_simple_command_manager/) and [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_simple_command_sequencer/).

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


This package may depend on other Robotnik or ROS standard packages in function of the handlers that are developed. The standard ROS packages can be installed using the rosdep install command:

```bash
# located in the workspace folder
rosdep install --from-path src --ignore-src -y -r
```

---

## 1. command_sequencer

Main features:

* Execute sequences (of commands). It can manage sequences of sequences. For Example:

  M1:
  - GOTO_TEST
  - GOTO_TEST2
  M1_BATLOW:
  - GOTO_TEST2
  M1_ERROR:
  - GOTO_TEST2

Where GOTO_TEST and GOTO_TEST2 are sequences of a simple command sequencer

Works as the simple command sequencer but with two differences: Error and BatteryLow sequences.
For example a M1 set of Sequences uses also M1_ERROR and M1_BATLOW.

- If Error in M1 then cancel M1 execution and executes M1_Error
- If charge battery service called during M1 execution Cancel M1 and executes M1_BATLOW 

### 1.1 Parameters

- **allow_error_sequence** if False in case of error the sequencer will not start error sequence, M1_ERROR in example


This node also has all the parameters available from its parent node, [RComponent](https://github.com/RobotnikAutomation/rcomponent/blob/master/src/rcomponent/rcomponent.py).

