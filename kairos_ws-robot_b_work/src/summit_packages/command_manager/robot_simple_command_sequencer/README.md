# robot_simple_command_sequencer

This package contains a node (**robot_simple_command_sequencer**) that provides a way to create easily simple sequences of commands. This commands are supported by the package [**robot_simple_command_manager**](https://github.com/RobotnikAutomation/robot_simple_command_manager/).

## Index

- [Installation](#Installation)
- [1. command_sequencer](#1.-command_sequencer)
  - [1.1 Parameters](#1.1-Parameters)
  - [1.2 Subscriptions](#1.2-Subscriptions)
  - [1.3 Publications](#1.3-Publications)
  - [1.4 Services](#1.4-Services)
  - [1.5 Action Server](#1.5-Action-Server)
  - [1.6 Examples of use](#1.6-Examples-of-use)
---

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

The main features:

* Execute sequences (of commands)
* The sequences can be queued
* Multiple input channels allowed: it is possible to interact with the node via topic, service and action server
* Sending a command does not overwrite/preempt a command that is currently running.To overwrite a command, you must first explicitly cancel it.


### 1.1 Parameters

- **allow_queuing** (bool, default: False): Enables the queuing of commands. Otherwise ignores any command unless the current one finishes or it's cancelled.

- **action_client_namespace** (string, default: command_manager/action): Action client namespace of the command managed node, in charge of the execution of the commands.

- **sequences** (dict, default: {}): Loads the list of available sequences. Example:

  ```yaml
  sequences:
    GOTO_TEST:
      - GOTO 1 0 0
      - GOTO 1 2 0
      - GOTO -1 0 0
    GOTO_TEST2:
      - GOTO 3 0 0
      - GOTO -3 0 0
      - GOTO 2 2 3.14
  ```

This node also has all the parameters available from its parent node, [RComponent](https://github.com/RobotnikAutomation/rcomponent/blob/master/src/rcomponent/rcomponent.py).

### 1.2 Subscriptions

- **command** (robot_simple_command_manager_msgs/CommandString): Topic to receive commands.

- **cancel** (std_msgs/Empty): Topic to cancel commands.

The command_sequencer_node may be subscribed also to the feedback, result and status topics that the actions handlers use.

### 1.3 Publications

- **state** (robotnik_msgs/State): Publishes the current state of the node

- **feedback** (robot_simple_command_manager_msgs/CommandStringFeedback): Publishes the feedback related to the current command sent.

The command_sequencer_node may also publish to the cancel and goal topics that the actions handlers use.

### 1.4 Services

- **command** (robot_simple_command_manager_msgs/SetCommandString): Service to receive commands.

- **cancel** (std_srvs/Trigger): Service to cancel commands.

### 1.5 Action Server

- **action** (robot_simple_command_manager_msgs/CommandStringActionAction): Action server to receive commands.

### 1.6 Examples of use

#### 1.6.1 Command types and format

The type of exchanged data with this component is string based. It means we have to compose commands into strings and read the feedback from strings.

The available commands are defined in the param *sequences*.

The command string can contain from 1 to n sequences. Examples:
* command: 'GOTO_TEST'
* command: 'GOTO_TEST GOTO_TEST2 GOTO_TEST'

**Keywords**
Reserved and special keywords can be combined as another command.

* ***LOOP***: it puts the execution of the command in a infinite loop. Send *cancel* to stop it.
  * example: command: 'GOTO_TEST GOTO_TEST2 LOOP'


#### 1.6.2 Send commands

You can send commands through **topic**:

```bash
rostopic pub /command_sequencer/command robot_simple_command_manager_msgs/CommandString "command: 'GOTO_TEST'"
```

**service**:

```bash
rosservice call /command_sequencer/command "command: 'GOTO_TEST'"
```

or **action**:

```bash
rostopic pub /command_sequencer/action/goal robot_simple_command_manager_msgs/RobotSimpleCommandActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  command:
    command: 'GOTO_TEST'"

```

#### 1.6.3 Cancel commands

You can cancel commands through *topic*:

```bash
rostopic pub /robot/command_sequencer/cancel std_msgs/Empty "{}"
```

*service*:

```bash
rosservice call /robot/command_sequencer/cancel "{}"
```

or *action* (this only cancels action goals):

```bash
rostopic pub /robot/command_sequencer/action/cancel actionlib_msgs/GoalID "stamp:
  secs: 0
  nsecs: 0
id: ''"
```
