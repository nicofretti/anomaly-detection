# Robot Manager System (RMS)

This package allows to create, modify, delete and launch missions. The `mission.yaml` is stored in the system home folder. Missions are a sequence of actions like: `move_to`, `pick`, `place`, `move_arm`... It has a queue of missions that are executed following a FIFO order.

Robot Manager System's main goal is performing and chaining repetitive missions. Example:
`Move_arm_position home` -> `Go_to_point 1` -> `Go_to_qr 4` -> `Move_arm_position up` -> `Go_to_point 0`

RMS works with a state machine, the states are:
- Error
- Pre-Idle
- Idle
- Charging mission
- Executing mission
- Pause mission (not soported yet)

The code is ready to manage **new types of actions**. For doing so, you will need to develop a new **control manager**.

Each **control manager** is an interface (inheriting from `manager_abstract_base_class.py`) between the ROS node that will execute the action and the RMS. This control manager is responsible for handling the action execution: it commands the action, knows if it is running / has finished / an error has appeared.

The managers have another state machine simpler than the RMS. The transitions between the states can be programmed, but it is necessary that for starting a mission the state must be `Idle` and it transitions to `Finalized` when it is finished.

RMS Controller Managers state machine states are:
- Error
- Pre-Idle
- Idle
- Executing
- Finalized
- Fail


## How to create a new manager

1. Create a new python file in the folder `bin/managers` with a descriptive name.
2. Add the following code:
```
#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract

class ManagerName(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self)

    #Erase if you won't define this method
    def recivedAction(self):
        None
    
    #Erase if you won't define this method
    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    #Erase if you won't define this method
    def recivedAction(self):
        print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    #Erase if you won't define this method
    def stateIdle(self):
        None

    #NECESSARY A DEFINITION
    def stateExecuting(self):
        print ("ManagerExample-->Executing: " + str(self.current_action_msg_))
        self.changeState(self.STATES_FINALIZED, "Not defined stateExecuting metoth")

    #Erase if you won't define this method
    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        #None

    #Erase if you won't define this method
    def stateError(self):
        print("Error")
        #None

    #Erase if you won't define this method
    def stateFinalized(self):
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")

```
3. Change the name of class to a more descriptive name.
4. When an action is added, it will have an id (`self.current_action_msg_`). 
    - First, the `recivedAction` method is executed. 
    - Then, if the state machine is in `Idle` state, the function `stateExecuting` is run. 
    - You can define the state machines transitions with the method `self.changeState`, which takes two parameters: 
        - The new state:
            - `STATES_FINALIZED`: mission finishes successfully.
            - `STATES_FAIL`: mission did not finish successfully.
            - `STATES_ERROR`: mission is aborted due to an error. All missions in the RMS queue are cancelled.
        - A message.
    - In the end of `stateFinalized` we use `self.changeState(self.STATES_IDLE, "Finalized")` to transition to the `Idle`state.
    - In `stateFail`, it is recomended to transition back to `stateExecuting` or `stateFinalized`.
    - If not defined, the method `stateError` the manager will not do anything.
6. Add the new manager in the `robotnik_rms.yaml`.
```
managers:
  - type_action: 0
    manager: "manager_poi_go_to"
    name_class: "ManagerPoiGoTo"
  - type_action: 1
    manager: "ManagerNameExampleReadme"
    name_class: "ManagerName"
```
- `type_action`: identification of your manager. It needs to be unique.
- `manager`: name of the Control Manager file responsible for handling this action (excluding the `.py` extension).
- `name_class`: name of the class.

You can see examples on how to do this in the included control managers.


## How to manage a mission
To manage a mission, you can use the RMS services and topics for monitoring missions status.

### Services

- `/robotnik_rms/add_mission_in_list`: adds a mission in the `.yaml` file. Srv message: `robotnik_rms_msgs/missionManager`
- `/robotnik_rms/add_mission_in_queue`: adds a mission to the execution queue. Srv message: `robotnik_rms_msgs/idMission`
- `/robotnik_rms/delete_mission`: removes a mission from the `.yaml` file. Srv message: `robotnik_rms_msgs/idMission`
- `/robotnik_rms/get_list_of_mission`: gets all the available missions in the `.yaml` file. Srv message: `robotnik_rms_msgs/getMissions`
- `/robotnik_rms/modify_mission`: modifies a mission in the `.yaml` file. Srv message: `robotnik_rms_msgs/missionManager`

### Topics
- `/robotnik_rms/status`: returns the status of the state machine and information about the current state.


## Authors

* **Juan Manuel Navarrete Carrascosa**
