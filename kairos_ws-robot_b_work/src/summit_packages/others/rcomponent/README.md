# RComponent

This component is used as a template for new ROS nodes.

There are two available scripts:

* **node_creator**
  * Simplifies the creation of a new RComponent based node. It creates a package with a functional node in C++ or Python.
  
* **subnode_creator**
  * Simplifies the creation of new C++ nodes that inherit from others on existing packages. The existing package requires to be created with the RComponent 'node_creator' script.

---

## Dependencies

- robotnik_msgs [🔗](https://github.com/RobotnikAutomation/robotnik_msgs)

```bash
git clone https://github.com/RobotnikAutomation/robotnik_msgs.git
```

---

## ROS

### Parameters

- **desired_freq (double, default: 40.0)**: The frequency of the node.

### Topics

* #### Publications
  - **~/state (robotnik_msgs/State)**: Publishes an overview of the state of the node.
