# poi_manager

A ROS node to manage the points of interest in a map. It reads a list of tagged positions from a YAML file and offers services to obtain the list and update it. The node offers the following services:
- ($id_robot)/read_pois
- ($id_robot)/update_pois

**arg id_robot will be the namespace under the node will be launched**

## Launch example:

```
> roslaunch poi_manager poi_manager.launch id_robot:=namespace
```
