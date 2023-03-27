rostful_tests
=============

ROS package to test rostful package


Run the tests topics and services.

```
rostopic pub -r 1 /hello std_msgs/String "hello"
rostopic pub -r 1 /bye std_msgs/String "bye"
rosrun rostful_tests fake_node.py
```

Run the rostful_server

```
roslaunch rosful_tests test_server.launch
```

Run the rostful_client

```
roslaunch rosful_tests test_client.launch
```

