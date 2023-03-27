# ping_time_monitor

Package intended to monitor the ping communication with any host

## subscribed topics

None

## published topics

**~/state** (robotnik_msgs/State)
Overall component state.

state: 300
desired_freq: 0.5
real_freq: 0.499733656645
state_description: "READY_STATE"



**~/status** (ping_time_monitor/PingStatus)

It provides information about the ping command result

```
header: 
  seq: 1
  stamp: 
    secs: 1554722013
    nsecs: 583703041
  frame_id: ''
time: 6.71000003815
host: "192.168.0.200"
count: 2
timeout: 1.0
packet_loss: 0.0
t_min: 2.14400005341
t_avg: 3.36100006104
t_max: 4.57800006866
t_mdev: 1.21700000763
```

## parameters

**host** (string, default: "127.0.0.1")

Host to ping

**count** (int, default: 4)

The number of times to send the ping request

**timeout** (int, default: 1)

Sets the time(seconds) to wait for a response

**node_name** (string, default: "ping_time_monitor")

Name for the node
