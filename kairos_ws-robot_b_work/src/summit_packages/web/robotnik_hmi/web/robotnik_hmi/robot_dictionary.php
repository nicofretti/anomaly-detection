<?php

$robot_dictionary = array();
$robot_dictionary["prefix"] = "/robot";
$robot_dictionary["global_frame"] = "robot_map";
$robot_dictionary["description"] = $robot_dictionary["prefix"]."/robot_description";
$robot_dictionary["topic_tf2_web_republisher"] = "/tf2_web_republisher";
$robot_dictionary["republisher_tfs"] = $robot_dictionary["prefix"]."/republisher_tfs";
$robot_dictionary["topic_poi_interactive_marker"] = $robot_dictionary["prefix"]."/poi_interactive_marker";
$robot_dictionary["topic_scan"] = $robot_dictionary["prefix"]."/throttle/front_laser/scan";
$robot_dictionary["topic_front_camera"]= $robot_dictionary["prefix"]."/front_ptz_camera/image_raw";
$robot_dictionary["topic_2d_scan"]= $robot_dictionary["prefix"]."/throttle/front_laser/scan";
$robot_dictionary["topic_map"] = $robot_dictionary["prefix"]."/throttle/map";
$robot_dictionary["topic_odometry"] = $robot_dictionary["prefix"]."/robotnik_base_control/odom";
$robot_dictionary["servic_reset_odometry"] = $robot_dictionary["prefix"]."/set_odometry";
$robot_dictionary["topic_joystick_robot_control"] = $robot_dictionary["prefix"]."/pad_teleop/cmd_vel";
$robot_dictionary["topic_joystick_axi_control"] = $robot_dictionary["prefix"]."/front_ptz_camera/front_ptz_camera/ptz_command";
$robot_dictionary["topic_system_status"] = $robot_dictionary["prefix"]."/system_monitor/diagnostics";
$robot_dictionary["topic_hw_battery"] = $robot_dictionary["prefix"]."/battery_estimator/data";
$robot_dictionary["topic_charge_manager_docking_status"]=$robot_dictionary["prefix"]."/charge_manager/docking_status";
$robot_dictionary["topic_hw_emergency_stop"] = $robot_dictionary["prefix"]."/robotnik_base_hw/emergency_stop";
$robot_dictionary["topic_hw_io"] = $robot_dictionary["prefix"]."/robotnik_base_hw/io";
$robot_dictionary["topic_hw_status"] = $robot_dictionary["prefix"]."/robotnik_base_hw/status";
$robot_dictionary["topic_fix"] = $robot_dictionary["prefix"]."/gps/fix";
$robot_dictionary["gpsOdometryFilterTopic"] = $robot_dictionary["prefix"]."/filtered_world";

$robot_dictionary["service_map_nav_manager_base"] = $robot_dictionary["prefix"]."/map_nav_manager";
$robot_dictionary["service_poi_interactive_marker_base"] = $robot_dictionary["prefix"]."/poi_manager";

$robot_dictionary["service_robotnik_gps_waypoint_get_route_base"] = $robot_dictionary["prefix"]."/robotnik_gps_waypoints";
$robot_dictionary["service_robotnik_rms_base"] = $robot_dictionary["prefix"]."/robotnik_rms";
$robot_dictionary["service_poi_interactive_marker_add_point"] = $robot_dictionary["prefix"]."/poi_interactive_marker/add_pois_2d";
$robot_dictionary["service_poi_manager_read_points"] =$robot_dictionary["prefix"]."/poi_manager/read_pois";
$robot_dictionary["service_poi_interactive_marker_goto_tag"] = $robot_dictionary["prefix"]."/poi_interactive_marker/goto_tag";
$robot_dictionary["service_poi_interactive_marker_stop_goto"] = $robot_dictionary["prefix"]."/poi_interactive_marker/stop_goto";
$robot_dictionary["service_poi_interactive_marker_load_points"] = $robot_dictionary["prefix"]."/poi_interactive_marker/load_poi";
$robot_dictionary["topic_robot_pose"]=$robot_dictionary["prefix"]."/robot_pose";
$robot_dictionary["topic_robot_local_state"]=$robot_dictionary["prefix"]."/robot_local_control/state";
$robot_dictionary["web_video_server_port"]="8081";
$robot_dictionary["service_poi_manager_get_poi_list"]=$robot_dictionary["prefix"]."/poi_manager/get_poi_list";

$robot_dictionary["service_poi_interactive_marker_add_point"] = $robot_dictionary["prefix"]."/poi_interactive_marker/add_pois_2d";
$robot_dictionary["service_poi_manager_read_points"] =$robot_dictionary["prefix"]."/poi_manager/read_pois";

$robot_dictionary["topic_robot_local_state"]=$robot_dictionary["prefix"]."/robot_local_control/state";
$robot_dictionary["service_rlc_localization_stop"] = $robot_dictionary["prefix"]."/robot_local_control/LocalizationComponent/stop_running_module";
$robot_dictionary["service_rlc_localization_start"] = $robot_dictionary["prefix"]."/robot_local_control/LocalizationComponent/start_module";
$robot_dictionary["service_rlc_localization_save_map"] = $robot_dictionary["prefix"]."/robot_local_control/LocalizationComponent/save_map";
$robot_dictionary["rlc_localization_ns"] = $robot_dictionary["prefix"]."/robot_local_control/LocalizationComponent/";


$robot_dictionary["web_video_server_port"]="8081";

$robot_dictionary["topic_complex_sequencer_status"]=$robot_dictionary["prefix"]."/complex_sequencer/status";
$robot_dictionary["topic_command_sequencer_status"]=$robot_dictionary["prefix"]."/command_sequencer/status";


$robot_dictionary["param_rlc_localization_components"]=$robot_dictionary["prefix"]."/robot_local_control/LocalizationComponent/components";

$robot_dictionary["service_command_sequencer_get_sequences"]=$robot_dictionary["prefix"]."/command_sequencer/get_sequences";
$robot_dictionary["service_command_manager_get_handlers"]=$robot_dictionary["prefix"]."/command_manager/get_handlers";
$robot_dictionary["service_command_manager_get_handler_info"]=$robot_dictionary["prefix"]."/command_manager/get_handler_info";
$robot_dictionary["service_command_manager_command"]=$robot_dictionary["prefix"]."/command_manager/command";
$robot_dictionary["service_command_sequencer_add_sequence"]=$robot_dictionary["prefix"]."/command_sequencer/add_sequence";
$robot_dictionary["service_command_sequencer_cancel"]=$robot_dictionary["prefix"]."/command_sequencer/cancel";

$robot_dictionary["service_command_sequencer_remove_sequence"]=$robot_dictionary["prefix"]."/command_sequencer/remove_sequence";




$robot_dictionary["service_complex_sequencer_get_sequences"]=$robot_dictionary["prefix"]."/complex_sequencer/get_sequences";
$robot_dictionary["service_complex_sequencer_command"]=$robot_dictionary["prefix"]."/complex_sequencer/command";
$robot_dictionary["service_command_sequencer_command"]=$robot_dictionary["prefix"]."/command_sequencer/command";
$robot_dictionary["service_complex_sequencer_add_sequence"]=$robot_dictionary["prefix"]."/complex_sequencer/add_sequence";
$robot_dictionary["service_complex_sequencer_remove_sequence"]=$robot_dictionary["prefix"]."/complex_sequencer/remove_sequence";
$robot_dictionary["service_complex_sequencer_cancel"]=$robot_dictionary["prefix"]."/complex_sequencer/cancel";


$robot_dictionary["service_get_scheduler"]=$robot_dictionary["prefix"]."/command_scheduler/get_schedules";
$robot_dictionary["service_add_scheduler"]=$robot_dictionary["prefix"]."/command_scheduler/add_schedule";
$robot_dictionary["service_remove_scheduler"]=$robot_dictionary["prefix"]."/command_scheduler/remove_schedule";


?>
